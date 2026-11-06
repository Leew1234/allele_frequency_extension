"""Observed/expected ratios by variant frequency category, with block-bootstrap
95% intervals (candidate selection diagnostics, not proof of selection).

Reported ratios:
  oe_any           (rare + common observed) / expected      ~ the original OVEP
  oe_rare          rare observed / expected
  oe_common        common observed / expected
  oe_common_share  observed common variants / expected common variants given
                   where variants occurred (sum of P(common | variant) over
                   the observed variant sites). Purifying selection pushes
                   variants toward low frequency, so this falls below 1 under
                   constraint. Conditioning on the variant sites keeps errors
                   in the mutability model out of it.
  oe_common_share_marginal  the older, unconditional version (observed common
                   share / sum P(common) / sum P(variant)); biased when the
                   mutability model is imperfect, reported for reference.

Intervals resample contiguous genomic blocks (default 100 kb) to respect
linkage and local correlation. They cover sampling noise in the scored region
only, not model, seed or calibration uncertainty; compare ensemble members or
calibration choices for those.

Arrays are read in chunks, so memory-mapped chromosome-scale inputs are fine.
"""
import argparse
import json
import numpy as np

CHUNK = 5_000_000


def load_class_probabilities(path):
    """(n_sites, 3) class probabilities from predict.py output.
    .npy files are memory-mapped; .npz (older versions) are loaded."""
    if str(path).endswith('.npz'):
        return np.load(path)['class_probabilities']
    return np.load(path, mmap_mode='r')


def _ratios(obs, exp, exp_common_at_variants):
    """obs, exp: (..., 3) arrays; exp_common_at_variants: (...) -> dict of ratio arrays."""
    obs = np.asarray(obs, dtype=np.float64)
    exp = np.asarray(exp, dtype=np.float64)
    ecv = np.asarray(exp_common_at_variants, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        o_var, e_var = obs[..., 1] + obs[..., 2], exp[..., 1] + exp[..., 2]
        return {'oe_any': o_var / e_var,
                'oe_rare': obs[..., 1] / exp[..., 1],
                'oe_common': obs[..., 2] / exp[..., 2],
                'oe_common_share': obs[..., 2] / ecv,
                'oe_common_share_marginal': (obs[..., 2] / o_var) / (exp[..., 2] / e_var)}


def _clean(v):
    v = float(v)
    return v if np.isfinite(v) else None


def category_ratios(labels, predictions, mask=None, block_size=100_000,
                    n_boot=1000, seed=0, chunk=CHUNK):
    n = len(labels)
    if predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError('predictions must have shape (n_sites, 3)')
    if len(predictions) != n:
        raise ValueError(f'labels ({n}) and predictions ({len(predictions)}) differ in length')
    if mask is not None and len(mask) != n:
        raise ValueError(f'mask ({len(mask)}) and labels ({n}) differ in length')

    n_blocks = -(-n // block_size)
    b_obs = np.zeros((n_blocks, 3))
    b_exp = np.zeros((n_blocks, 3))
    b_ecv = np.zeros(n_blocks)
    b_n = np.zeros(n_blocks, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        lab = np.asarray(labels[s:e])
        pred = np.asarray(predictions[s:e], dtype=np.float64)
        ok = (lab >= 0) & np.all(np.isfinite(pred), axis=1)
        if mask is not None:
            ok &= np.asarray(mask[s:e], dtype=bool)
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            continue
        block = (s + idx) // block_size
        lab, pred = lab[idx].astype(int), pred[idx]
        b_n += np.bincount(block, minlength=n_blocks)
        for k in range(3):
            b_obs[:, k] += np.bincount(block[lab == k], minlength=n_blocks)
            b_exp[:, k] += np.bincount(block, weights=pred[:, k], minlength=n_blocks)
        pv = pred[:, 1] + pred[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            pc = np.where(pv > 0, pred[:, 2] / pv, 0.0)
        b_ecv += np.bincount(block, weights=np.where(lab >= 1, pc, 0.0), minlength=n_blocks)
    keep = b_n > 0
    if not keep.any():
        raise ValueError('No valid positions')
    b_obs, b_exp, b_ecv = b_obs[keep], b_exp[keep], b_ecv[keep]

    obs, exp, ecv = b_obs.sum(axis=0), b_exp.sum(axis=0), b_ecv.sum()
    point = _ratios(obs, exp, ecv)
    result = {'n': int(b_n.sum()), 'observed': obs.astype(int).tolist(),
              'expected': exp.tolist(),
              'observed_common_share': _clean(obs[2] / (obs[1] + obs[2])) if obs[1] + obs[2] > 0 else None,
              'expected_common_share': _clean(ecv / (obs[1] + obs[2])) if obs[1] + obs[2] > 0 else None,
              'expected_common_share_marginal': _clean(exp[2] / (exp[1] + exp[2])) if exp[1] + exp[2] > 0 else None}
    result.update({k: _clean(v) for k, v in point.items()})

    if n_boot:
        nb = len(b_obs)
        rng = np.random.default_rng(seed)
        draws = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=n_boot)  # (n_boot, nb)
        boot = _ratios(draws @ b_obs, draws @ b_exp, draws @ b_ecv)
        result['n_blocks'] = nb
        result['ci95'] = {}
        for k, v in boot.items():
            v = v[np.isfinite(v)]
            result['ci95'][k] = ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                 if v.size and nb > 1 else None)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labels', required=True)
    p.add_argument('--predictions', required=True, help='.npy from predict.py (or older .npz)')
    p.add_argument('--mask', help='Optional boolean .npy annotation mask aligned to the chromosome')
    p.add_argument('--block-size', type=int, default=100_000)
    p.add_argument('--bootstrap', type=int, default=1000, help='replicates; 0 disables intervals')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    mask = np.load(args.mask, mmap_mode='r') if args.mask else None
    print(json.dumps(category_ratios(np.load(args.labels, mmap_mode='r'),
                                     load_class_probabilities(args.predictions), mask,
                                     args.block_size, args.bootstrap, args.seed), indent=2))


if __name__ == '__main__':
    main()
