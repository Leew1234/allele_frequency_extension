"""Fit isotonic calibration maps on a held-out chromosome (NumPy only).

Mirrors the original pipeline's calibration step, but separately for the two
quantities the O/E scores depend on:
  P(variant)            fitted on all scorable sites, target label >= 1
  P(common | variant)   fitted on variant sites only, target label == 2

Use a chromosome that was used neither for training nor for the regions you
score (the original code held out chr1 / chr2 for this). Run predict.py on it
WITHOUT --calibration, then:

  python calibrate.py --labels arrays/chr1_frequency_labels.npy \
      --predictions chr1_pred.npy --out calibration.json [--mask masks/chr1_neutral.npy]

and pass --calibration calibration.json to predict.py for other chromosomes.
With --mask, the curves are fitted on putatively neutral sites only; compare
against a fit on the whole chromosome to see how much selected regions move
the baseline.

Sites are subsampled in chunks (--max-sites), so full chromosome-length
float64 arrays are never built.
"""
import argparse
import os
import json
import numpy as np

from score import load_class_probabilities

CHUNK = 5_000_000


def pav(y, w):
    """Pool-adjacent-violators: non-decreasing weighted fit of y."""
    vals, wts, sizes = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi)); wts.append(float(wi)); sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            wt = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wt
            n = sizes[-2] + sizes[-1]
            vals[-2:], wts[-2:], sizes[-2:] = [v], [wt], [n]
    return np.repeat(vals, sizes)


def fit_isotonic(x, y, n_bins=1000):
    """Isotonic map from scores x to event rate y, fitted on quantile bins.
    Returns (bin_x, fitted_y), to be applied with np.interp."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x)
    x, y = x[keep], y[keep]
    if x.size == 0:
        raise ValueError('no finite scores to calibrate')
    order = np.argsort(x, kind='stable')
    chunks = np.array_split(order, min(n_bins, x.size))
    bx = np.array([x[c].mean() for c in chunks])
    by = np.array([y[c].mean() for c in chunks])
    bw = np.array([len(c) for c in chunks], dtype=np.float64)
    fy = pav(by, bw)
    # merge bins with identical x so np.interp gets increasing x
    ux, inv = np.unique(bx, return_inverse=True)
    uy = np.bincount(inv, weights=fy * bw) / np.bincount(inv, weights=bw)
    return ux, uy


def apply_isotonic(x, mapping):
    bx, by = (np.asarray(v, dtype=np.float64) for v in mapping)
    x = np.asarray(x, dtype=np.float64)
    out = np.interp(x, bx, by)  # clamps outside the fitted range
    out[~np.isfinite(x)] = np.nan
    return out


def split_classes(probs):
    probs = np.asarray(probs, dtype=np.float64)
    pv = probs[..., 1] + probs[..., 2]
    with np.errstate(invalid='ignore', divide='ignore'):
        pc = np.where(pv > 0, probs[..., 2] / pv, 0.0)
    pc[~np.isfinite(pv)] = np.nan
    return pv, pc


def calibrate_classes(probs, calibration):
    pv, pc = split_classes(probs)
    pv = apply_isotonic(pv, calibration['variant'])
    pc = apply_isotonic(pc, calibration['common_given_variant'])
    return np.stack([1 - pv, pv * (1 - pc), pv * pc], axis=-1).astype(np.float32)


def _eligible(labels, probs, mask, s, e):
    ok = (np.asarray(labels[s:e]) >= 0) & np.isfinite(np.asarray(probs[s:e, 0]))
    if mask is not None:
        ok &= np.asarray(mask[s:e], dtype=bool)
    return ok


def fit_calibration(labels, probs, max_sites=20_000_000, n_bins=1000, seed=0,
                    mask=None, chunk=CHUNK):
    """Fit both isotonic maps. Arrays may be memory-mapped; they are read in
    chunks and at most ~max_sites sites are kept (uniform random subsample)."""
    n = len(labels)
    if len(probs) != n:
        raise ValueError('labels and predictions differ in length')
    if mask is not None and len(mask) != n:
        raise ValueError('mask and labels differ in length')
    total = sum(int(_eligible(labels, probs, mask, s, s + chunk).sum()) for s in range(0, n, chunk))
    if total == 0:
        raise ValueError('no scorable sites')
    rate = min(1.0, max_sites / total)
    rng = np.random.default_rng(seed)
    pv_parts, pc_parts, y_parts = [], [], []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        ok = _eligible(labels, probs, mask, s, e)
        if rate < 1:
            ok &= rng.random(e - s) < rate
        idx = np.flatnonzero(ok)
        pv, pc = split_classes(np.asarray(probs[s:e])[idx])
        pv_parts.append(pv); pc_parts.append(pc)
        y_parts.append(np.asarray(labels[s:e])[idx])
    pv, pc, y = np.concatenate(pv_parts), np.concatenate(pc_parts), np.concatenate(y_parts)
    var = y >= 1
    if not var.any():
        raise ValueError('no variant sites to calibrate P(common | variant)')
    v = fit_isotonic(pv, var, n_bins)
    c = fit_isotonic(pc[var], y[var] == 2, n_bins)
    return {'variant': [v[0].tolist(), v[1].tolist()],
            'common_given_variant': [c[0].tolist(), c[1].tolist()],
            'n_sites': int(y.size), 'n_variant_sites': int(var.sum()),
            'masked': mask is not None}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labels', required=True)
    p.add_argument('--predictions', required=True,
                   help='.npy from predict.py run without --calibration')
    p.add_argument('--mask', help='boolean .npy (e.g. neutral sites); fit only where True')
    p.add_argument('--out', required=True)
    p.add_argument('--max-sites', type=int, default=20_000_000)
    p.add_argument('--bins', type=int, default=1000)
    args = p.parse_args()
    cal = fit_calibration(np.load(args.labels, mmap_mode='r'),
                          load_class_probabilities(args.predictions),
                          args.max_sites, args.bins,
                          mask=np.load(args.mask, mmap_mode='r') if args.mask else None)
    with open(args.out + '.partial', 'w') as f:
        json.dump(cal, f)
    os.replace(args.out + '.partial', args.out)
    print(f"Saved {args.out} ({cal['n_sites']:,} sites, {cal['n_variant_sites']:,} variant sites)")


if __name__ == '__main__':
    main()
