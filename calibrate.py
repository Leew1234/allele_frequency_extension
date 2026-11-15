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

Spectrum-head predictions (1 + K columns, analysis plan v2) are calibrated
the same way for P(variant); the class distribution given a variant is
calibrated through its cumulative tails, one isotonic map per boundary:
  P(class >= k | variant), k = 2..K, target label >= k, variant sites only
The classes are ordered by frequency, so this keeps the order, and the two
tails the scores rely on are each calibrated exactly as a binary event:
k = 2 is 1 - P(singleton | variant), and k = K is P(common | variant), the
same quantity (and the same fit) as the 3-class calibration above.

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
    # Bins are in order of x, so their means cannot really decrease; with many
    # tied scores (a k-mer table has few distinct values) rounding in the mean
    # can still make them differ in the last digit, in either direction. Remove
    # that, so tied bins merge below and stay in the order PAV fitted them in.
    bx = np.maximum.accumulate(bx)
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


def spectrum_tails(cond):
    """P(class >= k | variant) for k = 2..K from (..., K) conditional class
    probabilities -> (..., K-1)."""
    cond = np.asarray(cond, dtype=np.float64)
    return np.cumsum(cond[..., ::-1], axis=-1)[..., ::-1][..., 1:]


def calibrate_spectrum(probs, calibration):
    """Apply a spectrum calibration to (n, 1+K) class probabilities."""
    from spectrum import conditional_classes, variant_probability
    probs = np.asarray(probs, dtype=np.float64)
    pv = apply_isotonic(variant_probability(probs), calibration['variant'])
    tails = spectrum_tails(conditional_classes(probs))
    cal = np.stack([apply_isotonic(tails[..., j], m)
                    for j, m in enumerate(calibration['class_at_least'])], axis=-1)
    # tails must not increase with k; a crossing can only come from fitting
    # the maps separately and is removed by taking the running minimum
    cal = np.minimum.accumulate(np.clip(cal, 0.0, 1.0), axis=-1)
    edges = np.concatenate([np.ones_like(cal[..., :1]), cal, np.zeros_like(cal[..., :1])], axis=-1)
    cond = edges[..., :-1] - edges[..., 1:]
    return np.concatenate([(1 - pv)[..., None], pv[..., None] * cond], axis=-1).astype(np.float32)


def calibrate_classes(probs, calibration):
    if calibration.get('scheme') == 'spectrum':
        return calibrate_spectrum(probs, calibration)
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


def fit_calibration_spectrum(labels, probs, max_sites=20_000_000, n_bins=1000, seed=0,
                             mask=None, chunk=CHUNK):
    """Spectrum version of fit_calibration: P(variant) on all scorable sites,
    and one map per cumulative tail P(class >= k | variant) on variant sites."""
    from spectrum import conditional_classes, variant_probability
    n = len(labels)
    k = probs.shape[1] - 1
    if len(probs) != n:
        raise ValueError('labels and predictions differ in length')
    if mask is not None and len(mask) != n:
        raise ValueError('mask and labels differ in length')
    total = sum(int(_eligible(labels, probs, mask, s, s + chunk).sum()) for s in range(0, n, chunk))
    if total == 0:
        raise ValueError('no scorable sites')
    rate = min(1.0, max_sites / total)
    rng = np.random.default_rng(seed)
    pv_parts, y_parts, tail_parts, cls_parts = [], [], [], []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        ok = _eligible(labels, probs, mask, s, e)
        if rate < 1:
            ok &= rng.random(e - s) < rate
        idx = np.flatnonzero(ok)
        pr = np.asarray(probs[s:e])[idx]
        lab = np.asarray(labels[s:e])[idx]
        if lab.size and lab.max() > k:
            raise ValueError(f'labels contain class {int(lab.max())} but predictions have {k} variant classes')
        pv_parts.append(variant_probability(pr))
        y_parts.append(lab >= 1)
        var = lab >= 1
        tail_parts.append(spectrum_tails(conditional_classes(pr[var])))
        cls_parts.append(lab[var])
    pv, var = np.concatenate(pv_parts), np.concatenate(y_parts)
    tails, cls = np.concatenate(tail_parts), np.concatenate(cls_parts)
    if cls.size == 0:
        raise ValueError('no variant sites to calibrate the class distribution')
    v = fit_isotonic(pv, var, n_bins)
    maps = []
    for j in range(k - 1):
        m = fit_isotonic(tails[:, j], cls >= j + 2, n_bins)
        maps.append([m[0].tolist(), m[1].tolist()])
    return {'scheme': 'spectrum', 'n_classes': k + 1,
            'variant': [v[0].tolist(), v[1].tolist()], 'class_at_least': maps,
            'n_sites': int(var.size), 'n_variant_sites': int(cls.size),
            'class_counts': [int((cls == c).sum()) for c in range(1, k + 1)],
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
    preds = load_class_probabilities(args.predictions)
    fit = fit_calibration if preds.shape[1] == 3 else fit_calibration_spectrum
    cal = fit(np.load(args.labels, mmap_mode='r'), preds, args.max_sites, args.bins,
              mask=np.load(args.mask, mmap_mode='r') if args.mask else None)
    with open(args.out + '.partial', 'w') as f:
        json.dump(cal, f)
    os.replace(args.out + '.partial', args.out)
    print(f"Saved {args.out} ({cal['n_sites']:,} sites, {cal['n_variant_sites']:,} variant sites)")


if __name__ == '__main__':
    main()
