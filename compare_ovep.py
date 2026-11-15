"""Like-for-like comparison of a reference model and a new model on one
chromosome: the paper's original binary OVEP model vs this extension, or two
configurations of the extension (e.g. whole-genome vs neutral-trained).

Both models are scored against the SAME labels (variant = label >= 1 from
data.py), on the SAME sites (scorable under both), inside the SAME masks, with
the SAME genomic blocks for a paired bootstrap. Per mask it reports:

  oe_original, oe_new   observed / expected variant sites (OVEP) under the
                        reference ("original") and new model
  oe_diff               oe_new - oe_original, with a paired 95% interval
  auc_original, auc_new site-level ROC-AUC for variant vs no variant
  oe_common_share       observed common variants / expected common variants
                        given where variants occurred (sum of the new model's
                        P(common | variant) over observed variant sites)
  se, z                 bootstrap standard errors; z = (ratio - 1) / se
                        (for oe_diff: diff / se)

Inputs:
  --original      reference predictions: the original repo's 1-D .npy (e.g.
                  calibrated_predictions/NC_000002.12.pred.npy; usually ~2048
                  shorter than the chromosome, head-aligned, tail excluded),
                  or another (n_sites, 3) class-probability .npy from predict.py.
  --original-raw  optional, for a 1-D original: its UNcalibrated prediction.
                  The original predict_by_chrom_fast.py leaves the first 512 bp
                  and the last partial batch unpredicted (exactly 0 before
                  calibration); those sites are excluded.
  --new           class probabilities .npy from predict.py.
  --labels        data.py labels. To mimic the original label definition,
                  build them with --indel-mode span --keep-non-pass.
  --mask NAME=PATH  boolean .npy, repeatable (e.g. cds=..., unannotated=...).
  --save-blocks   write per-block sums (.npz) so several chromosomes can be
                  pooled later: score_annotations.py --pool a.npz b.npz ...

Example:
  python compare_ovep.py --labels arrays/chr2_frequency_labels.npy \\
      --original calibrated_predictions/NC_000002.12.pred.npy \\
      --original-raw code_halftrainmaskmissing/base_excludeN_final_extrapolate/NC_000002.12.pred.npy \\
      --new pred/chr2.npy --mask cds=masks/chr2_cds.npy --mask unannotated=masks/chr2_neutral.npy
"""
import argparse
import json
import os
import numpy as np

CHUNK = 5_000_000
COLUMNS = ('obs_variant', 'exp_original', 'exp_new', 'obs_common', 'exp_new_common',
           'exp_common_at_variants', 'var_common_at_variants')
# exp_common_at_variants = sum of P(common | variant) over the sites where a
# variant was observed; var_common_at_variants = sum of p(1-p) there (the
# Poisson-binomial variance). The frequency expectation is conditioned on where
# variants actually occurred, so errors in the mutability model cannot leak in.


class VariantScore:
    """Slice-able per-site P(variant): a 1-D score as is, or the sum of the
    variant classes from class probabilities ((n, 3) from the v1 heads, or
    (n, 1 + K) from the spectrum head). Works on memory-mapped arrays.

    kind='singleton_share' instead returns P(singleton | variant) from
    spectrum class probabilities (for descriptive use, e.g. how the expected
    frequency spectrum varies with mutability)."""

    def __init__(self, arr, kind='variant'):
        self.arr, self.kind = arr, kind
        if arr.ndim == 2 and arr.shape[1] < 3:
            raise ValueError('2-D predictions must have at least 3 columns (class probabilities)')
        if arr.ndim not in (1, 2):
            raise ValueError('predictions must be 1-D scores or (n, classes) class probabilities')
        if kind not in ('variant', 'singleton_share'):
            raise ValueError("kind must be 'variant' or 'singleton_share'")
        if kind == 'singleton_share' and (arr.ndim != 2 or arr.shape[1] < 4):
            raise ValueError('singleton_share needs spectrum class probabilities')

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, sl):
        a = np.asarray(self.arr[sl], dtype=np.float64)
        if a.ndim == 1:
            return a
        pv = a[:, 1] + a[:, 2] if a.shape[1] == 3 else a[:, 1:].sum(axis=1)
        if self.kind == 'variant':
            return pv
        with np.errstate(invalid='ignore', divide='ignore'):
            share = np.where(pv > 0, a[:, 1] / pv, 0.0)   # 0, not NaN: the site stays scorable
        share[~np.isfinite(pv)] = np.nan
        return share


class IntervalMask:
    """Boolean mask defined by merged half-open intervals, built per slice
    (no chromosome-length array)."""

    def __init__(self, intervals, length):
        iv = sorted((max(0, int(a)), min(length, int(b))) for a, b in intervals if b > a)
        merged = []
        for a, b in iv:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            elif a < b:
                merged.append([a, b])
        self.starts = np.array([m[0] for m in merged], dtype=np.int64)
        self.ends = np.array([m[1] for m in merged], dtype=np.int64)
        self.length = length

    def __len__(self):
        return self.length

    @property
    def n_sites(self):
        return int((self.ends - self.starts).sum())

    def complement(self):
        edges = np.r_[0, np.ravel(np.column_stack([self.starts, self.ends])), self.length]
        return IntervalMask([(a, b) for a, b in zip(edges[0::2], edges[1::2])], self.length)

    def __getitem__(self, sl):
        s, e, _ = sl.indices(self.length)
        out = np.zeros(max(e - s, 0), dtype=bool)
        i0 = np.searchsorted(self.ends, s, side='right')
        i1 = np.searchsorted(self.starts, e, side='left')
        for a, b in zip(self.starts[i0:i1], self.ends[i0:i1]):
            out[max(a, s) - s:min(b, e) - s] = True
        return out


def roc_auc(y, s):
    """ROC-AUC via average ranks (ties handled), NumPy only."""
    y = np.asarray(y, bool)
    s = np.asarray(s, np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = midrank(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def midrank(s):
    """1-based ranks with ties given their average rank."""
    s = np.asarray(s, np.float64)
    order = np.argsort(s, kind='mergesort')
    s_sorted = s[order]
    first = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    group = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], s.size]
    ranks = np.empty(s.size)
    ranks[order] = ((starts + ends + 1) / 2.0)[group]
    return ranks


def block_sums(labels, original, new, masks, original_raw=None, block_size=100_000,
               auc_sites=2_000_000, seed=0, chunk=CHUNK, unit_sites=None):
    """Per-mask, per-block sums (see COLUMNS) and an AUC subsample.
    `original` may be None (then only the new model is scored).
    With `unit_sites`, also sums over consecutive runs of exactly that many
    eligible sites inside each mask ("units", for discriminate.py); the last
    incomplete unit of the chromosome is dropped."""
    n = len(labels)
    if len(new) != n:
        raise ValueError(f'new predictions ({len(new)}) and labels ({n}) differ in length')
    if new.ndim != 2 or new.shape[1] != 3:
        raise ValueError('new predictions must be (n, 3) class probabilities; collapse a spectrum '
                         'model first (python spectrum.py predictions/labels ...) or use spectrum_score.py')
    ref = VariantScore(original) if original is not None else None
    if original_raw is not None and (original is None or len(original_raw) != len(original)):
        raise ValueError('original and original-raw differ in length')
    for name, m in masks.items():
        if len(m) != n:
            raise ValueError(f'mask {name!r} ({len(m)}) and labels ({n}) differ in length')
    n_ref = min(len(ref), n) if ref is not None else n
    n_blocks = -(-n // block_size)
    names = list(masks)
    acc = {k: np.zeros((n_blocks, len(COLUMNS))) for k in names}
    count = {k: np.zeros(n_blocks, np.int64) for k in names}
    auc_buf = {k: ([], [], []) for k in names}
    unit_seen = {k: 0 for k in names}
    unit_parts = {k: [] for k in names}
    unit_starts = {k: [] for k in names}
    rate = min(1.0, auc_sites / max(n, 1))
    rng = np.random.default_rng(seed)

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        lab = np.asarray(labels[s:e])
        pn = np.asarray(new[s:e], dtype=np.float64)
        if ref is None:
            po = np.zeros(e - s)
        else:
            po = np.full(e - s, np.nan)
            if s < n_ref:
                po[:min(e, n_ref) - s] = ref[s:min(e, n_ref)]
                if original_raw is not None:
                    raw = np.asarray(original_raw[s:min(e, n_ref)], dtype=np.float64)
                    po[:raw.size][raw == 0] = np.nan
        ok = (lab >= 0) & np.isfinite(po) & np.isfinite(pn).all(1)
        pick = rng.random(e - s) < rate
        for k in names:
            idx = np.flatnonzero(ok & np.asarray(masks[k][s:e], dtype=bool))
            if idx.size == 0:
                continue
            block = (s + idx) // block_size
            var = lab[idx] >= 1
            pv = pn[idx, 1] + pn[idx, 2]
            count[k] += np.bincount(block, minlength=n_blocks)
            with np.errstate(divide='ignore', invalid='ignore'):
                pc = np.where(pv > 0, pn[idx, 2] / pv, 0.0)
            pc_var = np.where(var, pc, 0.0)
            cols = (var, po[idx], pv, lab[idx] == 2, pn[idx, 2], pc_var, pc_var * (1 - pc_var))
            for j, w in enumerate(cols):
                acc[k][:, j] += np.bincount(block, weights=w, minlength=n_blocks)
            if unit_sites:
                uid = (unit_seen[k] + np.arange(idx.size)) // unit_sites
                u0 = uid[0]
                part = np.stack([np.bincount(uid - u0, weights=w) for w in cols], axis=1)
                unit_parts[k].append((u0, part))
                k_in_mask = unit_seen[k] + np.arange(idx.size)
                unit_starts[k].append(s + idx[k_in_mask % unit_sites == 0])
                unit_seen[k] += idx.size
            a = pick[idx]
            auc_buf[k][0].append(var[a]); auc_buf[k][1].append(po[idx][a]); auc_buf[k][2].append(pv[a])

    out = {}
    for k in names:
        keep = count[k] > 0
        cat = lambda parts, dt: np.concatenate(parts) if parts else np.zeros(0, dt)
        out[k] = {'blocks': acc[k][keep], 'counts': count[k][keep],
                  'auc_y': cat(auc_buf[k][0], bool), 'auc_ref': cat(auc_buf[k][1], float),
                  'auc_new': cat(auc_buf[k][2], float), 'has_reference': ref is not None}
        if unit_sites:
            n_full = unit_seen[k] // unit_sites
            units = np.zeros((n_full, len(COLUMNS)))
            for u0, part in unit_parts[k]:
                hi = min(u0 + len(part), n_full)
                if hi > u0:
                    units[u0:hi] += part[:hi - u0]
            out[k]['units'] = units
            starts = np.concatenate(unit_starts[k]) if unit_starts[k] else np.zeros(0, np.int64)
            out[k]['units_pos'] = starts[:n_full].astype(np.int64)  # first site of each unit
    return out


def pool(entries):
    """Concatenate block sums / AUC samples of the same mask across chromosomes."""
    entries = [e for e in entries if e is not None]
    if not entries:
        return None
    out = {'blocks': np.concatenate([e['blocks'] for e in entries]),
           'counts': np.concatenate([e['counts'] for e in entries]),
           'auc_y': np.concatenate([e['auc_y'] for e in entries]),
           'auc_ref': np.concatenate([e['auc_ref'] for e in entries]),
           'auc_new': np.concatenate([e['auc_new'] for e in entries]),
           'has_reference': all(e['has_reference'] for e in entries)}
    if all('units' in e for e in entries):
        out['units'] = np.concatenate([e['units'] for e in entries])
        if all('units_pos' in e for e in entries):
            out['units_pos'] = np.concatenate([e['units_pos'] for e in entries])
            # entry index = chromosome identity when pooling per-chromosome files
            out['units_chrom'] = np.concatenate([np.full(len(e['units']), i) for i, e in enumerate(entries)])
    return out


def _ratios(t):
    """t: (..., 5) column sums -> dict of ratio arrays."""
    with np.errstate(divide='ignore', invalid='ignore'):
        oo, on = t[..., 0] / t[..., 1], t[..., 0] / t[..., 2]
        return {'oe_original': oo, 'oe_new': on, 'oe_diff': on - oo,
                'oe_common_share': t[..., 3] / t[..., 5],
                'oe_common_share_marginal': (t[..., 3] / t[..., 0]) / (t[..., 4] / t[..., 2])}


def _f(v):
    v = float(v)
    return v if np.isfinite(v) else None


def summarize(entry, n_boot=1000, seed=0):
    if entry is None or len(entry['blocks']) == 0:
        return {'n': 0}
    b = entry['blocks']
    tot = b.sum(0)
    point = _ratios(tot)
    has_ref = entry['has_reference']
    r = {'n': int(entry['counts'].sum()), 'observed_variants': int(round(tot[0])),
         'expected_new': float(tot[2]), 'oe_new': _f(point['oe_new']),
         'oe_common_share': _f(point['oe_common_share']),
         'oe_common_share_marginal': _f(point['oe_common_share_marginal'])}
    if has_ref:
        r.update({'expected_original': float(tot[1]), 'oe_original': _f(point['oe_original']),
                  'oe_diff': _f(point['oe_diff'])})
    y = entry['auc_y']
    r['auc_sites'] = int(y.size)
    r['auc_new'] = roc_auc(y, entry['auc_new']) if y.size else None
    if has_ref:
        r['auc_original'] = roc_auc(y, entry['auc_ref']) if y.size else None
    if n_boot and len(b) > 1:
        rng = np.random.default_rng(seed)
        draws = rng.multinomial(len(b), np.full(len(b), 1 / len(b)), size=n_boot)
        boot = _ratios(draws @ b)
        keys = ['oe_new', 'oe_common_share'] + (['oe_original', 'oe_diff'] if has_ref else [])
        r['ci95'], r['se'], r['z'] = {}, {}, {}
        for k in keys:
            v = boot[k][np.isfinite(boot[k])]
            if v.size < 2 or r.get(k) is None:
                continue
            r['ci95'][k] = [float(x) for x in np.percentile(v, [2.5, 97.5])]
            se = float(v.std(ddof=1))
            r['se'][k] = se
            centre = 0.0 if k == 'oe_diff' else 1.0
            r['z'][k] = (r[k] - centre) / se if se > 0 else None
        r['n_blocks'] = int(len(b))
    return r


def compare(labels, original, new, masks, original_raw=None, block_size=100_000,
            n_boot=1000, auc_sites=2_000_000, seed=0, chunk=CHUNK):
    sums = block_sums(labels, original, new, masks, original_raw, block_size, auc_sites, seed, chunk)
    return {k: summarize(v, n_boot, seed) for k, v in sums.items()}


def save_blocks(path, sums, meta=None):
    arrays = {}
    for k, v in sums.items():
        for field in ('blocks', 'counts', 'auc_y', 'auc_ref', 'auc_new', 'units', 'units_pos'):
            if field in v:
                arrays[f'{k}::{field}'] = v[field]
        arrays[f'{k}::has_reference'] = np.array(v['has_reference'])
    arrays['__meta__'] = np.array(json.dumps(meta or {}))
    tmp = f'{path}.partial'
    with open(tmp, 'wb') as f:  # atomic: a killed run never leaves a truncated .npz
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def load_blocks(path):
    z = np.load(path)
    out = {}
    for key in z.files:
        if key == '__meta__':
            continue
        name, field = key.rsplit('::', 1)
        out.setdefault(name, {})[field] = bool(z[key]) if field == 'has_reference' else z[key]
    return out, json.loads(str(z['__meta__'])) if '__meta__' in z.files else {}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labels', required=True)
    p.add_argument('--original', required=True,
                   help="reference predictions: original repo's 1-D .npy, or (n,3) from predict.py")
    p.add_argument('--original-raw', help='uncalibrated original prediction (to drop unpredicted sites)')
    p.add_argument('--new', required=True, help='class probabilities .npy from predict.py')
    p.add_argument('--mask', action='append', default=[], metavar='NAME=PATH')
    p.add_argument('--block-size', type=int, default=100_000)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--auc-sites', type=int, default=2_000_000, help='random subsample size for AUC')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save-blocks', help='write per-block sums to this .npz for pooling')
    args = p.parse_args()
    labels = np.load(args.labels, mmap_mode='r')
    masks = {'all_scorable': np.ones(len(labels), bool)}
    for item in args.mask:
        name, _, path = item.partition('=')
        if not path:
            p.error(f'--mask expects NAME=PATH, got {item!r}')
        masks[name] = np.load(path, mmap_mode='r')
    sums = block_sums(labels, np.load(args.original, mmap_mode='r'), np.load(args.new, mmap_mode='r'),
                      masks, np.load(args.original_raw, mmap_mode='r') if args.original_raw else None,
                      args.block_size, args.auc_sites, args.seed)
    if args.save_blocks:
        save_blocks(args.save_blocks, sums, {'labels': args.labels, 'original': args.original, 'new': args.new})
    print(json.dumps({k: summarize(v, args.bootstrap, args.seed) for k, v in sums.items()}, indent=2))


if __name__ == '__main__':
    main()
