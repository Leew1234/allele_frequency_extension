"""Find out where a calibration miss comes from (e.g. criterion S1: observed /
expected variant sites on unannotated sequence outside 0.95-1.05).

Two commands. Neither needs a GPU.

  blocks  uses the per-chromosome block files a finished run already has
          (WORK/blocks/<half>/<config>/*.npz from score_annotations.py). For
          one category it prints observed/expected per chromosome with a
          block-bootstrap interval, the pooled value, the pooled value with
          each chromosome left out, and the spread across 100 kb blocks. This
          answers: is the miss everywhere, or is it a few chromosomes?

  sites   reads labels and predictions for one chromosome and breaks
          observed/expected down by (a) decile of the predicted probability,
          (b) position along the chromosome, (c) local GC content (with
          --dna). This answers: is the miss at particular mutability levels,
          in particular regions, or tied to base composition? With
          --calibration it first applies a calibrate.py map, so the
          calibration chromosome's own uncalibrated prediction (cal_*.npy)
          can be checked against its own fit.

A value near 1 in every row of the calibration chromosome but not on the
test chromosomes means the calibration fit is fine and the two sets of
chromosomes differ (large-scale variation in mutation rate or in the
neutral mask's composition). A miss confined to some deciles means the
isotonic map is extrapolating. A miss confined to some regions points at
those regions.

  python diagnose_calibration.py blocks --blocks WORK/blocks/half1/neutral/*.npz --category unannotated
  python diagnose_calibration.py sites --labels arrays/NC_000004.12_frequency_labels.npy \\
      --predictions pred/half1/neutral/NC_000004.12.npy --mask masks/NC_000004.12_neutralA.npy \\
      --dna arrays/NC_000004.12_dna.npy
"""
import argparse
import json
import numpy as np

from compare_ovep import VariantScore

CHUNK = 5_000_000


def _ratio_ci(obs, exp, n_boot=1000, seed=0):
    """Observed/expected over blocks with a block-bootstrap 95% interval."""
    obs, exp = np.asarray(obs, float), np.asarray(exp, float)
    keep = exp > 0
    obs, exp = obs[keep], exp[keep]
    if obs.size == 0:
        return None, None
    point = float(obs.sum() / exp.sum())
    if obs.size < 2 or not n_boot:
        return point, None
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(obs.size, np.full(obs.size, 1.0 / obs.size), size=n_boot)
    r = (draws @ obs) / (draws @ exp)
    return point, [float(x) for x in np.percentile(r, [2.5, 97.5])]


def diagnose_blocks(files, category, model='new', n_boot=1000, seed=0):
    """Per-chromosome and leave-one-out observed/expected from v1 block files."""
    from compare_ovep import load_blocks
    col = {'new': 2, 'original': 1}[model]
    rows, all_obs, all_exp = [], [], []
    for path in files:
        sums, meta = load_blocks(path)
        if category not in sums:
            raise ValueError(f'{path}: no category {category!r} (have: {", ".join(sorted(sums))})')
        b = sums[category]['blocks']
        if model == 'original' and not sums[category]['has_reference']:
            raise ValueError(f'{path}: no reference model in this file')
        obs, exp = b[:, 0], b[:, col]
        point, ci = _ratio_ci(obs, exp, n_boot, seed)
        rows.append({'file': path, 'seqid': meta.get('seqid'), 'n_blocks': int(len(b)),
                     'observed': int(round(obs.sum())), 'expected': float(exp.sum()),
                     'oe': point, 'ci95': ci})
        all_obs.append(obs)
        all_exp.append(exp)
    obs, exp = np.concatenate(all_obs), np.concatenate(all_exp)
    pooled, ci = _ratio_ci(obs, exp, n_boot, seed)
    for i, r in enumerate(rows):
        o = sum(a.sum() for j, a in enumerate(all_obs) if j != i)
        x = sum(a.sum() for j, a in enumerate(all_exp) if j != i)
        r['pooled_without_this'] = float(o / x) if x > 0 else None
    with np.errstate(divide='ignore', invalid='ignore'):
        per_block = obs[exp > 0] / exp[exp > 0]
    oes = np.array([r['oe'] for r in rows if r['oe'] is not None])
    return {'category': category, 'model': model, 'pooled_oe': pooled, 'pooled_ci95': ci,
            'chromosomes': rows,
            'across_chromosomes': {'min': float(oes.min()), 'max': float(oes.max()),
                                   'sd': float(oes.std(ddof=1)) if oes.size > 1 else None},
            'per_block_oe_quantiles_5_25_50_75_95': [float(x) for x in
                                                     np.percentile(per_block, [5, 25, 50, 75, 95])]}


def _gc_fraction(dna, s, e, half):
    """Share of G/C in the window of +/- `half` bp around each site of [s, e).
    Channel order A, C, T, G: columns 1 and 3 are C and G."""
    n = len(dna)
    a, b = max(0, s - half), min(n, e + half)
    x = np.asarray(dna[a:b])
    gc = np.r_[0, np.cumsum(x[:, 1].astype(np.int64) + x[:, 3])]
    known = np.r_[0, np.cumsum(x.max(axis=1).astype(np.int64))]
    lo = np.clip(np.arange(s, e) - half, a, b) - a
    hi = np.clip(np.arange(s, e) + half + 1, a, b) - a
    with np.errstate(divide='ignore', invalid='ignore'):
        return (gc[hi] - gc[lo]) / (known[hi] - known[lo])


def diagnose_sites(labels, predictions, mask=None, dna=None, calibration=None, n_bins=10,
                   region_bp=10_000_000, gc_half_window=50, sample=2_000_000, seed=0, chunk=CHUNK):
    """Observed/expected variant sites by predicted-probability decile, by
    region, and by local GC content."""
    n = len(labels)
    if len(predictions) != n:
        raise ValueError('labels and predictions differ in length')
    if calibration is not None:
        from calibrate import calibrate_classes
    score = None if calibration is not None else VariantScore(predictions)

    def chunk_values(s, e):
        lab = np.asarray(labels[s:e])
        if calibration is not None:
            pv = VariantScore(calibrate_classes(np.asarray(predictions[s:e]), calibration))[0:e - s]
        else:
            pv = score[s:e]
        ok = (lab >= 0) & np.isfinite(pv)
        if mask is not None:
            ok &= np.asarray(mask[s:e], dtype=bool)
        return lab, pv, ok

    # pass 1: decile edges from a subsample of eligible sites
    rng = np.random.default_rng(seed)
    total = 0
    parts = []
    for s in range(0, n, chunk):
        lab, pv, ok = chunk_values(s, min(s + chunk, n))
        total += int(ok.sum())
        v = pv[ok]
        parts.append(v[rng.random(v.size) < min(1.0, sample / max(n, 1) * 4)])
    if total == 0:
        raise ValueError('no scorable sites')
    edges = np.unique(np.quantile(np.concatenate(parts), np.arange(1, n_bins) / n_bins))
    gc_edges = np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    n_regions = -(-n // region_bp)
    acc = {'decile': np.zeros((len(edges) + 1, 3)), 'region': np.zeros((n_regions, 3)),
           'gc': np.zeros((len(gc_edges) + 1, 3))}
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        lab, pv, ok = chunk_values(s, e)
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            continue
        var = (lab[idx] >= 1).astype(float)
        keys = {'decile': np.searchsorted(edges, pv[idx], side='right'),
                'region': (s + idx) // region_bp}
        if dna is not None:
            gc = _gc_fraction(dna, s, e, gc_half_window)[idx]
            keys['gc'] = np.searchsorted(gc_edges, np.nan_to_num(gc, nan=0.0), side='right')
        for name, key in keys.items():
            m = len(acc[name])
            acc[name][:, 0] += np.bincount(key, minlength=m)
            acc[name][:, 1] += np.bincount(key, weights=var, minlength=m)
            acc[name][:, 2] += np.bincount(key, weights=pv[idx], minlength=m)

    def rows(a, label):
        out = []
        for i, (cnt, obs, exp) in enumerate(a):
            if cnt == 0:
                continue
            out.append({'bin': label(i), 'n': int(cnt), 'observed': int(obs), 'expected': float(exp),
                        'oe': float(obs / exp) if exp > 0 else None})
        return out

    dec_lab = lambda i: (f'<= {edges[0]:.4g}' if i == 0 else f'> {edges[-1]:.4g}' if i == len(edges)
                         else f'{edges[i - 1]:.4g} - {edges[i]:.4g}')
    gc_lab = lambda i: (f'< {gc_edges[0]:.2f}' if i == 0 else f'>= {gc_edges[-1]:.2f}' if i == len(gc_edges)
                        else f'{gc_edges[i - 1]:.2f} - {gc_edges[i]:.2f}')
    tot = acc['decile'].sum(axis=0)
    res = {'n': int(tot[0]), 'observed': int(tot[1]), 'expected': float(tot[2]),
           'oe': float(tot[1] / tot[2]) if tot[2] > 0 else None,
           'by_predicted_probability': rows(acc['decile'], dec_lab),
           'by_region': rows(acc['region'], lambda i: f'{i * region_bp / 1e6:.0f}-{(i + 1) * region_bp / 1e6:.0f} Mb')}
    if dna is not None:
        res['by_gc_content'] = rows(acc['gc'], gc_lab)
    return res


def _print_table(title, rows, cols):
    print(f'\n{title}')
    print('  ' + '  '.join(f'{c:>14}' for c in cols))
    for r in rows:
        print('  ' + '  '.join(f'{(f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c))):>14}'
                               for c in cols))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('blocks', help='per-chromosome breakdown from existing block files')
    b.add_argument('--blocks', nargs='+', required=True)
    b.add_argument('--category', default='unannotated')
    b.add_argument('--model', choices=['new', 'original'], default='new')
    b.add_argument('--bootstrap', type=int, default=1000)
    b.add_argument('--json')
    s = sub.add_parser('sites', help='breakdown by probability, region and GC for one chromosome')
    s.add_argument('--labels', required=True)
    s.add_argument('--predictions', required=True, help='1-D scores or class probabilities')
    s.add_argument('--mask')
    s.add_argument('--dna', help='one-hot .npy: adds the GC breakdown')
    s.add_argument('--calibration', help='apply this calibrate.py map first (for uncalibrated cal_*.npy)')
    s.add_argument('--bins', type=int, default=10)
    s.add_argument('--region-mb', type=float, default=10.0)
    s.add_argument('--json')
    args = p.parse_args()

    if args.cmd == 'blocks':
        res = diagnose_blocks(args.blocks, args.category, args.model, args.bootstrap)
        print(f"{res['category']} ({res['model']} model): pooled observed/expected = "
              f"{res['pooled_oe']:.4f}, 95% interval {res['pooled_ci95']}")
        _print_table('Per chromosome', res['chromosomes'],
                     ['seqid', 'n_blocks', 'observed', 'oe', 'pooled_without_this'])
        print(f"\nAcross chromosomes: {res['across_chromosomes']}")
        print(f"Per-block observed/expected, quantiles 5/25/50/75/95: "
              f"{[round(x, 3) for x in res['per_block_oe_quantiles_5_25_50_75_95']]}")
    else:
        cal = None
        if args.calibration:
            with open(args.calibration) as f:
                cal = json.load(f)
        res = diagnose_sites(np.load(args.labels, mmap_mode='r'), np.load(args.predictions, mmap_mode='r'),
                             np.load(args.mask, mmap_mode='r') if args.mask else None,
                             np.load(args.dna, mmap_mode='r') if args.dna else None, cal,
                             args.bins, int(args.region_mb * 1e6))
        print(f"{res['n']:,} sites: observed/expected = {res['oe']:.4f}")
        _print_table('By predicted probability', res['by_predicted_probability'], ['bin', 'n', 'observed', 'oe'])
        _print_table('By region', res['by_region'], ['bin', 'n', 'observed', 'oe'])
        if 'by_gc_content' in res:
            _print_table('By GC content (+/- 50 bp)', res['by_gc_content'], ['bin', 'n', 'observed', 'oe'])
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
