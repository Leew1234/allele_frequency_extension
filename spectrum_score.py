"""Frequency-spectrum scores per sequence category (analysis plan v2).

The quantity of interest is the share of variants that are singletons,
adjusted for sequence-dependent mutability. For a set of sites it is scored
conditionally on where variants were observed, as oe_common_share is in v1:

  oe_share[c]       observed variants of class c / sum over the observed
                    variant sites of the model's P(class c | variant).
                    Purifying selection keeps variants young and rare, so
                    oe_share['singleton'] rises above 1 under constraint and
                    oe_share['common'] falls below 1. Conditioning on the
                    variant sites keeps errors in P(variant) out of it.
  singleton_share   observed and expected share of variants that are
                    singletons, and their difference ('excess'); the
                    difference is the direct analogue of MAPS, with the
                    network's context model in place of trinucleotides.
  oe_any            observed / expected variant sites (the original OVEP).
  oe_marginal[c]    observed class-c sites / sum of P(class c) over all
                    sites. A diagnostic only: unlike oe_any and oe_share it
                    is not fixed by the calibration, because it also depends
                    on how P(variant) and the class distribution co-vary.
  nll_per_variant   mean -log P(observed class | variant): how well the model
                    predicts the frequency class of a variant (lower is
                    better; a proper scoring rule).

With --baseline (a second prediction file on the same sites, e.g. from
kmer_baseline.py) the same scores are computed for it in the same blocks, and
'vs_baseline' reports the paired gain in nll_per_variant (baseline minus
model, > 0 = the model predicts the spectrum better).

Mutability bins: for selected masks, variant sites are split by a model's
P(variant) (--edges from the `edges` command) and oe_share['singleton'] is
reported per bin. On neutral sequence a mutability-adjusted measure must be
flat across bins; a measure that still depends on mutability is not.

Three commands:
  chrom   per-block sums for one chromosome -> .npz
  edges   quantile bin edges of P(variant) at variant sites, pooled over chromosomes
  pool    pool .npz files -> scores with block-bootstrap 95% intervals,
          and paired contrasts between categories (--contrast A:B)

Intervals resample 100 kb genomic blocks; the same resampled blocks are used
for every category and for the baseline, so contrasts and gains are paired.
They cover sampling noise in the scored regions only.

  python spectrum_score.py chrom --gff ref.gff --seqid NC_000004.12 \\
      --labels arrays/NC_000004.12_spectrum_labels.npy --new pred/neutral/NC_000004.12.npy \\
      --baseline pred/kmer3/NC_000004.12.npy --mask neutralA=masks/NC_000004.12_neutralA.npy \\
      --bed phastcons=masks/phastcons_pad0.bed --tiers neutralA --edges edges.json \\
      --out blocks/NC_000004.12.npz
  python spectrum_score.py pool --blocks blocks/*.npz --contrast tier_cds:tier_neutral \\
      --json results/spectrum.json --tsv results/spectrum.tsv
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import numpy as np

from compare_ovep import IntervalMask
from spectrum import LazyMask, class_names, conditional_classes, variant_probability

CHUNK = 5_000_000
_EPS = 1e-12
TIERS = ('tier_cds', 'tier_conserved_noncoding', 'tier_enhancer', 'tier_neutral')


# ------------------------------------------------------------------ columns

def n_columns(k, has_baseline):
    return 2 + k + (2 * k + 4) * (2 if has_baseline else 1)


def _model_offset(k, which):
    return 2 + k + (2 * k + 4) * which


def _model_weights(pred, lab, ok, k):
    """Per-site terms of one model for a chunk (zeros outside `ok`).
    Returns (pv, class probs (n, k), cond at variants (n, k), var_s, nll, nll_s)."""
    pr = np.where(ok[:, None], np.asarray(pred, dtype=np.float64), 0.0)
    pv = variant_probability(pr)
    var = ok & (lab >= 1)
    cond = np.zeros((len(lab), k))
    nll = np.zeros(len(lab))
    nll_s = np.zeros(len(lab))
    vi = np.flatnonzero(var)
    if vi.size:
        c = conditional_classes(pr[vi])
        cond[vi] = c
        cls = lab[vi].astype(np.int64) - 1
        nll[vi] = -np.log(np.clip(c[np.arange(vi.size), cls], _EPS, 1.0))
        ps = np.where(cls == 0, c[:, 0], 1.0 - c[:, 0])
        nll_s[vi] = -np.log(np.clip(ps, _EPS, 1.0))
    var_s = cond[:, 0] * (1.0 - cond[:, 0])
    return pv, pr[:, 1:], cond, var_s, nll, nll_s


def block_sums(labels, new, masks, baseline=None, block_size=100_000, edges=None,
               bin_by=None, bin_masks=(), chunk=CHUNK):
    """Per-mask sums over genomic blocks. Returns {name: {'blocks': (n_blocks,
    columns)[, 'bins': (n_blocks, n_bins, 4)]}} and the number of variant
    classes k. Eligible sites: label >= 0 and finite predictions from every
    model given."""
    n = len(labels)
    if new.ndim != 2 or new.shape[1] < 4:
        raise ValueError('spectrum_score.py needs spectrum class probabilities (n, 1 + K, K >= 3); '
                         'score three-class predictions with score.py / score_annotations.py')
    k = new.shape[1] - 1
    for name, arr in (('new', new), ('baseline', baseline), ('bin_by', bin_by)):
        if arr is not None and len(arr) != n:
            raise ValueError(f'{name} predictions ({len(arr)}) and labels ({n}) differ in length')
    if baseline is not None and baseline.shape[1] != k + 1:
        raise ValueError('baseline and new predictions differ in the number of classes')
    for name, m in masks.items():
        if len(m) != n:
            raise ValueError(f'mask {name!r} ({len(m)}) and labels ({n}) differ in length')
    has_b = baseline is not None
    n_blocks = -(-n // block_size)
    n_bins = len(edges) + 1 if edges is not None else 0
    bin_masks = [m for m in bin_masks if m in masks] if n_bins else []
    acc = {name: np.zeros((n_blocks, n_columns(k, has_b))) for name in masks}
    bins = {name: np.zeros((n_blocks, n_bins, 4)) for name in bin_masks}

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        lab = np.asarray(labels[s:e]).astype(np.int64)
        pn = np.asarray(new[s:e], dtype=np.float64)
        ok = (lab >= 0) & np.isfinite(pn).all(axis=1)
        pb = None
        if has_b:
            pb = np.asarray(baseline[s:e], dtype=np.float64)
            ok &= np.isfinite(pb).all(axis=1)
        pq = None
        if bin_masks:
            pq = pn if bin_by is None else np.asarray(bin_by[s:e], dtype=np.float64)
            ok &= np.isfinite(pq).all(axis=1)
        if lab[ok].size and lab[ok].max() > k:
            raise ValueError(f'labels contain class {int(lab[ok].max())} but predictions have {k} variant classes')
        models = [_model_weights(pn, lab, ok, k)]
        if has_b:
            models.append(_model_weights(pb, lab, ok, k))
        var = ok & (lab >= 1)
        which_bin = None
        if bin_masks:
            pv_q = variant_probability(np.where(ok[:, None], pq, 0.0))
            which_bin = np.searchsorted(np.asarray(edges, dtype=np.float64), pv_q, side='right')
        for name, mask in masks.items():
            idx = np.flatnonzero(ok & np.asarray(mask[s:e], dtype=bool))
            if idx.size == 0:
                continue
            block = (s + idx) // block_size
            a = acc[name]
            count = lambda w=None: np.bincount(block, weights=w, minlength=n_blocks)
            a[:, 0] += count()
            a[:, 1] += count(var[idx].astype(float))
            for j in range(k):
                a[:, 2 + j] += count((lab[idx] == j + 1).astype(float))
            for w, (pv, pcls, cond, var_s, nll, nll_s) in enumerate(models):
                o = _model_offset(k, w)
                a[:, o] += count(pv[idx])
                for j in range(k):
                    a[:, o + 1 + j] += count(pcls[idx, j])
                    a[:, o + 1 + k + j] += count(cond[idx, j])
                a[:, o + 1 + 2 * k] += count(var_s[idx])
                a[:, o + 2 + 2 * k] += count(nll[idx])
                a[:, o + 3 + 2 * k] += count(nll_s[idx])
            if name in bins:
                vi = idx[var[idx]]
                if vi.size:
                    flat = ((s + vi) // block_size) * n_bins + which_bin[vi]
                    size = n_blocks * n_bins
                    cols = [np.ones(vi.size), (lab[vi] == 1).astype(float), models[0][2][vi, 0],
                            models[1][2][vi, 0] if has_b else np.zeros(vi.size)]
                    for j, w in enumerate(cols):
                        bins[name][:, :, j] += np.bincount(flat, weights=w, minlength=size).reshape(n_blocks, n_bins)
    out = {name: {'blocks': acc[name]} for name in masks}
    for name in bins:
        out[name]['bins'] = bins[name]
    return out, k


# ------------------------------------------------------------------- scores

def _stats(t, k, has_b, names):
    """Column sums (..., C) -> {key tuple: array}. `names` = variant class names."""
    out = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        obs_var = t[..., 1]
        obs = [t[..., 2 + j] for j in range(k)]

        def model(prefix, w):
            o = _model_offset(k, w)
            out[prefix + ('oe_any',)] = obs_var / t[..., o]
            for j, nm in enumerate(names):
                out[prefix + ('oe_marginal', nm)] = obs[j] / t[..., o + 1 + j]
                out[prefix + ('oe_share', nm)] = obs[j] / t[..., o + 1 + k + j]
            exp_s = t[..., o + 1 + k]
            out[prefix + ('singleton_share', 'expected')] = exp_s / obs_var
            out[prefix + ('singleton_share', 'excess')] = (obs[0] - exp_s) / obs_var
            out[prefix + ('nll_per_variant',)] = t[..., o + 2 + 2 * k] / obs_var
            out[prefix + ('singleton_nll_per_variant',)] = t[..., o + 3 + 2 * k] / obs_var

        out[('singleton_share', 'observed')] = obs[0] / obs_var
        model((), 0)
        if has_b:
            model(('baseline',), 1)
            o0, o1 = _model_offset(k, 0), _model_offset(k, 1)
            out[('vs_baseline', 'nll_gain')] = (t[..., o1 + 2 + 2 * k] - t[..., o0 + 2 + 2 * k]) / obs_var
            out[('vs_baseline', 'singleton_nll_gain')] = (t[..., o1 + 3 + 2 * k] - t[..., o0 + 3 + 2 * k]) / obs_var
            out[('vs_baseline', 'oe_share_singleton_diff')] = (
                out[('oe_share', names[0])] - out[('baseline', 'oe_share', names[0])])
    return out


def _put(d, key, value):
    for part in key[:-1]:
        d = d.setdefault(part, {})
    d[key[-1]] = value


def _f(v):
    v = float(v)
    return v if np.isfinite(v) else None


def _ci(v):
    v = v[np.isfinite(v)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])] if v.size > 1 else None


def summarize(blocks, k, has_b, draws=None, bins=None):
    """Scores for one category from its (n_blocks, C) block sums. `draws` is
    an (n_boot, n_blocks) matrix of bootstrap block counts shared by all
    categories (so that contrasts are paired)."""
    names = class_names(k + 1)[1:]
    tot = blocks.sum(axis=0)
    res = {'n': int(round(tot[0])), 'observed_variants': int(round(tot[1])),
           'observed': {nm: int(round(tot[2 + j])) for j, nm in enumerate(names)}}
    if tot[0] == 0:
        return res
    for key, v in _stats(tot, k, has_b, names).items():
        _put(res, key, _f(v))
    if draws is not None and len(blocks) > 1:
        boot = _stats(draws @ blocks, k, has_b, names)
        res['ci95'], res['se'] = {}, {}
        for key, v in boot.items():
            _put(res['ci95'], key, _ci(v))
            fin = v[np.isfinite(v)]
            _put(res['se'], key, float(fin.std(ddof=1)) if fin.size > 1 else None)
        res['n_blocks'] = int((blocks[:, 0] > 0).sum())
    if bins is not None:
        bt = bins.sum(axis=0)                                  # (n_bins, 4)
        bb = None
        if draws is not None and len(bins) > 1:
            bb = (draws @ bins.reshape(len(bins), -1)).reshape(len(draws), *bins.shape[1:])
        rows = []
        with np.errstate(divide='ignore', invalid='ignore'):
            for b in range(bt.shape[0]):
                row = {'n_variants': int(round(bt[b, 0])), 'observed_singletons': int(round(bt[b, 1])),
                       'oe_singleton_share': _f(bt[b, 1] / bt[b, 2])}
                if bb is not None:
                    row['ci95'] = _ci(bb[:, b, 1] / bb[:, b, 2])
                if has_b:
                    row['baseline_oe_singleton_share'] = _f(bt[b, 1] / bt[b, 3])
                    if bb is not None:
                        row['baseline_ci95'] = _ci(bb[:, b, 1] / bb[:, b, 3])
                rows.append(row)
        res['mutability_bins'] = rows
        dev = [abs(r['oe_singleton_share'] - 1) for r in rows if r['oe_singleton_share'] is not None]
        res['mutability_bins_summary'] = {'n_bins': len(rows), 'max_abs_deviation': max(dev) if dev else None}
        if has_b:
            dev_b = [abs(r['baseline_oe_singleton_share'] - 1) for r in rows
                     if r.get('baseline_oe_singleton_share') is not None]
            res['mutability_bins_summary']['baseline_max_abs_deviation'] = max(dev_b) if dev_b else None
    return res


def contrast(blocks_a, blocks_b, k, has_b, draws=None):
    """Paired difference (A - B) of the singleton scores of two categories."""
    names = class_names(k + 1)[1:]
    keys = [('oe_share', names[0]), ('singleton_share', 'excess')]
    if has_b:
        keys += [('baseline', 'oe_share', names[0]), ('baseline', 'singleton_share', 'excess')]
    sa, sb = _stats(blocks_a.sum(axis=0), k, has_b, names), _stats(blocks_b.sum(axis=0), k, has_b, names)
    res = {}
    for key in keys:
        _put(res, key, {'diff': _f(sa[key] - sb[key])})
    if draws is not None and len(blocks_a) > 1:
        ba, bb = _stats(draws @ blocks_a, k, has_b, names), _stats(draws @ blocks_b, k, has_b, names)
        for key in keys:
            d = ba[key] - bb[key]
            node = res
            for part in key:
                node = node[part]
            node['ci95'] = _ci(d)
            fin = d[np.isfinite(d)]
            node['se'] = float(fin.std(ddof=1)) if fin.size > 1 else None
    return res


# ---------------------------------------------------------------------- I/O

def save_blocks(path, sums, meta):
    arrays = {}
    for name, v in sums.items():
        for field, arr in v.items():
            arrays[f'{name}::{field}'] = arr
    arrays['__meta__'] = np.array(json.dumps(meta))
    tmp = f'{path}.partial'
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def load_blocks(path):
    z = np.load(path)
    out = {}
    for key in z.files:
        if key == '__meta__':
            continue
        name, field = key.rsplit('::', 1)
        out.setdefault(name, {})[field] = z[key]
    return out, json.loads(str(z['__meta__']))


def pool_files(paths, contrasts=(), n_boot=1000, seed=0):
    """Pool per-chromosome files. Returns the results dict."""
    loaded = [load_blocks(p) for p in paths]
    metas = [m for _, m in loaded]
    for field in ('k', 'has_baseline', 'block_size', 'edges'):
        if any(m.get(field) != metas[0].get(field) for m in metas):
            raise ValueError(f'the files were made with different {field!r} settings')
    k, has_b = metas[0]['k'], metas[0]['has_baseline']
    n_blocks = [m['n_blocks'] for m in metas]
    names = sorted(set().union(*(set(s) for s, _ in loaded)))
    c = n_columns(k, has_b)

    def stack(name, field, shape_tail):
        parts = [s[name][field] if name in s and field in s[name] else np.zeros((nb,) + shape_tail)
                 for (s, _), nb in zip(loaded, n_blocks)]
        return np.concatenate(parts)

    total = sum(n_blocks)
    draws = None
    if n_boot and total > 1:
        rng = np.random.default_rng(seed)
        draws = rng.multinomial(total, np.full(total, 1.0 / total), size=n_boot).astype(np.float64)
    blocks = {name: stack(name, 'blocks', (c,)) for name in names}
    n_bins = len(metas[0]['edges']) + 1 if metas[0].get('edges') else 0
    results = {}
    for name in names:
        has_bins = n_bins and any(name in s and 'bins' in s[name] for s, _ in loaded)
        results[name] = summarize(blocks[name], k, has_b, draws,
                                  stack(name, 'bins', (n_bins, 4)) if has_bins else None)
    out = {'meta': {'k': k, 'classes': class_names(k + 1), 'has_baseline': has_b,
                    'block_size': metas[0]['block_size'], 'n_blocks': total, 'bootstrap': n_boot,
                    'mutability_edges': metas[0].get('edges'), 'pooled_from': metas},
           'categories': results, 'contrasts': {}}
    for a, b in contrasts:
        if a not in blocks or b not in blocks:
            raise ValueError(f'contrast {a}:{b}: category missing from the block files '
                             f'(available: {", ".join(names)})')
        out['contrasts'][f'{a}_minus_{b}'] = contrast(blocks[a], blocks[b], k, has_b, draws)
    return out


def table_rows(results):
    cats = results['categories']
    names = results['meta']['classes'][1:]
    has_b = results['meta']['has_baseline']
    cols = (['category', 'n', 'observed_variants', 'oe_any', 'singleton_share_observed',
             'singleton_share_expected', 'oe_share_singleton', 'ci_lo', 'ci_hi']
            + [f'oe_share_{nm}' for nm in names[1:]]
            + (['baseline_oe_share_singleton', 'nll_gain_vs_baseline'] if has_b else []))
    rows = []
    order = sorted(cats, key=lambda c: (not c.startswith('tier_'), c))
    for name in order:
        r = cats[name]
        if not r.get('observed_variants'):
            continue
        ci = (r.get('ci95', {}).get('oe_share', {}).get(names[0]) or [None, None])
        row = [name, r['n'], r['observed_variants'], r.get('oe_any'),
               r['singleton_share']['observed'], r['singleton_share']['expected'],
               r['oe_share'][names[0]], ci[0], ci[1]] + [r['oe_share'][nm] for nm in names[1:]]
        if has_b:
            row += [r['baseline']['oe_share'][names[0]], r['vs_baseline']['nll_gain']]
        rows.append(row)
    return cols, rows


def write_outputs(results, json_path=None, tsv_path=None):
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path + '.partial', 'w') as f:
            json.dump(results, f, indent=2)
        os.replace(json_path + '.partial', json_path)
    if tsv_path:
        cols, rows = table_rows(results)
        with open(tsv_path, 'w') as f:
            f.write('\t'.join(cols) + '\n')
            for r in rows:
                f.write('\t'.join('' if v is None else (f'{v:.4g}' if isinstance(v, float) else str(v))
                                  for v in r) + '\n')


# -------------------------------------------------------------------- masks

def bed_intervals(bed_path, chromosome):
    out = []
    opener = gzip.open if str(bed_path).endswith('.gz') else open
    with opener(bed_path, 'rt') as fh:
        for line in fh:
            if not line.strip() or line.startswith(('#', 'track', 'browser')):
                continue
            chrom, start, end, *_ = line.split()
            if chrom == chromosome:
                out.append((int(start), int(end)))
    return out


def tier_masks(masks, neutral):
    """Mutually exclusive constraint tiers from masks named CDS, phastcons,
    enhancer (optional) and the neutral mask:
      tier_cds                   coding sequence
      tier_conserved_noncoding   phastCons elements outside CDS
      tier_enhancer              enhancers outside CDS and phastCons elements
      tier_neutral               the neutral mask outside phastCons elements
    """
    for need in ('CDS', 'phastcons', neutral):
        if need not in masks:
            raise ValueError(f'--tiers needs a mask named {need!r} (have: {", ".join(sorted(masks))})')
    cds, phast = masks['CDS'], masks['phastcons']
    tiers = {'tier_cds': cds,
             'tier_conserved_noncoding': LazyMask(lambda p, c: p & ~c, [phast, cds]),
             'tier_neutral': LazyMask(lambda n, p: n & ~p, [masks[neutral], phast])}
    if 'enhancer' in masks:
        tiers['tier_enhancer'] = LazyMask(lambda e, c, p: e & ~c & ~p, [masks['enhancer'], cds, phast])
    return tiers


def quantile_edges(triples, n_bins=10, max_sites=2_000_000, seed=0, chunk=CHUNK):
    """Interior quantile edges of P(variant) at observed variant sites, pooled
    over (labels, predictions, mask-or-None) triples."""
    rng = np.random.default_rng(seed)
    parts = []
    per = max(max_sites // max(len(triples), 1), 1)
    for labels, pred, mask in triples:
        vals = []
        for s in range(0, len(labels), chunk):
            lab = np.asarray(labels[s:s + chunk])
            pr = np.asarray(pred[s:s + chunk], dtype=np.float64)
            ok = (lab >= 1) & np.isfinite(pr).all(axis=1)
            if mask is not None:
                ok &= np.asarray(mask[s:s + chunk], dtype=bool)
            vals.append(variant_probability(pr[ok]))
        vals = np.concatenate(vals) if vals else np.zeros(0)
        if vals.size > per:
            vals = rng.choice(vals, per, replace=False)
        parts.append(vals)
    allv = np.concatenate(parts) if parts else np.zeros(0)
    if allv.size < n_bins * 10:
        raise ValueError('too few variant sites to define mutability bins')
    # tied quantiles (a predictor with few distinct values, e.g. a k-mer
    # table) are merged, which gives fewer bins than asked for
    # (a site whose P(variant) equals an edge falls in the bin above it, so an
    # edge at the minimum would leave the first bin empty)
    edges = np.unique(np.quantile(allv, np.arange(1, n_bins) / n_bins))
    edges = edges[edges > allv.min()]
    if edges.size == 0:
        raise ValueError('P(variant) is constant at the variant sites; no mutability bins can be formed')
    return [float(x) for x in edges], int(allv.size)


# --------------------------------------------------------------------- main

def _named(items, parser, what):
    out = {}
    for item in items:
        name, _, path = item.partition('=')
        if not path:
            parser.error(f'{what} expects NAME=PATH, got {item!r}')
        out[name] = path
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    c = sub.add_parser('chrom', help='per-block sums for one chromosome')
    c.add_argument('--labels', required=True, help='spectrum labels (data.py --scheme spectrum)')
    c.add_argument('--new', required=True, help='spectrum class probabilities from predict.py')
    c.add_argument('--baseline', help='second prediction file on the same sites (e.g. kmer_baseline.py)')
    c.add_argument('--gff')
    c.add_argument('--seqid', help='GFF column 1 / BED name of this chromosome')
    c.add_argument('--types', nargs='+', help='restrict to these GFF feature types (default: all)')
    c.add_argument('--exclude-source', nargs='*', default=['RefSeq'])
    c.add_argument('--mask', action='append', default=[], metavar='NAME=PATH', help='boolean .npy')
    c.add_argument('--bed', action='append', default=[], metavar='NAME=PATH',
                   help='BED intervals (contig names as --seqid)')
    c.add_argument('--tiers', metavar='NEUTRAL_MASK',
                   help='add the mutually exclusive constraint tiers; needs GFF CDS, --bed phastcons=... '
                        'and the named neutral mask')
    c.add_argument('--edges', help='JSON from the edges command: enables mutability bins')
    c.add_argument('--bin-by', help='prediction file whose P(variant) defines the bins (default: --new)')
    c.add_argument('--bin-masks', nargs='+', help='masks that get mutability bins '
                   '(default: tier_neutral and every --mask)')
    c.add_argument('--block-size', type=int, default=100_000)
    c.add_argument('--out', required=True, help='.npz for the pool command')

    e = sub.add_parser('edges', help='mutability bin edges')
    e.add_argument('--chrom', action='append', nargs=3, required=True,
                   metavar=('LABELS', 'PREDICTIONS', 'MASK'),
                   help="one chromosome; MASK may be '-' for none. Repeatable.")
    e.add_argument('--bins', type=int, default=10)
    e.add_argument('--max-sites', type=int, default=2_000_000)
    e.add_argument('--seed', type=int, default=0)
    e.add_argument('--out', required=True)

    q = sub.add_parser('pool', help='pool block files into scores')
    q.add_argument('--blocks', nargs='+', required=True)
    q.add_argument('--contrast', action='append', default=[], metavar='A:B',
                   help='paired difference of the singleton scores, category A minus B')
    q.add_argument('--bootstrap', type=int, default=1000)
    q.add_argument('--seed', type=int, default=0)
    q.add_argument('--json')
    q.add_argument('--tsv')
    args = p.parse_args()

    if args.cmd == 'edges':
        triples = [(np.load(l, mmap_mode='r'), np.load(pr, mmap_mode='r'),
                    None if m == '-' else np.load(m, mmap_mode='r')) for l, pr, m in args.chrom]
        edges, n = quantile_edges(triples, args.bins, args.max_sites, args.seed)
        with open(args.out + '.partial', 'w') as f:
            json.dump({'edges': edges, 'n_bins': len(edges) + 1, 'n_bins_requested': args.bins,
                       'n_variant_sites_used': n, 'predictions': [pr for _, pr, _ in args.chrom]}, f, indent=1)
        os.replace(args.out + '.partial', args.out)
        print(f'Saved {args.out}: {len(edges) + 1} bins from {n:,} variant sites'
              + ('' if len(edges) + 1 == args.bins else f' ({args.bins} requested; tied quantiles merged)'))
        return

    if args.cmd == 'pool':
        contrasts = []
        for item in args.contrast:
            a, _, b = item.partition(':')
            if not b:
                p.error(f'--contrast expects A:B, got {item!r}')
            contrasts.append((a, b))
        results = pool_files(args.blocks, contrasts, args.bootstrap, args.seed)
        write_outputs(results, args.json, args.tsv)
        if not args.json and not args.tsv:
            print(json.dumps(results, indent=2))
        else:
            print(f"{len(results['categories'])} categories -> "
                  f"{', '.join(x for x in (args.json, args.tsv) if x)}")
        return

    labels = np.load(args.labels, mmap_mode='r')
    n = len(labels)
    masks = {}
    if args.gff:
        if not args.seqid:
            p.error('--gff needs --seqid')
        from score_annotations import build_masks
        masks.update(build_masks(args.gff, args.seqid, n, set(args.types) if args.types else None,
                                 args.exclude_source))
    masks['all_scorable'] = IntervalMask([(0, n)], n)
    named = _named(args.mask, p, '--mask')
    for name, path in named.items():
        masks[name] = np.load(path, mmap_mode='r')
    for name, path in _named(args.bed, p, '--bed').items():
        if not args.seqid:
            p.error('--bed needs --seqid')
        masks[name] = IntervalMask(bed_intervals(path, args.seqid), n)
    if args.tiers:
        masks.update(tier_masks(masks, args.tiers))
    edges = None
    if args.edges:
        with open(args.edges) as f:
            edges = json.load(f)['edges']
    bin_masks = args.bin_masks or (['tier_neutral'] + list(named))
    sums, k = block_sums(labels, np.load(args.new, mmap_mode='r'), masks,
                         np.load(args.baseline, mmap_mode='r') if args.baseline else None,
                         args.block_size, edges,
                         np.load(args.bin_by, mmap_mode='r') if args.bin_by else None, bin_masks)
    meta = {'seqid': args.seqid, 'labels': args.labels, 'new': args.new, 'baseline': args.baseline,
            'k': k, 'has_baseline': args.baseline is not None, 'block_size': args.block_size,
            'n_blocks': -(-n // args.block_size), 'edges': edges, 'bin_by': args.bin_by}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_blocks(args.out, sums, meta)
    print(f'{args.out}: {len(sums)} categories, {meta["n_blocks"]} blocks')


if __name__ == '__main__':
    main()
