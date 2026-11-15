"""How close are two models' per-site predictions on one chromosome?

Written to check a retrained copy of the original OVEP model against the
published model's predictions, but it compares any two prediction files:
1-D scores (the original repository's <seqid>.pred.npy) or class
probabilities from predict.py (compared by P(variant)).

Reported on the sites both models predict:
  correlation    Pearson r of the probabilities and of their logits, and
                 Spearman rank correlation (on a random subsample)
  means          mean probability of each model and their ratio
  agreement      share of sites whose two probabilities differ by less than
                 10% and 25% (relative), and quantiles of b / a
  by decile of a mean of a and of b in each decile of a (do the two models
                 agree at low and at high mutability?)
With --labels (and optionally --mask) it adds, for each model, the ROC-AUC
for variant vs no variant and observed/expected variant sites, on the same
sites.

1-D files shorter than the chromosome are head-aligned, as the original
predict_by_chrom_fast.py writes them. --a-raw / --b-raw give a model's
uncalibrated output: sites it left unpredicted (exactly 0 there) are dropped.

For a yardstick, run it on two models that should agree (two seeds of your
own model): a wrong label definition, a shifted coordinate system or a
different window alignment shows up as a clearly lower correlation or a
different mean than that.

  python compare_predictions.py \\
      --a /local/disk4/gyuanfan/rate/calibrated_predictions/NC_000022.11.pred.npy \\
      --a-raw /local/disk4/gyuanfan/rate/code_halftrainmaskmissing/base_excludeN_final_extrapolate/NC_000022.11.pred.npy \\
      --b orig_repro/half1/calibrated/NC_000022.11.pred.npy \\
      --labels arrays/NC_000022.11_frequency_labels.npy --json chr22_original_vs_retrained.json
"""
import argparse
import json
import numpy as np

from compare_ovep import VariantScore, midrank, roc_auc

CHUNK = 5_000_000


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def compare_predictions(a, b, a_raw=None, b_raw=None, labels=None, mask=None,
                        sample=2_000_000, seed=0, chunk=CHUNK):
    sa, sb = VariantScore(a), VariantScore(b)
    n = min(len(sa), len(sb))
    if labels is not None:
        n = min(n, len(labels))
    if mask is not None:
        n = min(n, len(mask))
    for raw, s, name in ((a_raw, sa, 'a'), (b_raw, sb, 'b')):
        if raw is not None and len(raw) != len(s):
            raise ValueError(f'--{name}-raw and --{name} differ in length')
    rng = np.random.default_rng(seed)
    rate = min(1.0, sample / max(n, 1))
    # running sums for means / Pearson on all shared sites
    m = np.zeros(11)   # count, sa, sb, saa, sbb, sab, la, lb, laa, lbb, lab
    obs = exp_a = exp_b = 0.0
    n_lab = 0
    pa_s, pb_s, y_s = [], [], []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        pa, pb = sa[s:e], sb[s:e]
        ok = np.isfinite(pa) & np.isfinite(pb)
        if a_raw is not None:
            ok &= np.asarray(a_raw[s:e]) != 0
        if b_raw is not None:
            ok &= np.asarray(b_raw[s:e]) != 0
        if mask is not None:
            ok &= np.asarray(mask[s:e], dtype=bool)
        lab = None
        if labels is not None:
            lab = np.asarray(labels[s:e])
            ok &= lab >= 0
        pa, pb = pa[ok], pb[ok]
        la, lb = _logit(pa), _logit(pb)
        m += [pa.size, pa.sum(), pb.sum(), (pa * pa).sum(), (pb * pb).sum(), (pa * pb).sum(),
              la.sum(), lb.sum(), (la * la).sum(), (lb * lb).sum(), (la * lb).sum()]
        pick = rng.random(pa.size) < rate
        pa_s.append(pa[pick])
        pb_s.append(pb[pick])
        if lab is not None:
            y = lab[ok] >= 1
            obs += y.sum()
            exp_a += pa.sum()
            exp_b += pb.sum()
            n_lab += y.size
            y_s.append(y[pick])
    cnt = m[0]
    if cnt < 2:
        raise ValueError('fewer than two sites predicted by both models')

    def pearson(sx, sy, sxx, syy, sxy):
        cov = sxy - sx * sy / cnt
        vx, vy = sxx - sx * sx / cnt, syy - sy * sy / cnt
        return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else None

    pa, pb = np.concatenate(pa_s), np.concatenate(pb_s)
    ra, rb = midrank(pa), midrank(pb)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = pb / pa
    ratio = ratio[np.isfinite(ratio)]
    rel = np.abs(pb - pa) / np.maximum((pa + pb) / 2, 1e-12)
    edges = np.unique(np.quantile(pa, np.arange(1, 10) / 10))
    which = np.searchsorted(edges, pa, side='right')
    deciles = [{'decile_of_a': int(i + 1), 'n': int((which == i).sum()),
                'mean_a': float(pa[which == i].mean()), 'mean_b': float(pb[which == i].mean())}
               for i in range(len(edges) + 1) if (which == i).any()]
    res = {'sites_compared': int(cnt), 'sample_size': int(pa.size),
           'pearson_r': pearson(m[1], m[2], m[3], m[4], m[5]),
           'pearson_r_logit': pearson(m[6], m[7], m[8], m[9], m[10]),
           'spearman_r': float(np.corrcoef(ra, rb)[0, 1]),
           'mean_a': float(m[1] / cnt), 'mean_b': float(m[2] / cnt),
           'mean_b_over_mean_a': float(m[2] / m[1]) if m[1] > 0 else None,
           'share_within_10_percent': float((rel < 0.10).mean()),
           'share_within_25_percent': float((rel < 0.25).mean()),
           'b_over_a_quantiles_5_25_50_75_95': [float(x) for x in np.percentile(ratio, [5, 25, 50, 75, 95])],
           'by_decile_of_a': deciles}
    if labels is not None:
        y = np.concatenate(y_s)
        res['labels'] = {'sites': int(n_lab), 'observed_variants': int(obs),
                         'oe_a': float(obs / exp_a) if exp_a > 0 else None,
                         'oe_b': float(obs / exp_b) if exp_b > 0 else None,
                         'auc_a': roc_auc(y, pa), 'auc_b': roc_auc(y, pb)}
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', required=True, help='first prediction file (e.g. the published model)')
    p.add_argument('--b', required=True, help='second prediction file (e.g. the retrained model)')
    p.add_argument('--a-raw', help="first model's uncalibrated output (drops unpredicted sites)")
    p.add_argument('--b-raw', help="second model's uncalibrated output")
    p.add_argument('--labels', help='labels .npy: adds AUC and observed/expected for both models')
    p.add_argument('--mask', help='boolean .npy: restrict to these sites')
    p.add_argument('--sample', type=int, default=2_000_000, help='sites for rank statistics')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--json')
    args = p.parse_args()
    load = lambda path: np.load(path, mmap_mode='r') if path else None
    res = compare_predictions(load(args.a), load(args.b), load(args.a_raw), load(args.b_raw),
                              load(args.labels), load(args.mask), args.sample, args.seed)
    text = json.dumps(res, indent=2)
    if args.json:
        with open(args.json, 'w') as f:
            f.write(text)
    print(text)


if __name__ == '__main__':
    main()
