"""Discriminate de novo mutation (DNM) sites from background with several
models on the same sites (the paper's Figure 3 benchmark).

For each DNM set: positives = every DNM position scorable by all models;
negatives = a uniform random sample (--neg-sites) of the other scorable sites.
Per model it reports ROC-AUC and the Hanley-McNeil standard error (the
paper's method); per model pair, the AUC difference with DeLong's paired test
(correct for two models scored on the same sites; Hanley-McNeil treats them as
independent and is conservative).

Config (JSON):
{
  "models": {"original": "score", "base": "classes", "neutral": "classes"},
  "dnm": {"denovo_db": "dnm/denovo_db_ssc_sibs.grch38.tsv",
          "platinum": "dnm/platinum_pedigree.grch38.tsv"},
  "chromosomes": {
    "NC_000002.12": {
      "dnm_name": "chr2",
      "dna": "arrays/NC_000002.12_dna.npy",
      "predictions": {
        "original": {"path": "calibrated_predictions/NC_000002.12.pred.npy",
                     "raw": "raw_predictions/NC_000002.12.pred.npy"},
        "base": "pred/half1/base/NC_000002.12.npy",
        "neutral": "pred/half1/neutral/NC_000002.12.npy"}}}
}
"score" = a 1-D per-site score (the original model, gnomAD or Roulette
rates aligned to the chromosome); "classes" = (n, 3) from predict.py. 1-D
scores shorter than the chromosome are head-aligned; "raw" marks unpredicted
sites (exactly 0) to exclude. DNM files: tab-separated chrom, 1-based pos;
extra columns and '#' header lines are ignored.
"""
import argparse
import os
import json
import math
import numpy as np

from compare_ovep import VariantScore, midrank

CHUNK = 5_000_000


def read_dnms(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith('#'):
                continue
            f = line.split()
            try:
                out.setdefault(f[0], set()).add(int(f[1]) - 1)
            except (IndexError, ValueError):
                continue  # header or malformed line
    return out


def hanley_mcneil_se(auc, n_pos, n_neg):
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    return math.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
                      + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))


def delong(y, scores):
    """Fast DeLong (Sun & Xu 2014). scores: (k, n). Returns aucs (k,), cov (k, k)."""
    y = np.asarray(y, bool)
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    aucs, v01, v10 = np.empty(k), np.empty((k, m)), np.empty((k, n))
    for i in range(k):
        tx, ty = midrank(pos[i]), midrank(neg[i])
        tz = midrank(np.r_[pos[i], neg[i]])
        aucs[i] = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v01[i] = (tz[:m] - tx) / n
        v10[i] = 1.0 - (tz[m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def p_two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2))


def collect(config, dnm_sets, neg_sites, seed=0, chunk=CHUNK):
    """Scores at positives and sampled negatives for every DNM set."""
    models = list(config['models'])
    chroms = config['chromosomes']
    rng = np.random.default_rng(seed)
    total = sum(len(np.load(c['dna'], mmap_mode='r')) for c in chroms.values())
    rate = min(1.0, neg_sites / max(total, 1))
    out = {name: {'y': [], 'scores': [], 'n_dnm_listed': 0, 'n_dnm_unscorable': 0} for name in dnm_sets}
    for seqid, c in chroms.items():
        dna = np.load(c['dna'], mmap_mode='r')
        n = len(dna)
        preds, raws = {}, {}
        for mname in models:
            spec = c['predictions'][mname]
            spec = spec if isinstance(spec, dict) else {'path': spec}
            arr = np.load(spec['path'], mmap_mode='r')
            kind = config['models'][mname]
            if (kind == 'classes') != (arr.ndim == 2):
                raise ValueError(f'{seqid}/{mname}: kind {kind!r} but array has {arr.ndim} dims')
            if arr.ndim == 2 and len(arr) != n:
                raise ValueError(f'{seqid}/{mname}: {len(arr)} rows vs chromosome length {n}')
            preds[mname] = VariantScore(arr)
            raws[mname] = np.load(spec['raw'], mmap_mode='r') if 'raw' in spec else None
        dn = {name: np.array(sorted(p for p in s.get(c.get('dnm_name', seqid), set()) if 0 <= p < n),
                             dtype=np.int64)
              for name, s in dnm_sets.items()}
        for name in dnm_sets:
            out[name]['n_dnm_listed'] += dn[name].size
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            ok = np.asarray(dna[s:e]).max(axis=1) > 0
            cols = []
            for mname in models:
                v = np.full(e - s, np.nan)
                m_len = min(len(preds[mname]), n)
                if s < m_len:
                    v[:min(e, m_len) - s] = preds[mname][s:min(e, m_len)]
                    if raws[mname] is not None:
                        raw = np.asarray(raws[mname][s:min(e, m_len)], dtype=np.float64)
                        v[:raw.size][raw == 0] = np.nan
                cols.append(v)
                ok &= np.isfinite(v)
            cols = np.vstack(cols)
            neg_pick = rng.random(e - s) < rate
            for name in dnm_sets:
                d = dn[name]
                d = d[(d >= s) & (d < e)] - s
                is_dnm = np.zeros(e - s, bool)
                is_dnm[d] = True
                out[name]['n_dnm_unscorable'] += int((is_dnm & ~ok).sum())
                take = ok & (is_dnm | neg_pick)
                idx = np.flatnonzero(take)
                out[name]['y'].append(is_dnm[idx])
                out[name]['scores'].append(cols[:, idx])
    for name in out:
        o = out[name]
        o['y'] = np.concatenate(o['y']) if o['y'] else np.zeros(0, bool)
        o['scores'] = np.hstack(o['scores']) if o['scores'] else np.zeros((len(models), 0))
    return models, out


def evaluate(models, data):
    y, sc = data['y'], data['scores']
    m, n = int(y.sum()), int((~y).sum())
    res = {'n_pos': m, 'n_neg': n, 'n_dnm_listed': data['n_dnm_listed'],
           'n_dnm_unscorable': data['n_dnm_unscorable'], 'models': {}, 'pairs': {}}
    if m < 2 or n < 2:
        res['error'] = 'too few positives or negatives'
        return res
    aucs, cov = delong(y, sc)
    for i, name in enumerate(models):
        res['models'][name] = {'auc': float(aucs[i]), 'se_delong': float(math.sqrt(max(cov[i, i], 0))),
                               'se_hanley_mcneil': hanley_mcneil_se(aucs[i], m, n)}
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if j <= i:
                continue
            diff = float(aucs[j] - aucs[i])
            var = cov[i, i] + cov[j, j] - 2 * cov[i, j]
            se = math.sqrt(max(var, 0))
            se_hm = math.sqrt(res['models'][a]['se_hanley_mcneil'] ** 2 + res['models'][b]['se_hanley_mcneil'] ** 2)
            res['pairs'][f'{b}_minus_{a}'] = {
                'diff': diff, 'se_delong': se,
                'p_delong': p_two_sided(diff / se) if se > 0 else None,
                'ci95_delong': [diff - 1.96 * se, diff + 1.96 * se],
                'p_hanley_mcneil': p_two_sided(diff / se_hm) if se_hm > 0 else None}
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--neg-sites', type=int, default=5_000_000,
                   help='background sites sampled across all chromosomes (paper: ~5M)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--json', help='write results here')
    args = p.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    dnm_sets = {name: read_dnms(path) for name, path in config['dnm'].items()}
    models, data = collect(config, dnm_sets, args.neg_sites, args.seed)
    results = {name: evaluate(models, d) for name, d in data.items()}
    text = json.dumps(results, indent=2)
    if args.json:
        with open(args.json + '.partial', 'w') as f:
            f.write(text)
        os.replace(args.json + '.partial', args.json)
    print(text)


if __name__ == '__main__':
    main()
