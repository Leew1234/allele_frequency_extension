"""Discrimination endpoint: does adding a frequency score separate constrained
sequence (e.g. CDS) from neutral sequence better than OVEP alone?

Units are runs of a fixed number of scorable sites inside a mask (written by
score_annotations.py --unit-sites). For each unit:

  ovep_z      (observed - expected variants) / sqrt(expected), with the
              reference model's expectation (the original OVEP, as a z)
  freq_z      (observed common - sum of P(common | variant) over the unit's
              observed variant sites) / sqrt(sum p(1-p)) (Poisson-binomial);
              0 when the unit has no variant. Conditional on where variants
              occurred, so it is not a restatement of ovep_z and errors in the
              mutability model do not leak into it.
  combined    sum of the two after putting each on the neutral scale: centred
              and divided by its mean and SD on the control (neutral) units.
              Real allele frequencies are overdispersed relative to the
              binomial, so a raw freq_z would otherwise get too much weight.
              Only control units are used, so nothing is fitted to the case
              (CDS) units; 'combined_unscaled' (the raw sum) is reported too.

Lower = more constrained. 'neutral_calibration' reports each z's mean, SD,
quantiles and share beyond +/-1.96 on the control units: under a calibrated
model, about 0, 1 and 5%.

The primary analysis uses all case and control units (both z's are
standardized to each unit's own expectation). 'matched' repeats it on units
matched for expected variant count, equal numbers per quantile bin of the case
units, dropping case units with no control of comparable mutability.

Intervals: neighbouring units are not independent, so the primary 95%
intervals ('ci95_block') come from resampling whole genomic blocks (default
1 Mb; case and control units in a block move together; at least 30 blocks).
DeLong intervals ('ci95_delong') assume independent units and are secondary.
Per-chromosome AUCs show whether an improvement is stable.

  python discriminate.py --blocks blocks/half1/neutral/*.npz \\
      --case CDS --control neutralA --json results/half1/discrimination_neutral.json
"""
import argparse
import json
import math
import os
import numpy as np

from compare_ovep import load_blocks, pool, midrank
from denovo_auc import delong, p_two_sided

SCORES = ('ovep_z', 'freq_z', 'combined', 'combined_unscaled')
PAIRS = ((0, 1), (0, 2), (1, 2), (0, 3))


def raw_z(units):
    u = np.asarray(units, dtype=np.float64)
    obs, exp_ref, obs_c, ecv, vcv = u[:, 0], u[:, 1], u[:, 3], u[:, 5], u[:, 6]
    with np.errstate(divide='ignore', invalid='ignore'):
        ovep_z = (obs - exp_ref) / np.sqrt(exp_ref)
        freq_z = np.where((obs > 0) & (vcv > 0), (obs_c - ecv) / np.sqrt(vcv), 0.0)
    return ovep_z, freq_z


def neutral_scale(ctrl_units):
    """Mean and SD of each z on the control (neutral) units."""
    o, f = raw_z(ctrl_units)
    return {'ovep_z': (float(np.mean(o)), float(np.std(o, ddof=1))),
            'freq_z': (float(np.mean(f)), float(np.std(f, ddof=1)))}


def calibration_summary(ctrl_units):
    o, f = raw_z(ctrl_units)
    out = {}
    for name, z in (('ovep_z', o), ('freq_z', f)):
        q = np.percentile(z, [5, 25, 50, 75, 95])
        out[name] = {'mean': float(z.mean()), 'sd': float(z.std(ddof=1)),
                     'quantiles_5_25_50_75_95': [float(x) for x in q],
                     'share_beyond_1.96': float(np.mean(np.abs(z) > 1.96)), 'n_units': int(z.size)}
    return out


def unit_scores(units, scale=None):
    ovep_z, freq_z = raw_z(units)
    if scale is None:
        scale = {'ovep_z': (0.0, 1.0), 'freq_z': (0.0, 1.0)}
    (mo, so), (mf, sf) = scale['ovep_z'], scale['freq_z']
    combined = (ovep_z - mo) / (so if so > 0 else 1.0) + (freq_z - mf) / (sf if sf > 0 else 1.0)
    return {'ovep_z': ovep_z, 'freq_z': freq_z, 'combined': combined, 'combined_unscaled': ovep_z + freq_z}


def match_units(case_exp, ctrl_exp, n_bins=10, rng=None):
    """Exact-count matching on common support. Bins = quantiles of the case
    units' expected counts; in each bin, equal numbers of case and control
    units are drawn without replacement (min of the two). Case units in bins
    without controls are dropped, never paired with reused controls.
    Returns (case indices, control indices)."""
    rng = rng or np.random.default_rng(0)
    edges = np.unique(np.quantile(case_exp, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([case_exp.min(), case_exp.max() + 1e-12])
    nb = len(edges) - 1
    cb = np.clip(np.searchsorted(edges, case_exp, side='right') - 1, 0, nb - 1)
    kb = np.clip(np.searchsorted(edges, ctrl_exp, side='right') - 1, 0, nb - 1)
    inside = (ctrl_exp >= edges[0]) & (ctrl_exp <= edges[-1])
    ci, ki = [], []
    for b in range(nb):
        cases = np.flatnonzero(cb == b)
        ctrls = np.flatnonzero(inside & (kb == b))
        m = min(cases.size, ctrls.size)
        if m:
            ci.append(rng.choice(cases, m, replace=False))
            ki.append(rng.choice(ctrls, m, replace=False))
    cat = lambda x: np.concatenate(x) if x else np.zeros(0, int)
    return cat(ci), cat(ki)


def cohen_d(case, ctrl):
    s = math.sqrt(((case.size - 1) * case.var(ddof=1) + (ctrl.size - 1) * ctrl.var(ddof=1))
                  / max(case.size + ctrl.size - 2, 1))
    return float((ctrl.mean() - case.mean()) / s) if s > 0 else None


def _aucs_resampled(y, mat):
    """AUC of each row of mat (higher = case) via midranks."""
    m, n = int(y.sum()), int((~y).sum())
    return np.array([(midrank(r)[y].sum() - m * (m + 1) / 2) / (m * n) for r in mat])


def _block_bootstrap(case_sc, ctrl_sc, case_blk, ctrl_blk, n_boot, rng):
    """Resample whole genomic blocks (case and control units together) with
    replacement; returns (n_boot, 3) AUCs. Units in one block move together,
    so correlation between neighbouring units is respected."""
    blocks, inv = np.unique(np.r_[case_blk, ctrl_blk], return_inverse=True)
    ib_case, ib_ctrl = inv[:len(case_blk)], inv[len(case_blk):]
    nb = len(blocks)
    out = []
    for _ in range(n_boot):
        w = rng.multinomial(nb, np.full(nb, 1.0 / nb))
        rc = np.repeat(np.arange(len(case_blk)), w[ib_case])
        rk = np.repeat(np.arange(len(ctrl_blk)), w[ib_ctrl])
        if rc.size < 2 or rk.size < 2:
            continue
        y = np.r_[np.ones(rc.size, bool), np.zeros(rk.size, bool)]
        mat = np.vstack([-np.r_[case_sc[k][rc], ctrl_sc[k][rk]] for k in SCORES])
        out.append(_aucs_resampled(y, mat))
    return np.array(out), nb


MIN_BLOCKS = 30  # fewer blocks than this: no block-bootstrap interval (criteria read MISSING)


def _auc_block(case, ctrl, case_blk=None, ctrl_blk=None, case_chr=None, ctrl_chr=None,
               n_boot=500, rng=None, min_chrom_units=20, scale=None):
    sc_case, sc_ctrl = unit_scores(case, scale), unit_scores(ctrl, scale)
    y = np.r_[np.ones(len(case), bool), np.zeros(len(ctrl), bool)]
    mat = np.vstack([-np.r_[sc_case[k], sc_ctrl[k]] for k in SCORES])  # lower = constrained
    aucs, cov = delong(y, mat)
    res = {'n_case': int(len(case)), 'n_control': int(len(ctrl)),
           'mean_expected_case': float(case[:, 1].mean()),
           'mean_expected_control': float(ctrl[:, 1].mean()),
           'corr_ovep_freq_control': float(np.corrcoef(sc_ctrl['ovep_z'], sc_ctrl['freq_z'])[0, 1]),
           'sd_control': {k: float(np.std(sc_ctrl[k], ddof=1)) for k in SCORES},
           'scores': {}, 'pairs': {}}
    boot, nb = (None, 0)
    if case_blk is not None and n_boot:
        nb_all = len(np.unique(np.r_[case_blk, ctrl_blk]))
        if nb_all >= MIN_BLOCKS:
            boot, nb = _block_bootstrap(sc_case, sc_ctrl, case_blk, ctrl_blk, n_boot,
                                        rng or np.random.default_rng(0))
            res['bootstrap'] = {'n_blocks': int(nb), 'replicates': int(len(boot))}
        else:
            res['bootstrap'] = {'n_blocks': int(nb_all), 'error': f'fewer than {MIN_BLOCKS} genomic blocks; '
                                'no block-bootstrap interval (use smaller --block-bp)'}
    for i, k in enumerate(SCORES):
        se = math.sqrt(max(cov[i, i], 0))
        d = {'auc': float(aucs[i]), 'se_delong': se,
             'ci95_delong': [float(aucs[i] - 1.96 * se), float(aucs[i] + 1.96 * se)],
             'cohen_d': cohen_d(sc_case[k], sc_ctrl[k]),
             'mean_case': float(np.mean(sc_case[k])), 'mean_control': float(np.mean(sc_ctrl[k]))}
        if boot is not None and len(boot) > 1:
            d['ci95_block'] = [float(x) for x in np.percentile(boot[:, i], [2.5, 97.5])]
        res['scores'][k] = d
    for i, j in PAIRS:
        diff = float(aucs[j] - aucs[i])
        se = math.sqrt(max(cov[i, i] + cov[j, j] - 2 * cov[i, j], 0))
        d = {'diff': diff, 'se_delong': se, 'ci95_delong': [diff - 1.96 * se, diff + 1.96 * se],
             'p_delong': p_two_sided(diff / se) if se > 0 else None}
        if boot is not None and len(boot) > 1:
            bd = boot[:, j] - boot[:, i]
            d['ci95_block'] = [float(x) for x in np.percentile(bd, [2.5, 97.5])]
            d['se_block'] = float(np.std(bd, ddof=1))
        res['pairs'][f'{SCORES[j]}_minus_{SCORES[i]}'] = d
    if case_chr is not None:
        per = {}
        for c in np.unique(np.r_[case_chr, ctrl_chr]):
            a, b = case_chr == c, ctrl_chr == c
            if a.sum() < min_chrom_units or b.sum() < min_chrom_units:
                continue
            yy = np.r_[np.ones(a.sum(), bool), np.zeros(b.sum(), bool)]
            au = _aucs_resampled(yy, np.vstack([-np.r_[sc_case[k][a], sc_ctrl[k][b]] for k in SCORES]))
            per[str(int(c))] = {'n_case': int(a.sum()), 'n_control': int(b.sum()),
                                'auc': dict(zip(SCORES, map(float, au))),
                                'combined_minus_ovep_z': float(au[2] - au[0])}
        diffs = [v['combined_minus_ovep_z'] for v in per.values()]
        res['per_chromosome'] = per
        res['per_chromosome_summary'] = {
            'n_chromosomes': len(diffs),
            'share_positive_combined_minus_ovep_z': float(np.mean(np.array(diffs) > 0)) if diffs else None}
    return res


def discriminate(case_units, ctrl_units, n_bins=10, seed=0, min_matched=30, case_pos=None,
                 ctrl_pos=None, case_chr=None, ctrl_chr=None, block_bp=1_000_000, n_boot=500):
    """Primary analysis: all case vs all control units (the z-scores are
    already standardized to each unit's own expectation). With unit positions,
    95% intervals also come from a bootstrap over genomic blocks of
    `block_bp` (primary; DeLong treats units as independent and is secondary),
    and AUCs are reported per chromosome. 'matched': the same on mutability-
    matched units only; carries 'error' (and no AUCs) when fewer than
    `min_matched` units per group have comparable mutability."""
    case_all = np.asarray(case_units, float)
    ctrl_all = np.asarray(ctrl_units, float)
    kc, kk = case_all[:, 1] > 0, ctrl_all[:, 1] > 0
    have_pos = case_pos is not None and ctrl_pos is not None
    if have_pos:
        cc = np.zeros(len(case_all), int) if case_chr is None else np.asarray(case_chr)
        kch = np.zeros(len(ctrl_all), int) if ctrl_chr is None else np.asarray(ctrl_chr)
        cblk = cc.astype(np.int64) * 10**7 + np.asarray(case_pos) // block_bp
        kblk = kch.astype(np.int64) * 10**7 + np.asarray(ctrl_pos) // block_bp
        cblk, kblk, cc, kch = cblk[kc], kblk[kk], cc[kc], kch[kk]
    case_all, ctrl_all = case_all[kc], ctrl_all[kk]
    if len(case_all) < min_matched or len(ctrl_all) < min_matched:
        return {'error': 'too few units', 'n_case': int(len(case_all)), 'n_control': int(len(ctrl_all))}
    rng = np.random.default_rng(seed)
    pos_args = lambda ci, ki: ((cblk[ci], kblk[ki], cc[ci] if case_chr is not None else None,
                                kch[ki] if ctrl_chr is not None else None) if have_pos else (None,) * 4)
    every_c, every_k = np.arange(len(case_all)), np.arange(len(ctrl_all))
    scale = neutral_scale(ctrl_all)
    res = _auc_block(case_all, ctrl_all, *pos_args(every_c, every_k), n_boot=n_boot, rng=rng, scale=scale)
    res['block_bp'] = block_bp if have_pos else None
    res['neutral_scale'] = {k: {'mean': m, 'sd': sd} for k, (m, sd) in scale.items()}
    res['neutral_calibration'] = calibration_summary(ctrl_all)
    ci, ki = match_units(case_all[:, 1], ctrl_all[:, 1], n_bins, rng)
    matched = {'n_case_total': int(len(case_all)), 'share_of_case_units_matched': float(ci.size / len(case_all))}
    if ci.size < min_matched:
        matched['error'] = (f'only {ci.size} matched units per group (need {min_matched}); '
                            'case and control mutability barely overlap')
        matched['n_case'] = int(ci.size)
    else:
        matched |= _auc_block(case_all[ci], ctrl_all[ki], *pos_args(ci, ki), n_boot=n_boot, rng=rng,
                              scale=scale)
    res['matched'] = matched
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--blocks', nargs='+', required=True,
                   help='per-chromosome .npz from score_annotations.py --save-blocks --unit-sites N')
    p.add_argument('--case', default='CDS')
    p.add_argument('--control', default='neutralA')
    p.add_argument('--bins', type=int, default=10, help='quantile bins for mutability matching')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--block-bp', type=int, default=1_000_000, help='genomic block size for the bootstrap')
    p.add_argument('--bootstrap', type=int, default=500, help='block-bootstrap replicates (0 = DeLong only)')
    p.add_argument('--json')
    args = p.parse_args()
    loaded = [load_blocks(f)[0] for f in args.blocks]
    for name in (args.case, args.control):
        if not all(name in s and 'units' in s[name] for s in loaded):
            p.error(f'{name!r} units missing in some files (score_annotations.py --unit-sites)')
    if not all(s[args.case]['has_reference'] for s in loaded):
        p.error('a reference model (--original) is needed for ovep_z')
    case = pool([s[args.case] for s in loaded])
    ctrl = pool([s[args.control] for s in loaded])
    res = {'case': args.case, 'control': args.control, 'files': len(args.blocks)}
    res.update(discriminate(case['units'], ctrl['units'], args.bins, args.seed,
                            case_pos=case.get('units_pos'), ctrl_pos=ctrl.get('units_pos'),
                            case_chr=case.get('units_chrom'), ctrl_chr=ctrl.get('units_chrom'),
                            block_bp=args.block_bp, n_boot=args.bootstrap))
    text = json.dumps(res, indent=2)
    if args.json:
        with open(args.json + '.partial', 'w') as f:
            f.write(text)
        os.replace(args.json + '.partial', args.json)
    print(text)


if __name__ == '__main__':
    main()
