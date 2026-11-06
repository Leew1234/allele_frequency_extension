"""Tests for the analysis-plan tools: comparison engine (pooling, reference
models, interval masks), per-annotation scoring, de novo AUC with DeLong, and
the pre-registered criteria checker. NumPy only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'real_data'))
from compare_ovep import (IntervalMask, VariantScore, block_sums, summarize, pool, compare,  # noqa: E402
                          save_blocks, load_blocks, roc_auc)
from score_annotations import build_masks  # noqa: E402
from denovo_auc import delong, hanley_mcneil_se, collect, evaluate, read_dnms  # noqa: E402
import evaluate_plan  # noqa: E402
import prepare_dnm  # noqa: E402


def toy(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    rate = np.where(rng.random(n) < 0.3, 0.12, 0.02)
    var = rng.random(n) < rate
    labels = np.where(var, np.where(rng.random(n) < 0.4, 2, 1), 0).astype(np.int8)
    new = np.stack([1 - rate, rate * 0.6, rate * 0.4], 1)
    return labels, rate, new


class TestEngine(unittest.TestCase):
    def test_interval_mask_matches_dense(self):
        rng = np.random.default_rng(0)
        iv = [(int(a), int(a + rng.integers(1, 50))) for a in rng.integers(0, 1000, 60)]
        dense = np.zeros(1000, bool)
        for a, b in iv:
            dense[a:min(b, 1000)] = True
        m = IntervalMask(iv, 1000)
        got = np.concatenate([m[s:s + 37] for s in range(0, 1000, 37)])
        np.testing.assert_array_equal(got, dense)
        np.testing.assert_array_equal(np.concatenate([m.complement()[s:s + 101] for s in range(0, 1000, 101)]), ~dense)
        self.assertEqual(m.n_sites, int(dense.sum()))

    def test_variant_score(self):
        cls = np.array([[.9, .07, .03], [.5, .3, .2]])
        np.testing.assert_allclose(VariantScore(cls)[0:2], [.1, .5])
        np.testing.assert_allclose(VariantScore(np.array([.1, .2]))[0:2], [.1, .2])
        with self.assertRaises(ValueError):
            VariantScore(np.zeros((3, 2)))

    def test_pooling_equals_single_pass(self):
        labels, rate, new = toy()
        half = len(labels) // 2
        full = block_sums(labels, rate, new, {'all': np.ones(len(labels), bool)}, block_size=10_000, auc_sites=10**9)
        parts = [block_sums(labels[s], rate[s], new[s], {'all': np.ones(half, bool)}, block_size=10_000,
                            auc_sites=10**9) for s in (slice(0, half), slice(half, None))]
        pooled = pool([p['all'] for p in parts])
        np.testing.assert_allclose(pooled['blocks'].sum(0), full['all']['blocks'].sum(0))
        a, b = summarize(pooled, 0), summarize(full['all'], 0)
        for k in ('oe_new', 'oe_original', 'oe_common_share', 'auc_new'):
            self.assertAlmostEqual(a[k], b[k], places=9)

    def test_reference_can_be_another_class_model(self):
        labels, rate, new = toy()
        worse = np.stack([1 - rate / 2, rate / 2 * 0.6, rate / 2 * 0.4], 1)  # predicts half the variants
        r = compare(labels, worse, new, {'all': np.ones(len(labels), bool)}, block_size=10_000, n_boot=200)['all']
        self.assertAlmostEqual(r['oe_original'], 2.0, delta=0.08)
        self.assertAlmostEqual(r['oe_new'], 1.0, delta=0.04)
        self.assertLess(r['ci95']['oe_diff'][1], 0)
        self.assertLess(r['z']['oe_diff'], -5)
        self.assertAlmostEqual(r['z']['oe_new'], (r['oe_new'] - 1) / r['se']['oe_new'])

    def test_no_reference_and_save_load(self):
        labels, rate, new = toy(50_000)
        sums = block_sums(labels, None, new, {'all': np.ones(len(labels), bool)}, block_size=5_000)
        r = summarize(sums['all'], 50)
        self.assertNotIn('oe_original', r)
        self.assertIn('oe_new', r['ci95'])
        with tempfile.TemporaryDirectory() as d:
            save_blocks(Path(d) / 'b.npz', sums, {'x': 1})
            back, meta = load_blocks(Path(d) / 'b.npz')
            self.assertEqual(meta, {'x': 1})
            self.assertFalse(back['all']['has_reference'])
            np.testing.assert_array_equal(back['all']['blocks'], sums['all']['blocks'])


class TestAnnotations(unittest.TestCase):
    def test_build_masks_from_gff(self):
        with tempfile.TemporaryDirectory() as d:
            g = Path(d) / 'a.gff'
            g.write_text('##gff-version 3\n'
                         'S1\tRefSeq\tregion\t1\t100\t.\t+\t.\tID=r\n'      # excluded source
                         'S1\tBestRefSeq\tCDS\t11\t20\t.\t+\t0\tID=c\n'
                         'S1\tBestRefSeq\texon\t6\t25\t.\t+\t.\tID=e\n'
                         'S1\tGnomon\tbiological_region\t51\t60\t.\t.\t.\tID=b\n'  # skipped type, still annotated
                         'S2\tBestRefSeq\tCDS\t1\t5\t.\t+\t0\tID=x\n')
            m = build_masks(g, 'S1', 100)
            self.assertEqual(sorted(m), ['CDS', 'exon', 'unannotated'])
            self.assertEqual(np.flatnonzero(m['CDS'][0:100]).tolist(), list(range(10, 20)))
            un = m['unannotated'][0:100]
            self.assertFalse(un[5:25].any() or un[50:60].any())
            self.assertTrue(un[:5].all() and un[25:50].all() and un[60:].all())
            with self.assertRaises(ValueError):
                build_masks(g, 'S9', 100)


class TestDenovo(unittest.TestCase):
    def test_delong_matches_rank_auc_and_brute_force_variance(self):
        rng = np.random.default_rng(0)
        y = rng.random(400) < 0.2
        s1 = rng.normal(y * 1.0, 1)
        s2 = s1 + rng.normal(0, 1, 400)
        aucs, cov = delong(y, np.vstack([s1, s2]))
        self.assertAlmostEqual(aucs[0], roc_auc(y, s1), places=12)
        # brute-force DeLong structural components
        pos, neg = s1[y], s1[~y]
        psi = (pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])
        v10, v01 = psi.mean(1), psi.mean(0)
        brute = v10.var(ddof=1) / y.sum() + v01.var(ddof=1) / (~y).sum()
        self.assertAlmostEqual(cov[0, 0], brute, places=12)
        self.assertGreater(hanley_mcneil_se(0.7, 100, 1000), 0)

    def test_collect_and_evaluate_on_toy_genome(self):
        rng = np.random.default_rng(3)
        n = 60_000
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            dna = np.eye(4, dtype=np.uint8)[rng.integers(0, 4, n)]
            dna[100:200] = 0
            rate = np.where(rng.random(n) < 0.3, 0.1, 0.01)
            np.save(d / 'dna.npy', dna)
            np.save(d / 'good.npy', np.stack([1 - rate, rate * .5, rate * .5], 1))
            np.save(d / 'flat.npy', np.tile([.97, .02, .01], (n, 1)))
            orig = (rate * np.exp(rng.normal(0, .2, n)))[:-2048]
            raw = orig.copy(); raw[:512] = 0
            np.save(d / 'orig.npy', orig); np.save(d / 'raw.npy', raw)
            dnms = np.sort(rng.choice(n, 300, replace=False, p=rate / rate.sum()))
            (d / 'dnm.tsv').write_text('#chrom\tpos\n' + ''.join(f'chrX\t{p + 1}\n' for p in dnms))
            config = {'models': {'original': 'score', 'flat': 'classes', 'good': 'classes'},
                      'dnm': {'set': str(d / 'dnm.tsv')},
                      'chromosomes': {'SX': {'dnm_name': 'chrX', 'dna': str(d / 'dna.npy'),
                                             'predictions': {'original': {'path': str(d / 'orig.npy'),
                                                                          'raw': str(d / 'raw.npy')},
                                                             'flat': str(d / 'flat.npy'),
                                                             'good': str(d / 'good.npy')}}}}
            models, data = collect(config, {'set': read_dnms(d / 'dnm.tsv')}, 20_000, chunk=7_000)
            res = evaluate(models, data['set'])
            excluded = int(((dnms < 512) | ((dnms >= 100) & (dnms < 200)) | (dnms >= n - 2048)).sum())
            self.assertEqual(res['n_pos'], 300 - excluded)
            self.assertEqual(res['n_dnm_unscorable'], excluded)
            self.assertAlmostEqual(res['models']['flat']['auc'], 0.5)
            self.assertGreater(res['models']['good']['auc'], 0.65)
            self.assertLess(res['pairs']['good_minus_flat']['p_delong'], 1e-6)


class TestDiscrimination(unittest.TestCase):
    @staticmethod
    def units(rng, n, rate_mult, share_true, lo, hi):
        exp = rng.uniform(lo, hi, n) * 2000
        obs = rng.poisson(exp * rate_mult)
        return np.stack([obs, exp, exp, rng.binomial(obs, share_true), exp * 0.3, obs * 0.3, obs * 0.21], 1)

    def test_null_and_signal(self):
        from discriminate import discriminate
        rng = np.random.default_rng(1)
        ctrl = self.units(rng, 15000, 1, .3, .005, .045)  # partly overlaps the cases' mutability
        null = discriminate(self.units(rng, 1500, 1, .3, .02, .06), ctrl)
        m = null['matched']
        self.assertAlmostEqual(m['mean_expected_case'], m['mean_expected_control'], delta=1.5)
        self.assertLess(m['share_of_case_units_matched'], 1)
        for block in (null, m):
            for k in ('ovep_z', 'freq_z', 'combined'):
                self.assertLess(abs(block['scores'][k]['auc'] - 0.5), 0.04, k)
        for res in (discriminate(self.units(rng, 1500, 1, .15, .02, .06), ctrl),):
            self.assertGreater(res['pairs']['combined_minus_ovep_z']['ci95_delong'][0], 0)
            self.assertGreater(res['matched']['pairs']['combined_minus_ovep_z']['ci95_delong'][0], 0)
        depl = discriminate(self.units(rng, 1500, .7, .3, .02, .06), ctrl)
        self.assertLess(depl['pairs']['combined_minus_ovep_z']['ci95_delong'][1], 0)  # frequency adds nothing
        apart = discriminate(self.units(rng, 300, 1, .3, .08, .1), ctrl)  # no mutability overlap
        self.assertIn('error', apart['matched'])
        self.assertIn('pairs', apart)  # primary analysis still available

    def test_block_bootstrap_widens_only_when_signal_clusters(self):
        from discriminate import discriminate
        rng = np.random.default_rng(3)

        def make(n, clustered):
            pos = np.sort(rng.integers(0, 50_000_000, n))
            chrom = rng.integers(0, 5, n)
            s = np.clip(rng.normal(.15, .15, (5, 60)), -.2, .6)[chrom, pos // 1_000_000] if clustered else .15
            exp = rng.uniform(40, 80, n)
            obs = rng.poisson(exp)
            return np.stack([obs, exp, exp, rng.binomial(obs, .3 * (1 - s)), exp * .3, obs * .3, obs * .21], 1), pos, chrom

        widths = {}
        for clustered in (False, True):
            (c, cp, cc), (k, kp, kc) = make(1200, clustered), make(6000, False)
            k[:, 3] = rng.binomial(k[:, 0].astype(int), .3)
            r = discriminate(c, k, case_pos=cp, ctrl_pos=kp, case_chr=cc, ctrl_chr=kc, n_boot=200)
            p = r['pairs']['combined_minus_ovep_z']
            widths[clustered] = (p['ci95_block'][1] - p['ci95_block'][0]) / (p['ci95_delong'][1] - p['ci95_delong'][0])
            self.assertEqual(r['per_chromosome_summary']['n_chromosomes'], 5)
            self.assertGreater(r['per_chromosome_summary']['share_positive_combined_minus_ovep_z'], 0.5)
            self.assertIn('ci95_block', r['matched']['pairs']['combined_minus_ovep_z'])
        self.assertLess(abs(widths[False] - 1), 0.3)   # independent units: block ~ DeLong
        self.assertGreater(widths[True], 1.3)          # clustered signal: DeLong too narrow

    def test_neutral_scaling_protects_against_overdispersion(self):
        from discriminate import discriminate
        rng = np.random.default_rng(4)

        def units(n, rate_mult, shift, overdispersed):
            exp = rng.uniform(40, 80, n)
            obs = rng.poisson(exp * rate_mult)
            p = (rng.beta(.9, 2.1, n) if overdispersed else np.full(n, .3)) * (1 - shift)
            return np.stack([obs, exp, exp, rng.binomial(obs, p), exp * .3, obs * .3, obs * .21], 1)

        calm = discriminate(units(1500, .8, .1, False), units(8000, 1, 0, False), n_boot=0)
        self.assertAlmostEqual(calm['neutral_calibration']['freq_z']['sd'], 1, delta=0.05)
        self.assertAlmostEqual(calm['neutral_calibration']['freq_z']['share_beyond_1.96'], 0.05, delta=0.01)
        self.assertAlmostEqual(calm['scores']['combined']['auc'], calm['scores']['combined_unscaled']['auc'], delta=0.01)
        wild = discriminate(units(1500, .8, .1, True), units(8000, 1, 0, True), n_boot=0)
        self.assertGreater(wild['neutral_calibration']['freq_z']['sd'], 2)
        self.assertGreater(wild['scores']['combined']['auc'], wild['scores']['combined_unscaled']['auc'] + 0.1)
        self.assertIn('combined_unscaled_minus_ovep_z', wild['pairs'])

    def test_frequency_expectation_is_conditioned_on_variant_sites(self):
        # Truth: CpG-like sites mutate more AND their variants are more often common.
        # A mutability model that only partly sees the difference (calibrated, but
        # with imperfect ranking) biases the unconditional expected share low;
        # conditioning on where variants occurred removes the bias.
        rng = np.random.default_rng(7)
        n = 400_000
        hot = rng.random(n) < 0.1
        pv_true = np.where(hot, 0.16, 0.02)
        pc_true = np.where(hot, 0.6, 0.2)
        var = rng.random(n) < pv_true
        labels = np.where(var, np.where(rng.random(n) < pc_true, 2, 1), 0).astype(np.int8)
        seen = np.where(rng.random(n) < 0.5, hot, rng.random(n) < 0.1)   # model sees 'hot' only half the time
        pv = np.where(seen, var[seen].mean(), var[~seen].mean())          # calibrated given what it sees
        pc = np.where(hot, 0.6, 0.2)                                      # P(common | variant) exact
        new = np.stack([1 - pv, pv * (1 - pc), pv * pc], 1)
        r = summarize(block_sums(labels, None, new, {'all': np.ones(n, bool)})['all'], 0)
        self.assertAlmostEqual(r['oe_common_share'], 1.0, delta=0.03)
        self.assertGreater(r['oe_common_share_marginal'], 1.08)

    def test_units_flow_from_score_annotations_blocks(self):
        from discriminate import discriminate
        labels, rate, new = toy(100_000)
        m = np.zeros(len(labels), bool)
        m[::3] = True
        s = block_sums(labels, rate, new, {'a': m, 'b': ~m}, unit_sites=300)
        self.assertEqual(s['a']['units'].shape[1], 7)
        self.assertEqual(len(s['a']['units']), int(((labels >= 0) & m).sum()) // 300)
        self.assertTrue(np.array_equal(s['a']['units_pos'],
                                       np.flatnonzero((labels >= 0) & m)[::300][:len(s['a']['units'])]))
        pooled = pool([s['a'], s['a']])
        self.assertEqual(np.bincount(pooled['units_chrom']).tolist(), [len(s['a']['units'])] * 2)
        r = discriminate(s['a']['units'], s['b']['units'])
        self.assertLess(abs(r['scores']['combined']['auc'] - 0.5), 0.1)


class TestLabelDefinitions(unittest.TestCase):
    def test_original_indel_mode_reproduces_original_code(self):
        from data import vcf_categories
        ref = 'ACCTGTAGAAAACGTTCAGTACGTAGCT'
        R = lambda p, n: ref[p - 1:p - 1 + n]
        recs = [(2, R(2, 1), 'G'), (4, R(4, 3), R(4, 1)), (9, R(9, 1), R(9, 1) + 'T'),
                (12, R(12, 1), 'G'), (14, R(14, 2), 'CA'), (20, R(20, 1), 'A,C')]
        with tempfile.TemporaryDirectory() as d:
            v = Path(d) / 't.vcf'
            v.write_text('#h\n' + ''.join(
                f'chr1\t{p}\t.\t{r}\t{a}\t.\tPASS\tAC={",".join(["40"] * len(a.split(",")))};AN=200\n'
                for p, r, a in recs))
            ours = (vcf_categories(v, 'chr1', ref, indel_mode='original') >= 1).astype(int)
        orig = np.zeros(len(ref), int)
        for p, r, a in recs:  # 1000genome/select_indel.py + pickle_label_chr.py
            end = p + 1 if len(a) == len(r) else p + len(r) + 1
            orig[p - 1:end - 1] = 1
        np.testing.assert_array_equal(ours, orig)

    def test_compare_labels(self):
        from compare_labels import compare as cmp
        ext = np.array([-1, -1, 0, 0, 1, 2, 1, 0], np.int8)
        orig = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1.0])  # one longer
        r = cmp(ext, orig, chunk=3)
        self.assertEqual(r['counts'], {'both_variant': 2, 'original_only': 1, 'extension_only': 1, 'neither': 2,
                                       'excluded_original_variant': 1, 'excluded_original_invariant': 1})
        self.assertAlmostEqual(r['agreement_on_scored_sites'], 4 / 6)


class TestPrepareDnm(unittest.TestCase):
    def _run(self, name, content, *extra):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / name, Path(d) / 'out.tsv'
            if name.endswith('.xlsx'):
                import openpyxl
                wb = openpyxl.Workbook()
                for row in content:
                    wb.active.append(row)
                wb.save(src)
            else:
                src.write_text(content)
            stats = prepare_dnm.main([str(src), '-o', str(out), *extra])
            return read_dnms(out), stats

    def test_supplement_with_title_rows_and_builds(self):
        rows = [['Supplementary Table 4. De novo mutations'], [],
                ['Chr', 'Pos_hg19', 'Pos_hg38', 'Ref', 'Alt', 'Proband'],
                ['chr1', 900, 1000, 'A', 'G', 'p1'], ['1', 950, 1050, 'C', 'T', 'p2'],
                ['chr1', 900, 1000, 'A', 'G', 'p3'],          # duplicate site
                ['X', 5, 20, 'G', 'A', 'p1'], ['chrM', 1, 7, 'A', 'G', 'p1'],
                ['chr2', 10, 300, 'AT', 'A', 'p4']]           # indel
        got, stats = self._run('s4.xlsx', rows)
        self.assertEqual(stats['pos_column'], 'Pos_hg38')
        self.assertEqual(got, {'chr1': {999, 1049}, 'chrX': {19}})
        self.assertEqual((stats['indels'], stats['duplicates'], stats['non_autosomal_xy']), (1, 1, 1))
        got, _ = self._run('s4.xlsx', rows, '--keep-indels')
        self.assertEqual(got['chr2'], {299})

    def test_text_explicit_columns_and_zero_based(self):
        got, _ = self._run('d.csv', 'a,b,c\n17,x,99\n23,x,5\n', '--chrom-col', 'a', '--pos-col', '2')
        self.assertEqual(got, {'chr17': {98}, 'chrX': {4}})
        got, _ = self._run('d.bed', '1\t99\t100\n', '--no-header', '--zero-based')
        self.assertEqual(got, {'chr1': {99}})


class TestCriteria(unittest.TestCase):
    def test_freeze_detects_edits(self):
        with tempfile.TemporaryDirectory() as d:
            c = Path(d) / 'criteria.json'
            c.write_text('{"criteria": []}')
            Path(d, 'criteria.sha256').write_text(evaluate_plan.sha256(c) + '  criteria.json\n')
            self.assertTrue(evaluate_plan.freeze_status(c)[2])
            c.write_text('{"criteria": [] }')
            cur, frozen, ok = evaluate_plan.freeze_status(c)
            self.assertFalse(ok)
            self.assertIn('DOES NOT match', evaluate_plan.report({'criteria': [], 'decisions': {}}, {}, {},
                                                               (cur, frozen, ok)))

    def test_pass_fail_missing_and_require(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for h, share in (('half1', 0.9), ('half2', 0.95)):
                (d / h).mkdir()
                (d / h / 'neutral.json').write_text(json.dumps(
                    {'categories': {'CDS': {'ci95': {'oe_common_share': [0.8, share]},
                                            'z': {'oe_common_share': -4.0, 'oe_original': -3.0}}}}))
                for s, hi in enumerate((0.9, 0.95, 1.1, 0.97, 0.99)):
                    (d / h / f'neutral_seed{s}.json').write_text(json.dumps(
                        {'categories': {'CDS': {'ci95': {'oe_common_share': [0.5, hi]}}}}))
            doc = {'criteria': [
                {'id': 'A', 'group': 'g', 'claim': 'c', 'op': '<', 'right': 1.0, 'over': {'half': ['half1', 'half2']},
                 'left': {'file': '{half}/neutral.json', 'path': ['categories', 'CDS', 'ci95', 'oe_common_share', 1]}},
                {'id': 'B', 'group': 'g', 'claim': 'c', 'op': '<=', 'over': {'half': ['half1', 'half2']},
                 'left': {'file': '{half}/neutral.json', 'path': ['categories', 'CDS', 'z', 'oe_common_share']},
                 'right': {'file': '{half}/neutral.json', 'path': ['categories', 'CDS', 'z', 'oe_original']}},
                {'id': 'C', 'group': 'g', 'claim': 'c', 'op': '<', 'right': 0.92, 'over': {'half': ['half1', 'half2']},
                 'left': {'file': '{half}/neutral.json', 'path': ['categories', 'CDS', 'ci95', 'oe_common_share', 1]}},
                {'id': 'D', 'group': 'g', 'claim': 'c', 'op': 'between', 'right': [0.9, 1.1],
                 'left': {'file': 'half1/missing.json', 'path': ['x']}},
                {'id': 'E', 'group': 'g', 'claim': 'c', 'op': '<', 'right': 1.0, 'require': {'seed': 4},
                 'over': {'half': ['half1', 'half2'], 'seed': ['0', '1', '2', '3', '4']},
                 'left': {'file': '{half}/neutral_seed{seed}.json', 'path': ['categories', 'CDS', 'ci95', 'oe_common_share', 1]}},
                {'id': 'F', 'group': 'g', 'claim': 'c', 'op': '<', 'right': 0.96, 'require': {'seed': 4},
                 'over': {'half': ['half1'], 'seed': ['0', '1', '2', '3', '4']},
                 'left': {'file': '{half}/neutral_seed{seed}.json', 'path': ['categories', 'CDS', 'ci95', 'oe_common_share', 1]}}],
                'decisions': {'yes': ['A', 'B'], 'no': ['A', 'C'], 'incomplete': ['A', 'D']}}
            out, dec = evaluate_plan.evaluate(doc, d)
            self.assertEqual({k: v['status'] for k, v in out.items()},
                             {'A': 'PASS', 'B': 'PASS', 'C': 'FAIL', 'D': 'MISSING', 'E': 'PASS', 'F': 'FAIL'})
            self.assertEqual(dec, {'yes': 'YES', 'no': 'NO', 'incomplete': 'INCOMPLETE'})
            text = evaluate_plan.report(doc, out, dec)
            self.assertIn('| yes | **YES** | A PASS, B PASS |', text)

    def test_shipped_criteria_file_is_well_formed(self):
        doc = json.loads((ROOT / 'experiments' / 'real_data' / 'criteria.json').read_text())
        ids = {c['id'] for c in doc['criteria']}
        for c in doc['criteria']:
            self.assertIn(c['op'], evaluate_plan.OPS)
            for dim in c.get('require', {}):
                self.assertIn(dim, c['over'])
        for crit in doc['decisions'].values():
            self.assertTrue(set(crit) <= ids)
        self.assertTrue(evaluate_plan.freeze_status(ROOT / 'experiments' / 'real_data' / 'criteria.json')[2],
                        'criteria.json changed since it was frozen (criteria.sha256)')


if __name__ == '__main__':
    unittest.main()
