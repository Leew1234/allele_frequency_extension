import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import encode_dna, vcf_categories, apply_callable_bed, site_category, bed_to_mask
from score import category_ratios, load_class_probabilities
from train import valid_starts, ChromosomeSampler, make_batch, masked_labels
from predict import tile_predict, window_starts, accumulate_classes, finalize
from model import factorized_to_classes, unweight_softmax, to_class_probabilities
from calibrate import pav, fit_isotonic, apply_isotonic, calibrate_classes, fit_calibration

HAS_TF = importlib.util.find_spec('tensorflow') is not None
HEADER = '##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'


def write_vcf(d, records):
    path = Path(d) / 'tiny.vcf'
    path.write_text(HEADER + ''.join('\t'.join(r) + '\n' for r in records))
    return path


def rec(pos, ref, alt, info, filt='PASS', chrom='chr1'):
    return (chrom, str(pos), '.', ref, alt, '.', filt, info)


class TestLabels(unittest.TestCase):
    def test_encoding_matches_original(self):
        self.assertEqual(encode_dna('ACTGN').tolist(),
                         [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        self.assertEqual(encode_dna('actg').tolist(), encode_dna('ACTG').tolist())

    def test_basic_labels_and_callability(self):
        with tempfile.TemporaryDirectory() as d:
            vcf = write_vcf(d, [rec(2, 'C', 'G', 'AC=1;AN=200'),
                                rec(3, 'T', 'G', 'AC=40;AN=200'),
                                rec(5, 'N', 'A', 'AC=1;AN=200'),
                                rec(7, 'A', 'G', 'DP=20')])
            labels = vcf_categories(vcf, 'chr1', 'ACTGNAA', 0.01)
            self.assertEqual(labels.tolist(), [0, 1, 2, 0, -1, 0, -1])
            bed = Path(d) / 'callable.bed'
            bed.write_text('chr1\t0\t3\n')
            self.assertEqual(apply_callable_bed(labels, bed, 'chr1').tolist(),
                             [0, 1, 2, -1, -1, -1, -1])

    def test_unobserved_alt_is_not_a_variant(self):
        with tempfile.TemporaryDirectory() as d:
            vcf = write_vcf(d, [rec(1, 'A', 'G', 'AC=0;AN=200')])
            self.assertEqual(vcf_categories(vcf, 'chr1', 'ACTG').tolist(), [0, 0, 0, 0])

    def test_rare_reference_allele_is_rare(self):
        with tempfile.TemporaryDirectory() as d:
            vcf = write_vcf(d, [rec(1, 'A', 'G', 'AC=199;AN=200'),
                                rec(2, 'C', 'G', 'AC=150;AN=200')])
            self.assertEqual(vcf_categories(vcf, 'chr1', 'ACTG').tolist(), [1, 2, 0, 0])

    def test_record_order_does_not_matter(self):
        good, bad = rec(3, 'T', 'G', 'AC=40;AN=200'), rec(3, 'TX', 'G', 'AC=1;AN=200')
        with tempfile.TemporaryDirectory() as d:
            a = vcf_categories(write_vcf(d, [good, bad]), 'chr1', 'ACTG')
            b = vcf_categories(write_vcf(d, [bad, good]), 'chr1', 'ACTG')
            self.assertEqual(a.tolist(), b.tolist())
            self.assertEqual(a[2], 2)
            only_bad = vcf_categories(write_vcf(d, [bad]), 'chr1', 'ACTG')
            self.assertEqual(only_bad[2], -1)

    def test_split_multiallelic_records_are_merged(self):
        with tempfile.TemporaryDirectory() as d:
            # REF at 0.5%, two ALTs at ~50%: common
            vcf = write_vcf(d, [rec(1, 'A', 'G', 'AC=100;AN=200'), rec(1, 'A', 'C', 'AC=99;AN=200')])
            self.assertEqual(vcf_categories(vcf, 'chr1', 'ACTG')[0], 2)
            # counts exceeding AN once merged -> ignore
            vcf = write_vcf(d, [rec(1, 'A', 'G', 'AC=150;AN=200'), rec(1, 'A', 'C', 'AC=100;AN=200')])
            self.assertEqual(vcf_categories(vcf, 'chr1', 'ACTG')[0], -1)

    def test_star_allele_is_not_variation_here(self):
        with tempfile.TemporaryDirectory() as d:
            vcf = write_vcf(d, [rec(1, 'A', '*,G', 'AC=150,1;AN=200'), rec(2, 'C', '*', 'AC=5;AN=200')])
            self.assertEqual(vcf_categories(vcf, 'chr1', 'ACTG')[:2].tolist(), [1, 0])

    def test_filters(self):
        with tempfile.TemporaryDirectory() as d:
            vcf = write_vcf(d, [rec(1, 'A', 'G', 'AC=40;AN=200', filt='LowQual')])
            self.assertEqual(vcf_categories(vcf, 'chr1', 'ACTG')[0], -1)
            self.assertEqual(vcf_categories(vcf, 'chr1', 'ACTG', pass_only=False)[0], 2)

    def test_indel_modes(self):
        with tempfile.TemporaryDirectory() as d:
            vcf = write_vcf(d, [rec(2, 'CTG', 'C', 'AC=40;AN=200')])
            self.assertEqual(vcf_categories(vcf, 'chr1', 'ACTGA').tolist(), [0, 2, 0, 0, 0])
            self.assertEqual(vcf_categories(vcf, 'chr1', 'ACTGA', indel_mode='span').tolist(),
                             [0, 2, 2, 2, 0])

    def test_other_contigs_and_bad_bed(self):
        with tempfile.TemporaryDirectory() as d:
            vcf = write_vcf(d, [rec(1, 'A', 'G', 'AC=40;AN=200', chrom='chr10')])
            self.assertEqual(vcf_categories(vcf, 'chr1', 'ACTG').tolist(), [0, 0, 0, 0])
            bed = Path(d) / 'b.bed'
            bed.write_text('1\t0\t4\n')  # wrong naming convention
            with self.assertRaises(ValueError):
                apply_callable_bed(np.zeros(4, np.int8), bed, 'chr1')

    def test_site_category(self):
        self.assertEqual(site_category(200, [0]), 0)
        self.assertEqual(site_category(200, [1]), 1)
        self.assertEqual(site_category(200, [2]), 2)
        self.assertEqual(site_category(200, [200]), 0)  # ALT fixed in cohort
        self.assertEqual(site_category(200, [150, 50]), 2)  # REF absent, two ALTs
        with self.assertRaises(ValueError):
            site_category(200, [201])

    def test_fixed_alt_and_conflicting_an(self):
        with tempfile.TemporaryDirectory() as d:
            vcf = write_vcf(d, [rec(1, 'A', 'G', 'AC=200;AN=200'),
                                rec(2, 'C', 'G', 'AC=1;AN=200'), rec(2, 'C', 'T', 'AC=1;AN=180')])
            self.assertEqual(vcf_categories(vcf, 'chr1', 'ACTG')[:2].tolist(), [0, -1])


class TestLabelValidation(unittest.TestCase):
    def test_compare_with_table(self):
        from validate_labels import expected_from_table, compare
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / 't.tsv'
            t.write_text('chr1\t1\tA\tG\t200\t0.005\n'
                         'chr1\t2\tC\tG\t200\t0.2\n'
                         'chr1\t3\tT\tG\t200\t0\n'
                         'chr1\t4\tG\tA\t.\t.\n'
                         'chr2\t1\tA\tG\t200\t0.2\n')
            exp, maf = expected_from_table(t, 'chr1', 6, 0.01)
            self.assertEqual(exp.tolist(), [1, 2, 0, -2, -9, -9])
            labels = np.array([1, 1, 0, 0, 2, -1], np.int8)
            s, mism = compare(labels, exp, maf, 0.01)
            self.assertEqual(s['record_sites_compared'], 3)
            self.assertEqual(mism.tolist(), [1])
            self.assertEqual(s['variant_without_record'], 1)
            self.assertEqual(s['bcftools_records_without_usable_maf'], 1)

    @unittest.skipUnless(shutil.which('bcftools'), 'bcftools not installed')
    def test_labels_match_bcftools(self):
        """Random genotypes (multi-allelic, deletions, rare/fixed REF, missing
        calls): data.py on INFO AC/AN of split records must equal bcftools'
        genotype-derived MAF class on joined records."""
        from validate_labels import expected_from_table, compare
        rng = np.random.default_rng(0)
        L, n_s = 3000, 60
        seq = ''.join(rng.choice(list('ACGT'), L))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            lines = ['##fileformat=VCFv4.2', f'##contig=<ID=chr1,length={L}>',
                     '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
                     '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'
                     + '\t'.join(f's{i}' for i in range(n_s))]
            for pos in np.sort(rng.choice(np.arange(5, L - 5, 3), 300, replace=False)):
                kind = rng.integers(5)
                ref = seq[pos - 1]
                others = [b for b in 'ACGT' if b != ref]
                alts = list(rng.choice(others, 2, replace=False)) if kind == 1 else [rng.choice(others)]
                if kind == 2:
                    ref, alts = seq[pos - 1:pos + 2], [seq[pos - 1]]
                f = {3: [0.01, 0.99], 4: [1.0, 0.0]}.get(kind, None)
                f = np.array(f) if f is not None else rng.dirichlet(np.r_[10, np.ones(len(alts))])
                gts = ['./.' if rng.random() < .03 else '/'.join(map(str, rng.choice(len(f), 2, p=f)))
                       for _ in range(n_s)]
                lines.append(f'chr1\t{pos}\t.\t{ref}\t{",".join(alts)}\t.\tPASS\t.\tGT\t' + '\t'.join(gts))
            (d / 'gt.vcf').write_text('\n'.join(lines) + '\n')
            sh = lambda c: subprocess.run(c, shell=True, check=True, cwd=d, capture_output=True)
            sh("bcftools +fill-tags gt.vcf -- -t AC,AN | bcftools norm -m- | bcftools view -G -o sites.vcf")
            sh("bcftools norm -m+any gt.vcf | bcftools +fill-tags -- -t AC,AN,MAF | "
               "bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\t%INFO/AN\\t%INFO/MAF\\n' > table.tsv")
            labels = vcf_categories(d / 'sites.vcf', 'chr1', seq, 0.01)
            exp, maf = expected_from_table(d / 'table.tsv', 'chr1', L, 0.01)
            s, _ = compare(labels, exp, maf, 0.01)
            self.assertEqual(s['record_sites_compared'], 300)
            self.assertEqual(s['mismatches'], 0, s)
            self.assertEqual(s['variant_without_record'], 0)


class TestTraining(unittest.TestCase):
    def test_valid_starts_matches_brute_force(self):
        rng = np.random.default_rng(1)
        dna = np.eye(4, dtype=np.uint8)[rng.integers(0, 4, 300)]
        dna[rng.choice(300, 4, replace=False)] = 0
        labels = rng.choice([-1, 0, 1], 300, p=[.6, .3, .1]).astype(np.int8)
        labels[100:200] = -1
        w = 16
        ok = valid_starts(dna, labels, w, chunk=37)
        brute = [dna[s:s + w].max(1).min() > 0 and (labels[s:s + w] >= 0).any() for s in range(300 - w + 1)]
        self.assertEqual(ok.tolist(), brute)
        self.assertEqual(len(valid_starts(dna[:10], labels[:10], w)), 0)

    def test_sampler_only_returns_valid_starts(self):
        dna = np.tile(np.eye(4, dtype=np.uint8), (50, 1))
        dna[90] = 0
        labels = np.zeros(200, np.int8)
        s = ChromosomeSampler(dna, labels, 16)
        rng = np.random.default_rng(0)
        for _ in range(500):
            st = s.sample(rng)
            self.assertFalse(75 <= st <= 90)

    def test_softmax_batch_masks_ignored_sites(self):
        dna = np.tile(np.eye(4, dtype=np.uint8), (8, 1))
        labels = np.array([0, 1, 2, -1] * 8, dtype=np.int8)
        X, y, w = make_batch({'c': (dna, labels)}, [('c', 0)], 16, 'softmax', [1., 5., 10.])
        self.assertEqual(X.shape, (1, 16, 4))
        self.assertEqual(y.min(), 0)
        self.assertEqual(w[0, :4].tolist(), [1., 5., 10., 0.])
        X, y = make_batch({'c': (dna, labels)}, [('c', 0)], 16, 'factorized', None)
        self.assertEqual(y[0, 3], -1)


class TestNeutralMasks(unittest.TestCase):
    def test_bed_to_mask(self):
        with tempfile.TemporaryDirectory() as d:
            bed = Path(d) / 'n.bed'
            bed.write_text('chr1\t2\t4\nchr2\t0\t9\nchr1\t8\t20\n')
            self.assertEqual(bed_to_mask(bed, 'chr1', 10).astype(int).tolist(),
                             [0, 0, 1, 1, 0, 0, 0, 0, 1, 1])

    def test_masked_labels_scopes(self):
        y = np.array([-1, 0, 1, 2, 0, 1, 2])
        m = np.array([1, 1, 1, 1, 0, 0, 0], bool)
        self.assertEqual(masked_labels(y, m, 'none').tolist(), y.tolist())
        self.assertEqual(masked_labels(y, m, 'all').tolist(), [-1, 0, 1, 2, -1, -1, -1])
        self.assertEqual(masked_labels(y, m, 'common').tolist(), [-1, 0, 1, 2, 0, 3, 3])
        self.assertEqual(masked_labels(y, None, 'all').tolist(), y.tolist())

    def test_valid_starts_respects_mask(self):
        rng = np.random.default_rng(3)
        dna = np.eye(4, dtype=np.uint8)[rng.integers(0, 4, 400)]
        labels = rng.choice([-1, 0, 1], 400, p=[.1, .8, .1]).astype(np.int8)
        mask = np.zeros(400, bool)
        mask[150:170] = True
        w = 32
        ok = valid_starts(dna, labels, w, mask, chunk=50)
        brute = [((labels[s:s + w] >= 0) & mask[s:s + w]).any() for s in range(400 - w + 1)]
        self.assertEqual(ok.tolist(), brute)

    def test_batch_with_mask(self):
        dna = np.tile(np.eye(4, dtype=np.uint8), (4, 1))
        labels = np.array([0, 1, 2, 1] * 4, dtype=np.int8)
        mask = np.array([True] * 8 + [False] * 8)
        data = {'c': (dna, labels, mask)}
        _, y = make_batch(data, [('c', 0)], 16, 'factorized', None, 'common')
        self.assertEqual(y[0].tolist(), [0, 1, 2, 1] * 2 + [0, 3, 3, 3] * 2)
        _, y, w = make_batch(data, [('c', 0)], 16, 'softmax', [1, 1, 1], 'all')
        self.assertTrue(np.all(w[0, 8:] == 0) and np.all(w[0, :8] == 1))

    def test_calibration_mask_targets_masked_rate(self):
        rng = np.random.default_rng(2)
        n = 200_000
        labels = (rng.random(n) < np.where(np.arange(n) < n // 2, 0.02, 0.10)).astype(np.int8)
        probs = np.tile([0.95, 0.04, 0.01], (n, 1))
        mask = np.arange(n) < n // 2
        cal = fit_calibration(labels, probs, n_bins=10, mask=mask, chunk=7_777, max_sites=50_000)
        self.assertTrue(cal['masked'])
        self.assertLessEqual(cal['n_sites'], 60_000)
        pv = calibrate_classes(probs[:1], cal)[0, 1:].sum()
        self.assertAlmostEqual(pv, 0.02, delta=0.004)
        with self.assertRaises(ValueError):
            fit_calibration(labels, probs, mask=mask[:10])


class TestPrediction(unittest.TestCase):
    def test_window_starts_cover_end(self):
        self.assertEqual(window_starts(10, 16, 4), [])
        self.assertEqual(window_starts(16, 16, 4), [0])
        self.assertEqual(window_starts(23, 16, 4), [0, 4, 7])

    def test_tiling_is_aligned_and_complete(self):
        rng = np.random.default_rng(0)
        dna = np.eye(4, dtype=np.uint8)[rng.integers(0, 4, 1001)]
        # model that outputs "is this base A?" -> averaging must reproduce it exactly
        out = tile_predict(dna, lambda x: x[..., :1], 64, 1, batch_size=7)
        np.testing.assert_allclose(out[:, 0], dna[:, 0])

    def test_tiling_ambiguous_base_leaves_local_gap_only(self):
        dna = np.tile(np.eye(4, dtype=np.uint8), (250, 1))
        dna[500] = 0
        out = tile_predict(dna, lambda x: np.ones(x.shape[:2] + (1,)), 64, 1)
        nan = np.isnan(out[:, 0])
        self.assertTrue(nan[500])
        self.assertFalse(nan[:450].any() or nan[550:].any())

    def test_tiling_into_disk_memmap(self):
        from numpy.lib.format import open_memmap
        rng = np.random.default_rng(5)
        dna = np.eye(4, dtype=np.uint8)[rng.integers(0, 4, 777)]
        with tempfile.TemporaryDirectory() as d:
            out = open_memmap(str(Path(d) / 'raw.npy'), mode='w+', dtype=np.float32, shape=(777, 2))
            tile_predict(dna, lambda x: x[..., :2], 64, 2, batch_size=5, out=out, chunk=100)
            out.flush()
            np.testing.assert_allclose(np.load(Path(d) / 'raw.npy'), dna[:, :2])

    def test_ensemble_average_and_calibration(self):
        n = 1000
        raw_a = np.tile([0.2, 0.5], (n, 1)).astype(np.float32)          # factorized
        raw_b = np.tile([0.70, 0.20, 0.10], (n, 1)).astype(np.float32)  # softmax
        raw_a[3] = np.nan
        acc = np.zeros((n, 3), np.float32)
        accumulate_classes(acc, raw_a, {'head': 'factorized'}, chunk=128)
        accumulate_classes(acc, raw_b, {'head': 'softmax'}, chunk=128)
        finalize(acc, 2, chunk=128)
        np.testing.assert_allclose(acc[0], [(0.8 + 0.7) / 2, (0.1 + 0.2) / 2, (0.1 + 0.1) / 2], rtol=1e-6)
        self.assertTrue(np.isnan(acc[3]).all())
        # identity calibration leaves values unchanged
        ident = {'variant': [[0, 1], [0, 1]], 'common_given_variant': [[0, 1], [0, 1]]}
        acc2 = acc.copy()
        finalize(acc2, 1, ident, chunk=128)
        np.testing.assert_allclose(acc2[0], acc[0], rtol=1e-5)

    def test_weight_correction(self):
        p, w = 0.02, 10.0
        learned = w * p / (w * p + 1 - p)  # optimum of weighted BCE
        cls = factorized_to_classes([[learned, 0.3]], variant_pos_weight=w)
        self.assertAlmostEqual(float(cls[0, 1] + cls[0, 2]), p, places=6)
        self.assertAlmostEqual(float(cls.sum()), 1.0, places=6)
        truth = np.array([.97, .02, .01])
        cw = np.array([1, 10, 10])
        learned = truth * cw / (truth * cw).sum()
        np.testing.assert_allclose(unweight_softmax(learned, cw), truth, rtol=1e-6)
        np.testing.assert_allclose(to_class_probabilities(learned[None], {'class_weights': cw})[0],
                                   truth, rtol=1e-6)


class TestCalibration(unittest.TestCase):
    def test_pav(self):
        fit = pav([1, 3, 2, 4], [1, 1, 1, 1])
        self.assertEqual(fit.tolist(), [1, 2.5, 2.5, 4])

    def test_isotonic_recovers_rate(self):
        rng = np.random.default_rng(0)
        true = rng.uniform(0, 0.2, 200_000)
        y = rng.random(true.size) < true
        score = np.sqrt(true)  # monotone but miscalibrated
        m = fit_isotonic(score, y, 200)
        cal = apply_isotonic(score, m)
        self.assertLess(np.abs(cal - true).mean(), 0.01)
        self.assertTrue(np.isnan(apply_isotonic([np.nan], m)[0]))

    def test_calibrate_classes(self):
        rng = np.random.default_rng(1)
        n = 100_000
        pv, pc = rng.uniform(0, .3, n), rng.uniform(0, 1, n)
        var = rng.random(n) < pv
        labels = np.where(var, np.where(rng.random(n) < pc, 2, 1), 0)
        probs = np.stack([1 - pv / 2, pv / 2 * (1 - pc), pv / 2 * pc], 1)  # P(variant) halved
        probs[0] = np.nan
        cal = fit_calibration(labels, probs, n_bins=100)
        fixed = calibrate_classes(probs, cal)
        self.assertTrue(np.isnan(fixed[0]).all())
        np.testing.assert_allclose(np.nansum(fixed, 1)[1:], 1, rtol=1e-5)
        r = category_ratios(labels, fixed, n_boot=0)
        self.assertAlmostEqual(r['oe_any'], 1, delta=0.03)
        self.assertAlmostEqual(r['oe_common_share'], 1, delta=0.03)


class TestScore(unittest.TestCase):
    def test_score_and_mask(self):
        y = np.array([0, 1, 2, -1])
        pred = np.array([[.8, .1, .1], [.1, .5, .4], [.2, .2, .6], [.3, .3, .4]])
        r = category_ratios(y, pred, n_boot=0)
        self.assertEqual(r['observed'], [1, 1, 1])
        self.assertAlmostEqual(r['oe_rare'], 1 / .8)
        self.assertAlmostEqual(r['oe_common'], 1 / 1.1)
        # conditional on the two variant sites: expected common = .4/.9 + .6/.8
        self.assertAlmostEqual(r['oe_common_share'], 1 / (.4 / .9 + .6 / .8))
        self.assertAlmostEqual(r['oe_common_share_marginal'], 0.5 / (1.1 / 1.9))
        r = category_ratios(y, pred, mask=np.array([1, 1, 0, 0]), n_boot=0)
        self.assertEqual(r['observed'], [1, 1, 0])
        with self.assertRaises(ValueError):
            category_ratios(y, pred, mask=np.ones(3, bool))
        with self.assertRaises(ValueError):
            category_ratios(y[:3], pred)

    def test_chunked_scoring_matches_and_reads_files(self):
        rng = np.random.default_rng(4)
        n = 50_000
        pred = rng.dirichlet([30, 2, 1], n).astype(np.float32)
        labels = rng.choice([-1, 0, 1, 2], n, p=[.05, .85, .07, .03]).astype(np.int8)
        pred[100:200] = np.nan
        mask = rng.random(n) < 0.7
        a = category_ratios(labels, pred, mask, block_size=1000, n_boot=50)
        b = category_ratios(labels, pred, mask, block_size=1000, n_boot=50, chunk=777)
        self.assertEqual(a['observed'], b['observed'])
        np.testing.assert_allclose(a['expected'], b['expected'], rtol=1e-9)
        for k in a['ci95']:
            np.testing.assert_allclose(a['ci95'][k], b['ci95'][k], rtol=1e-12)
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d) / 'p.npy', pred)
            np.savez(Path(d) / 'p.npz', class_probabilities=pred)
            for f in ('p.npy', 'p.npz'):
                loaded = load_class_probabilities(str(Path(d) / f))
                c = category_ratios(labels, loaded, mask, block_size=1000, n_boot=0)
                self.assertEqual(c['observed'], a['observed'])

    def test_bootstrap_interval_brackets_estimate(self):
        rng = np.random.default_rng(0)
        n = 500_000
        pred = np.tile([.96, .03, .01], (n, 1))
        labels = rng.choice(3, n, p=[.96, .03, .01])
        r = category_ratios(labels, pred, block_size=10_000, n_boot=300)
        for k in ('oe_any', 'oe_rare', 'oe_common', 'oe_common_share'):
            lo, hi = r['ci95'][k]
            self.assertLessEqual(lo, r[k])
            self.assertGreaterEqual(hi, r[k])
            self.assertTrue(lo < 1 < hi, k)


class TestCompareOvep(unittest.TestCase):
    def test_roc_auc_matches_pairwise_definition(self):
        from compare_ovep import roc_auc
        rng = np.random.default_rng(0)
        y = rng.random(300) < 0.2
        s = np.round(rng.random(300) + 0.3 * y, 1)  # many ties
        pos, neg = s[y], s[~y]
        brute = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (pos.size * neg.size)
        self.assertAlmostEqual(roc_auc(y, s), brute, places=12)
        self.assertIsNone(roc_auc(np.zeros(5, bool), np.arange(5)))

    def test_paired_comparison(self):
        from compare_ovep import compare
        rng = np.random.default_rng(1)
        n = 300_000
        rate = np.where(rng.random(n) < 0.3, 0.12, 0.02)
        var = rng.random(n) < rate
        labels = np.where(var, np.where(rng.random(n) < 0.4, 2, 1), 0).astype(np.int8)
        labels[:1000] = -1
        original = rate[:-2048].copy()                    # repo predictions are shorter
        raw = original.copy()
        raw[:512] = 0                                     # unpredicted start
        new = np.stack([1 - rate / 2, rate / 2 * 0.6, rate / 2 * 0.4], 1)  # half the expected variants
        half = np.arange(n) < n // 2
        masks = {'all': np.ones(n, bool), 'first_half': half}
        a = compare(labels, original, new, masks, raw, block_size=10_000, n_boot=200, chunk=70_001)
        b = compare(labels, original, new, masks, raw, block_size=10_000, n_boot=200)
        self.assertEqual(a['all']['n'], n - 1000 - 2048)
        self.assertEqual(a['all']['n'], b['all']['n'])
        self.assertAlmostEqual(a['all']['oe_original'], 1.0, delta=0.03)
        self.assertAlmostEqual(a['all']['oe_new'], 2.0, delta=0.06)
        lo, hi = a['all']['ci95']['oe_diff']
        self.assertTrue(0.8 < lo < hi < 1.2)
        self.assertGreater(a['all']['auc_original'], 0.6)
        self.assertAlmostEqual(a['all']['auc_original'], a['all']['auc_new'], places=6)  # same ranking
        self.assertEqual(a['first_half']['n'], n // 2 - 1000)
        self.assertAlmostEqual(a['all']['oe_common_share'], 1.0, delta=0.05)
        with self.assertRaises(ValueError):
            compare(labels, original, new, {'bad': np.ones(10, bool)})


@unittest.skipUnless(HAS_TF, 'TensorFlow not installed')
class TestModel(unittest.TestCase):
    def test_factorized_loss_masks_correctly(self):
        import tensorflow as tf
        from model import factorized_loss
        y = np.array([[-1, 0, 1, 2]], np.float32)
        p = np.array([[[.5, .5], [.2, .9], [.6, .3], [.6, .7]]], np.float32)
        got = float(factorized_loss()(tf.constant(y), tf.constant(p)))
        want = (-(np.log(.8) + np.log(.6) + np.log(.6)) - (np.log(.7) + np.log(.7))) / 3
        self.assertAlmostEqual(got, want, places=5)
        # label 3: counts as a variant, but its frequency term is skipped
        y3 = np.array([[-1, 0, 1, 3]], np.float32)
        got = float(factorized_loss()(tf.constant(y3), tf.constant(p)))
        want = (-(np.log(.8) + np.log(.6) + np.log(.6)) - np.log(.7)) / 3
        self.assertAlmostEqual(got, want, places=5)

    def test_both_heads_train_one_step(self):
        import tensorflow as tf
        from model import build_model
        dna = np.tile(np.eye(4, dtype=np.uint8), (16, 1))
        labels = np.array([0, 1, 2, -1] * 16, dtype=np.int8)
        for head in ('softmax', 'factorized'):
            m = build_model(64, head=head)
            batch = make_batch({'c': (dna, labels)}, [('c', 0)] * 2, 64, head, [1., 1., 1.])
            loss = m.train_on_batch(*batch)
            self.assertTrue(np.all(np.isfinite(loss)))
            self.assertEqual(m.predict_on_batch(batch[0]).shape[-1], 3 if head == 'softmax' else 2)


if __name__ == '__main__':
    unittest.main()
