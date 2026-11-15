"""Tests for the frequency-spectrum analysis (plan v2): class definitions and
labels, the spectrum head and its loss, calibration, the k-mer baseline, the
singleton-share scores, label validation and the diagnosis tools."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'real_data'))
from spectrum import (UNKNOWN, LazyMask, class_names, collapse_labels, collapse_probabilities,  # noqa: E402
                      conditional_classes, spectrum_class, variant_probability)
from data import encode_dna, site_category, site_spectrum_class, vcf_categories  # noqa: E402
from model import n_output_classes, spectrum_to_classes, to_class_probabilities  # noqa: E402
from train import make_batch, masked_labels  # noqa: E402
from calibrate import (calibrate_classes, fit_calibration, fit_calibration_spectrum,  # noqa: E402
                       spectrum_tails)
from kmer_baseline import context_codes, count_contexts, predict_chromosome, table_probabilities  # noqa: E402
import spectrum_score as ss  # noqa: E402
from compare_ovep import IntervalMask, VariantScore, block_sums, save_blocks  # noqa: E402
from validate_labels import compare as compare_labels, expected_from_table  # noqa: E402
from compare_predictions import compare_predictions  # noqa: E402
from diagnose_calibration import diagnose_blocks, diagnose_sites  # noqa: E402
import evaluate_plan  # noqa: E402

HAS_TF = importlib.util.find_spec('tensorflow') is not None
HEADER = '##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'


def write_vcf(d, records):
    path = Path(d) / 'tiny.vcf'
    path.write_text(HEADER + ''.join('\t'.join(r) + '\n' for r in records))
    return path


def rec(pos, ref, alt, info, filt='PASS', chrom='chr1'):
    return (chrom, str(pos), '.', ref, alt, '.', filt, info)


def simulate(n=300_000, seed=0, selected=None, shift=0.0):
    """Sites with a mutable class (30%), true class probabilities, and labels
    drawn from them. In `selected` (boolean), variants are thinned by half and
    the singleton probability is raised by `shift` (taken from common)."""
    rng = np.random.default_rng(seed)
    mutable = rng.random(n) < 0.3
    pv = np.where(mutable, 0.15, 0.03)
    cond = np.where(mutable[:, None], [0.30, 0.25, 0.20, 0.25], [0.45, 0.25, 0.15, 0.15])
    true_pv, true_cond = pv.copy(), cond.copy()
    if selected is not None:
        true_pv = np.where(selected, pv * 0.5, pv)
        true_cond = cond.copy()
        true_cond[selected, 0] += shift
        true_cond[selected, 3] -= shift
    var = rng.random(n) < true_pv
    u = rng.random(n)
    cls = (u[:, None] > np.cumsum(true_cond, axis=1)).sum(axis=1) + 1
    labels = np.where(var, cls, 0).astype(np.int8)
    probs = np.concatenate([(1 - pv)[:, None], pv[:, None] * cond], axis=1).astype(np.float32)
    return labels, probs, mutable


class TestClasses(unittest.TestCase):
    def test_boundaries(self):
        an = 6404
        got = [site_spectrum_class(an, [c]) for c in (0, 1, 2, 5, 6, 64, 65, 3000)]
        self.assertEqual(got, [0, 1, 2, 2, 3, 3, 4, 4])

    def test_folded_and_multiallelic(self):
        self.assertEqual(site_spectrum_class(6404, [6403]), 1)       # REF is the singleton
        self.assertEqual(site_spectrum_class(6404, [6404]), 0)       # monomorphic for ALT
        self.assertEqual(site_spectrum_class(6404, [1, 500]), 4)     # second most common allele decides
        self.assertEqual(site_spectrum_class(6404, [1, 3]), 2)
        self.assertEqual(site_spectrum_class(6404, [1], star_count=10), 1)

    def test_common_takes_precedence_with_few_samples(self):
        self.assertEqual(site_spectrum_class(200, [2]), 4)           # 2/200 = 1%
        self.assertEqual(site_spectrum_class(200, [1]), 1)
        self.assertEqual(spectrum_class(3, 0.2, 0.05, (1, 5)), 4)

    def test_other_count_bins(self):
        self.assertEqual([site_spectrum_class(6404, [c], count_bins=(1, 2, 10)) for c in (1, 2, 3, 11, 100)],
                         [1, 2, 3, 4, 5])

    def test_pooling_matches_v1_category(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            an = int(rng.integers(50, 7000))
            alts = [int(x) for x in rng.integers(0, an // 3, rng.integers(1, 3))]
            self.assertEqual(int(collapse_labels(np.array([site_spectrum_class(an, alts)]))[0]),
                             site_category(an, alts))

    def test_collapse(self):
        np.testing.assert_array_equal(collapse_labels(np.array([-1, 0, 1, 2, 3, 4])), [-1, 0, 1, 1, 1, 2])
        # a chromosome without any common variant must not have its top class turned into common
        np.testing.assert_array_equal(collapse_labels(np.array([0, 1, 2, 3])), [0, 1, 1, 1])
        p = np.array([[.9, .04, .03, .02, .01], [np.nan] * 5])
        c = collapse_probabilities(p)
        np.testing.assert_allclose(c[0], [.9, .09, .01], atol=1e-6)
        self.assertTrue(np.isnan(c[1]).all())
        self.assertEqual(class_names(5), ['none', 'singleton', 'ultra_rare', 'rare', 'common'])

    def test_conditional_classes(self):
        p = np.array([[.9, .04, .03, .02, .01], [1, 0, 0, 0, 0], [np.nan] * 5])
        c = conditional_classes(p)
        np.testing.assert_allclose(c[0], [.4, .3, .2, .1])
        np.testing.assert_array_equal(c[1], 0)
        self.assertTrue(np.isnan(c[2]).all())
        np.testing.assert_allclose(variant_probability(p)[:2], [.1, 0])


class TestLabels(unittest.TestCase):
    def test_spectrum_labels(self):
        with tempfile.TemporaryDirectory() as d:
            vcf = write_vcf(d, [rec(1, 'A', 'G', 'AC=1;AN=6404'), rec(2, 'C', 'G', 'AC=4;AN=6404'),
                                rec(3, 'T', 'G', 'AC=40;AN=6404'), rec(4, 'G', 'A', 'AC=900;AN=6404'),
                                rec(5, 'A', 'G', 'AC=0;AN=6404')])
            lab = vcf_categories(vcf, 'chr1', 'ACTGAC', scheme='spectrum')
            self.assertEqual(lab.tolist(), [1, 2, 3, 4, 0, 0])
            self.assertEqual(vcf_categories(vcf, 'chr1', 'ACTGAC').tolist(), [1, 1, 1, 2, 0, 0])

    def test_pooled_spectrum_labels_equal_v1_labels(self):
        rng = np.random.default_rng(1)
        n = 3000
        ref = ''.join(rng.choice(list('ACGT'), n))
        recs = []
        for pos in sorted(rng.choice(np.arange(1, n - 10), 400, replace=False)):
            an = 6404
            kind = rng.random()
            ac = int(rng.choice([1, 2, 4, 7, 60, 70, 3000, 6403]))
            r = ref[pos - 1]
            alt = 'ACGT'[('ACGT'.index(r) + 1) % 4]
            if kind < 0.15:
                recs.append(rec(pos, ref[pos - 1:pos + 3], r, f'AC={ac};AN={an}'))            # deletion
            elif kind < 0.25:
                recs.append(rec(pos, r, f'{alt},*', f'AC={ac},3;AN={an}'))                    # star allele
            elif kind < 0.35:
                other = 'ACGT'[('ACGT'.index(r) + 2) % 4]
                recs.append(rec(pos, r, f'{alt},{other}', f'AC={ac},2;AN={an}'))              # multi-allelic
            elif kind < 0.4:
                recs.append(rec(pos, r, alt, f'AC={ac};AN={an}', filt='LowQual'))
            else:
                recs.append(rec(pos, r, alt, f'AC={ac};AN={an}'))
        with tempfile.TemporaryDirectory() as d:
            vcf = write_vcf(d, recs)
            for mode in ('anchor', 'span', 'original'):
                for pass_only in (True, False):
                    v1 = vcf_categories(vcf, 'chr1', ref, indel_mode=mode, pass_only=pass_only)
                    sp = vcf_categories(vcf, 'chr1', ref, indel_mode=mode, pass_only=pass_only,
                                        scheme='spectrum')
                    np.testing.assert_array_equal(collapse_labels(sp), v1)
                    self.assertEqual(sorted(set(sp[sp >= 1].tolist())), [1, 2, 3, 4])

    def test_snv_only(self):
        ref = 'ACGTACGTAC'
        with tempfile.TemporaryDirectory() as d:
            vcf = write_vcf(d, [rec(1, 'A', 'G', 'AC=1;AN=6404'),
                                rec(3, 'GTA', 'G', 'AC=1;AN=6404'),        # deletion over 3..5
                                rec(4, 'T', 'C', 'AC=900;AN=6404'),        # SNV under the deletion
                                rec(7, 'G', 'GA', 'AC=3;AN=6404'),         # insertion
                                rec(8, 'T', 'C,*', 'AC=3,2;AN=6404'),      # SNV with a spanning-deletion allele
                                rec(9, 'A', '<DEL>', 'AC=1;AN=6404'),      # symbolic
                                rec(10, 'C', 'T,CA', 'AC=1,1;AN=6404')])   # SNV + insertion in one record
            lab = vcf_categories(vcf, 'chr1', ref, scheme='spectrum', snv_only=True, indel_mode='original')
            self.assertEqual(lab.tolist(), [1, 0, -1, -1, -1, 0, -1, 2, -1, -1])
            v1 = vcf_categories(vcf, 'chr1', ref, snv_only=True)
            np.testing.assert_array_equal(collapse_labels(lab), v1)


class TestLabelCommandLine(unittest.TestCase):
    def run_data(self, *args):
        import subprocess
        return subprocess.run([sys.executable, str(ROOT / 'data.py'), *args], capture_output=True, text=True)

    def test_spectrum_files_and_dna_check(self):
        with tempfile.TemporaryDirectory() as d:
            ref = 'ACGTACGTACGTACGTACGT'
            Path(d, 'ref.fa').write_text(f'>chr1\n{ref}\n>chr2\n{ref[::-1]}\n')
            Path(d, 'ok.bed').write_text('chr1\t0\t20\nchr2\t0\t20\n')
            vcf = write_vcf(d, [rec(1, 'A', 'C', 'AC=1;AN=6404'), rec(6, 'C', 'T', 'AC=900;AN=6404'),
                                rec(9, 'AC', 'A', 'AC=3;AN=6404')])
            common = ['--fasta', f'{d}/ref.fa', '--vcf', str(vcf), '--callable-bed', f'{d}/ok.bed',
                      '--chromosome', 'chr1', '--scheme', 'spectrum', '--snv-only',
                      '--output-prefix', f'{d}/out/chr1']
            r = self.run_data(*common)
            self.assertEqual(r.returncode, 0, r.stderr)
            lab = np.load(f'{d}/out/chr1_spectrum_labels.npy')
            self.assertEqual(lab.tolist(), [1, 0, 0, 0, 0, 4, 0, 0, -1, -1] + [0] * 10)
            self.assertFalse(Path(f'{d}/out/chr1_frequency_labels.npy').exists())
            np.testing.assert_array_equal(np.load(f'{d}/out/chr1_dna.npy'), encode_dna(ref))
            # --labels-only keeps a matching DNA array ...
            before = Path(f'{d}/out/chr1_dna.npy').stat().st_mtime_ns
            r = self.run_data(*common, '--labels-only')
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(Path(f'{d}/out/chr1_dna.npy').stat().st_mtime_ns, before)
            # ... and refuses one that encodes a different sequence
            np.save(f'{d}/out/chr1_dna.npy', encode_dna(ref[::-1]))
            r = self.run_data(*common, '--labels-only')
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('does not encode', r.stderr)
            r = self.run_data(*common[:-2], '--count-bins', '5', '1', '--output-prefix', f'{d}/out/chr1')
            self.assertNotEqual(r.returncode, 0)


class TestModelHelpers(unittest.TestCase):
    def test_spectrum_to_classes(self):
        raw = np.array([[0.2, 0.5, 0.2, 0.2, 0.1], [0.0, 0.25, 0.25, 0.25, 0.25]])
        p = spectrum_to_classes(raw)
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(p[0], [0.8, 0.1, 0.04, 0.04, 0.02], atol=1e-6)
        # averaged windows need not sum to one exactly: renormalized
        p2 = spectrum_to_classes(np.array([[0.2, 0.25, 0.1, 0.1, 0.05]]))
        np.testing.assert_allclose(p2[0], p[0], atol=1e-6)
        # positive-class weight is undone
        w = 4.0
        learned = w * 0.2 / (w * 0.2 + 0.8)
        np.testing.assert_allclose(spectrum_to_classes(np.array([[learned, .5, .2, .2, .1]]), w)[0], p[0], atol=1e-6)
        meta = {'head': 'spectrum', 'n_freq_classes': 4}
        np.testing.assert_allclose(to_class_probabilities(raw, meta), p)
        self.assertEqual(n_output_classes(meta), 5)
        self.assertEqual(n_output_classes({'head': 'factorized'}), 3)
        self.assertTrue(np.isnan(spectrum_to_classes(np.full((1, 5), np.nan))).all())

    def test_masked_labels_and_batches(self):
        y = np.array([0, 1, 4, -1, 2])
        m = np.array([True, False, False, True, True])
        np.testing.assert_array_equal(masked_labels(y, m, 'common', UNKNOWN), [0, UNKNOWN, UNKNOWN, -1, 2])
        np.testing.assert_array_equal(masked_labels(y, m, 'common'), [0, 3, 3, -1, 2])
        np.testing.assert_array_equal(masked_labels(y, m, 'all', UNKNOWN), [0, -1, -1, -1, 2])
        x = encode_dna('ACGTACGT')
        lab = np.array([0, 1, 2, 3, 4, -1, 0, 4], dtype=np.int8)
        mask = np.array([1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
        X, Y = make_batch({'c': (x, lab, mask)}, [('c', 0)], 8, 'spectrum', None, 'common')
        self.assertEqual(X.shape, (1, 8, 4))
        np.testing.assert_array_equal(Y[0], [0, 1, UNKNOWN, UNKNOWN, 4, -1, 0, UNKNOWN])


@unittest.skipUnless(HAS_TF, 'TensorFlow not installed')
class TestSpectrumHead(unittest.TestCase):
    def test_loss_matches_numpy(self):
        from model import spectrum_loss
        rng = np.random.default_rng(0)
        y = np.array([[0, 1, 4, -1, UNKNOWN, 2, 3, 0]], dtype=np.float32)
        pv = rng.uniform(0.05, 0.95, (1, 8))
        pc = rng.dirichlet(np.ones(4), (1, 8))
        pred = np.concatenate([pv[..., None], pc], axis=-1).astype(np.float32)
        got = float(spectrum_loss(4, 2.0)(y, pred))
        total, valid = 0.0, 0
        for i, lab in enumerate(y[0].astype(int)):
            if lab < 0:
                continue
            valid += 1
            total += -(2.0 * np.log(pv[0, i]) if lab >= 1 else np.log(1 - pv[0, i]))
            if 1 <= lab <= 4:
                total += -np.log(pc[0, i, lab - 1])
        self.assertAlmostEqual(got, total / valid, places=4)

    def test_model_output_and_learning(self):
        import tensorflow as tf
        from model import build_model
        tf.keras.utils.set_random_seed(0)
        m = build_model(64, head='spectrum', n_freq=4, lr=3e-3)
        self.assertEqual(m.output_shape, (None, 64, 5))
        rng = np.random.default_rng(0)
        idx = rng.integers(0, 4, (256, 64))
        X = np.eye(4, dtype=np.float32)[idx]
        # channel 1 ("C") is more mutable and its variants are mostly
        # singletons; elsewhere variants are mostly common
        var = rng.random(idx.shape) < np.where(idx == 1, 0.5, 0.2)
        cls = np.where(rng.random(idx.shape) < np.where(idx == 1, 0.8, 0.1), 1, 4)
        Y = np.where(var, cls, 0).astype(np.float32)
        out = m.predict(X[:2], verbose=0)
        np.testing.assert_allclose(out[..., 1:].sum(-1), 1, atol=1e-5)
        m.fit(X, Y, epochs=20, batch_size=16, verbose=0)
        p = spectrum_to_classes(m.predict(X, verbose=0))
        pv = variant_probability(p)
        self.assertGreater(pv[idx == 1].mean(), 1.3 * pv[idx != 1].mean())
        cond = conditional_classes(p.reshape(-1, 5)).reshape(*idx.shape, 4)
        # the class softmax is learned from variant sites only, and separately from P(variant)
        self.assertGreater(cond[idx == 1][:, 0].mean(), cond[idx != 1][:, 0].mean() + 0.25)
        self.assertGreater(cond[idx != 1][:, 3].mean(), cond[idx == 1][:, 3].mean() + 0.25)


class TestCalibration(unittest.TestCase):
    def test_tails(self):
        np.testing.assert_allclose(spectrum_tails(np.array([[.4, .3, .2, .1]])), [[.6, .3, .1]])

    def test_recalibrates_distorted_predictions(self):
        labels, probs, _ = simulate(400_000, seed=3)
        pv = variant_probability(probs)
        cond = conditional_classes(probs)
        bad_pv = pv ** 0.7                                   # overconfident about variants
        bad_cond = cond ** 2
        bad_cond /= bad_cond.sum(axis=1, keepdims=True)      # too sharp
        bad = np.concatenate([(1 - bad_pv)[:, None], bad_pv[:, None] * bad_cond], axis=1).astype(np.float32)
        half = len(labels) // 2
        cal = fit_calibration_spectrum(labels[:half], bad[:half])
        self.assertEqual(cal['scheme'], 'spectrum')
        self.assertEqual(cal['n_classes'], 5)
        fixed = calibrate_classes(bad[half:], cal)
        np.testing.assert_allclose(fixed.sum(axis=1), 1, atol=1e-5)
        self.assertTrue((fixed >= -1e-7).all())
        lab = labels[half:]
        for k in range(1, 5):
            self.assertAlmostEqual((lab == k).sum() / fixed[:, k].sum(), 1.0, delta=0.05)
            self.assertGreater(abs((lab == k).sum() / bad[half:, k].sum() - 1.0), 0.08)
        # conditional singleton share at the observed variant sites
        var = lab >= 1
        exp_s = conditional_classes(fixed[var])[:, 0].sum()
        self.assertAlmostEqual((lab == 1).sum() / exp_s, 1.0, delta=0.03)

    def test_common_tail_equals_v1_calibration(self):
        labels, probs, _ = simulate(200_000, seed=4)
        sp = fit_calibration_spectrum(labels, probs)
        v1 = fit_calibration(collapse_labels(labels), collapse_probabilities(probs))
        np.testing.assert_allclose(sp['variant'][1], v1['variant'][1], atol=1e-6)
        np.testing.assert_allclose(sp['class_at_least'][-1][1], v1['common_given_variant'][1], atol=1e-6)

    def test_mask_and_errors(self):
        labels, probs, mutable = simulate(100_000, seed=5)
        cal = fit_calibration_spectrum(labels, probs, mask=mutable)
        self.assertTrue(cal['masked'])
        self.assertEqual(sum(cal['class_counts']), int((labels[mutable] >= 1).sum()))
        with self.assertRaises(ValueError):
            fit_calibration_spectrum(np.where(labels == 4, 5, labels), probs)
        out = calibrate_classes(np.full((2, 5), np.nan), cal)
        self.assertTrue(np.isnan(out).all())


class TestKmer(unittest.TestCase):
    def test_reverse_complement_symmetry(self):
        rng = np.random.default_rng(0)
        seq = ''.join(rng.choice(list('ACGT'), 500))
        rc = seq[::-1].translate(str.maketrans('ACGT', 'TGCA'))
        for k in (3, 5, 7):
            a = context_codes(encode_dna(seq), 0, 500, k)
            b = context_codes(encode_dna(rc), 0, 500, k)
            np.testing.assert_array_equal(a, b[::-1])
            self.assertTrue((a[:k // 2] == -1).all() and (a[-(k // 2):] == -1).all())
        with self.assertRaises(ValueError):
            context_codes(encode_dna(seq), 0, 10, 4)

    def test_table_recovers_context_rates(self):
        rng = np.random.default_rng(1)
        n = 400_000
        seq = rng.choice(list('ACGT'), n)
        seq[1000:1010] = 'N'
        text = ''.join(seq)
        dna = encode_dna(text)
        cpg = np.zeros(n, bool)
        m = (seq[:-1] == 'C') & (seq[1:] == 'G')
        cpg[:-1] |= m
        cpg[1:] |= m
        var = rng.random(n) < np.where(cpg, 0.2, 0.02)
        single = rng.random(n) < np.where(cpg, 0.2, 0.6)
        labels = np.where(var, np.where(single, 1, 4), 0).astype(np.int8)
        labels[seq == 'N'] = -1
        counts = count_contexts(dna, labels, 3, 5, chunk=70_000)
        np.testing.assert_array_equal(counts, count_contexts(dna, labels, 3, 5))     # chunk-invariant
        self.assertEqual(int(counts.sum()), int(((labels >= 0) & (context_codes(dna, 0, n, 3) >= 0)).sum()))
        probs = table_probabilities(counts)
        pred = predict_chromosome(dna, probs, 3, chunk=90_000)
        self.assertTrue(np.isnan(pred[995:1015]).all(axis=1).any())
        self.assertTrue(np.isnan(pred[0]).all())
        ok = np.isfinite(pred[:, 0])
        np.testing.assert_allclose(pred[ok].sum(axis=1), 1, atol=1e-5)
        pv = variant_probability(pred[ok])
        self.assertAlmostEqual(pv[cpg[ok]].mean(), 0.2, delta=0.02)
        self.assertAlmostEqual(pv[~cpg[ok]].mean(), 0.02, delta=0.005)
        cond = conditional_classes(pred[ok])
        self.assertLess(cond[cpg[ok], 0].mean(), 0.3)
        self.assertGreater(cond[~cpg[ok], 0].mean(), 0.5)
        # mask restricts the sites counted
        half = np.arange(n) < n // 2
        self.assertLess(count_contexts(dna, labels, 3, 5, mask=half).sum(), counts.sum())
        with self.assertRaises(ValueError):
            count_contexts(dna, labels, 3, 3)                # labels have class 4

    def test_sparse_contexts_shrink_to_global(self):
        counts = np.zeros((64, 5), dtype=np.int64)
        counts[0] = [9000, 400, 300, 200, 100]
        counts[1] = [1, 1, 0, 0, 0]
        p = table_probabilities(counts)
        np.testing.assert_allclose(p[5], p[0], atol=1e-3)     # unseen context = genome-wide rates
        self.assertLess(p[1, 1:].sum(), 0.2)                  # one variant in two sites is not 50%
        with self.assertRaises(ValueError):
            table_probabilities(np.zeros((64, 5)))


class TestSpectrumScore(unittest.TestCase):
    def setUp(self):
        n = 300_000
        self.n = n
        self.selected = np.zeros(n, bool)
        self.selected[50_000:110_000] = True
        self.labels, self.probs, self.mutable = simulate(n, seed=7, selected=self.selected, shift=0.2)
        self.masks = {'neutral': ~self.selected, 'selected': self.selected,
                      'all': IntervalMask([(0, n)], n)}
        # a baseline that ignores mutability: one average spectrum everywhere
        avg = self.probs.mean(axis=0, keepdims=True)
        self.flat = np.repeat(avg, n, axis=0).astype(np.float32)

    def run_score(self, baseline=None, edges=None, **kw):
        sums, k = ss.block_sums(self.labels, self.probs, self.masks, baseline, 10_000, edges,
                                bin_masks=['neutral'], **kw)
        return sums, k

    def test_scores_on_neutral_and_selected(self):
        sums, k = self.run_score()
        self.assertEqual(k, 4)
        rng = np.random.default_rng(0)
        nb = len(sums['neutral']['blocks'])
        draws = rng.multinomial(nb, np.full(nb, 1 / nb), size=400).astype(float)
        neutral = ss.summarize(sums['neutral']['blocks'], k, False, draws)
        sel = ss.summarize(sums['selected']['blocks'], k, False, draws)
        self.assertEqual(neutral['n'], int((~self.selected).sum()))
        self.assertEqual(neutral['observed_variants'], int((self.labels[~self.selected] >= 1).sum()))
        self.assertEqual(sum(neutral['observed'].values()), neutral['observed_variants'])
        for name in ('singleton', 'ultra_rare', 'rare', 'common'):
            self.assertAlmostEqual(neutral['oe_share'][name], 1.0, delta=0.04)
            self.assertAlmostEqual(neutral['oe_marginal'][name], 1.0, delta=0.04)
        lo, hi = neutral['ci95']['oe_share']['singleton']
        self.assertTrue(lo < 1 < hi)
        self.assertAlmostEqual(neutral['oe_any'], 1.0, delta=0.03)
        # selection: half the variants are gone and singletons are enriched
        self.assertAlmostEqual(sel['oe_any'], 0.5, delta=0.04)
        self.assertGreater(sel['ci95']['oe_share']['singleton'][0], 1.2)
        self.assertLess(sel['ci95']['oe_share']['common'][1], 0.9)
        self.assertAlmostEqual(sel['singleton_share']['excess'], 0.2, delta=0.04)
        self.assertAlmostEqual(sel['singleton_share']['observed'] - sel['singleton_share']['expected'],
                               sel['singleton_share']['excess'], places=9)
        con = ss.contrast(sums['selected']['blocks'], sums['neutral']['blocks'], k, False, draws)
        self.assertGreater(con['oe_share']['singleton']['ci95'][0], 0.2)
        self.assertAlmostEqual(con['oe_share']['singleton']['diff'],
                               sel['oe_share']['singleton'] - neutral['oe_share']['singleton'], places=9)

    def test_manual_conditional_expectation(self):
        sums, k = self.run_score()
        tot = sums['all']['blocks'].sum(axis=0)
        var = self.labels >= 1
        cond = conditional_classes(self.probs[var])
        o = ss._model_offset(k, 0)
        np.testing.assert_allclose(tot[o + 1 + k:o + 1 + 2 * k], cond.sum(axis=0), rtol=1e-6)
        np.testing.assert_allclose(tot[o], variant_probability(self.probs).sum(), rtol=1e-6)
        nll = -np.log(cond[np.arange(var.sum()), self.labels[var].astype(int) - 1]).sum()
        self.assertAlmostEqual(tot[o + 2 + 2 * k], nll, delta=1e-4 * abs(nll))
        # chunking does not change anything
        sums2, _ = ss.block_sums(self.labels, self.probs, self.masks, None, 10_000, chunk=37_000)
        np.testing.assert_allclose(sums2['all']['blocks'], sums['all']['blocks'], rtol=1e-9, atol=1e-9)

    def test_baseline_and_mutability_bins(self):
        edges, used = ss.quantile_edges([(self.labels, self.probs, self.masks['neutral'])], n_bins=2)
        self.assertEqual(len(edges), 1)
        sums, k = self.run_score(baseline=self.flat, edges=edges)
        nb = len(sums['neutral']['blocks'])
        draws = np.random.default_rng(1).multinomial(nb, np.full(nb, 1 / nb), size=400).astype(float)
        r = ss.summarize(sums['neutral']['blocks'], k, True, draws, sums['neutral']['bins'])
        # the true model is flat across mutability; the context-free baseline is not
        bins = r['mutability_bins']
        self.assertEqual(len(bins), 2)
        self.assertEqual(sum(b['n_variants'] for b in bins), r['observed_variants'])
        for b in bins:
            self.assertAlmostEqual(b['oe_singleton_share'], 1.0, delta=0.05)
        self.assertGreater(bins[0]['baseline_oe_singleton_share'], 1.1)   # low mutability: more singletons
        self.assertLess(bins[1]['baseline_oe_singleton_share'], 0.9)
        self.assertLess(r['mutability_bins_summary']['max_abs_deviation'],
                        r['mutability_bins_summary']['baseline_max_abs_deviation'])
        # and it predicts the class of a variant better (proper scoring rule)
        self.assertGreater(r['ci95']['vs_baseline']['nll_gain'][0], 0)
        self.assertGreater(r['vs_baseline']['singleton_nll_gain'], 0)
        self.assertAlmostEqual(r['baseline']['oe_any'], r['oe_any'], delta=0.02)

    def test_pool_files_and_outputs(self):
        edges, _ = ss.quantile_edges([(self.labels, self.probs, None)], n_bins=3)
        self.assertEqual(len(edges), 1)       # only two distinct mutability levels: tied quantiles merge
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, sl in enumerate((slice(0, 150_000), slice(150_000, 300_000))):
                masks = {k: np.asarray(m[sl]) for k, m in self.masks.items()}
                sums, k = ss.block_sums(self.labels[sl], self.probs[sl], masks, self.flat[sl], 10_000, edges,
                                        bin_masks=['neutral'])
                path = f'{d}/part{i}.npz'
                ss.save_blocks(path, sums, {'k': k, 'has_baseline': True, 'block_size': 10_000,
                                            'n_blocks': 15, 'edges': edges, 'seqid': f'c{i}'})
                paths.append(path)
            res = ss.pool_files(paths, [('selected', 'neutral')], n_boot=300)
            whole, k = ss.block_sums(self.labels, self.probs, self.masks, self.flat, 10_000, edges,
                                     bin_masks=['neutral'])
            direct = ss.summarize(whole['selected']['blocks'], k, True)
            self.assertAlmostEqual(res['categories']['selected']['oe_share']['singleton'],
                                   direct['oe_share']['singleton'], places=9)
            self.assertEqual(res['meta']['n_blocks'], 30)
            self.assertEqual(len(res['categories']['neutral']['mutability_bins']), 2)
            self.assertNotIn('mutability_bins', res['categories']['selected'])
            c = res['contrasts']['selected_minus_neutral']
            self.assertGreater(c['oe_share']['singleton']['ci95'][0], 0)
            self.assertIn('baseline', c)
            ss.write_outputs(res, f'{d}/r.json', f'{d}/r.tsv')
            back = json.loads(Path(f'{d}/r.json').read_text())
            self.assertEqual(back['categories']['neutral']['n'], res['categories']['neutral']['n'])
            header = Path(f'{d}/r.tsv').read_text().splitlines()[0].split('\t')
            self.assertIn('oe_share_singleton', header)
            self.assertIn('nll_gain_vs_baseline', header)
            with self.assertRaises(ValueError):
                ss.pool_files(paths, [('selected', 'missing')], n_boot=0)
            # files made with different settings must not be pooled
            sums, k = ss.block_sums(self.labels, self.probs, self.masks, None, 10_000)
            ss.save_blocks(f'{d}/other.npz', sums, {'k': k, 'has_baseline': False, 'block_size': 10_000,
                                                    'n_blocks': 30, 'edges': None})
            with self.assertRaises(ValueError):
                ss.pool_files([paths[0], f'{d}/other.npz'])

    def test_tiers_are_disjoint(self):
        n = 1000
        masks = {'CDS': IntervalMask([(100, 200)], n), 'phastcons': IntervalMask([(150, 300), (700, 720)], n),
                 'enhancer': IntervalMask([(280, 400)], n),
                 'neutralA': IntervalMask([(500, 1000)], n)}
        tiers = ss.tier_masks(masks, 'neutralA')
        dense = {k: np.concatenate([np.asarray(m[s:s + 90]) for s in range(0, n, 90)]) for k, m in tiers.items()}
        self.assertEqual(sum(d.astype(int) for d in dense.values()).max(), 1)
        self.assertEqual(int(dense['tier_cds'].sum()), 100)
        self.assertEqual(int(dense['tier_conserved_noncoding'].sum()), 120)
        self.assertEqual(int(dense['tier_enhancer'].sum()), 100)
        self.assertEqual(int(dense['tier_neutral'].sum()), 480)
        with self.assertRaises(ValueError):
            ss.tier_masks({'CDS': masks['CDS']}, 'neutralA')
        lm = LazyMask(lambda a, b: a | b, [masks['CDS'], masks['enhancer']])
        self.assertEqual(len(lm), n)
        self.assertEqual(int(lm[0:n].sum()), 220)

    def test_input_checks(self):
        with self.assertRaises(ValueError):
            ss.block_sums(self.labels, collapse_probabilities(self.probs), self.masks)
        with self.assertRaises(ValueError):
            ss.block_sums(self.labels, self.probs[:-5], self.masks)
        with self.assertRaises(ValueError):
            ss.block_sums(np.where(self.labels == 4, 6, self.labels), self.probs, self.masks)
        with self.assertRaises(ValueError):
            ss.quantile_edges([(self.labels[:50], self.probs[:50], None)], n_bins=10)

    def test_v1_tools_reject_spectrum_predictions(self):
        with self.assertRaises(ValueError):
            block_sums(collapse_labels(self.labels), None, self.probs, {'all': np.ones(self.n, bool)})
        vs = VariantScore(self.probs)
        np.testing.assert_allclose(vs[0:100], variant_probability(self.probs[:100]), rtol=1e-6)
        share = VariantScore(self.probs, 'singleton_share')[0:100]
        np.testing.assert_allclose(share, conditional_classes(self.probs[:100])[:, 0], rtol=1e-6)
        with self.assertRaises(ValueError):
            VariantScore(collapse_probabilities(self.probs), 'singleton_share')


class TestValidateSpectrum(unittest.TestCase):
    def test_per_class_agreement(self):
        rows = ['chr1\t1\tA\tG\t6404\t0.000156152',     # 1 copy
                'chr1\t2\tC\tG\t6404\t0.000624610',     # 4 copies
                'chr1\t3\tT\tG\t6404\t0.00624610',      # 40 copies
                'chr1\t4\tG\tA\t6404\t0.140537',
                'chr1\t5\tA\tG\t6404\t0',
                'chr1\t6\tAC\tA\t6404\t0.3',            # deletion: positions 6-7
                'chr1\t7\tC\tT\t6404\t0.2',
                'chr2\t1\tA\tG\t6404\t0.5']
        with tempfile.TemporaryDirectory() as d:
            table = Path(d) / 't.tsv'
            table.write_text('\n'.join(rows) + '\n')
            exp, maf = expected_from_table(table, 'chr1', 8, 0.01, 'spectrum')
            self.assertEqual(exp.tolist(), [1, 2, 3, 4, 0, 4, 4, -9])
            exp_snv, _ = expected_from_table(table, 'chr1', 8, 0.01, 'spectrum', snv_only=True)
            self.assertEqual(exp_snv.tolist(), [1, 2, 3, 4, 0, -3, -3, -9])
            labels = np.array([1, 2, 2, 4, 0, -1, -1, 0])
            summary, mismatch = compare_labels(labels, exp_snv, maf, n_classes=5)
            self.assertEqual(summary['record_sites_compared'], 5)
            self.assertEqual(summary['mismatches'], 1)
            self.assertEqual(mismatch.tolist(), [2])
            self.assertEqual(summary['agreement_by_bcftools_class'], [1.0, 1.0, 1.0, 0.0, 1.0])
            self.assertEqual(summary['non_snv_sites_not_ignored_by_data_py'], 0)
            bad, _ = compare_labels(np.array([1, 2, 3, 4, 0, 4, -1, 0]), exp_snv, maf, n_classes=5)
            self.assertEqual(bad['non_snv_sites_not_ignored_by_data_py'], 1)


class TestDiagnosisTools(unittest.TestCase):
    def test_compare_predictions(self):
        labels, probs, _ = simulate(120_000, seed=9)
        pv = variant_probability(probs).astype(np.float32)
        same = compare_predictions(pv, probs, labels=labels)
        self.assertAlmostEqual(same['pearson_r'], 1.0, places=5)
        self.assertAlmostEqual(same['spearman_r'], 1.0, places=5)
        self.assertAlmostEqual(same['mean_b_over_mean_a'], 1.0, places=5)
        self.assertEqual(same['share_within_10_percent'], 1.0)
        self.assertAlmostEqual(same['labels']['oe_a'], 1.0, delta=0.05)
        self.assertGreater(same['labels']['auc_a'], 0.6)
        rng = np.random.default_rng(0)
        noisy = pv * np.exp(rng.normal(0, 0.5, pv.size)) * 1.3
        raw = np.ones_like(pv)
        raw[:1000] = 0                                        # unpredicted head
        diff = compare_predictions(pv, noisy[:-2048], b_raw=raw[:-2048])
        self.assertEqual(diff['sites_compared'], pv.size - 2048 - 1000)
        self.assertLess(diff['pearson_r'], 0.95)
        self.assertGreater(diff['mean_b_over_mean_a'], 1.2)
        self.assertLess(diff['share_within_10_percent'], 0.5)
        with self.assertRaises(ValueError):
            compare_predictions(pv, noisy, b_raw=raw[:-1])

    def test_diagnose_sites_finds_a_bad_region(self):
        labels, probs, _ = simulate(200_000, seed=11)
        pv = variant_probability(probs)
        off = probs.copy()
        off[120_000:160_000, 1:] *= 0.5                      # the model expects too few variants here
        off[:, 0] = 1 - off[:, 1:].sum(axis=1)
        rng = np.random.default_rng(0)
        dna = encode_dna(''.join(rng.choice(list('ACGT'), 200_000)))
        res = diagnose_sites(labels, off, dna=dna, region_bp=40_000, chunk=60_000)
        by = {r['bin']: r['oe'] for r in res['by_region']}
        self.assertGreater(by['0-0 Mb'] if '0-0 Mb' in by else list(by.values())[3], 0.5)
        oes = [r['oe'] for r in res['by_region']]
        self.assertEqual(len(oes), 5)
        self.assertAlmostEqual(oes[0], 1.0, delta=0.06)
        self.assertGreater(oes[3], 1.7)                       # 120-160 kb
        self.assertIn('by_gc_content', res)
        self.assertEqual(sum(r['n'] for r in res['by_predicted_probability']), res['n'])
        good = diagnose_sites(labels, pv.astype(np.float32))
        self.assertAlmostEqual(good['oe'], 1.0, delta=0.03)
        for r in good['by_predicted_probability']:
            self.assertAlmostEqual(r['oe'], 1.0, delta=0.08)

    def test_diagnose_blocks_per_chromosome(self):
        with tempfile.TemporaryDirectory() as d:
            files = []
            for i, scale in enumerate((1.0, 1.0, 1.25)):      # the third chromosome is off
                labels, probs, _ = simulate(100_000, seed=20 + i)
                p3 = collapse_probabilities(probs).astype(np.float64)
                p3[:, 1:] /= scale
                p3[:, 0] = 1 - p3[:, 1:].sum(axis=1)
                sums = block_sums(collapse_labels(labels), None, p3, {'unannotated': np.ones(100_000, bool)},
                                  block_size=10_000)
                save_blocks(f'{d}/c{i}.npz', sums, {'seqid': f'c{i}'})
                files.append(f'{d}/c{i}.npz')
            res = diagnose_blocks(files, 'unannotated', n_boot=200)
            oes = [r['oe'] for r in res['chromosomes']]
            self.assertAlmostEqual(oes[0], 1.0, delta=0.06)
            self.assertAlmostEqual(oes[2], 1.25, delta=0.08)
            self.assertGreater(res['pooled_oe'], 1.04)
            self.assertAlmostEqual(res['chromosomes'][2]['pooled_without_this'], 1.0, delta=0.05)
            self.assertEqual(res['chromosomes'][2]['seqid'], 'c2')
            with self.assertRaises(ValueError):
                diagnose_blocks(files, 'nope')


class TestDraftCriteria(unittest.TestCase):
    def test_draft_is_well_formed_and_unfrozen(self):
        path = ROOT / 'experiments' / 'spectrum' / 'criteria_v2_draft.json'
        doc = json.loads(path.read_text())
        ids = [c['id'] for c in doc['criteria']]
        self.assertEqual(len(ids), len(set(ids)))
        for decision, members in doc['decisions'].items():
            self.assertTrue(set(members) <= set(ids), decision)
        with tempfile.TemporaryDirectory() as d:
            out, decisions = evaluate_plan.evaluate(doc, d)
            self.assertTrue(all(o['status'] == 'MISSING' for o in out.values()))
            self.assertTrue(all(v == 'INCOMPLETE' for v in decisions.values()))
        self.assertFalse(path.with_suffix('.sha256').exists())   # a draft must not look frozen


if __name__ == '__main__':
    unittest.main()
