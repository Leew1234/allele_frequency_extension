"""End-to-end test of the frequency-spectrum tools through their command
lines: spectrum-head training (both mask scopes) -> ensemble prediction ->
calibration -> calibrated prediction -> k-mer baseline (fit, calibrate,
predict) -> mutability edges -> per-chromosome scores -> pooled results ->
collapse to the v1 format and score with a v1 tool. Wrong pairings of labels
and heads are refused. About a minute on CPU; skipped without TensorFlow.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HAS_TF = importlib.util.find_spec('tensorflow') is not None


def make_chromosome(rng, n):
    seq = rng.integers(0, 4, n)                      # 0 A, 1 C, 2 T, 3 G (repo order)
    dna = np.eye(4, dtype=np.uint8)[seq]
    dna[500:540] = 0                                 # ambiguous bases
    cpg = np.zeros(n, bool)
    m = (seq[:-1] == 1) & (seq[1:] == 3)
    cpg[:-1] |= m
    cpg[1:] |= m
    selected = np.zeros(n, bool)
    selected[5000:9000] = True                       # constrained, "annotated" block
    var = rng.random(n) < np.where(cpg, 0.2, 0.03) * np.where(selected, 0.5, 1.0)
    p_single = np.where(cpg, 0.15, 0.35) + np.where(selected, 0.35, 0.0)
    u = rng.random(n)
    cls = np.where(u < p_single, 1, np.where(u < p_single + 0.2, 2, np.where(u < p_single + 0.35, 3, 4)))
    labels = np.where(var, cls, 0).astype(np.int8)
    labels[dna.max(1) == 0] = -1
    labels[-700:] = -1                               # uncallable tail
    return dna, labels, ~selected, selected


@unittest.skipUnless(HAS_TF, 'TensorFlow not installed')
class TestSpectrumEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = Path(cls.tmp.name)
        rng = np.random.default_rng(0)
        manifest = {}
        for c in ('chrA', 'chrB', 'chrC', 'chrD'):
            dna, labels, mask, selected = make_chromosome(rng, 40_000)
            for name, arr in (('dna', dna), ('labels', labels), ('mask', mask), ('selected', selected)):
                np.save(d / f'{c}_{name}.npy', arr)
            manifest[c] = {k: str(d / f'{c}_{k}.npy') for k in ('dna', 'labels', 'mask')}
        (d / 'manifest.json').write_text(json.dumps(manifest))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_script(self, script, *args, ok=True):
        r = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                           capture_output=True, text=True, cwd=ROOT)
        if ok:
            self.assertEqual(r.returncode, 0, f'{script} failed:\n{r.stderr[-3000:]}')
        return r

    def train(self, out, *extra, ok=True):
        return self.run_script('train.py', '--manifest', self.d / 'manifest.json',
                               '--train-chromosomes', 'chrA', '--val-chromosomes', 'chrB',
                               '--window', 64, '--batch-size', 8, '--steps-per-epoch', 60,
                               '--epochs', 2, '--max-val-windows', 50, '--verbose', 0,
                               '--out', self.d / out, *extra, ok=ok)

    def test_pipeline(self):
        d = self.d
        # ---- training: two members, one with each mask scope
        self.train('s0.keras', '--head', 'spectrum', '--mask-scope', 'all', '--seed', 0)
        self.train('s1.keras', '--head', 'spectrum', '--mask-scope', 'common', '--seed', 1)
        meta = json.loads((d / 's0.keras.meta.json').read_text())
        self.assertEqual((meta['head'], meta['n_freq_classes'], meta['mask_scope']), ('spectrum', 4, 'all'))
        # a v1 head must refuse spectrum labels instead of silently dropping classes
        r = self.train('bad.keras', '--head', 'factorized', ok=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('labels contain class', r.stderr)

        # ---- ensemble prediction, calibration on a held-out chromosome, calibrated prediction
        models = [d / 's0.keras', d / 's1.keras']
        self.run_script('predict.py', '--model', *models, '--dna', d / 'chrC_dna.npy', '--output', d / 'cal_C.npy')
        raw = np.load(d / 'cal_C.npy')
        self.assertEqual(raw.shape, (40_000, 5))
        ok = np.isfinite(raw[:, 0])
        self.assertTrue((~ok[500:540]).all())
        np.testing.assert_allclose(raw[ok].sum(axis=1), 1, atol=1e-4)
        self.run_script('calibrate.py', '--labels', d / 'chrC_labels.npy', '--predictions', d / 'cal_C.npy',
                        '--mask', d / 'chrC_mask.npy', '--out', d / 'cal.json')
        cal = json.loads((d / 'cal.json').read_text())
        self.assertEqual((cal['scheme'], cal['n_classes'], len(cal['class_at_least'])), ('spectrum', 5, 3))
        self.run_script('predict.py', '--model', *models, '--dna', d / 'chrD_dna.npy',
                        '--output', d / 'pred_D.npy', '--calibration', d / 'cal.json')
        pred = np.load(d / 'pred_D.npy')
        ok = np.isfinite(pred[:, 0])
        np.testing.assert_allclose(pred[ok].sum(axis=1), 1, atol=1e-4)
        self.assertTrue((pred[ok] >= -1e-6).all())
        # in-sample, the calibration reproduces the class counts on the neutral mask
        self.run_script('predict.py', '--model', *models, '--dna', d / 'chrC_dna.npy',
                        '--output', d / 'pred_C.npy', '--calibration', d / 'cal.json')
        pc, lab, mask = np.load(d / 'pred_C.npy'), np.load(d / 'chrC_labels.npy'), np.load(d / 'chrC_mask.npy')
        use = mask & (lab >= 0) & np.isfinite(pc[:, 0])
        self.assertAlmostEqual((lab[use] >= 1).sum() / pc[use, 1:].sum(), 1.0, delta=0.03)
        var = use & (lab >= 1)
        cond = pc[var, 1:] / pc[var, 1:].sum(axis=1, keepdims=True)
        for k in range(1, 5):       # what the scores use: class shares given where variants are
            self.assertAlmostEqual((lab[var] == k).sum() / cond[:, k - 1].sum(), 1.0, delta=0.03)
        # a 3-class calibration cannot be applied to a spectrum model
        (d / 'cal3.json').write_text(json.dumps({'variant': [[0, 1], [0, 1]], 'common_given_variant': [[0, 1], [0, 1]]}))
        r = self.run_script('predict.py', '--model', models[0], '--dna', d / 'chrD_dna.npy',
                            '--output', d / 'x.npy', '--calibration', d / 'cal3.json', ok=False)
        self.assertNotEqual(r.returncode, 0)

        # ---- k-mer baseline through the same calibration and file format
        self.run_script('kmer_baseline.py', 'fit', '--manifest', d / 'manifest.json', '--chromosomes', 'chrA', 'chrB',
                        '--mask-scope', 'all', '--k', 3, '--out', d / 'kmer3.json')
        self.run_script('kmer_baseline.py', 'predict', '--table', d / 'kmer3.json', '--dna', d / 'chrC_dna.npy',
                        '--output', d / 'kcal_C.npy')
        self.run_script('calibrate.py', '--labels', d / 'chrC_labels.npy', '--predictions', d / 'kcal_C.npy',
                        '--mask', d / 'chrC_mask.npy', '--out', d / 'kcal.json')
        for c in 'CD':
            self.run_script('kmer_baseline.py', 'predict', '--table', d / 'kmer3.json', '--dna', d / f'chr{c}_dna.npy',
                            '--output', d / f'kmer_{c}.npy', '--calibration', d / 'kcal.json')
        kd = np.load(d / 'kmer_D.npy')
        self.assertEqual(kd.shape, (40_000, 5))
        np.testing.assert_allclose(kd[np.isfinite(kd[:, 0])].sum(axis=1), 1, atol=1e-4)

        # ---- scores: edges, per-chromosome blocks, pooled results with a contrast
        self.run_script('spectrum_score.py', 'edges', '--chrom', d / 'chrD_labels.npy', d / 'kmer_D.npy',
                        d / 'chrD_mask.npy', '--chrom', d / 'chrC_labels.npy', d / 'kmer_C.npy', d / 'chrC_mask.npy',
                        '--bins', 2, '--out', d / 'edges.json')
        for c in 'CD':
            self.run_script('spectrum_score.py', 'chrom', '--labels', d / f'chr{c}_labels.npy',
                            '--new', d / f'kmer_{c}.npy', '--baseline', d / f'pred_{c}.npy',
                            '--mask', f'neutral={d}/chr{c}_mask.npy', '--mask', f'selected={d}/chr{c}_selected.npy',
                            '--edges', d / 'edges.json', '--block-size', 2000, '--out', d / f'blocks_{c}.npz')
        self.run_script('spectrum_score.py', 'pool', '--blocks', d / 'blocks_C.npz', d / 'blocks_D.npz',
                        '--contrast', 'selected:neutral', '--bootstrap', 300, '--json', d / 'res.json', '--tsv', d / 'res.tsv')
        res = json.loads((d / 'res.json').read_text())
        cats = res['categories']
        self.assertEqual(res['meta']['classes'], ['none', 'singleton', 'ultra_rare', 'rare', 'common'])
        # the trinucleotide table is the true neutral model of this toy: calibrated on neutral sites ...
        self.assertAlmostEqual(cats['neutral']['oe_any'], 1.0, delta=0.08)
        self.assertAlmostEqual(cats['neutral']['oe_share']['singleton'], 1.0, delta=0.1)
        # ... and the constrained block shows half the variants and a clear singleton excess
        self.assertAlmostEqual(cats['selected']['oe_any'], 0.5, delta=0.12)
        self.assertGreater(cats['selected']['ci95']['oe_share']['singleton'][0], 1.3)
        self.assertGreater(res['contrasts']['selected_minus_neutral']['oe_share']['singleton']['ci95'][0], 0.3)
        self.assertEqual(len(cats['neutral']['mutability_bins']), 2)
        self.assertIn('nll_gain', cats['neutral']['vs_baseline'])
        self.assertIn('all_scorable', cats)
        self.assertTrue((d / 'res.tsv').read_text().startswith('category\t'))

        # ---- collapse to the v1 format and score with a v1 tool
        self.run_script('spectrum.py', 'labels', '--input', d / 'chrD_labels.npy', '--output', d / 'chrD_v1_labels.npy')
        self.run_script('spectrum.py', 'predictions', '--input', d / 'pred_D.npy', '--output', d / 'pred_D_v1.npy')
        v1 = np.load(d / 'chrD_v1_labels.npy')
        self.assertEqual(sorted(set(v1.tolist())), [-1, 0, 1, 2])
        self.assertEqual(int((v1 == 2).sum()), int((np.load(d / 'chrD_labels.npy') == 4).sum()))
        out = self.run_script('score.py', '--labels', d / 'chrD_v1_labels.npy', '--predictions', d / 'pred_D_v1.npy',
                              '--mask', d / 'chrD_mask.npy', '--block-size', 2000, '--bootstrap', 100).stdout
        self.assertIn('oe_common_share', json.loads(out))
        # the other diagnosis tool runs on spectrum predictions too
        out = self.run_script('compare_predictions.py', '--a', d / 'kmer_D.npy', '--b', d / 'pred_D.npy',
                              '--labels', d / 'chrD_labels.npy').stdout
        cmp_res = json.loads(out)
        self.assertGreater(cmp_res['pearson_r'], 0)          # both saw that CpG sites are more mutable
        self.assertIn('auc_a', cmp_res['labels'])
        out = self.run_script('diagnose_calibration.py', 'sites', '--labels', d / 'chrD_labels.npy',
                              '--predictions', d / 'kmer_D.npy', '--mask', d / 'chrD_mask.npy',
                              '--dna', d / 'chrD_dna.npy', '--region-mb', 0.01, '--json', d / 'diag.json').stdout
        self.assertIn('By GC content', out)
        self.assertEqual(len(json.loads((d / 'diag.json').read_text())['by_region']), 4)


if __name__ == '__main__':
    unittest.main()
