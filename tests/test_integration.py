"""End-to-end TensorFlow integration test through the command-line scripts.

Small synthetic chromosomes (CpG sites more mutable, imbalanced classes,
ambiguous bases, an uncallable stretch, neutral masks) go through:
train (factorized + mask scopes, softmax + class weights) -> save -> load ->
chromosome-held-out ensemble prediction -> masked calibration -> calibrated
prediction -> scoring. Takes about a minute on CPU; skipped without TensorFlow.
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
    var = rng.random(n) < np.where(cpg, 0.2, 0.02)  # ~3-4% variant sites
    common = rng.random(n) < np.where(cpg, 0.6, 0.2)
    labels = np.where(var, np.where(common, 2, 1), 0).astype(np.int8)
    labels[dna.max(1) == 0] = -1
    labels[-700:] = -1                               # uncallable tail
    mask = np.ones(n, bool)
    mask[5000:9000] = False                          # "annotated" block
    return dna, labels, mask


@unittest.skipUnless(HAS_TF, 'TensorFlow not installed')
class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = Path(cls.tmp.name)
        rng = np.random.default_rng(0)
        manifest = {}
        for c in ('chrA', 'chrB', 'chrC'):
            dna, labels, mask = make_chromosome(rng, 30_000)
            for name, arr in (('dna', dna), ('labels', labels), ('mask', mask)):
                np.save(d / f'{c}_{name}.npy', arr)
            manifest[c] = {k: str(d / f'{c}_{k}.npy') for k in ('dna', 'labels', 'mask')}
        (d / 'manifest.json').write_text(json.dumps(manifest))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_script(self, script, *args):
        r = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                           capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, f'{script} failed:\n{r.stderr[-3000:]}')
        return r.stdout

    def train(self, out, *extra):
        self.run_script('train.py', '--manifest', self.d / 'manifest.json',
                        '--train-chromosomes', 'chrA', '--val-chromosomes', 'chrB',
                        '--window', 64, '--batch-size', 8, '--steps-per-epoch', 40,
                        '--epochs', 2, '--max-val-windows', 50, '--verbose', 0,
                        '--out', self.d / out, *extra)
        self.assertTrue((self.d / out).exists())
        return json.loads((self.d / f'{out}.meta.json').read_text())

    def check_probs(self, path, labels):
        p = np.load(path)
        self.assertEqual(p.shape, (len(labels), 3))
        ok = np.isfinite(p).all(1)
        self.assertGreater(ok.mean(), 0.95)
        self.assertTrue(np.isnan(p[510]).all())       # inside the N stretch
        np.testing.assert_allclose(p[ok].sum(1), 1, atol=1e-4)
        self.assertTrue(((p[ok] >= 0) & (p[ok] <= 1)).all())
        return p

    def test_full_pipeline(self):
        d = self.d
        labels = np.load(d / 'chrC_labels.npy')
        mask = np.load(d / 'chrC_mask.npy')

        # train: two seeds with neutral-only loss, one 'common' scope, one weighted softmax
        meta0 = self.train('f0.keras', '--head', 'factorized', '--mask-scope', 'all', '--seed', 0)
        self.train('f1.keras', '--head', 'factorized', '--mask-scope', 'all', '--seed', 1)
        meta_c = self.train('fc.keras', '--head', 'factorized', '--mask-scope', 'common')
        meta_s = self.train('s.keras', '--head', 'softmax', '--class-weights', 1, 5, 5)
        self.assertEqual(meta0['mask_scope'], 'all')
        self.assertEqual(meta_c['mask_scope'], 'common')
        self.assertEqual(meta_s['class_weights'], [1, 5, 5])

        # saved models reload without custom objects and have the right heads
        import tensorflow as tf
        self.assertEqual(tf.keras.models.load_model(d / 'f0.keras', compile=False).output_shape[-1], 2)
        self.assertEqual(tf.keras.models.load_model(d / 's.keras', compile=False).output_shape[-1], 3)

        # held-out chromosome: ensemble, single factorized ('common'), weighted softmax
        out = self.run_script('predict.py', '--model', d / 'f0.keras', d / 'f1.keras',
                              '--dna', d / 'chrC_dna.npy', '--output', d / 'ens.npy')
        self.assertIn('2 models', out)
        self.assertEqual(list(d.glob('*.tmp.npy')), [])
        ens = self.check_probs(d / 'ens.npy', labels)
        self.run_script('predict.py', '--model', d / 'fc.keras', '--dna', d / 'chrC_dna.npy',
                        '--output', d / 'fc.npy')
        self.check_probs(d / 'fc.npy', labels)
        self.run_script('predict.py', '--model', d / 's.keras', '--dna', d / 'chrC_dna.npy',
                        '--output', d / 's.npy')
        soft = self.check_probs(d / 's.npy', labels)
        # predict.py must apply exactly the class-weight correction to the raw
        # network output (a model this undertrained is not yet calibrated, so
        # absolute O/E is checked after calibration below, not here)
        sys.path.insert(0, str(ROOT))
        from predict import tile_predict
        from model import unweight_softmax
        m = tf.keras.models.load_model(d / 's.keras', compile=False)
        raw = tile_predict(np.load(d / 'chrC_dna.npy'), lambda x: m.predict_on_batch(x), 64, 3, 64)
        np.testing.assert_allclose(soft, unweight_softmax(raw, [1, 5, 5]), atol=1e-5)
        ok = np.isfinite(raw[:, 0])
        self.assertLess((soft[ok, 1] + soft[ok, 2]).sum(), (raw[ok, 1] + raw[ok, 2]).sum())

        # neutral-site calibration, calibrated prediction, scoring
        self.run_script('calibrate.py', '--labels', d / 'chrC_labels.npy', '--predictions', d / 'ens.npy',
                        '--mask', d / 'chrC_mask.npy', '--out', d / 'cal.json')
        cal = json.loads((d / 'cal.json').read_text())
        self.assertTrue(cal['masked'])
        self.assertEqual(cal['n_sites'], int(((labels >= 0) & mask & np.isfinite(ens[:, 0])).sum()))
        self.run_script('predict.py', '--model', d / 'f0.keras', d / 'f1.keras',
                        '--dna', d / 'chrC_dna.npy', '--output', d / 'ens_cal.npy',
                        '--calibration', d / 'cal.json')
        self.check_probs(d / 'ens_cal.npy', labels)
        res = json.loads(self.run_script('score.py', '--labels', d / 'chrC_labels.npy',
                                         '--predictions', d / 'ens_cal.npy', '--mask', d / 'chrC_mask.npy',
                                         '--block-size', 2000, '--bootstrap', 200))
        # calibrated on these same neutral sites, so O/E must be ~1 there
        self.assertAlmostEqual(res['oe_any'], 1.0, delta=0.05)
        self.assertAlmostEqual(res['oe_common_share'], 1.0, delta=0.1)
        lo, hi = res['ci95']['oe_any']
        self.assertTrue(lo <= res['oe_any'] <= hi)


if __name__ == '__main__':
    unittest.main()
