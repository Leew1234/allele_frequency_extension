"""Tests for reproduce_original/: the original labels and calibration filter
are compared with direct ports of the original repository's scripts."""
import gzip
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reproduce_original'))
from orig_model import original_labels, calibration_sites  # noqa: E402
from train_orig import generate_data  # noqa: E402


def original_label_script(records, length):
    """data/1000genome/select_indel.py + data/pickle_label_chr.py, verbatim logic."""
    label = np.zeros(length)
    for chrom, pos, ref, alt in records:
        if len(alt) == len(ref):
            a, b = pos, pos + 1
        else:
            a, b = pos, pos + len(ref) + 1
        label[a - 1:b - 1] = 1
    return label


def original_filter_data(labels, genome, window_size=512, step_size=256):
    """calibration.py filter_data, verbatim logic."""
    valid = set()
    for i in range(0, len(labels) - window_size + 1, step_size):
        wl, wg = labels[i:i + window_size], genome[i:i + window_size]
        if not (np.all(wl == 0) or np.any(np.all(wg == 0, axis=1))):
            valid.update(range(i, i + window_size))
    out = np.zeros(len(labels), dtype=bool)
    out[sorted(valid)] = True
    return out


class TestOriginalRules(unittest.TestCase):
    def test_labels_match_original_scripts(self):
        rng = np.random.default_rng(0)
        n, recs = 5000, []
        for pos in sorted(rng.choice(np.arange(1, n - 20), 300, replace=False)):
            ref = 'A' * int(rng.integers(1, 5))
            alt = rng.choice(['G', 'GT', 'C,T', 'AC,A', '<DEL>', '*'])
            recs.append(('NC_1', int(pos), ref, str(alt)))
        recs.append(('NC_1', n - 1, 'ACGT', 'A'))   # runs past the end
        with tempfile.TemporaryDirectory() as d:
            vcf = Path(d) / 'x.vcf.gz'
            with gzip.open(vcf, 'wt') as fh:
                fh.write('##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')
                fh.write('NC_2\t10\t.\tA\tG\t.\tPASS\t.\n')   # other contig ignored
                for c, p, r, a in recs:
                    fh.write(f'{c}\t{p}\t.\t{r}\t{a}\t.\tLowQual\tAC=0\n')
            got = original_labels(vcf, 'NC_1', n)
        np.testing.assert_array_equal(got, original_label_script(recs, n)[:n].astype(np.uint8))

    def test_calibration_filter_matches_original(self):
        rng = np.random.default_rng(1)
        n = 6000
        labels = (rng.random(n) < 0.002).astype(np.uint8)
        labels[2000:3200] = 0
        dna = np.eye(4, dtype=np.uint8)[rng.integers(0, 4, n)]
        dna[4100] = 0
        np.testing.assert_array_equal(calibration_sites(labels, dna), original_filter_data(labels, dna))

    def test_generator_drops_unlabeled_windows(self):
        n = 50_000
        dna = {'a': np.eye(4, dtype=np.uint8)[np.zeros(n, int)]}
        lab = {'a': np.zeros(n, np.uint8)}
        lab['a'][20_000:20_010] = 1
        random.seed(3.14)
        x, y = next(generate_data(dna, lab, ['a'], 16))
        self.assertEqual(x.shape[1:], (2048, 4))
        self.assertTrue(np.all(y.reshape(len(y), -1).sum(axis=1) > 0))
        self.assertLessEqual(len(x), 16)


class TestArchitecture(unittest.TestCase):
    def test_original_layout(self):
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            self.skipTest('TensorFlow not installed')
        from orig_model import cnn1d
        m = cnn1d(2048, 4)
        names = [type(l).__name__ for l in m.layers]
        self.assertEqual(names.count('BatchNormalization'), 10)   # encoder only
        self.assertEqual(names.count('Conv1DTranspose'), 4)
        self.assertEqual(m.output_shape, (None, 2048, 1))
        # parameter count of the original cnn_batchnorm.cnn1d(2048, 4), by hand:
        # encoder blocks (conv k=3 + BN, twice) 1,120 + 4,928 + 19,072 + 75,008 + 297,472;
        # decoder (transpose conv k=2, two convs) 623,360 + 188,800 + 47,296 + 8,865
        self.assertEqual(m.count_params(), 1_265_921)


if __name__ == '__main__':
    unittest.main()
