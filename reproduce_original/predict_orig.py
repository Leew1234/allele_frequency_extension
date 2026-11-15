#!/usr/bin/env python3
"""Predict with the reproduced original ensemble, fit the original isotonic
calibration, and write files in the layout run_plan.sh expects:

  <out>/raw/<seqid>.pred.npy          uncalibrated ensemble mean (HALFn_ORIG_RAW_DIR);
                                      0 where nothing was predicted
  <out>/calibrated/<seqid>.pred.npy   calibrated (HALFn_ORIG_DIR)

Prediction tiles 2,048-bp windows with a 512-bp step and keeps the central
1,024 bp, averaging the five members, as predict_by_chrom_fast.py did. Two
deliberate differences: overlapping windows are averaged correctly (the
original divided by a miscounted denominator, a monotone distortion that the
calibration removes), and the chromosome ends are covered (the original left
the last partial chunk unpredicted). Calibration follows calibration.py: an
isotonic fit of the label on the uncalibrated score, on the calibration
chromosome, using only 512-bp windows (step 256) with at least one label and no
N; the fit uses 1,000 quantile bins instead of every site.

  python predict_orig.py --work $WORK --models models/half1 --out orig/half1 \\
      --calibration NC_000001.11 NC_000004.12 NC_000006.12 ...
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from orig_model import WINDOW, calibration_sites, self_crossentropy  # noqa: E402
from predict import tile_predict  # noqa: E402
from calibrate import fit_isotonic, apply_isotonic  # noqa: E402


def save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + '.partial.npy'
    np.save(tmp, arr)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True)
    ap.add_argument('--models', required=True, help='folder with model0.keras ... model4.keras')
    ap.add_argument('--out', required=True)
    ap.add_argument('--calibration', required=True, help='calibration chromosome (chr1 for half 1, chr2 for half 2)')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('seqids', nargs='+', help='chromosomes to predict (besides the calibration one)')
    a = ap.parse_args(argv)

    import keras
    paths = sorted(glob.glob(f'{a.models}/model[0-9].keras'))
    if not paths:
        sys.exit(f'no model*.keras in {a.models}')
    models = [keras.models.load_model(p, custom_objects={'self_crossentropy': self_crossentropy}) for p in paths]
    for m in models:  # no XLA: its autotuner fails on V100 GPUs
        m.jit_compile = False
    print(f'ensemble of {len(models)}: {paths}')

    def predict_fn(x):
        return np.mean([m.predict_on_batch(x) for m in models], axis=0)

    for s in [a.calibration] + [s for s in a.seqids if s != a.calibration]:
        out = f'{a.out}/raw/{s}.pred.npy'
        if os.path.exists(out):
            print(f'{s}: raw prediction exists')
            continue
        dna = np.load(f'{a.work}/arrays/{s}_dna.npy', mmap_mode='r')
        pred = tile_predict(dna, predict_fn, WINDOW, 1, batch_size=a.batch_size)[:, 0]
        pred = np.where(np.isfinite(pred), pred, 0.0).astype(np.float32)
        save(out, pred)
        print(f'{s}: predicted {len(pred)} sites')

    cal_raw = np.load(f'{a.out}/raw/{a.calibration}.pred.npy', mmap_mode='r')
    cal_lab = np.load(f'{a.work}/arrays/{a.calibration}_orig_labels.npy', mmap_mode='r')
    cal_dna = np.load(f'{a.work}/arrays/{a.calibration}_dna.npy', mmap_mode='r')
    keep = calibration_sites(cal_lab, cal_dna) & (np.asarray(cal_raw) > 0)
    mapping = fit_isotonic(np.asarray(cal_raw)[keep], np.asarray(cal_lab)[keep])
    with open(f'{a.out}/calibration.json', 'w') as fh:
        json.dump({'chromosome': a.calibration, 'sites': int(keep.sum()),
                   'x': [float(v) for v in mapping[0]], 'y': [float(v) for v in mapping[1]]}, fh)
    for s in [a.calibration] + list(a.seqids):
        raw = np.load(f'{a.out}/raw/{s}.pred.npy', mmap_mode='r')
        save(f'{a.out}/calibrated/{s}.pred.npy', apply_isotonic(raw, mapping).astype(np.float32))
    print(f'calibrated on {a.calibration} ({int(keep.sum())} sites)')


if __name__ == '__main__':
    main()
