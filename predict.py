"""Predict per-site frequency-class probabilities for one chromosome.

Windows overlap: with window W the step is W/4 and only the central W/2 of each
window is kept (512-bp step, 1024-bp core for W=2048, as in the original
predict_by_chrom_fast.py), and overlapping cores are averaged. The first and
last windows also keep their outer edge, and a final window is aligned to the
chromosome end, so the whole chromosome is covered. Windows containing an
ambiguous base are skipped; positions covered only by such windows are NaN.

Several --model files form an ensemble (e.g. the paper's five seeds): each
model's output is converted to unweighted class probabilities using its own
<model>.meta.json, the models are averaged, and --calibration (from
calibrate.py, fitted on the same ensemble's uncalibrated output) is applied
last. A position is NaN if any member could not predict it.

Output is a single (n_sites, 3) float32 .npy (P(none), P(rare), P(common)),
written through a memory map; intermediate arrays also live on disk and all
post-processing is chunked, so peak RAM stays small even for chr1.
"""
import argparse
import json
import os
import numpy as np
from numpy.lib.format import open_memmap

from model import to_class_probabilities

CHUNK = 5_000_000


def window_starts(length, window, step):
    if length < window:
        return []
    starts = list(range(0, length - window + 1, step))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def tile_predict(dna, predict_fn, window, n_out, batch_size=64, out=None, chunk=CHUNK):
    """Run predict_fn over overlapping windows and average the kept cores.

    predict_fn maps a (B, window, 4) float32 array to (B, window, n_out).
    `out` may be a zero-filled (len(dna), n_out) float32 array or memmap that
    receives the result; otherwise one is allocated in memory.
    Returns (len(dna), n_out) with NaN where nothing was predicted.
    """
    n = len(dna)
    margin, step = window // 4, window // 4
    total = np.zeros((n, n_out), dtype=np.float32) if out is None else out
    count = np.zeros(n, dtype=np.uint8)
    starts = window_starts(n, window, step)
    last = n - window

    def flush(batch_x, batch_s):
        pred = np.asarray(predict_fn(np.stack(batch_x)), dtype=np.float32)
        for o, s in zip(pred, batch_s):
            lo = 0 if s == 0 else margin
            hi = window if s == last else window - margin
            total[s + lo:s + hi] += o[lo:hi]
            count[s + lo:s + hi] += 1

    batch_x, batch_s = [], []
    for s in starts:
        x = np.asarray(dna[s:s + window], dtype=np.float32)
        if x.max(axis=1).min() == 0:
            continue  # contains an ambiguous base
        batch_x.append(x)
        batch_s.append(s)
        if len(batch_x) == batch_size:
            flush(batch_x, batch_s)
            batch_x, batch_s = [], []
    if batch_x:
        flush(batch_x, batch_s)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        t = np.array(total[s:e])
        c = count[s:e]
        t[c > 0] /= c[c > 0, None]
        t[c == 0] = np.nan
        total[s:e] = t
    return total


def accumulate_classes(acc, raw, meta, chunk=CHUNK):
    """acc += unweighted class probabilities of one model's raw output (chunked)."""
    for s in range(0, len(acc), chunk):
        e = min(s + chunk, len(acc))
        acc[s:e] += to_class_probabilities(raw[s:e], meta)


def finalize(acc, n_models, calibration=None, chunk=CHUNK):
    """acc /= n_models, then optional calibration (chunked, in place)."""
    if calibration is not None:
        from calibrate import calibrate_classes
    for s in range(0, len(acc), chunk):
        e = min(s + chunk, len(acc))
        p = np.asarray(acc[s:e], dtype=np.float64) / n_models
        if calibration is not None:
            p = calibrate_classes(p, calibration)
        acc[s:e] = p


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, nargs='+', help='one or more .keras models (ensemble)')
    p.add_argument('--meta', nargs='+', help='defaults to <model>.meta.json for each model')
    p.add_argument('--dna', required=True, help='One-hot .npy for a chromosome')
    p.add_argument('--output', required=True, help='Output .npy path (n_sites x 3, float32)')
    p.add_argument('--calibration', help='JSON from calibrate.py')
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()

    if not args.output.endswith('.npy'):
        p.error('--output must end in .npy (written as a memory map)')
    metas = args.meta or [m + '.meta.json' for m in args.model]
    if len(metas) != len(args.model):
        p.error('give one --meta per --model')
    meta_objs = []
    for path in metas:
        if not os.path.exists(path):
            p.error(f'{path} not found; it is written by train.py and records '
                    'the head and class weights needed to interpret the output')
        with open(path) as f:
            meta_objs.append(json.load(f))
    calibration = None
    if args.calibration:
        with open(args.calibration) as f:
            calibration = json.load(f)

    import tensorflow as tf
    dna = np.load(args.dna, mmap_mode='r')
    n = len(dna)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    # Written under a temporary name and renamed only when complete, so an
    # interrupted run never leaves a full-size but unfinished output behind.
    partial = os.path.join(out_dir, os.path.basename(args.output) + '.partial.npy')
    acc = open_memmap(partial, mode='w+', dtype=np.float32, shape=(n, 3))  # zero-filled
    tmp = os.path.join(out_dir, os.path.basename(args.output) + '.raw.tmp.npy')
    try:
        for model_path, meta in zip(args.model, meta_objs):
            model = tf.keras.models.load_model(model_path, compile=False)
            window, n_out = model.input_shape[1], model.output_shape[-1]
            raw = open_memmap(tmp, mode='w+', dtype=np.float32, shape=(n, n_out))
            tile_predict(dna, lambda x: model.predict_on_batch(x), window, n_out,
                         args.batch_size, out=raw)
            accumulate_classes(acc, raw, meta)
            del raw
            tf.keras.backend.clear_session()
            print(f'  {model_path}: done')
        finalize(acc, len(args.model), calibration)
        acc.flush()
        del acc
        os.replace(partial, args.output)
    finally:
        for f in (tmp, partial):
            if os.path.exists(f):
                os.remove(f)
    acc = np.load(args.output, mmap_mode='r')
    covered = sum(int(np.isfinite(acc[s:s + CHUNK, 0]).sum()) for s in range(0, n, CHUNK))
    print(f'Saved {args.output}: {covered:,}/{n:,} sites predicted '
          f'({len(args.model)} model{"s" if len(args.model) > 1 else ""}'
          f'{", calibrated" if calibration else ""})')


if __name__ == '__main__':
    main()
