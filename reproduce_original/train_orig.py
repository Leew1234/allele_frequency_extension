#!/usr/bin/env python3
"""Train one member of the original OVEP ensemble, as the original
train_batchnorm.py did.

Faithful to the original: architecture and loss (orig_model.py), batch size
16, 1,000 steps per epoch, up to 50 epochs, early stopping on validation loss
with patience 2, keeping the best model. Windows of 2,048 bp are drawn by
picking a training chromosome uniformly at random and then a uniform start;
windows with no labeled base are dropped, so a batch can hold fewer than 16
windows; windows containing N are kept (the original's check for them did
nothing). Validation draws from the same pool, 20,000 batches per epoch.
Python's random module is seeded with 3.14 as in the original, so all five
members see the same windows and differ only in their weight initialization.

Differences, all technical: Keras 3 instead of standalone Keras 2 (fit instead
of fit_generator, .keras format), the weight initialization is seeded with the
member index so a run can be repeated, and data are read from the .npy arrays
that run_plan.sh made rather than from the original .pkl files (same one-hot
coding up to the order of the four bases, which the model learns either way).

  python train_orig.py --work $WORK --member 0 --out models/half1/model0.keras \\
      NC_000003.12 NC_000005.10 ... NC_000021.9
"""
import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orig_model import WINDOW, cnn1d  # noqa: E402


def generate_data(dna, labels, chroms, batch_size, rng=random):
    lengths = {c: len(dna[c]) - WINDOW for c in chroms}
    while True:
        xs, ys = [], []
        for _ in range(batch_size):
            c = chroms[int(rng.random() * len(chroms))]
            start = int(rng.random() * lengths[c])
            y = labels[c][start:start + WINDOW]
            if np.sum(y) == 0:
                continue
            xs.append(np.asarray(dna[c][start:start + WINDOW], dtype=np.float32))
            ys.append(np.asarray(y, dtype=np.float32))
        if not xs:
            continue
        yield np.stack(xs), np.stack(ys)[..., None]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True)
    ap.add_argument('--member', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--val-steps', type=int, default=20000)
    ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('seqids', nargs='+', help='training chromosomes')
    a = ap.parse_args(argv)

    import keras
    keras.utils.set_random_seed(a.member)   # weight initialization
    random.seed(3.14)                        # window sampling, as in the original

    dna = {s: np.load(f'{a.work}/arrays/{s}_dna.npy', mmap_mode='r') for s in a.seqids}
    lab = {s: np.load(f'{a.work}/arrays/{s}_orig_labels.npy', mmap_mode='r') for s in a.seqids}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out[:-len('.keras')] + '.partial.keras' if a.out.endswith('.keras') else a.out + '.partial.keras'

    model = cnn1d(WINDOW, 4)
    callbacks = [keras.callbacks.ModelCheckpoint(tmp, save_best_only=True, monitor='val_loss'),
                 keras.callbacks.EarlyStopping(monitor='val_loss', patience=a.patience),
                 keras.callbacks.CSVLogger(a.out + '.history.csv')]
    model.fit(generate_data(dna, lab, a.seqids, a.batch_size),
              steps_per_epoch=a.steps, epochs=a.epochs,
              validation_data=generate_data(dna, lab, a.seqids, a.batch_size),
              validation_steps=a.val_steps, callbacks=callbacks, verbose=2)
    os.replace(tmp, a.out)
    print(f'saved {a.out}')


if __name__ == '__main__':
    main()
