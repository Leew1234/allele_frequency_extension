"""Train the frequency-aware U-Net with chromosome-disjoint validation.

Input is a JSON mapping chromosome IDs to {dna: path, labels: path}; all paths
point at arrays emitted by data.py and are memory-mapped.

Training windows start at random positions on every draw (as in the paper),
chromosomes are drawn in proportion to their usable length, and windows
containing an ambiguous base are never used. Validation uses a fixed,
non-overlapping grid on separate chromosomes so val_loss is comparable across
epochs. A <out>.meta.json file records the head and any class weights so that
predict.py can undo the weighting.

Neutral-region training: give manifest entries a "mask" (boolean .npy from
make_mask.py, True = putatively neutral) and choose --mask-scope:
  all     labels outside the mask are ignored entirely; the network still sees
          the full sequence as input context (the original repo instead
          zeroed the input).
  common  factorized head only: P(variant) learns from every site, but
          P(common | variant) learns only from variants inside the mask.
"""
import argparse
import os
import json
import math
import numpy as np


MASK_SCOPES = ('none', 'all', 'common')


def valid_starts(dna, labels, window, mask=None, chunk=10_000_000):
    """Boolean array over window start positions: True if the window has no
    ambiguous base and at least one scorable site (label >= 0, inside `mask`
    when one is given)."""
    n = len(labels)
    if n < window:
        return np.zeros(0, dtype=bool)
    flag = np.empty(n, dtype=bool)
    for s in range(0, n, chunk):
        flag[s:s + chunk] = np.asarray(dna[s:s + chunk]).max(axis=1) == 0
    csum = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(flag, out=csum[1:])
    ok = (csum[window:] - csum[:-window]) == 0
    for s in range(0, n, chunk):
        f = np.asarray(labels[s:s + chunk]) >= 0
        if mask is not None:
            f &= np.asarray(mask[s:s + chunk], dtype=bool)
        flag[s:s + chunk] = f
    np.cumsum(flag, out=csum[1:])
    ok &= (csum[window:] - csum[:-window]) > 0
    return ok


def masked_labels(y, m, scope):
    """Apply a neutral mask to one window of labels (int32 copy)."""
    y = np.array(y, dtype=np.int32)
    if m is None or scope == 'none':
        return y
    m = np.asarray(m, dtype=bool)
    if scope == 'all':
        y[~m] = -1
    elif scope == 'common':
        y[(~m) & (y >= 1)] = 3  # variant, frequency class not learned here
    else:
        raise ValueError(f'mask scope must be one of {MASK_SCOPES}')
    return y


class ChromosomeSampler:
    """Draws uniformly random valid window starts from one chromosome."""

    def __init__(self, dna, labels, window, mask=None):
        self.ok = valid_starts(dna, labels, window, mask)
        self.n_valid = int(self.ok.sum())
        # Rejection sampling is fast when most starts are valid; otherwise
        # fall back to an explicit index list.
        self.index = np.flatnonzero(self.ok) if self.n_valid and self.n_valid < 0.05 * len(self.ok) else None

    def sample(self, rng):
        if self.index is not None:
            return int(self.index[rng.integers(len(self.index))])
        while True:
            s = int(rng.integers(len(self.ok)))
            if self.ok[s]:
                return s


def make_batch(data, coords, window, head, class_weights, mask_scope='none'):
    """data[chrom] is (dna, labels) or (dna, labels, mask)."""
    xs, ys = [], []
    for chrom, start in coords:
        x, y, *rest = data[chrom]
        m = rest[0] if rest else None
        xs.append(np.asarray(x[start:start + window], dtype=np.float32))
        ys.append(masked_labels(y[start:start + window],
                                None if m is None else m[start:start + window], mask_scope))
    X, Y = np.stack(xs), np.stack(ys)
    if head == 'factorized':
        return X, Y.astype(np.float32)  # the loss masks -1 itself
    valid = Y >= 0
    W = np.where(valid, np.asarray(class_weights, dtype=np.float32)[np.where(valid, Y, 0)], 0.0)
    return X, np.where(valid, Y, 0), W.astype(np.float32)


def random_batches(data, samplers, window, batch_size, head, class_weights, seed,
                   mask_scope='none'):
    names = [c for c in samplers if samplers[c].n_valid > 0]
    if not names:
        raise ValueError('No usable training windows; check callable intervals and window size')
    prob = np.array([samplers[c].n_valid for c in names], dtype=np.float64)
    prob /= prob.sum()
    rng = np.random.default_rng(seed)
    while True:
        picks = rng.choice(len(names), size=batch_size, p=prob)
        coords = [(names[i], samplers[names[i]].sample(rng)) for i in picks]
        yield make_batch(data, coords, window, head, class_weights, mask_scope)


def _window_mask(data, chrom, mask_scope):
    """Mask that restricts which windows are usable (only scope 'all')."""
    entry = data[chrom]
    return entry[2] if mask_scope == 'all' and len(entry) > 2 else None


def grid_windows(data, chromosomes, window, max_windows=None, seed=43, mask_scope='none'):
    coords = []
    for chrom in chromosomes:
        ok = valid_starts(data[chrom][0], data[chrom][1], window,
                          _window_mask(data, chrom, mask_scope))
        starts = np.arange(0, len(ok), window)
        coords += [(chrom, int(s)) for s in starts[ok[starts]]]
    if not coords:
        raise ValueError('No usable validation windows')
    if max_windows and len(coords) > max_windows:
        keep = np.sort(np.random.default_rng(seed).choice(len(coords), max_windows, replace=False))
        coords = [coords[i] for i in keep]
    return coords


def fixed_batches(data, coords, window, batch_size, head, class_weights, mask_scope='none'):
    while True:
        for i in range(0, len(coords), batch_size):
            yield make_batch(data, coords[i:i + batch_size], window, head, class_weights, mask_scope)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', required=True)
    p.add_argument('--train-chromosomes', nargs='+', required=True)
    p.add_argument('--val-chromosomes', nargs='+', required=True)
    p.add_argument('--head', choices=['softmax', 'factorized'], default='softmax')
    p.add_argument('--window', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--steps-per-epoch', type=int, default=1000)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--max-val-windows', type=int, default=5000)
    p.add_argument('--class-weights', nargs=3, type=float, default=[1, 1, 1],
                   help='softmax head only; predict.py undoes the weighting')
    p.add_argument('--variant-pos-weight', type=float, default=1.0, help='factorized head only')
    p.add_argument('--common-pos-weight', type=float, default=1.0, help='factorized head only')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0, help='change per ensemble member')
    p.add_argument('--verbose', type=int, choices=[0, 1, 2], default=2,
                   help='Keras verbosity; 2 = one line per epoch (log-friendly)')
    p.add_argument('--mask-scope', choices=MASK_SCOPES, default='none',
                   help="use the manifest 'mask' arrays (see module docstring)")
    p.add_argument('--out', default='frequency_model.keras')
    args = p.parse_args()
    if set(args.train_chromosomes) & set(args.val_chromosomes):
        p.error('Validation chromosomes must be disjoint from training chromosomes')
    if args.mask_scope == 'common' and args.head != 'factorized':
        p.error('--mask-scope common requires --head factorized')
    with open(args.manifest) as f:
        sources = json.load(f)
    needed = set(args.train_chromosomes + args.val_chromosomes)
    missing = needed - sources.keys()
    if missing:
        p.error(f'Not in manifest: {sorted(missing)}')
    data = {}
    for c in needed:
        x = np.load(sources[c]['dna'], mmap_mode='r')
        y = np.load(sources[c]['labels'], mmap_mode='r')
        if len(x) != len(y):
            p.error(f'{c}: dna and labels lengths differ ({len(x)} vs {len(y)})')
        if args.mask_scope == 'none':
            data[c] = (x, y)
            continue
        if 'mask' not in sources[c]:
            p.error(f"{c}: --mask-scope {args.mask_scope} needs a 'mask' entry in the manifest")
        m = np.load(sources[c]['mask'], mmap_mode='r')
        if len(m) != len(y):
            p.error(f'{c}: mask and labels lengths differ ({len(m)} vs {len(y)})')
        data[c] = (x, y, m)

    samplers = {c: ChromosomeSampler(data[c][0], data[c][1], args.window,
                                     _window_mask(data, c, args.mask_scope))
                for c in args.train_chromosomes}
    val = grid_windows(data, args.val_chromosomes, args.window, args.max_val_windows,
                       mask_scope=args.mask_scope)
    print(f'train starts: {sum(s.n_valid for s in samplers.values()):,}; val windows: {len(val):,}')

    import tensorflow as tf
    from model import build_model
    tf.keras.utils.set_random_seed(args.seed)
    model = build_model(args.window, lr=args.lr, head=args.head,
                        variant_pos_weight=args.variant_pos_weight,
                        common_pos_weight=args.common_pos_weight)
    train_gen = random_batches(data, samplers, args.window, args.batch_size, args.head,
                               args.class_weights, args.seed, args.mask_scope)
    val_gen = fixed_batches(data, val, args.window, args.batch_size, args.head,
                            args.class_weights, args.mask_scope)
    model.fit(train_gen, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs,
              validation_data=val_gen, validation_steps=math.ceil(len(val) / args.batch_size),
              verbose=args.verbose,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience,
                                                          restore_best_weights=True)])
    partial = args.out[:-len('.keras')] + '.partial.keras' if args.out.endswith('.keras') else args.out + '.partial'
    model.save(partial)
    os.replace(partial, args.out)
    meta = {'head': args.head, 'window': args.window,
            'class_weights': args.class_weights,
            'variant_pos_weight': args.variant_pos_weight,
            'common_pos_weight': args.common_pos_weight,
            'train_chromosomes': args.train_chromosomes,
            'val_chromosomes': args.val_chromosomes, 'seed': args.seed,
            'mask_scope': args.mask_scope}
    with open(args.out + '.meta.json.partial', 'w') as f:  # written last: marks completion
        json.dump(meta, f, indent=2)
    os.replace(args.out + '.meta.json.partial', args.out + '.meta.json')
    print(f'Saved {args.out} and {args.out}.meta.json')


if __name__ == '__main__':
    main()
