"""Frequency-spectrum classes (analysis plan v2) shared by labelling, the
model, calibration and scoring.

Per reference base:
   0  callable, no variant allele observed
   1  singleton      minor allele count == 1
   2  ultra-rare     minor allele count 2..5
   3  rare           minor allele count > 5 and minor allele frequency < threshold
   4  common         minor allele frequency >= threshold
  -1  ignore

"Minor allele" is the second most common allele at the site, REF included
(the folded spectrum, as in data.py), so no ancestral-state call is needed.
The common boundary is a frequency so that it stays identical to the v1
rare/common boundary; it takes precedence over the count classes (with few
samples a count of 2 can already be >= the threshold). Pooling 1-3 -> rare
and 4 -> common therefore gives the v1 label of the same site.

The count boundaries are configurable (`count_bins`, upper bounds of the
count-defined classes); everything downstream reads the number of classes
from the arrays, and only assumes class 1 = singleton, last class = common.
"""
import numpy as np

DEFAULT_COUNT_BINS = (1, 5)
CLASS_NAMES = ('none', 'singleton', 'ultra_rare', 'rare', 'common')
UNKNOWN = 9   # training only: a variant whose frequency class is not learned


def class_names(n_classes):
    """Names for an (n_classes)-column probability array (incl. 'none')."""
    if n_classes == len(CLASS_NAMES):
        return list(CLASS_NAMES)
    return ['none'] + [f'count_class_{i}' for i in range(1, n_classes - 2)] + ['rare', 'common']


def spectrum_class(mac, maf, threshold=0.01, count_bins=DEFAULT_COUNT_BINS):
    """Class of a polymorphic site from its minor allele count and frequency."""
    if mac <= 0:
        return 0
    n_freq = len(count_bins) + 2
    if maf >= threshold:
        return n_freq
    mac = int(round(mac))
    for i, upper in enumerate(count_bins):
        if mac <= upper:
            return i + 1
    return n_freq - 1


def collapse_labels(labels, n_freq=len(DEFAULT_COUNT_BINS) + 2):
    """Spectrum labels -> v1 labels (0 none, 1 rare, 2 common, -1 ignore).
    `n_freq` is the number of variant classes; the last one is common."""
    labels = np.asarray(labels)
    out = np.where(labels >= 1, 1, labels).astype(np.int8)
    out[labels == n_freq] = 2
    return out


def collapse_probabilities(probs):
    """(n, 1+K) spectrum probabilities -> (n, 3): none, rare (all but the last
    class), common (the last class). NaN rows stay NaN."""
    probs = np.asarray(probs)
    if probs.ndim != 2 or probs.shape[1] < 3:
        raise ValueError('expected (n, >=3) class probabilities')
    return np.stack([probs[:, 0], probs[:, 1:-1].sum(axis=1), probs[:, -1]], axis=1).astype(np.float32)


def variant_probability(probs):
    """P(variant) from class probabilities with any number of classes."""
    probs = np.asarray(probs, dtype=np.float64)
    return probs[..., 1:].sum(axis=-1)


def conditional_classes(probs):
    """P(class k | variant) for k = 1..K, shape (..., K). 0 where P(variant) is
    0, NaN where the input is NaN."""
    probs = np.asarray(probs, dtype=np.float64)
    pv = probs[..., 1:].sum(axis=-1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        cond = np.where(pv > 0, probs[..., 1:] / pv, 0.0)
    cond[~np.isfinite(np.broadcast_to(pv, cond.shape))] = np.nan
    return cond


class LazyMask:
    """A mask computed per slice from other masks (arrays, memmaps or
    IntervalMask), e.g. LazyMask(lambda a, b: a & ~b, [phastcons, cds]).
    Never builds a chromosome-length array."""

    def __init__(self, fn, parts):
        self.fn, self.parts = fn, list(parts)
        self.length = len(self.parts[0])
        if any(len(p) != self.length for p in self.parts):
            raise ValueError('mask parts differ in length')

    def __len__(self):
        return self.length

    def __getitem__(self, sl):
        return np.asarray(self.fn(*(np.asarray(p[sl], dtype=bool) for p in self.parts)), dtype=bool)


def main():
    import argparse
    from numpy.lib.format import open_memmap
    import os
    p = argparse.ArgumentParser(description='Collapse spectrum labels or predictions to the v1 '
                                'three-class form, so every v1 tool (score.py, compare_ovep.py, '
                                'score_annotations.py, discriminate.py) can be run on a spectrum model.')
    p.add_argument('kind', choices=['labels', 'predictions'])
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--n-freq-classes', type=int, default=len(DEFAULT_COUNT_BINS) + 2,
                   help='labels only: number of variant classes (the last is common)')
    a = p.parse_args()
    src = np.load(a.input, mmap_mode='r')
    tmp = a.output + '.partial.npy'
    chunk = 5_000_000
    if a.kind == 'labels':
        out = open_memmap(tmp, mode='w+', dtype=np.int8, shape=(len(src),))
        for s in range(0, len(src), chunk):
            out[s:s + chunk] = collapse_labels(src[s:s + chunk], a.n_freq_classes)
    else:
        out = open_memmap(tmp, mode='w+', dtype=np.float32, shape=(len(src), 3))
        for s in range(0, len(src), chunk):
            out[s:s + chunk] = collapse_probabilities(src[s:s + chunk])
    out.flush()
    del out
    os.replace(tmp, a.output)
    print(f'Saved {a.output}')


if __name__ == '__main__':
    main()
