"""Sequence-context (k-mer) baseline for the frequency spectrum.

The standard way to adjust the proportion of singletons for mutability (MAPS,
Lek et al. 2016) conditions on the trinucleotide context. This script builds
that baseline in the same file format as the network's predictions, so every
scoring tool can be run on it and compared with the network like for like:

  fit      count, for every k-mer centred on a site (strand-symmetric: a
           k-mer and its reverse complement are one context), how many
           scorable sites carry no variant / each frequency class. Uses the
           same manifest, chromosomes and neutral mask as train.py, so the
           baseline learns from exactly the sites the network learns from.
  predict  write (n_sites, classes) class probabilities for one chromosome:
           P(variant | context) and P(class | variant, context). Sites whose
           k-mer contains an ambiguous base or runs off the chromosome end
           are NaN. --calibration applies a calibrate.py map, fitted on the
           baseline's own uncalibrated output for the calibration chromosome,
           exactly as for the network.

k = 3 is the trinucleotide context; 5 and 7 are stronger baselines (a
heptamer table has 8,192 contexts). Sparse contexts are shrunk toward the
genome-wide rates (--prior-sites, --prior-variants pseudo-observations).

  python kmer_baseline.py fit --manifest manifest_A.json --chromosomes chr3 chr5 \\
      --mask-scope all --k 3 --out kmer3.json
  python kmer_baseline.py predict --table kmer3.json --dna arrays/chr4_dna.npy \\
      --output pred/kmer3/chr4.npy [--calibration cal/kmer3.json]
"""
import argparse
import json
import os
import numpy as np
from numpy.lib.format import open_memmap
from numpy.lib.stride_tricks import sliding_window_view

CHUNK = 5_000_000


def context_codes(dna, start, end, k):
    """Canonical k-mer code (int64) of every site in [start, end): the k-mer
    centred on the site, or its reverse complement, whichever code is smaller.
    -1 where the k-mer contains an ambiguous base or leaves the chromosome.
    `dna` is the (n, 4) one-hot array of data.py (channel order A, C, T, G, so
    the complement of channel i is (i + 2) % 4)."""
    if k < 1 or k % 2 == 0:
        raise ValueError('k must be a positive odd number')
    n, h = len(dna), k // 2
    a, b = max(0, start - h), min(n, end + h)
    x = np.asarray(dna[a:b])
    idx = np.where(x.max(axis=1) > 0, x.argmax(axis=1), -1).astype(np.int64)
    idx = np.concatenate([np.full(h - (start - a), -1, np.int64), idx,
                          np.full(h - (b - end), -1, np.int64)])
    win = sliding_window_view(idx, k)
    weights = 4 ** np.arange(k, dtype=np.int64)
    fwd = win @ weights
    # reverse complement: complement each base, read the window backwards
    rev = sliding_window_view((idx + 2) % 4, k) @ weights[::-1]
    code = np.minimum(fwd, rev)
    code[(win < 0).any(axis=1)] = -1
    return code


def count_contexts(dna, labels, k, n_classes, mask=None, counts=None, chunk=CHUNK):
    """counts[context, class] += number of scorable sites (label >= 0, inside
    `mask` when given, unambiguous k-mer)."""
    if counts is None:
        counts = np.zeros((4 ** k, n_classes), dtype=np.int64)
    n = len(labels)
    if len(dna) != n:
        raise ValueError('dna and labels differ in length')
    if mask is not None and len(mask) != n:
        raise ValueError('mask and labels differ in length')
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        lab = np.asarray(labels[s:e]).astype(np.int64)
        code = context_codes(dna, s, e, k)
        ok = (lab >= 0) & (code >= 0)
        if mask is not None:
            ok &= np.asarray(mask[s:e], dtype=bool)
        if lab[ok].size and lab[ok].max() >= n_classes:
            raise ValueError(f'labels contain class {int(lab[ok].max())} but n_classes is {n_classes}')
        flat = np.bincount(code[ok] * n_classes + lab[ok], minlength=counts.size)
        counts += flat.reshape(counts.shape)
    return counts


def table_probabilities(counts, prior_sites=100.0, prior_variants=20.0):
    """(contexts, classes) class probabilities from the count table, each
    context shrunk toward the genome-wide rates. Contexts never seen (the
    non-canonical half of the code space) get the genome-wide rates too."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total == 0 or counts[:, 1:].sum() == 0:
        raise ValueError('the table holds no scorable sites or no variants')
    g_var = counts[:, 1:].sum() / total
    g_cond = counts[:, 1:].sum(axis=0) / counts[:, 1:].sum()
    sites = counts.sum(axis=1)
    var = counts[:, 1:].sum(axis=1)
    pv = (var + prior_sites * g_var) / (sites + prior_sites)
    cond = (counts[:, 1:] + prior_variants * g_cond) / (var + prior_variants)[:, None]
    return np.concatenate([(1 - pv)[:, None], pv[:, None] * cond], axis=1)


def predict_chromosome(dna, probs_table, k, out=None, calibration=None, chunk=CHUNK):
    """Fill `out` (n, classes) float32 with the table's class probabilities."""
    n = len(dna)
    if out is None:
        out = np.zeros((n, probs_table.shape[1]), dtype=np.float32)
    if calibration is not None:
        from calibrate import calibrate_classes
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        code = context_codes(dna, s, e, k)
        p = probs_table[np.where(code >= 0, code, 0)]
        p[code < 0] = np.nan
        if calibration is not None:
            p = calibrate_classes(p, calibration)
        out[s:e] = p
    return out


def _load_table(path):
    with open(path) as f:
        t = json.load(f)
    return t, table_probabilities(np.array(t['counts']), t['prior_sites'], t['prior_variants'])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fit', help='count contexts on training chromosomes')
    f.add_argument('--manifest', required=True, help='same JSON as train.py')
    f.add_argument('--chromosomes', nargs='+', required=True)
    f.add_argument('--mask-scope', choices=['none', 'all'], default='none',
                   help="'all' counts only sites inside the manifest 'mask' (neutral sites)")
    f.add_argument('--k', type=int, default=3)
    f.add_argument('--n-classes', type=int, default=5,
                   help='class columns incl. "none": 5 for spectrum labels, 3 for v1 labels')
    f.add_argument('--prior-sites', type=float, default=100.0)
    f.add_argument('--prior-variants', type=float, default=20.0)
    f.add_argument('--out', required=True)
    q = sub.add_parser('predict', help='write class probabilities for one chromosome')
    q.add_argument('--table', required=True)
    q.add_argument('--dna', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--calibration', help='JSON from calibrate.py')
    args = p.parse_args()

    if args.cmd == 'fit':
        with open(args.manifest) as fh:
            sources = json.load(fh)
        missing = set(args.chromosomes) - sources.keys()
        if missing:
            p.error(f'Not in manifest: {sorted(missing)}')
        counts = None
        for c in args.chromosomes:
            if args.mask_scope == 'all' and 'mask' not in sources[c]:
                p.error(f"{c}: --mask-scope all needs a 'mask' entry in the manifest")
            mask = np.load(sources[c]['mask'], mmap_mode='r') if args.mask_scope == 'all' else None
            counts = count_contexts(np.load(sources[c]['dna'], mmap_mode='r'),
                                    np.load(sources[c]['labels'], mmap_mode='r'),
                                    args.k, args.n_classes, mask, counts)
            print(f'  {c}: {int(counts.sum()):,} sites so far')
        table_probabilities(counts, args.prior_sites, args.prior_variants)  # fails early if empty
        seen = int((counts.sum(axis=1) > 0).sum())
        table = {'k': args.k, 'n_classes': args.n_classes, 'prior_sites': args.prior_sites,
                 'prior_variants': args.prior_variants, 'chromosomes': args.chromosomes,
                 'mask_scope': args.mask_scope, 'n_sites': int(counts.sum()),
                 'n_variant_sites': int(counts[:, 1:].sum()), 'contexts_seen': seen,
                 'counts': counts.tolist()}
        with open(args.out + '.partial', 'w') as fh:
            json.dump(table, fh)
        os.replace(args.out + '.partial', args.out)
        print(f"Saved {args.out}: k={args.k}, {seen:,} contexts, {table['n_sites']:,} sites, "
              f"{table['n_variant_sites']:,} variant sites")
        return

    if not args.output.endswith('.npy'):
        p.error('--output must end in .npy (written as a memory map)')
    table, probs = _load_table(args.table)
    calibration = None
    if args.calibration:
        with open(args.calibration) as fh:
            calibration = json.load(fh)
        if calibration.get('n_classes', 3) != table['n_classes']:
            p.error('--calibration was fitted for a different number of classes')
    dna = np.load(args.dna, mmap_mode='r')
    partial = args.output + '.partial.npy'
    try:
        out = open_memmap(partial, mode='w+', dtype=np.float32, shape=(len(dna), table['n_classes']))
        predict_chromosome(dna, probs, table['k'], out, calibration)
        out.flush()
        del out
        os.replace(partial, args.output)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    print(f"Saved {args.output} (k={table['k']}{', calibrated' if calibration else ''})")


if __name__ == '__main__':
    main()
