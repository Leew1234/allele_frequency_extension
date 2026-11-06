"""Quantify how this extension's labels differ from the original
repository's binary labels on one chromosome.

The original labels (data/<seqid>_label.pkl, written by pickle_label_chr.py)
mark every position touched by any VCF record as polymorphic. The extension
marks a site polymorphic only if a second allele is actually observed in the
cohort, and sets uncallable, ambiguous or malformed sites to -1. Comparisons
between the original OVEP model and the extension use the extension's labels
for both models, so this report states how far that definition is from the
one the original model was trained on.

  python compare_labels.py --labels arrays/NC_000002.12_frequency_labels.npy \\
      --original-labels polymorphism-main/data/NC_000002.12_label.pkl --json labels_vs_original.json
"""
import argparse
import os
import json
import pickle
import numpy as np

CHUNK = 10_000_000


def compare(labels, original, chunk=CHUNK):
    n = min(len(labels), len(original))
    c = dict.fromkeys(['both_variant', 'original_only', 'extension_only', 'neither',
                       'excluded_original_variant', 'excluded_original_invariant'], 0)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        lab = np.asarray(labels[s:e])
        ov = np.asarray(original[s:e]) > 0
        ok = lab >= 0
        nv = lab >= 1
        c['both_variant'] += int((ok & nv & ov).sum())
        c['original_only'] += int((ok & ~nv & ov).sum())
        c['extension_only'] += int((ok & nv & ~ov).sum())
        c['neither'] += int((ok & ~nv & ~ov).sum())
        c['excluded_original_variant'] += int((~ok & ov).sum())
        c['excluded_original_invariant'] += int((~ok & ~ov).sum())
    scored = c['both_variant'] + c['original_only'] + c['extension_only'] + c['neither']
    orig_var_scored = c['both_variant'] + c['original_only']
    ext_var = c['both_variant'] + c['extension_only']
    return {
        'length_extension': int(len(labels)), 'length_original': int(len(original)),
        'compared_sites': int(n), 'counts': c,
        'scored_sites': scored,
        'agreement_on_scored_sites': (c['both_variant'] + c['neither']) / scored if scored else None,
        'original_variant_rate_scored': orig_var_scored / scored if scored else None,
        'extension_variant_rate_scored': ext_var / scored if scored else None,
        'share_of_original_variants_kept': c['both_variant'] / orig_var_scored if orig_var_scored else None,
        'share_of_original_variants_excluded': (c['excluded_original_variant']
                                                / max(orig_var_scored + c['excluded_original_variant'], 1)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labels', required=True, help='data.py labels .npy')
    p.add_argument('--original-labels', required=True, help="original repo's <seqid>_label.pkl (or .npy)")
    p.add_argument('--json')
    args = p.parse_args()
    labels = np.load(args.labels, mmap_mode='r')
    if args.original_labels.endswith('.npy'):
        original = np.load(args.original_labels, mmap_mode='r')
    else:
        with open(args.original_labels, 'rb') as f:
            original = np.asarray(pickle.load(f))
    res = compare(labels, original)
    text = json.dumps(res, indent=2)
    if args.json:
        with open(args.json + '.partial', 'w') as f:
            f.write(text)
        os.replace(args.json + '.partial', args.json)
    print(text)


if __name__ == '__main__':
    main()
