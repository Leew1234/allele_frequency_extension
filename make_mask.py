"""Turn a BED file into a boolean per-site mask aligned to a chromosome's arrays.

Masks are used by train.py (--mask-scope), calibrate.py (--mask) and
score.py (--mask). True = use this site.

Example: an "unannotated" (putatively neutral) mask, matching the original
repository's masklabel experiment, from GFF features not sourced 'RefSeq':

  awk -F'\\t' '!/^#/ && $1=="NC_000001.11" && $2!="RefSeq" {print $1"\\t"$4-1"\\t"$5}' \\
      GRCh38_latest_genomic.gff > annotated_chr1.bed
  python make_mask.py --bed annotated_chr1.bed --chromosome NC_000001.11 \\
      --labels arrays/chr1_frequency_labels.npy --invert --out masks/chr1_neutral.npy
"""
import argparse
from pathlib import Path
import numpy as np

from data import bed_to_mask, save_npy_atomic


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bed', required=True)
    p.add_argument('--chromosome', required=True, help='contig name as written in the BED')
    p.add_argument('--labels', required=True, help='labels .npy from data.py (sets the length)')
    p.add_argument('--invert', action='store_true', help='mask = sites NOT in the BED')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    n = len(np.load(args.labels, mmap_mode='r'))
    mask = bed_to_mask(args.bed, args.chromosome, n)
    if args.invert:
        mask = ~mask
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_npy_atomic(args.out, mask)
    print(f'{args.out}: {int(mask.sum()):,}/{n:,} sites ({mask.mean():.1%}) selected')


if __name__ == '__main__':
    main()
