#!/usr/bin/env python3
"""Write <arrays>/<seqid>_orig_labels.npy (uint8 0/1) with the original
repository's label definition, from the sites VCF that run_plan.sh made.

  python make_orig_labels.py --work $WORK NC_000003.12 NC_000005.10 ...
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orig_model import original_labels  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', required=True, help='run_plan.sh WORK directory (has vcf/ and arrays/)')
    ap.add_argument('seqids', nargs='+')
    a = ap.parse_args(argv)
    for s in a.seqids:
        out = f'{a.work}/arrays/{s}_orig_labels.npy'
        dna = f'{a.work}/arrays/{s}_dna.npy'
        vcf = f'{a.work}/vcf/{s}.sites.vcf.gz'
        if os.path.exists(out) and os.path.getmtime(out) > max(os.path.getmtime(dna), os.path.getmtime(vcf)):
            print(f'{s}: up to date')
            continue
        n = len(np.load(dna, mmap_mode='r'))
        lab = original_labels(vcf, s, n)
        tmp = out + '.partial.npy'
        np.save(tmp, lab)
        os.replace(tmp, out)
        print(f'{s}: {n} bases, {int(lab.sum())} labeled 1 ({lab.mean():.4f})')


if __name__ == '__main__':
    main()
