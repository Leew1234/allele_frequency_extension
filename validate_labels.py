"""Cross-check data.py labels against an independent bcftools calculation.

bcftools recomputes AC/AN from the genotypes and reports MAF as the frequency
of the second most common allele (checked with bcftools 1.19), the same
definition data.py uses. Build the table from the SAME VCF, region and FILTER
policy you labeled, keeping genotypes and joining all records at a position
into one (-m+any):

  bcftools view -r chr1 -f PASS,. 1kGP.chr1.vcf.gz \\
    | bcftools norm -m+any -f GRCh38.fa \\
    | bcftools +fill-tags -- -t AC,AN,MAF \\
    | bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\t%INFO/AN\\t%INFO/MAF\\n' > chr1.bcftools.tsv

  python validate_labels.py --labels arrays/chr1_frequency_labels.npy \\
      --table chr1.bcftools.tsv --chromosome chr1 --threshold 0.01 \\
      --disagreements chr1_label_disagreements.tsv

Reported:
  record_sites     3x3 agreement (rows = data.py 0/1/2, columns = bcftools
                   0/1/2) at positions that have a bcftools record and a
                   scorable data.py label. Off-diagonal counts are bugs or
                   INFO/genotype inconsistencies worth inspecting.
  ignored_by_data_py   bcftools sites that data.py set to -1 (uncallable,
                   ambiguous, failed FILTER, conflicting AN, ...).
  variant_without_record   positions data.py calls variant with no bcftools
                   record: expected with --indel-mode span (bases inside
                   deletions), otherwise should be 0.

Spectrum labels (data.py --scheme spectrum): add --scheme spectrum (and the
same --count-bins). The expected class then comes from the bcftools minor
allele count, round(MAF * AN), and the agreement table is 5x5. With
--snv-only labels, also pass --snv-only here: records with a non-SNV allele
are then expected to be ignored (-1) by data.py and are counted separately
instead of being compared.
"""
import argparse
import json
import numpy as np

from spectrum import DEFAULT_COUNT_BINS, spectrum_class

_SNV = frozenset('ACGT*')


def expected_from_table(table_path, chromosome, length, threshold, scheme='v1',
                        count_bins=DEFAULT_COUNT_BINS, snv_only=False):
    """Per-position expected category from a bcftools query table.
    -9 = no record, -2 = record without a usable MAF/AN, -3 = record with a
    non-SNV allele (only with snv_only), else the class (0/1/2, or the
    spectrum classes). Also returns {pos: MAF} for record sites."""
    exp = np.full(length, -9, dtype=np.int8)
    maf_at = {}
    with open(table_path) as fh:
        for line in fh:
            if not line.strip() or line.startswith('#'):
                continue
            chrom, pos, ref, alt, an, maf = line.rstrip('\n').split('\t')[:6]
            if chrom != chromosome:
                continue
            p = int(pos) - 1
            if not 0 <= p < length:
                continue
            if snv_only and not (len(ref) == 1 and all(a in _SNV for a in alt.split(','))):
                exp[p:min(p + len(ref), length)] = -3
                continue
            if exp[p] == -3:
                continue  # an SNV record under an earlier indel's span stays ignored
            try:
                an_v, maf_v = float(an), float(maf)
            except ValueError:
                exp[p] = -2
                continue
            if an_v <= 0:
                exp[p] = -2
                continue
            maf_at[p] = maf_v
            if scheme == 'spectrum':
                exp[p] = spectrum_class(round(maf_v * an_v), maf_v, threshold, count_bins)
            else:
                exp[p] = 0 if maf_v <= 0 else (1 if maf_v < threshold else 2)
    return exp, maf_at


def compare(labels, expected, maf_at=None, threshold=0.01, boundary_tol=1e-4, n_classes=3):
    labels = np.asarray(labels)
    if len(labels) != len(expected):
        raise ValueError('labels and table length differ')
    rec = expected >= 0
    both = rec & (labels >= 0)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (labels[both].astype(int), expected[both].astype(int)), 1)
    mismatch = np.flatnonzero(both & (labels != expected))
    boundary = 0
    if maf_at is not None:
        boundary = sum(1 for p in mismatch if abs(maf_at.get(int(p), -1) - threshold) < boundary_tol)
    agree = int(np.trace(confusion))
    total = int(confusion.sum())
    diag = np.diag(confusion)
    per_class = [(int(diag[c]) / int(confusion[:, c].sum())) if confusion[:, c].sum() else None
                 for c in range(n_classes)]
    return {
        'record_sites_compared': total,
        'agreement': agree / total if total else None,
        'agreement_by_bcftools_class': per_class,
        'non_snv_sites_not_ignored_by_data_py': int(((expected == -3) & (labels >= 0)).sum()),
        'confusion_rows_datapy_cols_bcftools': confusion.tolist(),
        'mismatches': int(mismatch.size),
        'mismatches_within_rounding_of_threshold': int(boundary),
        'ignored_by_data_py': int((rec & (labels < 0)).sum()),
        'ignored_by_data_py_but_bcftools_variant': int((rec & (labels < 0) & (expected >= 1)).sum()),
        'bcftools_records_without_usable_maf': int((expected == -2).sum()),
        'variant_without_record': int(((expected == -9) & (labels >= 1)).sum()),
    }, mismatch


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labels', required=True)
    p.add_argument('--table', required=True, help='bcftools query output (see above)')
    p.add_argument('--chromosome', required=True)
    p.add_argument('--threshold', type=float, default=0.01, help='same value given to data.py')
    p.add_argument('--disagreements', help='write mismatching positions to this TSV')
    p.add_argument('--scheme', choices=['v1', 'spectrum'], default='v1')
    p.add_argument('--count-bins', nargs='+', type=int, default=list(DEFAULT_COUNT_BINS))
    p.add_argument('--snv-only', action='store_true', help='the labels were made with data.py --snv-only')
    args = p.parse_args()
    labels = np.load(args.labels, mmap_mode='r')
    expected, maf_at = expected_from_table(args.table, args.chromosome, len(labels), args.threshold,
                                           args.scheme, tuple(args.count_bins), args.snv_only)
    n_classes = len(args.count_bins) + 3 if args.scheme == 'spectrum' else 3
    summary, mismatch = compare(labels, expected, maf_at, args.threshold, n_classes=n_classes)
    print(json.dumps(summary, indent=2))
    if args.disagreements:
        with open(args.disagreements, 'w') as f:
            f.write('pos1\tdata_py\tbcftools\tbcftools_maf\n')
            for q in mismatch:
                f.write(f'{q + 1}\t{labels[q]}\t{expected[q]}\t{maf_at.get(int(q), "")}\n')
        print(f'Wrote {mismatch.size} disagreements to {args.disagreements}')


if __name__ == '__main__':
    main()
