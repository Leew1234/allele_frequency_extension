"""Generate allele-frequency category labels from a matching FASTA and VCF.

Categories per reference base:
   0  callable, no variant allele observed in the cohort
   1  rare polymorphism   (minor allele frequency <  threshold)
   2  common polymorphism (minor allele frequency >= threshold)
  -1  ignore (ambiguous base, uncallable, failed filter, malformed record)

"Minor allele frequency" is the frequency of the second most common allele at
the site (REF included). So a site where REF is at 0.5% and ALT at 99.5% is a
*rare* polymorphism, while a site whose ALT count is 0, or where every called
allele is the same ALT (the reference genome carries an allele absent from the
cohort), is *not* a polymorphism in this cohort and is labeled 0.

Records at the same position (e.g. multi-allelic sites split with
`bcftools norm -m-`) are merged before classification, so the result does not
depend on record order. Merging requires the records to report the same AN;
otherwise the site is set to -1. validate_labels.py cross-checks the labels
against bcftools.

--scheme spectrum (analysis plan v2) writes finer classes instead, to
<prefix>_spectrum_labels.npy: 1 singleton, 2 ultra-rare (2-5 copies), 3 rare,
4 common, with the same "minor allele" definition (see spectrum.py). Pooling
classes 1-3 gives the rare label above. --snv-only labels single-nucleotide
variants only. Singletons need a call set that contains them: the 1000
Genomes phased panel does not (experiments/spectrum/fetch_fullcallset.sh).
"""
import argparse
import os
import gzip
from pathlib import Path
import numpy as np

from spectrum import DEFAULT_COUNT_BINS, spectrum_class

BASES = 'ACTG'  # preserve the original repository's one-hot channel order
INDEL_MODES = ('anchor', 'span', 'original')
SCHEMES = ('v1', 'spectrum')
_SNV_ALLELES = frozenset('ACGT*')
_LUT = np.full(256, -1, dtype=np.int8)
for _i, _b in enumerate(BASES):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i


def base_index(sequence):
    """Channel index (0-3) per base, -1 for N or any other character."""
    return _LUT[np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)]


def encode_dna(sequence):
    idx = base_index(sequence)
    output = np.zeros((len(idx), 4), dtype=np.uint8)
    ok = np.flatnonzero(idx >= 0)
    output[ok, idx[ok]] = 1
    return output


def read_fasta(path, chromosome):
    opener = gzip.open if str(path).endswith('.gz') else open
    pieces, keep, found = [], False, False
    with opener(path, 'rt') as f:
        for line in f:
            if line.startswith('>'):
                if found and keep:
                    break  # finished the requested contig
                name = line[1:].split()[0]
                keep = name == chromosome
                found |= keep
            elif keep:
                pieces.append(line.strip())
    if not found:
        raise ValueError(f'Chromosome {chromosome!r} not found in FASTA')
    return ''.join(pieces).upper()


def minor_allele(an, alt_counts, star_count=0):
    """(count, frequency) of the second most common allele at a site, REF
    included; (0, 0) if the site is not polymorphic in the cohort.

    `star_count` is the AC of '*' (spanning-deletion) alleles: those haplotypes
    carry neither REF nor a variant starting here, so they reduce the REF count
    but are not variation at this site. Raises ValueError on inconsistent counts.
    """
    if an <= 0 or star_count < 0 or any(c < 0 for c in alt_counts):
        raise ValueError('invalid allele counts')
    ref_count = an - sum(alt_counts) - star_count
    if ref_count < -1e-9:
        raise ValueError('allele counts exceed AN')
    if sum(alt_counts) == 0:
        return 0.0, 0.0
    counts = sorted([max(ref_count, 0.0)] + list(alt_counts), reverse=True)
    # counts[1] == 0: every called allele is the same ALT, monomorphic in the cohort
    return counts[1], counts[1] / an


def site_category(an, alt_counts, star_count=0, threshold=0.01):
    """Classify one site from AN and the counts of its real ALT alleles.
    Returns 0 (no ALT observed), 1 (rare) or 2 (common)."""
    mac, maf = minor_allele(an, alt_counts, star_count)
    if mac == 0:
        return 0
    return 2 if maf >= threshold else 1


def site_spectrum_class(an, alt_counts, star_count=0, threshold=0.01,
                        count_bins=DEFAULT_COUNT_BINS):
    """Frequency-spectrum class of one site (see spectrum.py): 0 none,
    1 singleton, 2 ultra-rare, 3 rare, 4 common with the default bins."""
    mac, maf = minor_allele(an, alt_counts, star_count)
    return spectrum_class(mac, maf, threshold, count_bins)


def _ac_an(info):
    ac = an = None
    for chunk in info.split(';'):
        if chunk.startswith('AC='):
            ac = chunk[3:]
        elif chunk.startswith('AN='):
            an = chunk[3:]
    if ac is None or an is None:
        raise ValueError('missing AC/AN')
    return [float(x) for x in ac.split(',')], float(an)


def vcf_categories(vcf_path, chromosome, reference, threshold=0.01,
                   indel_mode='anchor', pass_only=True, scheme='v1',
                   count_bins=DEFAULT_COUNT_BINS, snv_only=False):
    """Return an int8 label array for one chromosome (see module docstring).

    scheme='v1' gives 0/1/2 (none/rare/common); scheme='spectrum' gives the
    finer classes of spectrum.py (0 none, 1 singleton, 2 ultra-rare, 3 rare,
    4 common with the default `count_bins`). Where several records label the
    same base, the highest class wins under both schemes, so pooling the
    spectrum classes reproduces the v1 label.

    snv_only=True labels single-nucleotide variants only: every reference
    base covered by a record with an indel, MNP or symbolic allele is set to
    -1 (neither a variant nor "no variant"). Singleton indel calls are the
    least reliable genotypes, and frequency-spectrum analyses (e.g. MAPS)
    are conventionally SNV-only. `indel_mode` then has no effect.

    indel_mode='anchor' labels only the VCF POS base of an indel.
    indel_mode='span' also labels every REF base a deletion covers.
    indel_mode='original' reproduces the original repository's slice
    (1000genome/select_indel.py + pickle_label_chr.py): a record whose ALT
    field is as long as REF marks POS only; any other record marks POS plus
    len(REF) further bases (one more than 'span').
    A site with no VCF record is labeled 0; pass the result through
    apply_callable_bed() so that uncallable sites become -1.
    """
    if not 0 < threshold < 0.5:
        raise ValueError('threshold must be between 0 and 0.5')
    if indel_mode not in INDEL_MODES:
        raise ValueError(f'indel_mode must be one of {INDEL_MODES}')
    if scheme not in SCHEMES:
        raise ValueError(f'scheme must be one of {SCHEMES}')
    n = len(reference)
    blocked = np.zeros(n, dtype=bool) if snv_only else None
    labels = np.zeros(n, dtype=np.int8)
    labels[base_index(reference) < 0] = -1

    sites = {}      # pos -> [AN, [alt ACs], star AC, end]
    invalid = set()
    prefix = chromosome + '\t'
    opener = gzip.open if str(vcf_path).endswith('.gz') else open
    with opener(vcf_path, 'rt') as fh:
        for line in fh:
            if not line.startswith(prefix):
                continue  # header lines and other contigs
            fields = line.split('\t', 8)
            if len(fields) < 8:
                continue
            _, pos, _, ref, alts, _, filt, info = fields[:8]
            pos = int(pos) - 1
            if pos < 0 or pos >= n:
                continue
            ref = ref.upper()
            if pass_only and filt not in ('PASS', '.'):
                invalid.add(pos)
                continue
            if reference[pos:pos + len(ref)] != ref:
                invalid.add(pos)
                continue
            if alts == '.':
                continue  # explicitly invariant record
            alt_list = alts.split(',')
            if snv_only and not (len(ref) == 1 and all(a in _SNV_ALLELES for a in alt_list)):
                blocked[pos:min(pos + len(ref), n)] = True
                continue
            try:
                ac, an = _ac_an(info.rstrip('\n'))
                if len(ac) != len(alt_list):
                    raise ValueError('AC/ALT length mismatch')
            except ValueError:
                invalid.add(pos)
                continue
            real = [c for a, c in zip(alt_list, ac) if a != '*']
            star = sum(c for a, c in zip(alt_list, ac) if a == '*')
            if indel_mode == 'original':
                end = pos + 1 if len(alts) == len(ref) else pos + len(ref) + 1
            else:
                end = pos + len(ref)
            end = min(end, n)
            rec = sites.get(pos)
            if rec is None:
                sites[pos] = [an, real, star, end, False]
            else:
                # Split records of one site (bcftools norm -m-) share AN. A
                # different AN means the records come from different call
                # sets or sample subsets, so their counts cannot be combined.
                rec[4] |= rec[0] != an
                rec[1] = rec[1] + real
                rec[2] += star
                rec[3] = max(rec[3], end)

    # A valid record wins over a malformed one at the same position.
    for pos in invalid - sites.keys():
        labels[pos] = -1
    for pos, (an, real, star, end, conflict) in sites.items():
        if conflict:
            labels[pos] = -1
            continue
        try:
            if scheme == 'spectrum':
                category = site_spectrum_class(an, real, star, threshold, count_bins)
            else:
                category = site_category(an, real, star, threshold)
        except ValueError:
            labels[pos] = -1
            continue
        span = (pos,) if indel_mode == 'anchor' or snv_only else range(pos, end)
        for p in span:
            if labels[p] >= 0:
                labels[p] = max(labels[p], category)
    if snv_only:
        labels[blocked] = -1
    return labels


def bed_to_mask(bed_path, chromosome, length):
    """Boolean array of `length`, True inside zero-based half-open BED intervals."""
    mask = np.zeros(length, dtype=bool)
    n_intervals = 0
    opener = gzip.open if str(bed_path).endswith('.gz') else open
    with opener(bed_path, 'rt') as fh:
        for line in fh:
            if not line.strip() or line.startswith(('#', 'track', 'browser')):
                continue
            chrom, start, end, *_ = line.split()
            if chrom == chromosome:
                a, b = int(start), int(end)
                mask[max(0, a):min(length, b)] = True
                n_intervals += 1
    if n_intervals == 0:
        raise ValueError(f'No intervals for {chromosome!r} in {bed_path}; '
                         'check that contig names match the FASTA')
    return mask


def apply_callable_bed(labels, bed_path, chromosome):
    """Restrict to zero-based, half-open callable BED intervals."""
    callable_mask = bed_to_mask(bed_path, chromosome, len(labels))
    result = labels.copy()
    result[~callable_mask] = -1
    return result


def save_npy_atomic(path, arr):
    """np.save under a temporary name, then rename: never a truncated file."""
    tmp = str(path) + '.partial'
    with open(tmp, 'wb') as f:
        np.save(f, arr)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fasta', required=True)
    p.add_argument('--vcf', required=True,
                   help='Tip: strip genotypes first (bcftools view -G) for speed')
    p.add_argument('--chromosome', required=True, help='Must match FASTA, VCF and BED contig naming')
    p.add_argument('--callable-bed', required=True, help='Callable sites in BED0 coordinates, same genome build')
    p.add_argument('--threshold', type=float, default=0.01, help='Common if minor allele frequency >= threshold')
    p.add_argument('--indel-mode', choices=INDEL_MODES, default='anchor',
                   help="'original' reproduces the original repository's labeled span")
    p.add_argument('--keep-non-pass', action='store_true',
                   help='Use records regardless of FILTER (the original repo did this)')
    p.add_argument('--scheme', choices=SCHEMES, default='v1',
                   help="'spectrum' writes <prefix>_spectrum_labels.npy with the finer classes "
                        '(singleton / ultra-rare / rare / common; see spectrum.py)')
    p.add_argument('--count-bins', nargs='+', type=int, default=list(DEFAULT_COUNT_BINS),
                   help='spectrum scheme: upper minor-allele-count bounds of the count classes')
    p.add_argument('--snv-only', action='store_true',
                   help='label SNVs only; bases under indel/MNP/symbolic records become -1')
    p.add_argument('--labels-only', action='store_true',
                   help='do not (re)write <prefix>_dna.npy')
    p.add_argument('--output-prefix', required=True)
    args = p.parse_args()
    if sorted(set(args.count_bins)) != args.count_bins or args.count_bins[0] < 1:
        p.error('--count-bins must be increasing positive integers')
    dna = read_fasta(args.fasta, args.chromosome)
    y = vcf_categories(args.vcf, args.chromosome, dna, args.threshold,
                       args.indel_mode, pass_only=not args.keep_non_pass, scheme=args.scheme,
                       count_bins=tuple(args.count_bins), snv_only=args.snv_only)
    y = apply_callable_bed(y, args.callable_bed, args.chromosome)
    out = Path(args.output_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.labels_only:
        save_npy_atomic(str(out) + '_dna.npy', encode_dna(dna))
    else:
        # the labels are only meaningful next to the sequence they were made from
        existing = np.load(str(out) + '_dna.npy', mmap_mode='r')
        step = 5_000_000
        same = existing.shape == (len(dna), 4) and all(
            np.array_equal(existing[s:s + step], encode_dna(dna[s:s + step])) for s in range(0, len(dna), step))
        if not same:
            p.error(f'{out}_dna.npy does not encode {args.chromosome} of {args.fasta}; '
                    'delete it (or run without --labels-only) so it is rebuilt')
    suffix = '_spectrum_labels.npy' if args.scheme == 'spectrum' else '_frequency_labels.npy'
    save_npy_atomic(str(out) + suffix, y)  # written last: marks completion
    values, counts = np.unique(y, return_counts=True)
    print(f'{args.chromosome}: ' + ', '.join(f'{int(v)}={int(c)}' for v, c in zip(values, counts)))


if __name__ == '__main__':
    main()
