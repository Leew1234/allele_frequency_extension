#!/usr/bin/env python3
"""Convert a published de novo mutation table into the 'chrom<TAB>pos' file used by denovo_auc.py.

Reads .tsv/.csv/.txt (optionally .gz) or .xlsx. Chromosome and position columns
are found by name (chrom/chr/chromosome, pos/position/start, preferring names
that mention hg38/GRCh38) unless given with --chrom-col / --pos-col (a name or a
0-based index). If reference and alternate allele columns are found, only SNVs
are kept unless --keep-indels. Chromosome names are written as chr1..chr22,
chrX, chrY; duplicates are removed and the output is sorted.

Example (Jonsson et al. 2017, Supplementary Table 4):
  python prepare_dnm.py nature24018-s4.xlsx -o dnm/jonsson2017.grch38.tsv
Check the summary it prints: the number of DNMs should be close to 108,778
(fewer if indels are dropped), and the columns it picked should be hg38 ones.
"""
import argparse
import csv
import gzip
import io
import os
import re
import sys

CHROM_NAMES = ('chrom', 'chr', 'chromosome', 'contig', 'seqname', '#chrom', '#chr')
POS_NAMES = ('pos', 'position', 'start', 'pos_hg38', 'position_hg38', 'hg38_pos', 'grch38_pos', 'bp')
REF_NAMES = ('ref', 'reference', 'ref_allele', 'refallele')
ALT_NAMES = ('alt', 'alternate', 'alt_allele', 'altallele', 'var', 'variant_allele')
BUILD_HINT = re.compile(r'hg38|grch38|b38', re.I)
OTHER_BUILD = re.compile(r'hg19|grch37|b37|hg18', re.I)


def read_rows(path, sheet=None, sep=None):
    """Return a list of rows (lists of strings) from a text or Excel table."""
    if path.lower().endswith(('.xlsx', '.xlsm')):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb[sheet] if sheet else wb.worksheets[0]
        return [['' if v is None else str(v).strip() for v in row] for row in ws.iter_rows(values_only=True)]
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', newline='') as fh:
        text = fh.read()
    if sep is None:
        sample = '\n'.join(l for l in text.splitlines()[:50] if not l.startswith('##'))
        sep = '\t' if sample.count('\t') >= sample.count(',') else ','
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith('##')]
    return [[c.strip() for c in row] for row in csv.reader(io.StringIO('\n'.join(lines)), delimiter=sep)]


def _norm(name):
    return re.sub(r'[\s.\-]+', '_', name.strip().lower())


def find_header(rows, max_scan=30):
    """Index of the first row that names both a chromosome and a position column."""
    for i, row in enumerate(rows[:max_scan]):
        names = [_norm(c) for c in row]
        if any(n in CHROM_NAMES or n.startswith('chr') for n in names) and \
           any(n in POS_NAMES or 'pos' in n or n.startswith('start') for n in names):
            return i
    return None


def pick_column(header, wanted, given, kind):
    if given is not None:
        if given.lstrip('-').isdigit():
            return int(given)
        names = [_norm(c) for c in header]
        if _norm(given) not in names:
            sys.exit(f'{kind} column {given!r} not in header: {header}')
        return names.index(_norm(given))
    names = [_norm(c) for c in header]
    cands = [i for i, n in enumerate(names)
             if n in wanted or (kind == 'position' and ('pos' in n or n.startswith('start')))
             or (kind == 'chromosome' and n.startswith('chr') and 'pos' not in n)]
    if kind == 'position':  # prefer hg38 columns, avoid other builds
        cands = [i for i in cands if not OTHER_BUILD.search(header[i])] or cands
        cands.sort(key=lambda i: (not BUILD_HINT.search(header[i]), names[i] not in wanted))
    else:
        cands.sort(key=lambda i: names[i] not in wanted)
    return cands[0] if cands else None


def norm_chrom(c):
    c = c.strip()
    c = re.sub(r'^chr', '', c, flags=re.I)
    c = {'23': 'X', '24': 'Y', 'M': 'MT', 'm': 'MT'}.get(c, c).upper()
    if c in [str(i) for i in range(1, 23)] + ['X', 'Y']:
        return 'chr' + c
    return None  # mitochondria, contigs, junk


def convert(rows, chrom_col=None, pos_col=None, keep_indels=False, no_header=False, one_based=True):
    if no_header:
        header, body = [str(i) for i in range(len(rows[0]))], rows
        ci = int(chrom_col or 0)
        pi = int(pos_col or 1)
        ri = ai = None
    else:
        h = find_header(rows)
        if h is None and chrom_col is not None and pos_col is not None:
            h = 0  # columns given explicitly; take the first row as the header
        if h is None:
            sys.exit('could not find a header row naming chromosome and position; use --chrom-col/--pos-col '
                     '(with --no-header for a headerless file)')
        header, body = rows[h], rows[h + 1:]
        ci = pick_column(header, CHROM_NAMES, chrom_col, 'chromosome')
        pi = pick_column(header, POS_NAMES, pos_col, 'position')
        if ci is None or pi is None:
            sys.exit(f'could not identify chromosome/position columns in header: {header}')
        names = [_norm(c) for c in header]
        ri = next((i for i, n in enumerate(names) if n in REF_NAMES), None)
        ai = next((i for i, n in enumerate(names) if n in ALT_NAMES), None)
    stats = {'rows': 0, 'bad': 0, 'non_autosomal_xy': 0, 'indels': 0}
    sites = set()
    for row in body:
        if len(row) <= max(ci, pi) or not any(row):
            continue
        stats['rows'] += 1
        chrom = norm_chrom(row[ci])
        try:
            pos = int(float(row[pi].replace(',', '')))
        except ValueError:
            stats['bad'] += 1
            continue
        if chrom is None:
            stats['non_autosomal_xy'] += 1
            continue
        if not keep_indels and ri is not None and ai is not None and len(row) > max(ri, ai):
            if len(row[ri]) != 1 or len(row[ai]) != 1 or row[ri] == '-' or row[ai] == '-':
                stats['indels'] += 1
                continue
        sites.add((chrom, pos if one_based else pos + 1))
    order = {f'chr{i}': i for i in range(1, 23)} | {'chrX': 23, 'chrY': 24}
    out = sorted(sites, key=lambda s: (order[s[0]], s[1]))
    stats.update(kept=len(out), duplicates=stats['rows'] - stats['bad'] - stats['non_autosomal_xy']
                 - stats['indels'] - len(out),
                 chrom_column=header[ci], pos_column=header[pi],
                 allele_columns=None if ri is None or ai is None else [header[ri], header[ai]])
    return out, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('table')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--sheet', help='Excel sheet name (default: first sheet)')
    ap.add_argument('--sep', help='field separator for text input (default: guess tab or comma)')
    ap.add_argument('--chrom-col'); ap.add_argument('--pos-col')
    ap.add_argument('--no-header', action='store_true')
    ap.add_argument('--zero-based', action='store_true', help='positions in the table are 0-based (e.g. BED starts)')
    ap.add_argument('--keep-indels', action='store_true')
    a = ap.parse_args(argv)
    rows = read_rows(a.table, a.sheet, a.sep)
    if not rows:
        sys.exit('empty table')
    sites, stats = convert(rows, a.chrom_col, a.pos_col, a.keep_indels, a.no_header, not a.zero_based)
    if not sites:
        sys.exit(f'no sites kept: {stats}')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + '.partial'
    with open(tmp, 'w') as fh:
        fh.write('#chrom\tpos\n')
        fh.writelines(f'{c}\t{p}\n' for c, p in sites)
    os.replace(tmp, a.out)
    for k, v in stats.items():
        print(f'{k}: {v}', file=sys.stderr)
    return stats


if __name__ == '__main__':
    main()
