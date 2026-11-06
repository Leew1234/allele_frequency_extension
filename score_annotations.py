"""Observed vs expected polymorphism per GFF feature type (the paper's
Figure 5 categories), for a reference model and a new model, on one
chromosome, then pooled across chromosomes.

Per chromosome:
  python score_annotations.py --gff GRCh38_latest_genomic.gff --seqid NC_000002.12 \\
      --labels arrays/NC_000002.12_frequency_labels.npy \\
      --original calibrated_predictions/NC_000002.12.pred.npy --original-raw raw/NC_000002.12.pred.npy \\
      --new pred/neutral/NC_000002.12.npy --save-blocks blocks/orig_vs_neutral/NC_000002.12.npz

Pool test chromosomes into one table:
  python score_annotations.py --pool blocks/orig_vs_neutral/*.npz \\
      --json results/orig_vs_neutral.json --tsv results/orig_vs_neutral.tsv

Categories:
  * every GFF feature type (column 3) whose source (column 2) is not in
    --exclude-source (default RefSeq, as in the original repo), minus the
    container types the original analysis skipped (region, chromosome,
    match, ...), each as the union of its intervals;
  * `unannotated` = sites outside every non-excluded-source feature of any
    type (the original repo's masklabel definition);
  * any --mask NAME=PATH you add (boolean .npy).
--original may be the original 1-D prediction or another (n, 3) class
probability file (e.g. the whole-genome configuration as reference for the
neutral one). Omit it to score the new model alone.
"""
import argparse
import os
import gzip
import json
from pathlib import Path
import numpy as np

from compare_ovep import (IntervalMask, block_sums, summarize, pool, save_blocks,
                          load_blocks)

SKIP_TYPES = {'biological_region', 'region', 'chromosome', 'conserved_region',
              'sequence_comparison', 'sequence_feature', 'sequence_alteration',
              'match', 'cDNA_match', 'scaffold'}
UNANNOTATED = 'unannotated'


def gff_intervals(gff_path, seqid, exclude_sources=('RefSeq',)):
    """{type: [(start0, end)]} for one seqid, plus all intervals (any type)."""
    by_type, every = {}, []
    exclude = set(exclude_sources)
    opener = gzip.open if str(gff_path).endswith('.gz') else open
    prefix = seqid + '\t'
    with opener(gff_path, 'rt') as fh:
        for line in fh:
            if not line.startswith(prefix):
                continue
            f = line.split('\t', 8)
            if len(f) < 5 or f[1] in exclude:
                continue
            try:
                a, b = int(f[3]) - 1, int(f[4])  # GFF 1-based inclusive -> half-open
            except ValueError:
                continue
            if b <= a:
                continue
            every.append((a, b))
            if f[2] not in SKIP_TYPES:
                by_type.setdefault(f[2], []).append((a, b))
    return by_type, every


def build_masks(gff_path, seqid, length, types=None, exclude_sources=('RefSeq',)):
    by_type, every = gff_intervals(gff_path, seqid, exclude_sources)
    if not every:
        raise ValueError(f'no GFF features for {seqid!r}; check the seqid matches the GFF')
    masks = {t: IntervalMask(iv, length) for t, iv in sorted(by_type.items())
             if types is None or t in types}
    masks[UNANNOTATED] = IntervalMask(every, length).complement()
    return masks


def table_rows(results):
    cols = ['category', 'n', 'observed_variants', 'oe_original', 'oe_new', 'oe_diff',
            'oe_diff_ci_lo', 'oe_diff_ci_hi', 'oe_common_share', 'share_ci_lo', 'share_ci_hi',
            'auc_original', 'auc_new']
    rows = []
    for k, r in sorted(results.items(), key=lambda kv: (kv[0] != UNANNOTATED, kv[0])):
        ci = r.get('ci95', {})
        get = lambda key, i: ci.get(key, [None, None])[i]
        rows.append([k, r.get('n'), r.get('observed_variants'), r.get('oe_original'), r.get('oe_new'),
                     r.get('oe_diff'), get('oe_diff', 0), get('oe_diff', 1), r.get('oe_common_share'),
                     get('oe_common_share', 0), get('oe_common_share', 1),
                     r.get('auc_original'), r.get('auc_new')])
    return cols, rows


def write_outputs(results, meta, json_path=None, tsv_path=None):
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path + '.partial', 'w') as f:
            json.dump({'meta': meta, 'categories': results}, f, indent=2)
        os.replace(json_path + '.partial', json_path)
    if tsv_path:
        cols, rows = table_rows(results)
        with open(tsv_path, 'w') as f:
            f.write('\t'.join(cols) + '\n')
            for r in rows:
                f.write('\t'.join('' if v is None else (f'{v:.4g}' if isinstance(v, float) else str(v))
                                  for v in r) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pool', nargs='+', help='pool saved per-chromosome .npz block files')
    p.add_argument('--gff')
    p.add_argument('--seqid', help='GFF column 1 name of this chromosome')
    p.add_argument('--labels')
    p.add_argument('--new', help='class probabilities .npy from predict.py')
    p.add_argument('--original', help='reference: original 1-D .npy or (n,3) .npy')
    p.add_argument('--original-raw')
    p.add_argument('--types', nargs='+', help='restrict to these GFF types (default: all)')
    p.add_argument('--exclude-source', nargs='*', default=['RefSeq'])
    p.add_argument('--mask', action='append', default=[], metavar='NAME=PATH')
    p.add_argument('--block-size', type=int, default=100_000)
    p.add_argument('--auc-sites', type=int, default=1_000_000)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save-blocks', help='per-chromosome block sums (.npz) for --pool')
    p.add_argument('--unit-sites', type=int,
                   help='also save sums over runs of this many sites per category (for discriminate.py)')
    p.add_argument('--json')
    p.add_argument('--tsv')
    args = p.parse_args()

    if args.pool:
        loaded = [load_blocks(f) for f in args.pool]
        names = sorted(set().union(*(set(s) for s, _ in loaded)))
        results = {k: summarize(pool([s.get(k) for s, _ in loaded]), args.bootstrap, args.seed)
                   for k in names}
        meta = {'pooled_from': [m for _, m in loaded]}
    else:
        for req in ('gff', 'seqid', 'labels', 'new'):
            if getattr(args, req) is None:
                p.error(f'--{req} is required unless --pool is given')
        labels = np.load(args.labels, mmap_mode='r')
        masks = build_masks(args.gff, args.seqid, len(labels), set(args.types) if args.types else None,
                            args.exclude_source)
        masks['all_scorable'] = IntervalMask([(0, len(labels))], len(labels))
        for item in args.mask:
            name, _, path = item.partition('=')
            if not path:
                p.error(f'--mask expects NAME=PATH, got {item!r}')
            masks[name] = np.load(path, mmap_mode='r')
        sums = block_sums(labels, np.load(args.original, mmap_mode='r') if args.original else None,
                          np.load(args.new, mmap_mode='r'), masks,
                          np.load(args.original_raw, mmap_mode='r') if args.original_raw else None,
                          args.block_size, args.auc_sites, args.seed, unit_sites=args.unit_sites)
        meta = {'seqid': args.seqid, 'labels': args.labels, 'new': args.new, 'original': args.original}
        if args.save_blocks:
            Path(args.save_blocks).parent.mkdir(parents=True, exist_ok=True)
            save_blocks(args.save_blocks, sums, meta)
        results = {k: summarize(v, args.bootstrap, args.seed) for k, v in sums.items()}
    write_outputs(results, meta, args.json, args.tsv)
    if not args.json and not args.tsv:
        print(json.dumps(results, indent=2))
    else:
        print(f"{len(results)} categories -> {', '.join(x for x in (args.json, args.tsv) if x)}")


if __name__ == '__main__':
    main()
