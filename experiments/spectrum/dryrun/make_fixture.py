"""Small synthetic stand-in for the inputs of run_spectrum.sh, so the whole
frequency-spectrum pipeline can be dry-run in about 15 minutes on a CPU.
Same file types and naming conventions as the human data:

  genotype VCFs named by contig (c1..c8) standing in for the full call set,
  FASTA/GFF by seqid (S1..S8), a contig->seqid map, a callable BED and a
  phastCons-like BED in contig names, a de novo mutation list, and fake
  "original model" predictions per half.

Biology (deliberately simple). 150 diploid samples. The minor allele count of
a variant is drawn from a spectrum proportional to count^-a:
  neutral sequence        a = 1.0
  CpG sites               8x more mutable, and a is 0.25 lower: recurrent
                          mutation pushes variants to higher frequency, so
                          mutable sites have FEWER singletons without any
                          selection (what a mutability adjustment must remove)
  enhancers               a + 0.15, 10% of variants removed
  conserved non-coding    a + 0.40, 30% of variants removed
  coding sequence (CDS)   a + 0.60, 50% of variants removed; GC-rich
Purifying selection therefore raises the singleton share in the order
CDS > conserved non-coding > enhancer > neutral. About 1% of records are
indels, 0.5% fail FILTER, and 0.5% have many missing genotypes, to exercise
the SNV-only, PASS-only and missingness handling. With 300 chromosomes the
dry run uses a 5% common threshold (15 copies) so that every class is filled.
"""
import argparse
from pathlib import Path
import numpy as np

N_SAMPLES = 150


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--length', type=int, default=300_000)
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    out = Path(a.out)
    for d in ('vcf', 'orig/half1', 'orig/half2', 'orig_raw/half1', 'orig_raw/half2'):
        (out / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    n = a.length
    seqids = [f'S{i}' for i in range(1, 9)]
    contig = {s: f'c{s[1:]}' for s in seqids}
    (out / 'chr_map.tsv').write_text(''.join(f'{contig[s]}\t{s}\n' for s in seqids))
    fa = open(out / 'ref.fa', 'w')
    gff = open(out / 'ref.gff', 'w')
    gff.write('##gff-version 3\n')
    callable_bed = open(out / 'callable.bed', 'w')
    phast = open(out / 'phastcons.bed', 'w')
    dnm = open(out / 'dnm.tsv', 'w')
    dnm.write('#chrom\tpos\n')
    hdr = ('##fileformat=VCFv4.2\n##FILTER=<ID=PASS,Description="All filters passed">\n'
           '##FILTER=<ID=LowQual,Description="Low quality">\n'
           '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
           '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'
           + '\t'.join(f's{i}' for i in range(N_SAMPLES)) + '\n')
    counts = np.arange(1, N_SAMPLES + 1)
    for s in seqids:
        c = contig[s]
        starts = rng.choice(np.arange(4000, n - 6000, 2000), 62, replace=False)
        cds_blocks, cons_blocks, enh_blocks = starts[:30], starts[30:42], starts[42:]
        cds = np.zeros(n, bool)
        cons = np.zeros(n, bool)
        enh = np.zeros(n, bool)
        for b in cds_blocks:
            cds[b:b + 2000] = True
        for b in cons_blocks:
            cons[b + 200:b + 1700] = True
        for b in enh_blocks:
            enh[b + 500:b + 1100] = True
        is_gc = rng.random(n) < np.where(cds, 0.65, 0.4)
        seq = np.where(is_gc, np.where(rng.random(n) < .5, 'C', 'G'), np.where(rng.random(n) < .5, 'A', 'T'))
        seq[10_000:10_200] = 'N'
        text = ''.join(seq)
        fa.write(f'>{s} synthetic\n' + '\n'.join(text[i:i + 60] for i in range(0, n, 60)) + '\n')
        gff.write(f'{s}\tRefSeq\tregion\t1\t{n}\t.\t+\t.\tID={s}\n')
        for k, b in enumerate(cds_blocks):
            gff.write(f'{s}\tBestRefSeq\texon\t{b - 99}\t{b + 2100}\t.\t+\t.\tID=exon-{s}-{k}\n')
            gff.write(f'{s}\tBestRefSeq\tCDS\t{b + 1}\t{b + 2000}\t.\t+\t0\tID=cds-{s}-{k}\n')
            phast.write(f'{c}\t{b + 200}\t{b + 1700}\n')
        for b in cons_blocks:    # conserved, but not annotated in the GFF
            phast.write(f'{c}\t{b + 200}\t{b + 1700}\n')
        for k, b in enumerate(enh_blocks):
            gff.write(f'{s}\tRefSeqFE\tenhancer\t{b + 501}\t{b + 1100}\t.\t.\t.\tID=enh-{s}-{k}\n')
        callable_bed.write(f'{c}\t0\t{n - 2000}\n')
        cpg = np.zeros(n, bool)
        m = (seq[:-1] == 'C') & (seq[1:] == 'G')
        cpg[:-1] |= m
        cpg[1:] |= m
        pvar = np.where(cpg, 0.16, 0.02) * (seq != 'N')
        keep = np.where(cds, 0.5, np.where(cons, 0.7, np.where(enh, 0.9, 1.0)))
        slope = (1.0 - 0.25 * cpg + np.where(cds, 0.6, np.where(cons, 0.4, np.where(enh, 0.15, 0.0))))
        with open(out / 'vcf' / f'{c}.vcf', 'w') as v:
            v.write(hdr.replace('#CHROM', f'##contig=<ID={c},length={n}>\n#CHROM'))
            for i in np.flatnonzero(rng.random(n) < np.clip(pvar * keep, 0, 1)):
                w = counts ** -slope[i]
                mac = int(rng.choice(counts, p=w / w.sum()))
                gt = np.zeros(2 * N_SAMPLES, dtype=int)
                gt[rng.choice(2 * N_SAMPLES, mac, replace=False)] = 1
                calls = [f'{x}/{y}' for x, y in gt.reshape(-1, 2)]
                u = rng.random()
                ref = seq[i]
                alt = 'ACGT'[('ACGT'.index(ref) + 1) % 4]
                filt = 'PASS'
                if u < 0.01 and i + 2 < n and 'N' not in text[i:i + 2]:
                    ref, alt = text[i:i + 2], text[i]          # 1-bp deletion
                elif u < 0.015:
                    filt = 'LowQual'
                elif u < 0.02:
                    for j in rng.choice(N_SAMPLES, N_SAMPLES // 8, replace=False):
                        calls[j] = './.'                        # 12.5% missing genotypes
                v.write(f'{c}\t{i + 1}\t.\t{ref}\t{alt}\t.\t{filt}\t.\tGT\t' + '\t'.join(calls) + '\n')
        w = pvar / pvar.sum()
        for pos in np.sort(rng.choice(n, 300, replace=False, p=w)):
            dnm.write(f'{c}\t{pos + 1}\n')
        # fake original model: a noisy neutral rate that partly absorbed selection
        orig = pvar * np.exp(rng.normal(0, 0.3, n)) * np.where(cds, 0.75, 1.0)
        orig = orig[:n - 2048].astype(np.float32)
        raw = orig.copy()
        raw[:512] = 0
        for h in ('half1', 'half2'):
            np.save(out / 'orig' / h / f'{s}.pred.npy', orig)
            np.save(out / 'orig_raw' / h / f'{s}.pred.npy', raw)
    print(f'fixture written to {out}')


if __name__ == '__main__':
    main()
