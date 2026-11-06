"""Small synthetic stand-in for the real inputs of run_plan.sh, so the whole
plan can be dry-run in ~15 minutes on a CPU. Same file types and naming
conventions as the human data:

  genotype VCFs named by contig (c1..c8), FASTA/GFF by seqid (S1..S8), a
  contig->seqid map, a callable BED and a phastCons-like BED in contig names,
  two de novo mutation lists, and fake "original model" predictions per half
  (1-D, 2048 shorter than the chromosome, first 512 bp unpredicted).

Biology (deliberately simple): CpG sites 8x more mutable; GC-rich CDS/exon
blocks under purifying selection (half of variants removed, common variants
x0.3); V_gene_segment blocks under diversifying selection (variants x1.5,
more common). 100 diploid samples, so "rare" (< 1%) means one copy.
"""
import argparse
import pickle
from pathlib import Path
import numpy as np

N_SAMPLES = 100


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--length', type=int, default=300_000)
    p.add_argument('--seed', type=int, default=5)
    a = p.parse_args()
    out = Path(a.out)
    for d in ('vcf', 'orig/half1', 'orig/half2', 'orig_raw/half1', 'orig_raw/half2', 'orig_labels'):
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
    dnm = {k: open(out / f'dnm_{k}.tsv', 'w') for k in ('denovo_db', 'platinum')}
    for f in dnm.values():
        f.write('#chrom\tpos\n')
    hdr = ('##fileformat=VCFv4.2\n##FILTER=<ID=PASS,Description="All filters passed">\n'
           '##FILTER=<ID=LowQual,Description="Low quality">\n'
           '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
           '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'
           + '\t'.join(f's{i}' for i in range(N_SAMPLES)) + '\n')
    for s in seqids:
        c = contig[s]
        starts = rng.choice(np.arange(4000, n - 6000, 2000), 34, replace=False)
        cds_blocks, v_blocks = starts[:30], starts[30:]
        func = np.zeros(n, bool)
        vseg = np.zeros(n, bool)
        for b in cds_blocks:
            func[b:b + 2000] = True
        for b in v_blocks:
            vseg[b:b + 2000] = True
        is_gc = rng.random(n) < np.where(func, 0.65, 0.4)
        seq = np.where(is_gc, np.where(rng.random(n) < .5, 'C', 'G'), np.where(rng.random(n) < .5, 'A', 'T'))
        seq[10_000:10_200] = 'N'
        text = ''.join(seq)
        fa.write(f'>{s} synthetic\n' + '\n'.join(text[i:i + 60] for i in range(0, n, 60)) + '\n')
        gff.write(f'{s}\tRefSeq\tregion\t1\t{n}\t.\t+\t.\tID={s}\n')
        for k, b in enumerate(cds_blocks):
            gff.write(f'{s}\tBestRefSeq\texon\t{b - 99}\t{b + 2100}\t.\t+\t.\tID=exon-{s}-{k}\n')
            gff.write(f'{s}\tBestRefSeq\tCDS\t{b + 1}\t{b + 2000}\t.\t+\t0\tID=cds-{s}-{k}\n')
            phast.write(f'{c}\t{b + 200}\t{b + 1700}\n')
        for k, b in enumerate(v_blocks):
            gff.write(f'{s}\tCurated Genomic\tV_gene_segment\t{b + 1}\t{b + 2000}\t.\t+\t.\tID=v-{s}-{k}\n')
        for k in range(20):
            e = int(rng.integers(20_000, n - 20_000))
            gff.write(f'{s}\tRefSeqFE\tenhancer\t{e + 1}\t{e + 600}\t.\t.\t.\tID=enh-{s}-{k}\n')
        callable_bed.write(f'{c}\t0\t{n - 2000}\n')
        cpg = np.zeros(n, bool)
        m = (seq[:-1] == 'C') & (seq[1:] == 'G')
        cpg[:-1] |= m
        cpg[1:] |= m
        pvar = np.where(cpg, 0.16, 0.02) * (seq != 'N')
        pcom = np.where(cpg, 0.6, 0.2)
        keep = np.where(func, 0.5, 1.0) * np.where(vseg, 1.5, 1.0)
        pcom_sel = np.clip(np.where(func, pcom * 0.3, np.where(vseg, pcom * 1.4, pcom)), 0, 1)
        orig_label = np.zeros(n)  # original repo's rule: every VCF record is a variant
        with open(out / 'vcf' / f'{c}.vcf', 'w') as v:
            v.write(hdr.replace('#CHROM', f'##contig=<ID={c},length={n}>\n#CHROM'))
            for i in np.flatnonzero(rng.random(n) < np.clip(pvar * keep, 0, 1)):
                ref = seq[i]
                alt = 'ACGT'[('ACGT'.index(ref) + 1) % 4]
                alleles = np.zeros((N_SAMPLES, 2), int)
                if rng.random() < pcom_sel[i]:
                    alleles = (rng.random((N_SAMPLES, 2)) < rng.uniform(0.03, 0.5)).astype(int)
                    if alleles.sum() < 2:
                        alleles[0] = 1
                else:
                    alleles[rng.integers(N_SAMPLES), 0] = 1
                gts = '\t'.join(f'{x}/{y}' for x, y in alleles)
                filt = 'LowQual' if rng.random() < 0.005 else 'PASS'
                v.write(f'{c}\t{i + 1}\t.\t{ref}\t{alt}\t.\t{filt}\t.\tGT\t{gts}\n')
                orig_label[i] = 1
        with open(out / 'orig_labels' / f'{s}_label.pkl', 'wb') as f:
            pickle.dump(orig_label, f)
        for k, f in dnm.items():
            w = pvar / pvar.sum()
            for pos in np.sort(rng.choice(n, 150, replace=False, p=w)):
                f.write(f'{c}\t{pos + 1}\n')
        # fake original model: noisy neutral rate that partly absorbed selection
        orig = pvar * np.exp(rng.normal(0, 0.3, n)) * np.where(func, 0.75, 1.0)
        orig = orig[:n - 2048].astype(np.float32)
        raw = orig.copy()
        raw[:512] = 0
        for h in ('half1', 'half2'):
            np.save(out / 'orig' / h / f'{s}.pred.npy', orig)
            np.save(out / 'orig_raw' / h / f'{s}.pred.npy', raw)
    print(f'fixture written to {out}')


if __name__ == '__main__':
    main()
