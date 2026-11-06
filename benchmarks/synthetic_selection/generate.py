"""Generate the synthetic selection benchmark genome.

Five chromosomes. Neutral process: CpG sites are 8x more mutable than other
sites (P(variant) 0.16 vs 0.02) and their variants are more often common
(P(common | variant) 0.6 vs 0.2). 20% of each chromosome lies in GC-rich 2 kb
"functional" blocks under purifying selection: each variant survives with
probability 0.5 and P(common | variant) is multiplied by 0.3.

Writes g.fa, v.vcf (sites-only, AC/AN), callable.bed, functional.bed and
oracle.npz (per-chromosome true neutral class probabilities).
"""
import argparse
from pathlib import Path
import numpy as np

CHROMS = ['chrA', 'chrB', 'chrC', 'chrD', 'chrE']


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--length', type=int, default=400_000)
    p.add_argument('--seed', type=int, default=11)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n = args.length
    oracle = {}
    with open(out / 'g.fa', 'w') as fa, open(out / 'v.vcf', 'w') as vcf, \
            open(out / 'callable.bed', 'w') as bed, open(out / 'functional.bed', 'w') as ann:
        vcf.write('##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')
        for c in CHROMS:
            func = np.zeros(n, bool)
            n_blocks = n // 10_000  # 20% of sequence in 2 kb blocks
            for s in rng.choice(np.arange(5000, n - 5000, 2000), n_blocks, replace=False):
                func[s:s + 2000] = True
            is_gc = rng.random(n) < np.where(func, 0.7, 0.4)
            seq = np.where(is_gc, np.where(rng.random(n) < .5, 'C', 'G'),
                           np.where(rng.random(n) < .5, 'A', 'T'))
            s = ''.join(seq)
            fa.write(f'>{c}\n' + '\n'.join(s[i:i + 60] for i in range(0, n, 60)) + '\n')
            cpg = np.zeros(n, bool)
            m = (seq[:-1] == 'C') & (seq[1:] == 'G')
            cpg[:-1] |= m
            cpg[1:] |= m
            pvar = np.where(cpg, 0.16, 0.02)
            pcom = np.where(cpg, 0.6, 0.2)
            oracle[c] = np.stack([1 - pvar, pvar * (1 - pcom), pvar * pcom], 1).astype(np.float32)
            keep = np.where(func, 0.5, 1.0)
            pcom_sel = np.where(func, pcom * 0.3, pcom)
            for i in np.flatnonzero((rng.random(n) < pvar) & (rng.random(n) < keep)):
                ref = seq[i]
                alt = 'ACGT'[('ACGT'.index(ref) + 1) % 4]
                ac = rng.integers(64, 3000) if rng.random() < pcom_sel[i] else rng.integers(1, 60)
                vcf.write(f'{c}\t{i + 1}\t.\t{ref}\t{alt}\t.\tPASS\tAC={ac};AN=6404\n')
            bed.write(f'{c}\t0\t{n}\n')
            d = np.diff(np.r_[0, func.astype(int), 0])
            for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                ann.write(f'{c}\t{a}\t{b}\n')
    np.savez_compressed(out / 'oracle.npz', **oracle)
    print(f'Wrote synthetic genome to {out}')


if __name__ == '__main__':
    main()
