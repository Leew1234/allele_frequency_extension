#!/usr/bin/env bash
# Dry-run the whole frequency-spectrum pipeline on a small synthetic fixture
# (about 15 minutes on a CPU; needs TensorFlow and bcftools). Checks that every
# step of fetch_fullcallset.sh and run_spectrum.sh works and that
# criteria_report.md is produced; the numbers only reflect the toy biology in
# make_fixture.py (where the right answer is known, see its docstring).
#   bash experiments/spectrum/dryrun/dryrun.sh [work_dir]
#
# What to expect in WORK/results/criteria_report.md on the fixture:
#   F1, F2a-e  PASS   the tiers come out in the built-in order CDS > conserved
#                     non-coding > enhancer > neutral, and observed/expected
#                     variants near 0.5 / 0.7 / 0.9 / 1.0
#   T2, T1v    PASS   neutral singleton share and variant count near 1
#   T1, F4     may fail by a few percent: the fixture is tiny and the model is
#                     trained for minutes
#   F3         MISSING (4 bins here, 10 in the criteria); the bins scatter by
#                     about 10% for the same reason
#   F5         FAIL   in the fixture the neutral spectrum depends only on CpG
#                     status, which a trinucleotide table describes exactly, so
#                     the network cannot beat it here
#   R2         MISSING (2 seeds here, 5 in the criteria)
# Also: validation/*.json agreement 1.0 in every class; denovo.json AUC near
# 0.70 for the network and the k-mer tables, and well below 0.5 for
# *_singleton_share (mutable sites are expected to have fewer singletons).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D="${1:-$HERE/work}"
mkdir -p "$D"
"${PYTHON:-python3}" "$HERE/make_fixture.py" --out "$D/fixture"
F="$D/fixture"
CONTIGS="c1 c2 c3 c4 c5 c6 c7 c8"
# shellcheck disable=SC2086
KGP_FULL_LOCAL="$F/vcf/{contig}.vcf" bash "$HERE/../fetch_fullcallset.sh" "$D/data" sites $CONTIGS
KGP_FULL_LOCAL="$F/vcf/{contig}.vcf" bash "$HERE/../fetch_fullcallset.sh" "$D/data" tables c1 c2
cat > "$D/spectrum.env" <<EOT
WORK=$D/run
V1_WORK=
FASTA=$F/ref.fa
GFF=$F/ref.gff
CHR_MAP=$F/chr_map.tsv
CALLABLE_BED=$F/callable.bed
PHASTCONS_BED=$F/phastcons.bed
FULL_SITES_TEMPLATE=$D/data/sites_full/{contig}.sites.vcf.gz
FULL_TABLE_TEMPLATE=$D/data/sites_full/{contig}.bcftools.tsv
VALIDATE_SEQIDS="S1 S2"
MAF_THRESHOLD=0.05           # 300 chromosomes: common = 15 copies or more
COUNT_BINS="1 5"
SNV_ONLY=1
KEEP_NON_PASS=0
HALF1_ORIG_DIR=$F/orig/half1
HALF1_ORIG_RAW_DIR=$F/orig_raw/half1
HALF2_ORIG_DIR=$F/orig/half2
HALF2_ORIG_RAW_DIR=$F/orig_raw/half2
DNM_SETS="iceland_2017=$F/dnm.tsv"
DNM_NEG_SITES=200000
HALF1_TRAIN="S3 S5"
HALF1_VAL="S7"
HALF1_CAL="S1"
HALF1_TEST="S4 S6 S8"
HALF1_SEED_TEST="S4"
HALF2_TRAIN="S4 S6"
HALF2_VAL="S8"
HALF2_CAL="S2"
HALF2_TEST="S3 S5 S7"
HALF2_SEED_TEST="S3"
SPECTRUM_CONFIGS="neutralAB"
KMER_K="3 5"
MUT_BINS=4
SEEDS="0 1"
TRAIN_ARGS="--window 256 --steps-per-epoch 150 --epochs 4 --max-val-windows 200 --verbose 0"
EOT
bash "$HERE/../run_spectrum.sh" "$D/spectrum.env"
