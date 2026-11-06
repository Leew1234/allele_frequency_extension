#!/usr/bin/env bash
# Dry-run the whole analysis plan on a small synthetic fixture (~15 min on a
# CPU; needs TensorFlow and bcftools). Checks that every step of run_plan.sh
# works and that criteria_report.md is produced; the numbers themselves only
# reflect the toy biology in make_fixture.py.
#   bash experiments/real_data/dryrun/dryrun.sh [work_dir]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D="${1:-$HERE/work}"
mkdir -p "$D"
"${PYTHON:-python3}" "$HERE/make_fixture.py" --out "$D/fixture"
F="$D/fixture"
cat > "$D/plan.env" <<EOF
WORK=$D/run
FASTA=$F/ref.fa
GFF=$F/ref.gff
CHR_MAP=$F/chr_map.tsv
VCF_TEMPLATE=$F/vcf/{contig}.vcf
CALLABLE_BED=$F/callable.bed
PHASTCONS_BED=$F/phastcons.bed
HALF1_ORIG_DIR=$F/orig/half1
HALF1_ORIG_RAW_DIR=$F/orig_raw/half1
HALF2_ORIG_DIR=$F/orig/half2
HALF2_ORIG_RAW_DIR=$F/orig_raw/half2
ORIG_LABEL_DIR=$F/orig_labels
DNM_SETS="iceland_2017=$F/dnm_denovo_db.tsv extra=$F/dnm_platinum.tsv"
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
VALIDATE_SEQIDS="S1 S2"
SEEDS="0 1"
DISC_BLOCK_BP=20000          # toy chromosomes are 300 kb; real ones use the 1 Mb default
TRAIN_ARGS="--window 256 --steps-per-epoch 60 --epochs 3 --max-val-windows 200 --verbose 0"
EOF
bash "$HERE/../run_plan.sh" "$D/plan.env"
