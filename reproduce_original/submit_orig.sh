#!/usr/bin/env bash
# Retrain the original OVEP model (both halves, 5 members each) on Great Lakes
# and write its predictions where run_plan.sh expects the original model's:
#
#   bash submit_orig.sh plan.env slurm.env            # submit
#   DRY_RUN=1 bash submit_orig.sh plan.env slurm.env  # print job scripts only
#
# Needs: run_plan.sh plan.env labels  (DNA arrays and sites VCFs for all
# chromosomes) to have finished first. Output, under $WORK/orig_repro:
#   models/half1/model0-4.keras, models/half2/...
#   half1/raw/<seqid>.pred.npy, half1/calibrated/<seqid>.pred.npy, half2/...
# Then set in plan.env (and pilot.env):
#   HALF1_ORIG_DIR=$WORK/orig_repro/half1/calibrated  HALF1_ORIG_RAW_DIR=$WORK/orig_repro/half1/raw
#   HALF2_ORIG_DIR=$WORK/orig_repro/half2/calibrated  HALF2_ORIG_RAW_DIR=$WORK/orig_repro/half2/raw
#
# Chromosome roles are the original repository's: half 1 trains on chr3-21
# (odd) and calibrates on chr1; half 2 trains on chr4-22 (even) and calibrates
# on chr2; each half predicts the other half's chromosomes.
set -euo pipefail
[[ $# -eq 2 ]] || { sed -n '2,20p' "$0"; exit 1; }
PLAN_ENV="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
SLURM_ENV="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SLURM_ENV"
WORK=$(bash -c "source '$PLAN_ENV'; echo \"\$WORK\"")
PY="${PYTHON:-python3}"
OUT="$WORK/orig_repro"
ODD="NC_000003.12 NC_000005.10 NC_000007.14 NC_000009.12 NC_000011.10 NC_000013.11 NC_000015.10 NC_000017.11 NC_000019.10 NC_000021.9"
EVEN="NC_000004.12 NC_000006.12 NC_000008.11 NC_000010.11 NC_000012.12 NC_000014.9 NC_000016.10 NC_000018.10 NC_000020.11 NC_000022.11"
ALL="NC_000001.11 NC_000002.12 $ODD $EVEN"

for s in $ALL; do
  for f in "$WORK/arrays/${s}_dna.npy" "$WORK/vcf/$s.sites.vcf.gz"; do
    [[ -s "$f" || -n "${DRY_RUN:-}" ]] || { echo "missing $f: run 'run_plan.sh plan.env labels' first" >&2; exit 1; }
  done
done

D="$OUT/slurm/$(date +%Y%m%d_%H%M%S)"; mkdir -p "$D" "$OUT/logs"
hdr() {  # name partition time mem cpus gpu [array]
  echo "#!/bin/bash"
  echo "#SBATCH --job-name=orig_$1 --account=$ACCOUNT --partition=$2 --time=$3 --mem=$4"
  echo "#SBATCH --cpus-per-task=$5 --output=$OUT/logs/slurm-orig_$1-%A_%a.out"
  [[ $6 == 1 ]] && echo "#SBATCH --gpus=1"
  [[ -n "${7:-}" ]] && echo "#SBATCH --array=$7"
  echo "set -euo pipefail"
  echo "$SETUP_CMD"
  echo "export TF_CPP_MIN_LOG_LEVEL=2"
}

{ hdr labels "${PARTITION_CPU:-standard}" 4:00:00 8G 1 0
  echo "$PY '$HERE/make_orig_labels.py' --work '$WORK' $ALL"; } > "$D/labels.sh"

{ hdr train "$PARTITION_GPU" "${ORIG_TRAIN_TIME:-8:00:00}" "${ORIG_TRAIN_MEM:-32G}" 4 1 0-9
  echo 'h=$(( SLURM_ARRAY_TASK_ID / 5 + 1 )); m=$(( SLURM_ARRAY_TASK_ID % 5 ))'
  echo "if [[ \$h == 1 ]]; then chroms='$ODD'; else chroms='$EVEN'; fi"
  echo "out='$OUT'/models/half\$h/model\$m.keras"
  echo '[[ -s "$out" ]] && { echo "exists: $out"; exit 0; }'
  echo "$PY '$HERE/train_orig.py' --work '$WORK' --member \$m --out \"\$out\" \$chroms"; } > "$D/train.sh"

{ hdr predict "$PARTITION_GPU" "${ORIG_PREDICT_TIME:-8:00:00}" "${ORIG_PREDICT_MEM:-32G}" 4 1 1-2
  echo 'h=$SLURM_ARRAY_TASK_ID'
  echo "if [[ \$h == 1 ]]; then cal=NC_000001.11; targets='$EVEN'; else cal=NC_000002.12; targets='$ODD'; fi"
  echo "$PY '$HERE/predict_orig.py' --work '$WORK' --models '$OUT'/models/half\$h --out '$OUT'/half\$h --calibration \$cal \$targets"; } > "$D/predict.sh"

if [[ -n "${DRY_RUN:-}" ]]; then
  for f in labels train predict; do echo "== $D/$f.sh"; cat "$D/$f.sh"; done; exit 0
fi
j1=$(sbatch --parsable "$D/labels.sh")
j2=$(sbatch --parsable --dependency=afterok:$j1 "$D/train.sh")
j3=$(sbatch --parsable --dependency=afterok:$j2 "$D/predict.sh")
echo "labels $j1, train $j2 (10 models), predict+calibrate $j3"
echo "Logs: $OUT/logs   Outputs: $OUT/half1, $OUT/half2"
