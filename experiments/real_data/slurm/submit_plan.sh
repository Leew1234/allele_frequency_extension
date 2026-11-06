#!/usr/bin/env bash
# Submit the analysis plan to SLURM as four dependent stages:
#
#   prep     (CPU)        labels, masks, label validation
#   train    (GPU array)  one task per (half, configuration, seed)
#   predict  (GPU array)  one task per (half, configuration); 'neutral' also
#                         runs the single-seed predictions
#   post     (CPU)        per-category scores, discrimination, de novo, criteria
#
# Each stage starts only if the previous one finished without error
# (afterok). Every task calls run_plan.sh, which skips outputs that are
# already fresh, so resubmitting after a failure redoes only what is missing.
#
#   bash submit_plan.sh plan.env slurm.env            # submit
#   DRY_RUN=1 bash submit_plan.sh plan.env slurm.env  # write job scripts, print, submit nothing
#   bash submit_plan.sh pilot.env slurm.env           # the pilot (half 1, 2 seeds)
set -euo pipefail

[[ $# -eq 2 ]] || { sed -n '2,17p' "$0"; exit 1; }
PLAN_ENV="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
SLURM_ENV="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PLAN="$(cd "$HERE/.." && pwd)/run_plan.sh"
# shellcheck disable=SC1090
source "$SLURM_ENV"
# shellcheck disable=SC1090
( source "$PLAN_ENV"; : "${WORK:?WORK is not set in $PLAN_ENV}" )
WORK=$(bash -c "source '$PLAN_ENV'; echo \"\$WORK\"")
HALVES=$(bash -c "source '$PLAN_ENV'; echo \"\${HALVES:-half1 half2}\"")
SEEDS=$(bash -c "source '$PLAN_ENV'; echo \"\${SEEDS:-0 1 2 3 4}\"")
CONFIGS="base neutral neutralB"
JOBDIR="$WORK/slurm/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$JOBDIR" "$WORK/logs"

# Task lists: line N = array task N
: > "$JOBDIR/train_tasks.txt"; : > "$JOBDIR/predict_tasks.txt"
for h in $HALVES; do for c in $CONFIGS; do
  for s in $SEEDS; do echo "$h $c $s" >> "$JOBDIR/train_tasks.txt"; done
  echo "$h $c" >> "$JOBDIR/predict_tasks.txt"
done; done
N_TRAIN=$(wc -l < "$JOBDIR/train_tasks.txt"); N_PREDICT=$(wc -l < "$JOBDIR/predict_tasks.txt")

header() {  # name partition time mem cpus gpu(0/1) [array]
  echo "#!/usr/bin/env bash"
  echo "#SBATCH --job-name=af_$1"
  echo "#SBATCH --account=$ACCOUNT"
  echo "#SBATCH --partition=$2"
  echo "#SBATCH --time=$3"
  echo "#SBATCH --mem=$4"
  echo "#SBATCH --cpus-per-task=$5"
  [[ $6 == 1 ]] && echo "#SBATCH --gpus=${GPUS_PER_JOB:-1}"
  [[ -n "${7:-}" ]] && echo "#SBATCH --array=$7" && echo "#SBATCH --output=$WORK/logs/slurm-%x-%A_%a.out" \
                    || echo "#SBATCH --output=$WORK/logs/slurm-%x-%j.out"
  [[ -n "${MAIL_USER:-}" ]] && echo "#SBATCH --mail-user=$MAIL_USER" && echo "#SBATCH --mail-type=END,FAIL"
  echo "set -euo pipefail"
  echo "$SETUP_CMD"
}

{ header prep "$PARTITION_CPU" "$PREP_TIME" "$PREP_MEM" "$PREP_CPUS" 0
  echo "bash '$RUN_PLAN' '$PLAN_ENV' labels masks validate"; } > "$JOBDIR/prep.sh"
{ header train "$PARTITION_GPU" "$TRAIN_TIME" "$TRAIN_MEM" "$TRAIN_CPUS" 1 "1-$N_TRAIN"
  echo "read -r H C S < <(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" '$JOBDIR/train_tasks.txt')"
  echo "ONLY_HALVES=\$H ONLY_CONFIGS=\$C ONLY_SEEDS=\$S bash '$RUN_PLAN' '$PLAN_ENV' train"; } > "$JOBDIR/train.sh"
{ header predict "$PARTITION_GPU" "$PREDICT_TIME" "$PREDICT_MEM" "$PREDICT_CPUS" 1 "1-$N_PREDICT"
  echo "read -r H C < <(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" '$JOBDIR/predict_tasks.txt')"
  echo "ONLY_HALVES=\$H ONLY_CONFIGS=\$C bash '$RUN_PLAN' '$PLAN_ENV' predict"; } > "$JOBDIR/predict.sh"
{ header post "$PARTITION_CPU" "$POST_TIME" "$POST_MEM" "$POST_CPUS" 0
  echo "bash '$RUN_PLAN' '$PLAN_ENV' score denovo evaluate"; } > "$JOBDIR/post.sh"

submit() {  # [dependency] script -> job id
  local dep=$1 script=$2
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "sbatch --parsable ${dep:+--dependency=afterok:$dep} $script" >&2
    echo "DRY$RANDOM"
  else
    sbatch --parsable ${dep:+--dependency=afterok:$dep} "$script" | cut -d';' -f1
  fi
}
J1=$(submit "" "$JOBDIR/prep.sh")
J2=$(submit "$J1" "$JOBDIR/train.sh")
J3=$(submit "$J2" "$JOBDIR/predict.sh")
J4=$(submit "$J3" "$JOBDIR/post.sh")
cat <<EOF
Job scripts: $JOBDIR
  prep     $J1
  train    $J2  (array of $N_TRAIN: $(echo $HALVES | wc -w) halves x 3 configurations x $(echo $SEEDS | wc -w) seeds)
  predict  $J3  (array of $N_PREDICT)
  post     $J4
Report when done: $WORK/results/criteria_report.md
Watch: squeue -u \$USER ; logs in $WORK/logs/
If a stage fails, fix the cause and run this script again: finished outputs are kept.
EOF
