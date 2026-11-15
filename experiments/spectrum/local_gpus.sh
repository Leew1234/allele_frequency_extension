#!/usr/bin/env bash
# Run the GPU steps on a machine without a scheduler (e.g. a lab server with
# several GPUs): one worker per GPU, each working through its share of the
# tasks one after another.
#
#   bash local_gpus.sh spectrum.env train            # all (half, configuration, seed) trainings
#   bash local_gpus.sh spectrum.env predict          # all (half, configuration) predictions
#   GPUS="1 3" bash local_gpus.sh spectrum.env train # only these GPUs
#   RUN_SCRIPT=../real_data/run_plan.sh CONFIGS="base neutral neutralB" bash local_gpus.sh plan.env train
#
# The workers run in the background and survive logout (nohup). Each task calls
# the run script, which skips outputs that are already fresh, so running this
# again after a crash only redoes what is missing.
#
# A GPU that already has more than BUSY_MIB (default 1000) MiB in use is left
# alone, because someone else is probably using it; FORCE=1 uses it anyway.
# Watch:  tail -f WORK/logs/gpu*.log ;  nvidia-smi
# Stop:   kill $(cat WORK/logs/local_gpus.pids)
set -euo pipefail

[[ $# -eq 2 ]] || { sed -n '2,18p' "$0"; exit 1; }
ENV_FILE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
STEP=$2
case $STEP in train|predict) ;; *) echo "step must be train or predict" >&2; exit 1 ;; esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${RUN_SCRIPT:-$HERE/run_spectrum.sh}"
RUN_SCRIPT="$(cd "$(dirname "$RUN_SCRIPT")" && pwd)/$(basename "$RUN_SCRIPT")"
get() { bash -c "source '$ENV_FILE'; echo \"$1\""; }
WORK=$(get '$WORK'); HALVES=$(get '${HALVES:-half1 half2}'); SEEDS=$(get '${SEEDS:-0 1 2 3 4}')
CONFIGS="${CONFIGS:-$(get '${SPECTRUM_CONFIGS:-neutralAB}')}"
[[ -n "$WORK" ]] || { echo "WORK is not set in $ENV_FILE" >&2; exit 1; }
mkdir -p "$WORK/logs"

if [[ -z "${GPUS:-}" ]]; then
  command -v nvidia-smi > /dev/null || { echo "nvidia-smi not found; set GPUS=\"0 1 ...\"" >&2; exit 1; }
  GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', *' -v lim="${BUSY_MIB:-1000}" -v force="${FORCE:-0}" '$2 <= lim || force == 1 {print $1}' | tr '\n' ' ')
fi
read -r -a GPU_LIST <<< "$GPUS"
[[ ${#GPU_LIST[@]} -gt 0 ]] || { echo "no free GPU (all in use; FORCE=1 to override)" >&2; exit 1; }

TASKS=()
for h in $HALVES; do for c in $CONFIGS; do
  if [[ $STEP == train ]]; then for s in $SEEDS; do TASKS+=("$h $c $s"); done; else TASKS+=("$h $c -"); fi
done; done
echo "${#TASKS[@]} $STEP tasks on GPUs: ${GPU_LIST[*]}"

: > "$WORK/logs/local_gpus.pids"
for i in "${!GPU_LIST[@]}"; do
  g=${GPU_LIST[$i]}
  queue="$WORK/logs/gpu$g.$STEP.tasks"
  : > "$queue"
  for j in "${!TASKS[@]}"; do (( j % ${#GPU_LIST[@]} == i )) && echo "${TASKS[$j]}" >> "$queue"; done
  [[ -s "$queue" ]] || continue
  nohup bash -c '
    while read -r H C S; do
      echo "[$(date +%H:%M:%S)] gpu '"$g"': '"$STEP"' $H $C $S"
      if [[ $S == - ]]; then
        CUDA_VISIBLE_DEVICES='"$g"' ONLY_HALVES=$H ONLY_CONFIGS=$C bash "'"$RUN_SCRIPT"'" "'"$ENV_FILE"'" '"$STEP"' < /dev/null \
          || echo "FAILED: '"$STEP"' $H $C"
      else
        CUDA_VISIBLE_DEVICES='"$g"' ONLY_HALVES=$H ONLY_CONFIGS=$C ONLY_SEEDS=$S bash "'"$RUN_SCRIPT"'" "'"$ENV_FILE"'" '"$STEP"' < /dev/null \
          || echo "FAILED: '"$STEP"' $H $C $S"
      fi
    done < "'"$queue"'"
    echo "[$(date +%H:%M:%S)] gpu '"$g"': all tasks done"
  ' > "$WORK/logs/gpu$g.$STEP.log" 2>&1 &
  echo $! >> "$WORK/logs/local_gpus.pids"
  echo "  gpu $g: $(wc -l < "$queue") tasks, log $WORK/logs/gpu$g.$STEP.log"
done
echo "Running in the background. Finished when every log ends with 'all tasks done';"
echo "grep FAILED $WORK/logs/gpu*.$STEP.log shows tasks to look at (their own logs are in $WORK/logs/)."
