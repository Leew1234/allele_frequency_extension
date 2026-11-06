#!/usr/bin/env bash
# Reproduce the synthetic selection benchmark (about 8 minutes on a laptop CPU).
#   bash benchmarks/synthetic_selection/run.sh [work_dir]
# Compare the final table with expected_output.txt. Small differences (about
# +/-0.03) are expected across TensorFlow versions and hardware.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/../.." && pwd)"
W="${1:-$HERE/work}"
PY="${PYTHON:-python3}"
mkdir -p "$W/arr" "$W/masks" "$W/models" "$W/pred"
export TF_CPP_MIN_LOG_LEVEL=2

"$PY" "$HERE/generate.py" --out "$W"

echo "== labels and masks"
python_manifest="{"
for c in chrA chrB chrC chrD chrE; do
  "$PY" "$CODE/data.py" --fasta "$W/g.fa" --vcf "$W/v.vcf" --callable-bed "$W/callable.bed" \
      --chromosome $c --output-prefix "$W/arr/$c"
  "$PY" "$CODE/make_mask.py" --bed "$W/functional.bed" --chromosome $c \
      --labels "$W/arr/${c}_frequency_labels.npy" --out "$W/masks/${c}_functional.npy" > /dev/null
  "$PY" "$CODE/make_mask.py" --bed "$W/functional.bed" --chromosome $c --invert \
      --labels "$W/arr/${c}_frequency_labels.npy" --out "$W/masks/${c}_neutral.npy" > /dev/null
  python_manifest+="\"$c\": {\"dna\": \"$W/arr/${c}_dna.npy\", \"labels\": \"$W/arr/${c}_frequency_labels.npy\", \"mask\": \"$W/masks/${c}_neutral.npy\"},"
done
echo "${python_manifest%,}}" > "$W/manifest.json"

# Split: train chrA chrB, validate chrC, calibrate chrD, test chrE
train() {  # name, extra args...
  local name=$1; shift
  echo "== train $name"
  "$PY" "$CODE/train.py" --manifest "$W/manifest.json" --train-chromosomes chrA chrB \
      --val-chromosomes chrC --head factorized --window 256 --steps-per-epoch 150 --epochs 6 \
      --max-val-windows 400 --verbose 0 --out "$W/models/$name.keras" "$@"
}
train none_s0 --mask-scope none --seed 0
train common_s0 --mask-scope common --seed 0
train common_s1 --mask-scope common --seed 1
train all_s0 --mask-scope all --seed 0

declare -A MODELS=(
  [none]="$W/models/none_s0.keras"
  [common]="$W/models/common_s0.keras $W/models/common_s1.keras"
  [all]="$W/models/all_s0.keras")
for scope in none common all; do
  echo "== predict/calibrate $scope"
  # shellcheck disable=SC2086
  "$PY" "$CODE/predict.py" --model ${MODELS[$scope]} --dna "$W/arr/chrD_dna.npy" --output "$W/pred/${scope}_chrD.npy"
  "$PY" "$CODE/calibrate.py" --labels "$W/arr/chrD_frequency_labels.npy" --predictions "$W/pred/${scope}_chrD.npy" \
      --out "$W/pred/${scope}_cal_whole.json"
  "$PY" "$CODE/calibrate.py" --labels "$W/arr/chrD_frequency_labels.npy" --predictions "$W/pred/${scope}_chrD.npy" \
      --mask "$W/masks/chrD_neutral.npy" --out "$W/pred/${scope}_cal_neutral.json"
  for cal in whole neutral; do
    # shellcheck disable=SC2086
    "$PY" "$CODE/predict.py" --model ${MODELS[$scope]} --dna "$W/arr/chrE_dna.npy" \
        --output "$W/pred/${scope}_${cal}_chrE.npy" --calibration "$W/pred/${scope}_cal_${cal}.json"
  done
done

echo
"$PY" "$HERE/summarize.py" --work "$W" | tee "$W/results.txt"
