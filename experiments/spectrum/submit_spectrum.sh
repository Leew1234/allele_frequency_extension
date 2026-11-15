#!/usr/bin/env bash
# Submit the frequency-spectrum analysis to SLURM (Great Lakes): the same four
# dependent stages as experiments/real_data/slurm/submit_plan.sh.
#
#   prep     (CPU)        sites, labels, masks, label validation, k-mer baselines
#   train    (GPU array)  one task per (half, configuration, seed)
#   predict  (GPU array)  one task per (half, configuration)
#   post     (CPU)        scores, de novo benchmark, criteria report
#
#   bash submit_spectrum.sh spectrum.env ../real_data/slurm/slurm.env
#   DRY_RUN=1 bash submit_spectrum.sh spectrum.env ../real_data/slurm/slurm.env
set -euo pipefail
[[ $# -eq 2 ]] || { sed -n '2,11p' "$0"; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_SCRIPT="$HERE/run_spectrum.sh"
export SUBMIT_CONFIGS="$(bash -c "source '$1'; echo \"\${SPECTRUM_CONFIGS:-neutralAB}\"")"
export PREP_STEPS="sites labels masks validate kmer"
export POST_STEPS="score denovo evaluate"
exec bash "$HERE/../real_data/slurm/submit_plan.sh" "$@"
