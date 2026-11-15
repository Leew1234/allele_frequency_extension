#!/usr/bin/env bash
# Run the frequency-spectrum analysis (plan v2) end to end.
#
#   bash run_spectrum.sh spectrum.env                 # every step
#   bash run_spectrum.sh spectrum.env train predict   # selected steps, in this order
#
# Steps: sites labels masks validate kmer train predict score denovo evaluate
#
#   sites     rename the full-call-set sites VCFs (fetch_fullcallset.sh) to the FASTA names
#   labels    one-hot DNA (reused from a finished v1 run if V1_WORK has it) and
#             spectrum labels: none / singleton / ultra-rare / rare / common
#   masks     neutral masks A (unannotated), B (outside phastCons) and AB (both), manifests
#   validate  labels against the bcftools tables, per class
#   kmer      k-mer context baselines (trinucleotide, ...): fit, calibrate, predict. CPU
#   train     spectrum-head models, one per (half, configuration, seed). GPU
#   predict   ensemble prediction, calibration on the held-out chromosome. GPU
#   score     singleton-share scores per category and tier, mutability bins,
#             paired comparison with each k-mer baseline
#   denovo    de novo mutation benchmark
#   evaluate  check the criteria file
#
# Same conventions as experiments/real_data/run_plan.sh: an output is rebuilt
# only if it is missing or older than an input, every output is written under a
# temporary name and renamed when complete, and settings that change outputs
# are recorded in WORK/settings.env (a run with different values stops).
#
# Slices for parallel jobs (local_gpus.sh and submit_spectrum.sh use these):
#   ONLY_HALVES="half1"  ONLY_CONFIGS="neutralAB"  ONLY_SEEDS="3"  (train, predict)
set -euo pipefail

[[ $# -ge 1 ]] || { sed -n '2,29p' "$0"; exit 1; }
# shellcheck disable=SC1090
source "$1"; shift
STEPS="${*:-sites labels masks validate kmer train predict score denovo evaluate}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python3}"
W="$WORK"
SEEDS="${SEEDS:-0 1 2 3 4}"
HALVES="${HALVES:-half1 half2}"
# A configuration is named after the neutral mask it is trained and calibrated
# on: neutralAB (unannotated AND outside conserved elements; the default),
# neutralA (unannotated, as v1's "neutral") or neutralB (outside conserved elements).
CONFIGS="${SPECTRUM_CONFIGS:-neutralAB}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
MAF_THRESHOLD="${MAF_THRESHOLD:-0.01}"
COUNT_BINS="${COUNT_BINS:-1 5}"               # singleton = 1, ultra-rare = 2..5
SNV_ONLY="${SNV_ONLY:-1}"                     # 1 = label SNVs only (indel/MNP bases ignored)
KEEP_NON_PASS="${KEEP_NON_PASS:-0}"           # 0 = records failing FILTER become ignored sites
INDEL_MODE="${INDEL_MODE:-anchor}"            # only used when SNV_ONLY=0
PHASTCONS_PAD="${PHASTCONS_PAD:-100}"
KMER_K="${KMER_K:-3 7}"                       # k-mer baselines; the first is the pre-specified one
MUT_BINS="${MUT_BINS:-10}"                    # mutability bins for the flatness check
SEED_PREDICTIONS="${SEED_PREDICTIONS:-1}"     # 1 = also predict and score each seed alone
DNM_NEG_SITES="${DNM_NEG_SITES:-5000000}"
CRITERIA="${CRITERIA:-$HERE/criteria_v2_draft.json}"
N_FREQ=$(( $(wc -w <<< "$COUNT_BINS") + 2 ))
mkdir -p "$W"/{vcf,arrays,masks,models,pred,cal,kmer,blocks,results,logs}
export TF_CPP_MIN_LOG_LEVEL=2

settings_now() {
  local v
  for v in MAF_THRESHOLD COUNT_BINS SNV_ONLY KEEP_NON_PASS INDEL_MODE PHASTCONS_PAD \
           TRAIN_ARGS KMER_K MUT_BINS DNM_NEG_SITES; do
    printf '%s=%q\n' "$v" "${!v:-}"
  done
}
if [[ -s "$W/settings.env" ]] && ! diff -q <(settings_now) "$W/settings.env" > /dev/null; then
  if [[ "${SETTINGS_CHANGED_OK:-0}" != 1 ]]; then
    echo "Settings differ from the ones $W was built with (< now, > recorded):" >&2
    diff <(settings_now) "$W/settings.env" | grep '^[<>]' >&2 || true
    echo "Use a new WORK directory, or delete the affected outputs and rerun with SETTINGS_CHANGED_OK=1." >&2
    exit 1
  fi
  echo "[settings] recording changed settings (SETTINGS_CHANGED_OK=1)" >&2
fi
settings_now > "$W/settings.env.partial.$$" && mv "$W/settings.env.partial.$$" "$W/settings.env"

have() { [[ -s "$1" ]]; }
fresh() { local out=$1; shift; have "$out" || return 1; local d; for d in "$@"; do [[ -z "$d" ]] && continue; [[ -e "$d" && "$out" -nt "$d" ]] || return 1; done; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
half_var() { local v="${1^^}_$2"; echo "${!v:-}"; }
contig_of() { awk -v s="$1" '$2 == s {print $1}' "$CHR_MAP"; }
tmpl() { local c; c=$(contig_of "$2"); echo "${1//\{contig\}/$c}"; }
all_seqids() { for h in $HALVES; do for r in TRAIN VAL CAL TEST; do half_var $h $r; done; done | tr ' ' '\n' | sed '/^$/d' | sort -u; }
models_for() { local h=$1 c=$2 out=""; for s in $SEEDS; do out+=" $W/models/$h/${c}_s$s.keras"; done; echo "$out"; }
mask_letter() { case $1 in neutralA) echo A;; neutralB) echo B;; neutralAB) echo AB;; *) echo "unknown configuration $1 (use neutralA, neutralB or neutralAB)" >&2; return 1;; esac; }
mask_for() { echo "$W/masks/$1_$2.npy"; }
for c in $CONFIGS; do mask_letter "$c" > /dev/null; done
labels_of() { echo "$W/arrays/$1_spectrum_labels.npy"; }
CALLABLE="$W/callable.bed"
PHAST_PADDED="$W/masks/phastcons_pad${PHASTCONS_PAD}.bed"
PHAST_RAW="$W/masks/phastcons_pad0.bed"

step_sites() {
  for s in $(all_seqids); do
    local src out="$W/vcf/$s.sites.vcf.gz"
    src=$(tmpl "$FULL_SITES_TEMPLATE" "$s")
    have "$src" || { echo "missing $src: run fetch_fullcallset.sh DATA sites $(contig_of "$s")" >&2; exit 1; }
    fresh "$out" "$src" "$CHR_MAP" && continue
    log "sites $s"
    bcftools annotate --rename-chrs "$CHR_MAP" -Oz -o "$out.partial" "$src"
    mv "$out.partial" "$out"
  done
}

step_labels() {
  if ! fresh "$CALLABLE" "$CALLABLE_BED" "$CHR_MAP"; then
    awk 'NR == FNR {m[$1] = $2; next} ($1 in m) {$1 = m[$1]; print}' OFS='\t' "$CHR_MAP" "$CALLABLE_BED" > "$CALLABLE.partial"
    mv "$CALLABLE.partial" "$CALLABLE"
  fi
  for s in $(all_seqids); do
    local dna="$W/arrays/${s}_dna.npy" sites="$W/vcf/$s.sites.vcf.gz" only=""
    # reuse the one-hot DNA of a finished v1 run instead of rebuilding it
    if [[ ! -e "$dna" && -n "${V1_WORK:-}" && -s "$V1_WORK/arrays/${s}_dna.npy" ]]; then
      ln -s "$V1_WORK/arrays/${s}_dna.npy" "$dna"
    fi
    # an existing array is kept only if it is newer than the FASTA; data.py
    # --labels-only also verifies that it encodes this FASTA sequence
    fresh "$dna" "$FASTA" && only=--labels-only
    fresh "$(labels_of "$s")" "$sites" "$CALLABLE" "$FASTA" && [[ -n "$only" ]] && continue
    [[ -z "$only" && -L "$dna" ]] && rm "$dna"   # never write through a link into the v1 run
    log "labels $s"
    # shellcheck disable=SC2086
    "$PY" "$CODE/data.py" --fasta "$FASTA" --vcf "$sites" --callable-bed "$CALLABLE" \
      --chromosome "$s" --threshold "$MAF_THRESHOLD" --scheme spectrum --count-bins $COUNT_BINS \
      --indel-mode "$INDEL_MODE" $([[ $SNV_ONLY == 1 ]] && echo --snv-only) \
      $([[ $KEEP_NON_PASS == 1 ]] && echo --keep-non-pass) $only --output-prefix "$W/arrays/$s"
  done
}

step_masks() {
  if ! fresh "$PHAST_PADDED" "$PHASTCONS_BED" "$CHR_MAP"; then
    awk 'NR == FNR {m[$1] = $2; next} ($1 in m) {a = $2 - p; if (a < 0) a = 0; print m[$1] "\t" a "\t" $3 + p}' \
      p="$PHASTCONS_PAD" OFS='\t' "$CHR_MAP" "$PHASTCONS_BED" > "$PHAST_PADDED.partial"
    mv "$PHAST_PADDED.partial" "$PHAST_PADDED"
  fi
  if ! fresh "$PHAST_RAW" "$PHASTCONS_BED" "$CHR_MAP"; then   # unpadded elements, for the conserved tier
    awk 'NR == FNR {m[$1] = $2; next} ($1 in m) {print m[$1] "\t" $2 "\t" $3}' OFS='\t' "$CHR_MAP" "$PHASTCONS_BED" > "$PHAST_RAW.partial"
    mv "$PHAST_RAW.partial" "$PHAST_RAW"
  fi
  for s in $(all_seqids); do
    local lab; lab=$(labels_of "$s")
    if ! fresh "$W/masks/${s}_neutralA.npy" "$lab" "$GFF"; then
      awk -F'\t' -v s="$s" '!/^#/ && $1 == s && $2 != "RefSeq" {print $1 "\t" $4 - 1 "\t" $5}' "$GFF" > "$W/masks/$s.annotated.bed"
      "$PY" "$CODE/make_mask.py" --bed "$W/masks/$s.annotated.bed" --chromosome "$s" --labels "$lab" \
        --invert --out "$W/masks/${s}_neutralA.npy"
    fi
    if ! fresh "$W/masks/${s}_neutralB.npy" "$lab" "$PHAST_PADDED"; then
      "$PY" "$CODE/make_mask.py" --bed "$PHAST_PADDED" --chromosome "$s" --labels "$lab" \
        --invert --out "$W/masks/${s}_neutralB.npy"
    fi
    if ! fresh "$W/masks/${s}_neutralAB.npy" "$W/masks/${s}_neutralA.npy" "$W/masks/${s}_neutralB.npy"; then
      # Mask AB: unannotated AND outside conserved elements. Mask A alone still
      # contains conserved sequence that no annotation covers; training on it
      # lets that constraint into the neutral expectation.
      "$PY" - "$CODE" "$W/masks/${s}_neutral" <<'EOF'
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from data import save_npy_atomic
a, b = (np.load(sys.argv[2] + x + '.npy', mmap_mode='r') for x in 'AB')
both = np.asarray(a) & np.asarray(b)
save_npy_atomic(sys.argv[2] + 'AB.npy', both)
print(f'{sys.argv[2]}AB.npy: {int(both.sum()):,}/{both.size:,} sites ({both.mean():.1%}) selected')
EOF
    fi
  done
  for m in A B AB; do  # rewritten only when the content changes
    "$PY" - "$W" "$m" $(all_seqids) <<'EOF'
import json, os, sys
w, m, seqids = sys.argv[1], sys.argv[2], sys.argv[3:]
text = json.dumps({s: {'dna': f'{w}/arrays/{s}_dna.npy', 'labels': f'{w}/arrays/{s}_spectrum_labels.npy',
                       'mask': f'{w}/masks/{s}_neutral{m}.npy'} for s in seqids}, indent=1)
path = f'{w}/manifest_{m}.json'
if not os.path.exists(path) or open(path).read() != text:
    open(path + '.partial', 'w').write(text)
    os.replace(path + '.partial', path)
EOF
  done
}

step_validate() {
  mkdir -p "$W/results/validation"
  for s in ${VALIDATE_SEQIDS:-$(for h in $HALVES; do half_var $h CAL; done)}; do
    local table out="$W/results/validation/$s.json"
    table=$(tmpl "${FULL_TABLE_TEMPLATE:-}" "$s")
    if ! have "$table"; then
      log "validate $s: no table at '$table' (fetch_fullcallset.sh DATA tables $(contig_of "$s")); skipped"
      continue
    fi
    fresh "$out" "$(labels_of "$s")" "$table" && continue
    log "validate labels $s against bcftools, per class"
    # shellcheck disable=SC2086
    "$PY" "$CODE/validate_labels.py" --labels "$(labels_of "$s")" --table "$table" \
      --chromosome "$(contig_of "$s")" --threshold "$MAF_THRESHOLD" --scheme spectrum \
      --count-bins $COUNT_BINS $([[ $SNV_ONLY == 1 ]] && echo --snv-only) \
      --disagreements "$W/results/validation/$s.disagreements.tsv" > "$W/results/validation/$s.out"
    sed '/^Wrote/d' "$W/results/validation/$s.out" > "$out.partial"
    mv "$out.partial" "$out"
  done
}

train_deps() {  # half config -> every array the model (or k-mer table) learns from
  local h=$1 c=$2 s
  for s in $(half_var $h TRAIN) $(half_var $h VAL); do
    echo "$W/arrays/${s}_dna.npy $(labels_of "$s") $(mask_for "$s" "$c")"
  done
}

# calibrate_and_predict half name calmask targets cmd...: uncalibrated prediction
# of the calibration chromosome, isotonic fit, calibrated prediction of the
# targets. "cmd" is a command that takes: --dna X --output Y [--calibration Z];
# remaining file arguments in DEPS (global array) decide freshness.
calibrate_and_predict() {
  local h=$1 name=$2 calmask=$3 targets=$4; shift 4
  local cal; cal=$(half_var $h CAL)
  local calpred="$W/pred/$h/$name/cal_$cal.npy" caljson="$W/cal/$h/$name.json"
  mkdir -p "$W/pred/$h/$name" "$W/cal/$h"
  fresh "$calpred" "${DEPS[@]}" "$W/arrays/${cal}_dna.npy" \
    || "$@" --dna "$W/arrays/${cal}_dna.npy" --output "$calpred"
  fresh "$caljson" "$calpred" "$(labels_of "$cal")" "$calmask" || "$PY" "$CODE/calibrate.py" \
    --labels "$(labels_of "$cal")" --predictions "$calpred" --mask "$calmask" --out "$caljson"
  local s
  for s in $targets; do
    fresh "$W/pred/$h/$name/$s.npy" "${DEPS[@]}" "$caljson" "$W/arrays/${s}_dna.npy" \
      || "$@" --dna "$W/arrays/${s}_dna.npy" --output "$W/pred/$h/$name/$s.npy" --calibration "$caljson"
  done
}

step_kmer() {
  for h in $HALVES; do
    local cal; cal=$(half_var $h CAL)
    mkdir -p "$W/kmer/$h"
    for c in $CONFIGS; do
      for k in $KMER_K; do
        local table="$W/kmer/$h/${c}_k$k.json"
        # shellcheck disable=SC2046
        if ! fresh "$table" $(train_deps $h $c); then
          log "k-mer table $h $c k=$k"
          # shellcheck disable=SC2046
          "$PY" "$CODE/kmer_baseline.py" fit --manifest "$W/manifest_$(mask_letter $c).json" \
            --chromosomes $(half_var $h TRAIN) $(half_var $h VAL) --mask-scope all --k "$k" \
            --n-classes $(( N_FREQ + 1 )) --out "$table" > "$W/logs/kmer_${h}_${c}_k$k.log"
        fi
        log "k-mer predict $h $c k=$k"
        DEPS=("$table")
        calibrate_and_predict $h "${c}_kmer$k" "$(mask_for "$cal" "$c")" "$(half_var $h TEST)" \
          "$PY" "$CODE/kmer_baseline.py" predict --table "$table"
      done
    done
  done
}

step_train() {
  for h in ${ONLY_HALVES:-$HALVES}; do
    mkdir -p "$W/models/$h"
    for c in ${ONLY_CONFIGS:-$CONFIGS}; do
      for seed in ${ONLY_SEEDS:-$SEEDS}; do
        local out="$W/models/$h/${c}_s$seed.keras"
        # shellcheck disable=SC2046
        fresh "$out" $(train_deps $h $c) && fresh "$out.meta.json" "$out" && continue
        log "train $h $c seed $seed"
        # shellcheck disable=SC2086,SC2046
        "$PY" "$CODE/train.py" --manifest "$W/manifest_$(mask_letter $c).json" \
          --train-chromosomes $(half_var $h TRAIN) --val-chromosomes $(half_var $h VAL) \
          --head spectrum --n-freq-classes "$N_FREQ" --mask-scope all --seed "$seed" $TRAIN_ARGS \
          --out "$out" > "$W/logs/train_${h}_${c}_s$seed.log" 2>&1
      done
    done
  done
}

step_predict() {
  for h in ${ONLY_HALVES:-$HALVES}; do
    local cal; cal=$(half_var $h CAL)
    for c in ${ONLY_CONFIGS:-$CONFIGS}; do
      log "predict $h $c"
      # shellcheck disable=SC2207
      DEPS=($(models_for $h $c))
      # shellcheck disable=SC2046
      calibrate_and_predict $h $c "$(mask_for "$cal" "$c")" "$(half_var $h TEST)" \
        "$PY" "$CODE/predict.py" --model $(models_for $h $c)
      [[ $SEED_PREDICTIONS == 1 && -n "$(half_var $h SEED_TEST)" ]] || continue
      for seed in $SEEDS; do
        log "predict $h $c seed $seed"
        DEPS=("$W/models/$h/${c}_s$seed.keras")
        calibrate_and_predict $h "${c}_s$seed" "$(mask_for "$cal" "$c")" "$(half_var $h SEED_TEST)" \
          "$PY" "$CODE/predict.py" --model "$W/models/$h/${c}_s$seed.keras"
      done
    done
  done
}

CONTRASTS=(tier_cds:tier_neutral tier_conserved_noncoding:tier_neutral tier_enhancer:tier_neutral
           tier_cds:tier_enhancer tier_conserved_noncoding:tier_enhancer tier_cds:tier_conserved_noncoding)

# score_set half name newdir baselinedir(or "") config targets: per-chromosome
# block files, then the pooled result (results/<half>/<name>.json and .tsv)
score_set() {
  local h=$1 name=$2 newdir=$3 basedir=$4 c=$5 targets=$6 files=() chroms=() s
  local edges="$W/blocks/$h/$name/edges.json" edge_deps=()
  mkdir -p "$W/blocks/$h/$name" "$W/results/$h"
  for s in $targets; do
    chroms+=(--chrom "$(labels_of "$s")" "$newdir/$s.npy" "$(mask_for "$s" "$c")")
    edge_deps+=("$(labels_of "$s")" "$newdir/$s.npy" "$(mask_for "$s" "$c")")
  done
  fresh "$edges" "${edge_deps[@]}" || "$PY" "$CODE/spectrum_score.py" edges "${chroms[@]}" \
    --bins "$MUT_BINS" --out "$edges" > /dev/null
  for s in $targets; do
    local out="$W/blocks/$h/$name/$s.npz" extra=()
    local deps=("$newdir/$s.npy" "$(labels_of "$s")" "$GFF" "$PHAST_RAW" "$edges"
                "$W/masks/${s}_neutralA.npy" "$W/masks/${s}_neutralB.npy" "$W/masks/${s}_neutralAB.npy")
    if [[ -n "$basedir" ]]; then extra=(--baseline "$basedir/$s.npy"); deps+=("$basedir/$s.npy"); fi
    fresh "$out" "${deps[@]}" || "$PY" "$CODE/spectrum_score.py" chrom --gff "$GFF" --seqid "$s" \
      --labels "$(labels_of "$s")" --new "$newdir/$s.npy" "${extra[@]}" \
      --mask "neutralA=$W/masks/${s}_neutralA.npy" --mask "neutralB=$W/masks/${s}_neutralB.npy" \
      --mask "neutralAB=$W/masks/${s}_neutralAB.npy" \
      --bed "phastcons=$PHAST_RAW" --tiers "$c" --edges "$edges" --out "$out" > /dev/null
    files+=("$out")
  done
  local cargs=() x
  for x in "${CONTRASTS[@]}"; do cargs+=(--contrast "$x"); done
  "$PY" "$CODE/spectrum_score.py" pool --blocks "${files[@]}" "${cargs[@]}" \
    --json "$W/results/$h/$name.json" --tsv "$W/results/$h/$name.tsv" > /dev/null
}

step_score() {
  for h in $HALVES; do
    for c in $CONFIGS; do
      log "score $h $c"
      for k in $KMER_K; do   # the network, with each k-mer baseline scored on the same sites
        score_set $h "${c}_vs_kmer$k" "$W/pred/$h/$c" "$W/pred/$h/${c}_kmer$k" $c "$(half_var $h TEST)"
      done
      if [[ $SEED_PREDICTIONS == 1 && -n "$(half_var $h SEED_TEST)" ]]; then
        for seed in $SEEDS; do
          score_set $h "${c}_seed$seed" "$W/pred/$h/${c}_s$seed" "" $c "$(half_var $h SEED_TEST)"
        done
      fi
    done
  done
}

step_denovo() {
  local sets="${DNM_SETS:-}" item
  [[ -n "${sets// /}" ]] || { log "no de novo sets configured (DNM_SETS); skipping"; return; }
  for item in $sets; do
    [[ "$item" == *=* && -n "${item%%=*}" ]] || { echo "DNM_SETS entry '$item' is not name=path" >&2; exit 1; }
  done
  for h in $HALVES; do
    local od rawd v1 deps=("$CHR_MAP")
    od=$(half_var $h ORIG_DIR); rawd=$(half_var $h ORIG_RAW_DIR); v1="${V1_PRED_TEMPLATE:-}"
    for item in $sets; do deps+=("${item#*=}"); done
    for s in $(half_var $h TEST); do
      deps+=("$W/arrays/${s}_dna.npy")
      for c in $CONFIGS; do
        deps+=("$W/pred/$h/$c/$s.npy")
        for k in $KMER_K; do deps+=("$W/pred/$h/${c}_kmer$k/$s.npy"); done
      done
      [[ -n "$od" ]] && deps+=("$od/$s.pred.npy")
      [[ -n "$rawd" ]] && deps+=("$rawd/$s.pred.npy")
    done
    fresh "$W/results/$h/denovo.json" "${deps[@]}" && continue
    log "de novo benchmark $h ($sets)"
    "$PY" - "$W" "$h" "$od" "$rawd" "$v1" "$CHR_MAP" "$sets" "$CONFIGS" "$KMER_K" $(half_var $h TEST) <<'EOF'
import json, os, sys
w, h, od, rawd, v1, chr_map, sets, configs, ks, *test = sys.argv[1:]
contig = {l.split()[1]: l.split()[0] for l in open(chr_map) if l.strip()}
dnm = dict(item.split('=', 1) for item in sets.split())
models = {'original': 'score'} if od else {}
have_v1 = bool(v1) and all(os.path.exists(v1.replace('{half}', h).replace('{seqid}', s)) for s in test)
if have_v1:
    models['v1_neutral'] = 'classes'
for c in configs.split():
    models[c] = 'classes'
    for k in ks.split():
        models[f'{c}_kmer{k}'] = 'classes'
    models[f'{c}_singleton_share'] = 'singleton_share'   # descriptive: spectrum shape vs mutability
chroms = {}
for s in test:
    preds = {}
    if od:
        preds['original'] = {'path': f'{od}/{s}.pred.npy'} | ({'raw': f'{rawd}/{s}.pred.npy'} if rawd else {})
    if have_v1:
        preds['v1_neutral'] = v1.replace('{half}', h).replace('{seqid}', s)
    for c in configs.split():
        preds[c] = f'{w}/pred/{h}/{c}/{s}.npy'
        preds[f'{c}_singleton_share'] = preds[c]
        for k in ks.split():
            preds[f'{c}_kmer{k}'] = f'{w}/pred/{h}/{c}_kmer{k}/{s}.npy'
    chroms[s] = {'dnm_name': contig.get(s, s), 'dna': f'{w}/arrays/{s}_dna.npy', 'predictions': preds}
json.dump({'models': models, 'dnm': dnm, 'chromosomes': chroms},
          open(f'{w}/results/{h}/denovo_config.json', 'w'), indent=1)
EOF
    "$PY" "$CODE/denovo_auc.py" --config "$W/results/$h/denovo_config.json" \
      --neg-sites "$DNM_NEG_SITES" --json "$W/results/$h/denovo.json" > /dev/null
  done
}

step_evaluate() {
  "$PY" "$CODE/experiments/real_data/evaluate_plan.py" --criteria "$CRITERIA" --results "$W/results" \
    --report "$W/results/criteria_report.md" --json "$W/results/criteria_report.json"
}

for st in $STEPS; do
  log "== $st"
  "step_$st"
done
