#!/usr/bin/env bash
# Run the pre-registered analysis plan end to end.
#
#   bash run_plan.sh plan.env                 # every step
#   bash run_plan.sh plan.env train predict   # selected steps, in this order
#
# Steps: labels masks validate train predict score denovo evaluate
# An output is rebuilt only if it is missing or older than any input it is
# made from (source VCF, FASTA, GFF, BEDs, labels, masks, models, ...), and
# every output is written under a temporary name and renamed when complete.
# So an interrupted run can simply be restarted, and a changed input redoes
# exactly what depends on it. Not tracked: changes to settings in plan.env
# (e.g. TRAIN_ARGS, MAF_THRESHOLD) -- delete the affected outputs yourself.
#
# Slices for parallel jobs (slurm/submit_plan.sh uses these):
#   ONLY_HALVES="half1"  ONLY_CONFIGS="neutral"  ONLY_SEEDS="3"  (train, predict)
set -euo pipefail

[[ $# -ge 1 ]] || { sed -n '2,17p' "$0"; exit 1; }
# shellcheck disable=SC1090
source "$1"; shift
STEPS="${*:-labels masks validate train predict score denovo evaluate}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python3}"
W="$WORK"
SEEDS="${SEEDS:-0 1 2 3 4}"
CONFIGS="base neutral neutralB"
TRAIN_ARGS="${TRAIN_ARGS:-}"
HALVES="${HALVES:-half1 half2}"
INDEL_MODE="${INDEL_MODE:-original}"          # matches the original repo's labeled span
KEEP_NON_PASS="${KEEP_NON_PASS:-1}"           # the original repo ignored FILTER
UNIT_SITES="${UNIT_SITES:-2000}"              # unit size for the discrimination endpoint (P2)
DISC_BLOCK_BP="${DISC_BLOCK_BP:-1000000}"     # genomic block size for the P2 block bootstrap
if [[ -z "${SPLIT_MULTIALLELIC:-}" ]]; then [[ $INDEL_MODE == original ]] && SPLIT_MULTIALLELIC=0 || SPLIT_MULTIALLELIC=1; fi
PHASTCONS_PAD="${PHASTCONS_PAD:-100}"
mkdir -p "$W"/{vcf,arrays,masks,models,pred,cal,blocks,results,logs}
export TF_CPP_MIN_LOG_LEVEL=2

have() { [[ -s "$1" ]]; }
# fresh OUT DEP...: OUT exists and is newer than every DEP (empty DEPs ignored),
# so a rebuilt input invalidates everything derived from it.
fresh() { local out=$1; shift; have "$out" || return 1; local d; for d in "$@"; do [[ -z "$d" ]] && continue; [[ -e "$d" && "$out" -nt "$d" ]] || return 1; done; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
half_var() { local v="${1^^}_$2"; echo "${!v:-}"; }   # half_var half1 TRAIN -> $HALF1_TRAIN
contig_of() { awk -v s="$1" '$2 == s {print $1}' "$CHR_MAP"; }
vcf_of() { local c; c=$(contig_of "$1"); echo "${VCF_TEMPLATE//\{contig\}/$c}"; }
all_seqids() { for h in $HALVES; do for r in TRAIN VAL CAL TEST; do half_var $h $r; done; done | tr ' ' '\n' | sed '/^$/d' | sort -u; }
models_for() { local h=$1 c=$2 out=""; for s in $SEEDS; do out+=" $W/models/$h/${c}_s$s.keras"; done; echo "$out"; }
mask_for() { case $2 in neutral) echo "$W/masks/$1_neutralA.npy";; neutralB) echo "$W/masks/$1_neutralB.npy";; *) echo "";; esac; }
orig_dir() { half_var "$1" ORIG_DIR; }
CALLABLE="$W/callable.bed"
PHAST_PADDED="$W/masks/phastcons_pad${PHASTCONS_PAD}.bed"

step_labels() {
  if ! fresh "$CALLABLE" "$CALLABLE_BED" "$CHR_MAP"; then
    awk 'NR == FNR {m[$1] = $2; next} ($1 in m) {$1 = m[$1]; print}' OFS='\t' "$CHR_MAP" "$CALLABLE_BED" > "$CALLABLE.partial"
    mv "$CALLABLE.partial" "$CALLABLE"
  fi
  for s in $(all_seqids); do
    local vcf sites
    vcf=$(vcf_of "$s"); sites="$W/vcf/$s.sites.vcf.gz"
    if ! fresh "$sites" "$vcf" "$CHR_MAP" "${SAMPLES:-}"; then
      log "sites $s (from $vcf)"
      # AC/AN recomputed from genotypes (optionally a sample subset), genotypes dropped,
      # contigs renamed to the FASTA/GFF names; multi-allelic sites split unless
      # INDEL_MODE=original (the original repo read records as they appear).
      local split_cmd=(bcftools view -Oz -o "$sites.partial")
      [[ $SPLIT_MULTIALLELIC == 1 ]] && split_cmd=(bcftools norm -m- -f "$FASTA" -Oz -o "$sites.partial")
      bcftools view ${SAMPLES:+-S "$SAMPLES"} -Ou "$vcf" \
        | bcftools +fill-tags -Ou -- -t AC,AN \
        | bcftools view -G -Ou \
        | bcftools annotate --rename-chrs "$CHR_MAP" -Ou \
        | "${split_cmd[@]}"
      mv "$sites.partial" "$sites"
    fi
    fresh "$W/arrays/${s}_dna.npy" "$FASTA" && fresh "$W/arrays/${s}_frequency_labels.npy" "$sites" "$CALLABLE" "$FASTA" && continue
    log "labels $s"
    "$PY" "$CODE/data.py" --fasta "$FASTA" --vcf "$sites" --callable-bed "$CALLABLE" \
      --chromosome "$s" --threshold "${MAF_THRESHOLD:-0.01}" --indel-mode "$INDEL_MODE" \
      $([[ $KEEP_NON_PASS == 1 ]] && echo --keep-non-pass) --output-prefix "$W/arrays/$s"
  done
}

step_masks() {
  if ! fresh "$PHAST_PADDED" "$PHASTCONS_BED" "$CHR_MAP"; then
    awk 'NR == FNR {m[$1] = $2; next} ($1 in m) {a = $2 - p; if (a < 0) a = 0; print m[$1] "\t" a "\t" $3 + p}' \
      p="$PHASTCONS_PAD" OFS='\t' "$CHR_MAP" "$PHASTCONS_BED" > "$PHAST_PADDED.partial"
    mv "$PHAST_PADDED.partial" "$PHAST_PADDED"
  fi
  for s in $(all_seqids); do
    local lab="$W/arrays/${s}_frequency_labels.npy"
    if ! fresh "$W/masks/${s}_neutralA.npy" "$lab" "$GFF"; then
      # Mask A (annotation-based): outside every GFF feature not sourced RefSeq,
      # the original repo's "unannotated" definition.
      awk -F'\t' -v s="$s" '!/^#/ && $1 == s && $2 != "RefSeq" {print $1 "\t" $4 - 1 "\t" $5}' "$GFF" > "$W/masks/$s.annotated.bed"
      "$PY" "$CODE/make_mask.py" --bed "$W/masks/$s.annotated.bed" --chromosome "$s" --labels "$lab" \
        --invert --out "$W/masks/${s}_neutralA.npy"
    fi
    if ! fresh "$W/masks/${s}_neutralB.npy" "$lab" "$PHAST_PADDED"; then
      # Mask B (conservation-based, independent of the GFF): outside phastCons
      # 100-way conserved elements padded by PHASTCONS_PAD bp.
      "$PY" "$CODE/make_mask.py" --bed "$PHAST_PADDED" --chromosome "$s" --labels "$lab" \
        --invert --out "$W/masks/${s}_neutralB.npy"
    fi
  done
  for m in A B; do  # rewritten only when the content changes
    "$PY" - "$W" "$m" $(all_seqids) <<'EOF'
import json, os, sys
w, m, seqids = sys.argv[1], sys.argv[2], sys.argv[3:]
text = json.dumps({s: {'dna': f'{w}/arrays/{s}_dna.npy', 'labels': f'{w}/arrays/{s}_frequency_labels.npy',
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
    local vcf; vcf=$(vcf_of "$s")
    fresh "$W/results/validation/$s.json" "$W/arrays/${s}_frequency_labels.npy" "$vcf" "${SAMPLES:-}" \
      && have "$W/results/validation/$s.disagreements.tsv" && continue
    log "validate labels $s against bcftools"
    bcftools view ${SAMPLES:+-S "$SAMPLES"} $([[ $KEEP_NON_PASS == 1 ]] || echo "-f PASS,.") -Ou "$vcf" \
      | bcftools annotate --rename-chrs "$CHR_MAP" -Ou \
      | bcftools norm -m+any -f "$FASTA" -Ou \
      | bcftools +fill-tags -Ou -- -t AC,AN,MAF \
      | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t%INFO/AN\t%INFO/MAF\n' > "$W/vcf/$s.bcftools.tsv.partial"
    mv "$W/vcf/$s.bcftools.tsv.partial" "$W/vcf/$s.bcftools.tsv"
    "$PY" "$CODE/validate_labels.py" --labels "$W/arrays/${s}_frequency_labels.npy" \
      --table "$W/vcf/$s.bcftools.tsv" --chromosome "$s" --threshold "${MAF_THRESHOLD:-0.01}" \
      --disagreements "$W/results/validation/$s.disagreements.tsv" > "$W/results/validation/$s.out"
    sed '/^Wrote/d' "$W/results/validation/$s.out" > "$W/results/validation/$s.json.partial"
    mv "$W/results/validation/$s.json.partial" "$W/results/validation/$s.json"
  done
  # How far the labels are from the original repo's binary labels (definition differences, reported as such)
  if [[ -n "${ORIG_LABEL_DIR:-}" ]]; then
    for s in $(all_seqids); do
      local orig_lab="$ORIG_LABEL_DIR/${s}_label.pkl"
      [[ -f "$orig_lab" ]] || continue
      fresh "$W/results/validation/labels_vs_original_$s.json" "$W/arrays/${s}_frequency_labels.npy" "$orig_lab" \
        || "$PY" "$CODE/compare_labels.py" --labels "$W/arrays/${s}_frequency_labels.npy" \
             --original-labels "$orig_lab" --json "$W/results/validation/labels_vs_original_$s.json" > /dev/null
    done
  fi
}

train_deps() {  # half config -> every array the model is trained or validated on
  local h=$1 c=$2 s
  for s in $(half_var $h TRAIN) $(half_var $h VAL); do
    echo "$W/arrays/${s}_dna.npy $W/arrays/${s}_frequency_labels.npy"
    case $c in neutral) echo "$W/masks/${s}_neutralA.npy";; neutralB) echo "$W/masks/${s}_neutralB.npy";; esac
  done
}

step_train() {
  for h in ${ONLY_HALVES:-$HALVES}; do
    mkdir -p "$W/models/$h"
    for c in ${ONLY_CONFIGS:-$CONFIGS}; do
      local scope manifest
      case $c in base) scope=none manifest=A;; neutral) scope=all manifest=A;; neutralB) scope=all manifest=B;; esac
      for seed in ${ONLY_SEEDS:-$SEEDS}; do
        local out="$W/models/$h/${c}_s$seed.keras"
        # shellcheck disable=SC2046
        fresh "$out" $(train_deps $h $c) && fresh "$out.meta.json" "$out" && continue
        log "train $h $c seed $seed"
        # shellcheck disable=SC2086
        "$PY" "$CODE/train.py" --manifest "$W/manifest_$manifest.json" \
          --train-chromosomes $(half_var $h TRAIN) --val-chromosomes $(half_var $h VAL) \
          --head factorized --mask-scope $scope --seed "$seed" $TRAIN_ARGS --out "$out" \
          > "$W/logs/train_${h}_${c}_s$seed.log" 2>&1
      done
    done
  done
}

predict_set() {  # half name models calmask targets -> calibrated predictions
  local h=$1 name=$2 models=$3 calmask=$4 targets=$5
  local cal; cal=$(half_var $h CAL)
  local calpred="$W/pred/$h/$name/cal_$cal.npy" caljson="$W/cal/$h/$name.json"
  mkdir -p "$W/pred/$h/$name" "$W/cal/$h"
  # shellcheck disable=SC2086
  fresh "$calpred" $models "$W/arrays/${cal}_dna.npy" || "$PY" "$CODE/predict.py" --model $models \
    --dna "$W/arrays/${cal}_dna.npy" --output "$calpred"
  fresh "$caljson" "$calpred" "$W/arrays/${cal}_frequency_labels.npy" "$calmask" || "$PY" "$CODE/calibrate.py" \
    --labels "$W/arrays/${cal}_frequency_labels.npy" --predictions "$calpred" \
    ${calmask:+--mask "$calmask"} --out "$caljson"
  for s in $targets; do
    # shellcheck disable=SC2086
    fresh "$W/pred/$h/$name/$s.npy" $models "$caljson" "$W/arrays/${s}_dna.npy" || "$PY" "$CODE/predict.py" \
      --model $models --dna "$W/arrays/${s}_dna.npy" --output "$W/pred/$h/$name/$s.npy" --calibration "$caljson"
  done
}

step_predict() {
  for h in ${ONLY_HALVES:-$HALVES}; do
    local cal; cal=$(half_var $h CAL)
    for c in ${ONLY_CONFIGS:-$CONFIGS}; do
      log "predict $h $c"
      predict_set $h $c "$(models_for $h $c)" "$(mask_for $cal $c)" "$(half_var $h TEST)"
    done
    if [[ " ${ONLY_CONFIGS:-neutral} " == *" neutral "* ]]; then
      for seed in $SEEDS; do
        log "predict $h neutral seed $seed"
        predict_set $h "neutral_s$seed" "$W/models/$h/neutral_s$seed.keras" "$(mask_for $cal neutral)" "$(half_var $h SEED_TEST)"
      done
    fi
  done
}

score_cmp() {  # half name reference_kind(orig|base|none) new_dir targets [extra args]
  local h=$1 name=$2 ref=$3 newdir=$4 targets=$5 files=() extra=("${@:6}")
  local od rawd; od=$(orig_dir $h); rawd=$(half_var $h ORIG_RAW_DIR)
  [[ $ref == orig && -z "$od" ]] && ref=none   # no original predictions: score the new model alone
  mkdir -p "$W/blocks/$h/$name" "$W/results/$h"
  for s in $targets; do
    local out="$W/blocks/$h/$name/$s.npz" refargs=()
    local deps=("$newdir/$s.npy" "$W/arrays/${s}_frequency_labels.npy" "$GFF"
                "$W/masks/${s}_neutralA.npy" "$W/masks/${s}_neutralB.npy")
    case $ref in
      orig) refargs=(--original "$od/$s.pred.npy"); deps+=("$od/$s.pred.npy")
            if [[ -n "$rawd" ]]; then refargs+=(--original-raw "$rawd/$s.pred.npy"); deps+=("$rawd/$s.pred.npy"); fi;;
      base) refargs=(--original "$W/pred/$h/base/$s.npy"); deps+=("$W/pred/$h/base/$s.npy");;
    esac
    fresh "$out" "${deps[@]}" || "$PY" "$CODE/score_annotations.py" --gff "$GFF" --seqid "$s" \
      --labels "$W/arrays/${s}_frequency_labels.npy" --new "$newdir/$s.npy" "${refargs[@]}" \
      --mask "neutralA=$W/masks/${s}_neutralA.npy" --mask "neutralB=$W/masks/${s}_neutralB.npy" \
      --bootstrap 0 --save-blocks "$out" "${extra[@]}" > /dev/null
    files+=("$out")
  done
  "$PY" "$CODE/score_annotations.py" --pool "${files[@]}" \
    --json "$W/results/$h/$name.json" --tsv "$W/results/$h/$name.tsv" > /dev/null
}

step_score() {
  for h in $HALVES; do
    local test; test=$(half_var $h TEST)
    log "score annotations $h"
    score_cmp $h base orig "$W/pred/$h/base" "$test"
    score_cmp $h neutral orig "$W/pred/$h/neutral" "$test" --unit-sites "$UNIT_SITES"
    score_cmp $h neutralB orig "$W/pred/$h/neutralB" "$test" --unit-sites "$UNIT_SITES"
    # P2 / R1c / R3 / R4: CDS vs neutral units (recomputed every run; cheap)
    for c in neutral neutralB; do
      local ctl=neutralA; [[ $c == neutralB ]] && ctl=neutralB
      if [[ -n "$(orig_dir $h)" ]]; then
        "$PY" "$CODE/discriminate.py" --blocks "$W/blocks/$h/$c"/*.npz --case CDS --control $ctl \
          --block-bp "$DISC_BLOCK_BP" --json "$W/results/$h/discrimination_$c.json" > /dev/null
      fi
    done
    score_cmp $h base_vs_neutral base "$W/pred/$h/neutral" "$test"
    score_cmp $h base_vs_neutralB base "$W/pred/$h/neutralB" "$test"
    for seed in $SEEDS; do
      score_cmp $h "neutral_seed$seed" orig "$W/pred/$h/neutral_s$seed" "$(half_var $h SEED_TEST)"
    done
  done
}

dnm_sets() {  # "name=path ..." from DNM_SETS, or the older DNM_DENOVO_DB / DNM_PLATINUM settings
  if [[ -n "${DNM_SETS:-}" ]]; then echo "$DNM_SETS"; return; fi
  [[ -n "${DNM_DENOVO_DB:-}" ]] && echo -n "denovo_db=$DNM_DENOVO_DB "
  [[ -n "${DNM_PLATINUM:-}" ]] && echo -n "platinum=$DNM_PLATINUM"
  echo
}

step_denovo() {
  local sets item; sets=$(dnm_sets)
  [[ -n "${sets// /}" ]] || { log "no de novo sets configured (DNM_SETS); skipping"; return; }
  for item in $sets; do
    [[ "$item" == *=* && -n "${item%%=*}" ]] || { echo "DNM_SETS entry '$item' is not name=path" >&2; exit 1; }
  done
  for h in $HALVES; do
    local od rawd deps=("$CHR_MAP")
    od=$(orig_dir $h); rawd=$(half_var $h ORIG_RAW_DIR)
    for item in $sets; do deps+=("${item#*=}"); done
    for s in $(half_var $h TEST); do
      deps+=("$W/arrays/${s}_dna.npy")
      for c in base neutral neutralB; do deps+=("$W/pred/$h/$c/$s.npy"); done
      [[ -n "$od" ]] && deps+=("$od/$s.pred.npy")
      [[ -n "$rawd" ]] && deps+=("$rawd/$s.pred.npy")
    done
    fresh "$W/results/$h/denovo.json" "${deps[@]}" && continue
    log "de novo benchmark $h ($sets)"
    "$PY" - "$W" "$h" "$od" "$rawd" "$CHR_MAP" "$sets" $(half_var $h TEST) <<'EOF'
import json, sys
w, h, od, ord_, chr_map, sets, *test = sys.argv[1:]
contig = {l.split()[1]: l.split()[0] for l in open(chr_map) if l.strip()}
dnm = dict(item.split('=', 1) for item in sets.split())
models = ({'original': 'score'} if od else {}) | {'base': 'classes', 'neutral': 'classes', 'neutralB': 'classes'}
chroms = {}
for s in test:
    preds = {c: f'{w}/pred/{h}/{c}/{s}.npy' for c in ('base', 'neutral', 'neutralB')}
    if od:
        preds['original'] = {'path': f'{od}/{s}.pred.npy'} | ({'raw': f'{ord_}/{s}.pred.npy'} if ord_ else {})
    chroms[s] = {'dnm_name': contig.get(s, s), 'dna': f'{w}/arrays/{s}_dna.npy', 'predictions': preds}
json.dump({'models': models, 'dnm': dnm,'chromosomes': chroms},
          open(f'{w}/results/{h}/denovo_config.json', 'w'), indent=1)
EOF
    "$PY" "$CODE/denovo_auc.py" --config "$W/results/$h/denovo_config.json" \
      --neg-sites "${DNM_NEG_SITES:-5000000}" --json "$W/results/$h/denovo.json" > /dev/null
  done
}

step_evaluate() {
  "$PY" "$HERE/evaluate_plan.py" --criteria "$HERE/criteria.json" --results "$W/results" \
    --report "$W/results/criteria_report.md" --json "$W/results/criteria_report.json"
}

for st in $STEPS; do
  log "== $st"
  "step_$st"
done
