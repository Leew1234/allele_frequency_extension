#!/usr/bin/env bash
# Sites-only VCFs (and bcftools validation tables) from the FULL 1000 Genomes
# high-coverage call set, which, unlike the phased panel, contains singletons.
#
#   fetch_fullcallset.sh DATA_DIR sites  chr21 chr22 ...   # or 'all' for chr1-22
#   fetch_fullcallset.sh DATA_DIR tables chr22 ...         # for label validation
#
# Why not the phased panel that fetch_data.sh downloads: sites with minor
# allele count < 2 were removed from it before phasing (Byrska-Bishop et al.
# 2022, Cell 185:3426, Methods), so it has no singletons at all.
#
# The full call set is one large genotype VCF per chromosome. This script does
# not store it: each file is streamed through bcftools, which recomputes AC and
# AN from the genotypes (optionally for a sample subset), marks sites with too
# many missing genotypes, drops the genotypes, and writes a small sites-only
# VCF. Nothing is resumable inside a chromosome; a chromosome that did not
# finish is redone on the next run, finished ones are skipped.
#
#   sites   DATA_DIR/sites_full/<contig>.sites.vcf.gz
#           FILTER keeps the call set's own VQSR result and gains 'HighMissing'
#           where more than MAX_MISSING of genotypes are missing. data.py turns
#           every non-PASS record into an ignored site when KEEP_NON_PASS=0.
#   tables  DATA_DIR/sites_full/<contig>.bcftools.tsv
#           chrom, pos, ref, alt, AN, MAF with all records at a position joined
#           (-m+any), computed by bcftools alone: the independent reference
#           that validate_labels.py compares the labels with. Only PASS records
#           with few missing genotypes (all records if KEEP_NON_PASS=1).
#
# Settings (environment):
#   KGP_FULL_BASE / KGP_FULL_FILE   where the call set is; {contig} is replaced
#   KGP_FULL_LOCAL   a local template (/path/{contig}.vcf.gz) used instead of
#                    downloading, if you already have the files
#   SAMPLES          file with one sample ID per line (e.g. unrelated founders);
#                    empty = all 3,202 samples
#   MAX_MISSING      default 0.05, the threshold the 1000 Genomes paper used
#
# The default URL is the directory the 1000 Genomes FTP site lists for the
# 3,202-sample call set. Check that it still lists these files before a long
# run (open KGP_FULL_BASE in a browser); set KGP_FULL_BASE / KGP_FULL_FILE if not.
set -euo pipefail

[[ $# -ge 3 ]] || { sed -n '2,36p' "$0"; exit 1; }
D=$(mkdir -p "$1" && cd "$1" && pwd)
MODE=$2; shift 2
case $MODE in sites|tables) ;; *) echo "second argument must be sites or tables" >&2; exit 1 ;; esac
[[ $1 == all ]] && set -- $(seq -f 'chr%g' 1 22)

KGP_FULL_BASE=${KGP_FULL_BASE:-https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_raw_GT_with_annot}
[[ -n "${KGP_FULL_FILE:-}" ]] || KGP_FULL_FILE='20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_{contig}.recalibrated_variants.vcf.gz'
MAX_MISSING=${MAX_MISSING:-0.05}
OUT="$D/sites_full"; mkdir -p "$OUT"
# The table must describe the records the labels use: with KEEP_NON_PASS=1 in
# spectrum.env (FILTER ignored), export KEEP_NON_PASS=1 here too.
TABLE_FILTER=(-f PASS,. -e "F_MISSING>$MAX_MISSING")
[[ "${KEEP_NON_PASS:-0}" == 1 ]] && TABLE_FILTER=()

log() { echo "[$(date +%H:%M:%S)] $*"; }
command -v bcftools > /dev/null || { echo "missing tool: bcftools" >&2; exit 1; }
if [[ -n "${SAMPLES:-}" && ! -s "$SAMPLES" ]]; then echo "SAMPLES file not found: $SAMPLES" >&2; exit 1; fi

# source CONTIG: writes the genotype VCF of one chromosome to stdout
source_vcf() {
  local c=$1
  if [[ -n "${KGP_FULL_LOCAL:-}" ]]; then
    local f=${KGP_FULL_LOCAL//\{contig\}/$c}
    [[ -s "$f" ]] || { echo "missing local file: $f" >&2; return 1; }
    cat "$f"
  else
    command -v curl > /dev/null || { echo "missing tool: curl" >&2; return 1; }
    curl -fsSL --retry 3 --retry-delay 10 "$KGP_FULL_BASE/${KGP_FULL_FILE//\{contig\}/$c}"
  fi
}

for c in "$@"; do
  if [[ $MODE == sites ]]; then
    out="$OUT/$c.sites.vcf.gz"
    if [[ -s "$out" ]]; then log "have $(basename "$out")"; continue; fi
    log "sites $c (streaming; AC/AN recomputed${SAMPLES:+ for the samples in $SAMPLES})"
    source_vcf "$c" \
      | bcftools view ${SAMPLES:+-S "$SAMPLES"} -Ou - \
      | bcftools +fill-tags -Ou - -- -t AC,AN,F_MISSING \
      | bcftools filter -m+ -s HighMissing -e "F_MISSING>$MAX_MISSING" -Ou - \
      | bcftools view -G -Oz -o "$out.partial" -
    # a truncated download ends the pipe with an error (pipefail), so a file
    # that reaches this line is complete
    mv "$out.partial" "$out"
    log "  $(bcftools view -H "$out" | wc -l) records"
  else
    out="$OUT/$c.bcftools.tsv"
    if [[ -s "$out" ]]; then log "have $(basename "$out")"; continue; fi
    log "validation table $c (streaming)"
    source_vcf "$c" \
      | bcftools view ${SAMPLES:+-S "$SAMPLES"} -Ou - \
      | bcftools +fill-tags -Ou - -- -t F_MISSING \
      | bcftools view "${TABLE_FILTER[@]}" -Ou - \
      | bcftools norm -m+any -Ou - \
      | bcftools +fill-tags -Ou - -- -t AC,AN,MAF \
      | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t%INFO/AN\t%INFO/MAF\n' - > "$out.partial"
    mv "$out.partial" "$out"
    log "  $(wc -l < "$out") positions"
  fi
done

cat <<EOF

Done ($MODE: $*). Lines for spectrum.env:
  FULL_SITES_TEMPLATE=$OUT/{contig}.sites.vcf.gz
  FULL_TABLE_TEMPLATE=$OUT/{contig}.bcftools.tsv
EOF
