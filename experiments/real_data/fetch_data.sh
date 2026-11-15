#!/usr/bin/env bash
# Download the public inputs for run_plan.sh (GRCh38) into a data directory.
#
#   fetch_data.sh DATA_DIR [pilot|all|refs]     (default: pilot)
#
# pilot = chr1, chr17, chr19, chr20, chr21, chr22 (the chromosomes in
# pilot.env.example); all = chr1-22 (the chromosomes in
# chr_map_grch38_refseq.tsv); refs = no VCFs, only the accessibility mask,
# RefSeq genome/annotation and phastCons (enough for experiments/spectrum,
# which takes its variants from fetch_fullcallset.sh). Run it on a login or transfer
# node (Great Lakes compute nodes may not have internet). Downloads resume
# (curl -C -), files that are already complete are skipped, and any failure
# stops the script. Total size: roughly 5 GB pilot, 30 GB all (mostly VCFs).
#
# Every URL can be overridden from the environment (e.g. KGP_BASE=...), in
# case a host moves files. Afterwards it prints the plan.env lines to use and
# the steps that cannot be automated (the de novo table).
set -euo pipefail

[[ $# -ge 1 ]] || { sed -n '2,15p' "$0"; exit 1; }
D=$(mkdir -p "$1" && cd "$1" && pwd)
SET=${2:-pilot}
case $SET in
  pilot) CHRS="chr1 chr17 chr19 chr20 chr21 chr22" ;;
  all)   CHRS=$(seq -f 'chr%g' 1 22 | tr '\n' ' ') ;;
  refs)  CHRS="" ;;
  *) echo "second argument must be pilot, all or refs" >&2; exit 1 ;;
esac

KGP_BASE=${KGP_BASE:-https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV}
[[ -n "${KGP_FILE:-}" ]] || KGP_FILE='1kGP_high_coverage_Illumina.{contig}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz'
MASK_URL=${MASK_URL:-https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/working/20160622_genome_mask_GRCh38/StrictMask/20160622.allChr.mask.bed}
REFSEQ_BASE=${REFSEQ_BASE:-https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers}
PHASTCONS_URL=${PHASTCONS_URL:-https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/phastConsElements100way.txt.gz}

log() { echo "[$(date +%H:%M:%S)] $*"; }
need() { command -v "$1" > /dev/null || { echo "missing tool: $1" >&2; exit 1; }; }
need curl; need gzip

# fetch URL DEST: resumable download to DEST.partial, renamed when curl succeeds
fetch() {
  local url=$1 dest=$2
  if [[ -s "$dest" ]]; then log "have $(basename "$dest")"; return; fi
  log "get  $url"
  mkdir -p "$(dirname "$dest")"
  curl -fL --retry 5 --retry-delay 10 -C - -o "$dest.partial" "$url" \
    || { echo "download failed: $url (rerun to resume; set the *_BASE/_URL variable if the file moved)" >&2; exit 1; }
  mv "$dest.partial" "$dest"
}

# 1. 1000 Genomes high-coverage phased VCFs (3,202 samples) and indexes
for c in $CHRS; do
  f=${KGP_FILE//\{contig\}/$c}
  fetch "$KGP_BASE/$f" "$D/vcf/$f"
  fetch "$KGP_BASE/$f.tbi" "$D/vcf/$f.tbi"
done

# 2. 1kGP strict accessibility mask
fetch "$MASK_URL" "$D/mask/$(basename "$MASK_URL")"

# 3. RefSeq GRCh38 genome and annotation (NC_ names, as in the original repo)
for f in GRCh38_latest_genomic.fna GRCh38_latest_genomic.gff; do
  if [[ ! -s "$D/refseq/$f" ]]; then
    fetch "$REFSEQ_BASE/$f.gz" "$D/refseq/$f.gz"
    log "unpack $f"
    gzip -dc "$D/refseq/$f.gz" > "$D/refseq/$f.partial" && mv "$D/refseq/$f.partial" "$D/refseq/$f"
  fi
done
if command -v samtools > /dev/null && [[ ! -s "$D/refseq/GRCh38_latest_genomic.fna.fai" ]]; then
  samtools faidx "$D/refseq/GRCh38_latest_genomic.fna"
fi

# 4. UCSC phastCons 100-way conserved elements -> 3-column BED
if [[ ! -s "$D/phastcons/phastConsElements100way.bed" ]]; then
  fetch "$PHASTCONS_URL" "$D/phastcons/phastConsElements100way.txt.gz"
  gzip -dc "$D/phastcons/phastConsElements100way.txt.gz" | cut -f2-4 \
    > "$D/phastcons/phastConsElements100way.bed.partial"
  mv "$D/phastcons/phastConsElements100way.bed.partial" "$D/phastcons/phastConsElements100way.bed"
fi

# Checks: every file present and non-empty; gzip files intact
bad=0
shopt -s nullglob   # the .gz copies may have been deleted after unpacking
for f in "$D"/vcf/*.vcf.gz "$D"/refseq/*.gz "$D"/phastcons/*.gz; do
  gzip -t "$f" 2> /dev/null || { echo "corrupt: $f (delete it and rerun)" >&2; bad=1; }
done
grep -q '^NC_000001' <(grep -m1 '^>' "$D/refseq/GRCh38_latest_genomic.fna" | tr -d '>') \
  || { echo "unexpected FASTA names (expected NC_...)" >&2; bad=1; }
[[ $(head -1 "$D/phastcons/phastConsElements100way.bed" | cut -f1) == chr* ]] \
  || { echo "unexpected phastCons BED format" >&2; bad=1; }
(( bad == 0 )) || exit 1

cat <<EOF

Done ($SET). Lines for plan.env / pilot.env:
  FASTA=$D/refseq/GRCh38_latest_genomic.fna
  GFF=$D/refseq/GRCh38_latest_genomic.gff
  VCF_TEMPLATE=$D/vcf/$KGP_FILE
  CALLABLE_BED=$D/mask/$(basename "$MASK_URL")
  PHASTCONS_BED=$D/phastcons/phastConsElements100way.bed
  DNM_SETS="iceland_2017=$D/dnm/jonsson2017.grch38.tsv"

Manual step (the de novo table is a journal supplement, so download it in a browser):
  1. Jonsson et al. 2017, "Parental influence on human germline de novo mutations
     in 1,548 trios from Iceland", Nature 549:519-522, doi:10.1038/nature24018.
     Download Supplementary Table 4 (the list of 108,778 DNMs, hg38) from the
     article's Supplementary Information and copy it to $D/dnm/.
  2. python $(dirname "$0")/prepare_dnm.py $D/dnm/<supplementary file> \\
       -o $D/dnm/jonsson2017.grch38.tsv
     It prints the columns it used and the counts; check the position column is
     the hg38 one and that roughly 100,000 SNVs are kept.
Optional, descriptive only: denovo-db SSC siblings (needs approval for non-autism
research; hg19, lift over first), or Gnocchi constraint scores (gnomAD v3.1.2,
https://gnomad.broadinstitute.org/downloads#v3-genomic-constraint) for comparison.
EOF
