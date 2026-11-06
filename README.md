# Experimental allele-frequency extension (not part of the published model)

Adds a per-nucleotide U-Net that predicts the **frequency class** of standing variation, leaving the original research scripts untouched. Per-site labels:

- `0`: callable site, no variant allele observed in the cohort
- `1`: rare polymorphism (minor allele frequency < 1%, configurable)
- `2`: common polymorphism (minor allele frequency >= 1%)
- `-1`: ignore (ambiguous base, uncallable, failed FILTER, malformed record)

"Minor allele frequency" is the frequency of the second most common allele, REF included; this is also how `bcftools +fill-tags` defines MAF. A site whose REF allele is at 0.5% is therefore *rare*. A record with `AC=0`, or one where every called allele is the same ALT (the reference genome carries an allele absent from the cohort), is *not* a polymorphism.

Two output heads (`--head`):

- `softmax` (default): P(none), P(rare), P(common).
- `factorized` (recommended): P(variant) and P(common | variant). The first output is the same target as the original binary model, so it can be benchmarked against it directly. The second is trained only at variant sites.

Both are converted to the same three class probabilities by `predict.py`. The per-category observed/expected ratios are **experimental diagnostics**, not proven tests of selection.

## The key caveat: selection can leak into the baseline

Observed allele frequencies are shaped by selection. If constrained regions share learnable sequence features, a model trained on the whole genome learns "this kind of sequence has few, rare variants" as part of its *baseline*. That hides the selection O/E is meant to detect. The original binary model has the same exposure; it is stronger for frequency, because a shift toward rare alleles is exactly the signal.

The extension therefore supports **neutral-region training and calibration** (`--mask-scope`, `calibrate.py --mask`). Comparing these settings on real data is the main experiment to run before making claims. See the synthetic demonstration at the end.

## Dependencies

Python 3.10+, NumPy, TensorFlow 2.x (tested with TF 2.21 / Keras 3). Labeling, masks, calibration, scoring and most tests need NumPy only. A GPU is strongly advised for chromosome-scale training. Genome, VCF and callable-site data are not included; `experiments/real_data/fetch_data.sh` downloads them (needs curl), and `prepare_dnm.py` needs openpyxl to read an .xlsx supplement. Never treat sites absent from a VCF as invariant without callability information. For 1000 Genomes, the accessibility ("strict") mask works as the callable BED.

## Step 1: label each chromosome

Reference, VCF and BED must use the **same genome build and contig names**. The VCF needs per-ALT `AC` and site `AN` in INFO. Strip genotypes first, and split multi-allelic sites:

```bash
bcftools view -G -r chr1 1kGP.chr1.vcf.gz | bcftools norm -m- -f GRCh38.fa -Oz -o chr1.sites.vcf.gz

python allele_frequency_extension/data.py \
  --fasta GRCh38.fa --vcf chr1.sites.vcf.gz \
  --callable-bed callable.bed --chromosome chr1 \
  --threshold 0.01 --output-prefix arrays/chr1
```

Outputs: `arrays/chr1_dna.npy`, `arrays/chr1_frequency_labels.npy`.

- `--indel-mode anchor` (default) labels only the VCF POS base of an indel. `--indel-mode span` also labels every REF base a deletion covers. `--indel-mode original` reproduces the original repo's labeled positions exactly: a record whose ALT field is as long as REF marks POS only; any other record marks POS plus len(REF) further bases. It was checked against the original two-step code on SNVs, deletions, insertions, MNPs and multi-allelic records. For full fidelity, don't split multi-allelic records first; the original read them as they appear in the VCF.
- `compare_labels.py` counts, per chromosome, where these labels differ from the original repo's `<seqid>_label.pkl` files: both variant, original only, extension only, and excluded. Report these counts alongside any comparison with the original model.
- Non-PASS records make the site `-1` by default. `--keep-non-pass` uses them anyway, as the original repo did.
- Records at the same position are merged before classification, so record order does not matter. A valid record wins over a malformed one. Records at one position must report the same `AN`; if they don't, they come from different call sets and the site is set to `-1`.
- `*` (spanning-deletion) alleles reduce the REF count but are not counted as variation at that site. bcftools counts them as an allele, so the two can disagree at such sites.

### Validate the labels against bcftools

`validate_labels.py` compares the labels with an independent bcftools calculation from the genotypes. Run it on at least one real chromosome before training:

```bash
bcftools view -r chr1 -f PASS,. 1kGP.chr1.vcf.gz \
  | bcftools norm -m+any -f GRCh38.fa \
  | bcftools +fill-tags -- -t AC,AN,MAF \
  | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t%INFO/AN\t%INFO/MAF\n' > chr1.bcftools.tsv
python allele_frequency_extension/validate_labels.py --labels arrays/chr1_frequency_labels.npy \
  --table chr1.bcftools.tsv --chromosome chr1 --disagreements chr1_disagreements.tsv
```

It prints a 3×3 agreement table (data.py vs bcftools) and writes every disagreeing position for inspection. On random genotype VCFs with tri-allelic sites, deletions, missing calls, rare and fixed REF alleles and MAFs at the 1% boundary, agreement was 100% (9,000 sites, bcftools 1.19). This check is also a unit test whenever bcftools is installed. It is how the fixed-ALT bug described above was found.

## Step 2: neutral masks (optional, recommended)

A mask is a boolean `.npy` aligned to the chromosome; True = putatively neutral. `make_mask.py` builds one from any BED. The original repo's "unannotated" definition (all GFF features not sourced `RefSeq`) becomes:

```bash
awk -F'\t' '!/^#/ && $1=="NC_000001.11" && $2!="RefSeq" {print $1"\t"$4-1"\t"$5}' \
    GRCh38_latest_genomic.gff > annotated_chr1.bed
python allele_frequency_extension/make_mask.py --bed annotated_chr1.bed --chromosome NC_000001.11 \
    --labels arrays/chr1_frequency_labels.npy --invert --out masks/chr1_neutral.npy
```

(Contig names in the BED must match `--chromosome`. Rename them if your FASTA uses `chr1`.)

## Step 3: manifest

```json
{
  "chr1": {"dna": "arrays/chr1_dna.npy", "labels": "arrays/chr1_frequency_labels.npy",
           "mask": "masks/chr1_neutral.npy"},
  "chr2": {"dna": "arrays/chr2_dna.npy", "labels": "arrays/chr2_frequency_labels.npy",
           "mask": "masks/chr2_neutral.npy"}
}
```

`mask` is needed only when training with `--mask-scope all|common`.

## Step 4: train

Reserve one chromosome for **validation** (early stopping), one for **calibration**, and leave the chromosomes you score untouched. Example mirroring the original odd/even split:

```bash
for seed in 0 1 2 3 4; do
  python allele_frequency_extension/train.py --manifest manifest.json \
    --train-chromosomes chr5 chr7 chr9 chr11 chr13 chr15 chr17 chr19 chr21 \
    --val-chromosomes chr3 --head factorized --mask-scope all \
    --seed $seed --out models/neutral_s$seed.keras
done
```

`--mask-scope`:

| scope | P(variant) learns from | P(common \| variant) learns from |
|---|---|---|
| `none` (default) | all callable sites | all variants |
| `all` | masked (neutral) sites only | masked variants only |
| `common` (factorized only) | all callable sites | masked variants only |

With a mask, the network still sees the **full sequence as input context**; only the loss is restricted. The original repo instead zeroed masked input and dropped those windows. Validation uses the same scope, so `val_loss` measures neutral-site fit.

Other details:
- Each training window starts at a random position on every draw, as in the paper. Chromosomes are drawn in proportion to their usable length. Windows with an ambiguous base are never used.
- Defaults follow the paper: 2048-bp window, batch 16, 1,000 steps/epoch, up to 50 epochs (early stopping, `--patience 4`).
- **Class weighting is off by default.** If you use `--class-weights` (softmax) or `--variant-pos-weight` / `--common-pos-weight` (factorized), `predict.py` undoes it using `<out>.meta.json`. Keep that file next to the model.

## Step 5: predict, calibrate, score

```bash
M="models/neutral_s0.keras models/neutral_s1.keras models/neutral_s2.keras models/neutral_s3.keras models/neutral_s4.keras"

# 1. Uncalibrated ensemble prediction on the calibration chromosome, then fit
python allele_frequency_extension/predict.py --model $M --dna arrays/chr1_dna.npy --output pred/chr1.npy
python allele_frequency_extension/calibrate.py --labels arrays/chr1_frequency_labels.npy \
    --predictions pred/chr1.npy --mask masks/chr1_neutral.npy --out cal_neutral.json

# 2. Calibrated ensemble prediction on chromosomes to analyse
python allele_frequency_extension/predict.py --model $M --dna arrays/chr2_dna.npy \
    --output pred/chr2.npy --calibration cal_neutral.json

# 3. Observed/expected with 95% block-bootstrap intervals, e.g. inside CDS
python allele_frequency_extension/score.py --labels arrays/chr2_frequency_labels.npy \
    --predictions pred/chr2.npy --mask masks/chr2_cds.npy
```

**Prediction.** Output is one `(n_sites, 3)` float32 `.npy` (P(none), P(rare), P(common)) written through a memory map. Intermediates live on disk and all post-processing runs in 5 Mb chunks, so allocated memory stays well under 2 GB even for chr1. Windows overlap with a W/4 step and only the central W/2 is kept, as in the original `predict_by_chrom_fast.py`. Chromosome ends are covered. Only positions reached solely by windows containing an `N` are NaN.

**Ensembles.** Pass several `--model` files. Each is converted with its own `.meta.json`, then the models are averaged, and calibration is applied last. Fit the calibration on the same ensemble's uncalibrated output.

**Calibration.** Isotonic maps are fitted separately for P(variant) and P(common | variant) on a subsample of sites (`--max-sites`), read in chunks. With `--mask`, only neutral sites are used. Fit both versions and compare them: the gap shows how much selected regions move the baseline.

**Scoring.** `score.py` reports `oe_any` (≈ the original OVEP), `oe_rare`, `oe_common`, and `oe_common_share`: observed common variants divided by the expected number given where variants occurred, which is the sum of P(common | variant) over the observed variant sites. Purifying selection shifts variants toward rare frequencies, so `oe_common_share` < 1 under constraint. Conditioning on the observed variant sites keeps errors in the mutability model out of it. An earlier, unconditional version (still reported as `oe_common_share_marginal`) averaged over all sites weighted by predicted mutability; on the dry run it expected a 34.5% common share at neutral sites against 37% observed. Variants concentrate where true mutability is high, and those sites carry more common variants than the model's weighting allows. Sanity check S3 caught this. `ci95` intervals resample contiguous blocks (`--block-size`, default 100 kb). They cover sampling noise in the scored region only, not seed or calibration uncertainty. For those, score each ensemble member and each calibration choice separately. `.npz` outputs from earlier versions are still accepted.

## Step 6: compare against the original OVEP model

`compare_ovep.py` scores the paper's binary predictions and this extension's P(variant) against the **same labels, sites, masks and genomic blocks**. It reports both OVEPs, their paired difference with a 95% interval, site-level AUC for each, and `oe_common_share`:

```bash
python allele_frequency_extension/compare_ovep.py --labels arrays/chr2_frequency_labels.npy \
  --original calibrated_predictions/NC_000002.12.pred.npy \
  --original-raw code_halftrainmaskmissing/base_excludeN_final_extrapolate/NC_000002.12.pred.npy \
  --new pred/chr2.npy --mask cds=masks/chr2_cds.npy --mask unannotated=masks/chr2_neutral.npy
```

`--original-raw` (the uncalibrated original prediction) lets the script drop sites the original pipeline never predicted. Those are the first 512 bp and the final partial batch of each chromosome, which are 0 before calibration. To match the original label definition, build the labels with `--indel-mode span --keep-non-pass`. The original files are named by RefSeq accession (`NC_…`); make sure the arrays describe the same chromosome.

`--original` can also be another model's `(n, 3)` output from `predict.py`, e.g. the whole-genome configuration as the reference for the neutral one. Add `--save-blocks` to write per-block sums that can be pooled across chromosomes. Every ratio comes with a bootstrap standard error and z = (ratio − 1) / SE.

## Step 7: per-category scores and de novo benchmark

- **`score_annotations.py`** reproduces the paper's per-category analysis (Figure 5) straight from the GFF. It scores every feature type and the `unannotated` remainder, plus any masks you add, for a reference and a new model. GFF intervals are read lazily, so no mask files are needed. Run it per chromosome with `--save-blocks`, then `--pool` the files into one genome-wide JSON/TSV table.
- **`denovo_auc.py`** reproduces the de novo benchmark (Figure 3). It scores several models on the same de novo and background sites, from a small JSON config. It reports AUC with the paper's Hanley-McNeil SE, plus DeLong's paired test for each pair of models. Hanley-McNeil treats the two AUCs as independent; on a test set of 4,000 sites it overstated the paired SE about 2.3× compared with DeLong and a bootstrap, which agreed (0.0094 vs 0.0093).

## Step 8: the pre-registered analysis plan

- **`discriminate.py`** (endpoint P2) asks whether frequency adds discrimination beyond OVEP. CDS and neutral sequence are cut into units of N scorable sites (`score_annotations.py --unit-sites N`). Each unit gets an OVEP z and a frequency z, both standardized to the unit's own expectation. Before they are added, each is put on the neutral scale: centred and divided by its mean and SD on the neutral (control) units. Real allele frequencies are overdispersed relative to the binomial; in a simulation with overdispersion the neutral SD of the frequency z was 3.97, and the raw sum lost most of OVEP's discrimination (AUC 0.63 vs 0.87), while the scaled sum kept most of it (0.81). Only control units are used, so nothing is fitted to the CDS units. The sum is compared with OVEP alone; all AUCs are reported, including the raw sum, and `neutral_calibration` gives each z's neutral mean, SD, quantiles and share beyond ±1.96. The primary 95% intervals come from a bootstrap over whole genomic blocks (`--block-bp`, default 1 Mb), because neighbouring units are not independent; DeLong's paired test is secondary. With fewer than 30 blocks there is no block interval. In simulations where the frequency signal varied by region, DeLong's interval was 1.6× too narrow, and the two agreed when units were independent. Per-chromosome AUC differences and the correlation between the two scores are reported to show stability. The primary analysis uses all units. A `matched` analysis (criterion R3) repeats it on CDS and neutral units of equal expected variant count, keeping equal numbers per bin and dropping CDS units without comparable controls. It reports an error instead of a number when too few match. In simulations the difference was −0.010 (CI −0.026 to +0.007) with no signal, −0.079 when constraint only removes variants, and +0.030 to +0.40 when frequencies shift.

`experiments/real_data/` turns the analysis plan into commands:

- `criteria.json` (v1.6) holds the success criteria: primary endpoints P1–P3, guardrails G1–G2 (de novo AUC on the Icelandic DNMs of Jónsson et al. 2017), sanity checks S1–S4 (S3: the expected common share is calibrated on unannotated sites; S4: the frequency z is not grossly overdispersed on neutral units), robustness checks R1a–R1c and R2–R4, and the decision rules for H1 and H2. Its SHA-256 is recorded in `criteria.sha256`. `evaluate_plan.py` reports whether the file still matches, and `evaluate_plan.py --freeze` re-records it; only do that before real results exist. Commit and tag both files, or upload them to OSF, before the first real-data step.
- **Getting the data.** `fetch_data.sh DATA_DIR [pilot|all]` downloads the public inputs into one directory: the 1000 Genomes high-coverage phased VCFs (3,202 samples) with indexes, the GRCh38 strict accessibility mask, the RefSeq GRCh38 FASTA and GFF, and UCSC phastCons 100-way elements (converted to a 3-column BED). `pilot` fetches the six pilot chromosomes, about 5 GB; `all` fetches chr1–22. Downloads resume after an interruption, finished files are skipped, gzip files are integrity-checked, and any failed download stops the script with the URL. Every URL can be overridden from the environment if a host moves a file. At the end it prints the `plan.env` lines to paste. Run it on a login or transfer node. It was tested against local copies of the files (including a resumed partial download and a bad URL); the real hosts could not be reached from where it was written, so check the first run.
- **De novo benchmark.** The required set is the Icelandic DNMs of Jónsson et al. 2017 (Nature 549:519; 108,778 DNMs from 1,548 trios, on hg38, with no stated access restriction). The table is a journal supplement (Supplementary Table 4), so download it in a browser, then run `prepare_dnm.py <table> -o jonsson2017.grch38.tsv`. It reads .xlsx, .csv or .tsv (optionally gzipped), skips title rows, finds the chromosome and position columns (preferring hg38 ones; or give `--chrom-col`/`--pos-col`), keeps SNVs, normalizes names to chr1…chrY, removes duplicates, and prints what it did. `DNM_SETS="iceland_2017=… other=…"` in `plan.env` lists the sets; only `iceland_2017` enters G1 and G2, others are reported descriptively. The paper's denovo-db SSC set can be added as an extra, but its terms limit use to autism-related research and its positions are hg19, so get approval and lift it over first. The older `DNM_DENOVO_DB`/`DNM_PLATINUM` settings still work.
- `run_plan.sh plan.env [steps]`: runs every step: labels, masks, validate, train, predict, score, denovo, evaluate. A step skips an output only if it exists and is newer than every input it was built from: the source VCF, sample list, FASTA, GFF, callable and phastCons BEDs, labels, masks, models and predictions, as applicable. Every script writes under a temporary name and renames on completion. Changes to settings in `plan.env` (e.g. `TRAIN_ARGS`) are not tracked; delete the affected outputs. So an interrupted run resumes cleanly, and a rebuilt input (e.g. a retrained model) redoes exactly what depends on it and nothing else. This was tested two ways. First, a prediction was killed mid-chromosome: only temporary files were left, and the resumed run rebuilt the file byte-identically and gave an identical criteria report. Second, a deleted model file was retrained and invalidated only its own predictions and scores; a further rerun did no work. Copy `plan.env.example`, which defines the odd/even chromosome roles mirroring the original repo, and fill in the paths. Its defaults match the original model's labels: all 3,202 samples, FILTER ignored, `INDEL_MODE=original`.
- `slurm/submit_plan.sh plan.env slurm.env` submits the plan to SLURM (e.g. Great Lakes) as four stages chained with `afterok`. The stages are prep on CPU (labels, masks, validation), a GPU training array with one task per half × configuration × seed, a GPU prediction array with one task per half × configuration, and post on CPU (scores, discrimination, de novo, criteria report). Copy `slurm/slurm.env.example` and set your account, partitions, setup commands (bcftools and a Python with TensorFlow ≥ 2.13) and resources. `DRY_RUN=1` writes the job scripts without submitting. Resubmitting after a failure redoes only what is missing. The generated job scripts were run locally in dependency order on the synthetic fixture, task by task, and reproduced the full pipeline.
- `pilot.env.example` runs a small pilot before the full run: half 1 only, train on chr17 and chr19, test on chr20 and chr22, 2 seeds. It reuses the paths in your `plan.env`.
- `evaluate_plan.py`: checks the results against `criteria.json` and writes `criteria_report.md`, one PASS / FAIL / MISSING row per criterion plus the decisions. A missing value never counts as a pass.
- `dryrun/dryrun.sh`: runs the whole plan on a small synthetic fixture that mimics every real input. The fixture has genotype VCFs under different contig names, a RefSeq-style GFF, a callable BED, a phastCons-style BED, de novo lists and fake original-model predictions. Run it before the real thing to check your environment (TensorFlow + bcftools).

## Tests

```bash
python -m unittest discover -s allele_frequency_extension/tests -v
```

- `tests/test_extension.py` (38 tests) checks the components. It runs with NumPy alone; the bcftools and TensorFlow cases skip automatically when those tools are missing.
- `tests/test_plan_tools.py` (20 tests, NumPy only; openpyxl for one) checks the plan tooling. It covers interval masks against dense masks, pooling against a single pass, another model as the reference, and saving and loading block sums. It also covers GFF category masks, DeLong against a brute-force calculation, a toy de novo benchmark, the criteria checker (pass, fail, missing, k-of-n) and freezing. On the discrimination endpoint it tests null, depletion-only and frequency-shift scenarios. It also checks that `--indel-mode original` matches the original code, that label comparison works, that `prepare_dnm.py` handles a supplement-style table (title rows, hg19 and hg38 columns, duplicates, indels, chrM) and headerless or 0-based input, and that the shipped criteria still match their frozen hash.
- `tests/test_integration.py` runs the real command-line scripts end to end on small synthetic chromosomes, in about 1.5 minutes on CPU. It trains factorized models with `--mask-scope all` and `common` and a class-weighted softmax model, then saves and reloads them. It then runs a held-out ensemble prediction (checking shapes, NaN gaps, and that probabilities sum to 1), confirms the class-weight correction is applied exactly, and runs neutral-masked calibration, calibrated prediction and scoring. It needs TensorFlow; it was run with TF 2.21 / Keras 3.

The synthetic selection experiment below is fully reproducible: see `benchmarks/synthetic_selection/`.

The component tests cover:
- labeling edge cases: AC=0, rare and fixed REF/ALT alleles, conflicting AN, record order, split multi-allelics, `*` alleles, FILTER, indel modes, contig mismatch;
- agreement with bcftools on random genotype VCFs;
- masks: BED to mask, mask scopes, mask-aware window validity, masked calibration;
- training batches and window sampling;
- prediction: tiling into disk memmaps, ensemble averaging, NaN propagation, weight correction;
- isotonic calibration, chunked sampling, and chunked scoring (identical to unchunked);
- `.npy` and `.npz` input, and bootstrap intervals;
- the OVEP comparison: tie-aware AUC, paired intervals, alignment of the shorter original predictions;
- TensorFlow: loss masking incl. scope `common`, one training step per head.

## Synthetic demonstration of baseline contamination

A 5-chromosome synthetic genome (400 kb each) was built with:
- neutral mutability where CpG sites are 8× more mutable and their variants skew common;
- 20% of sequence in GC-rich "functional" blocks under purifying selection: half of variants removed, P(common) × 0.3.

The truth inside functional blocks is therefore `oe_any` ≈ 0.49 and `oe_common_share` ≈ 0.29. Factorized models were trained on 2 chromosomes, validated on 1, calibrated on 1, and scored on the 5th, inside the functional blocks:

| training scope | calibration | functional `oe_any` (95% CI) | functional `oe_common_share` (95% CI) | neutral `oe_any` |
|---|---|---|---|---|
| oracle (true neutral rates) | — | 0.49 (0.47–0.51) | 0.29 (0.27–0.32) | 1.00 |
| `none` (whole genome) | whole chr | 0.75 (0.71–0.78) | 0.90 (0.82–0.98) | 1.06 |
| `none` | neutral | 0.60 (0.57–0.63) | 0.72 (0.66–0.79) | 0.99 |
| `common` (2-seed ensemble) | whole chr | 0.97 (0.93–1.02) | 0.57 (0.53–0.62) | 1.00 |
| `common` (2-seed ensemble) | neutral | 0.95 (0.91–1.00) | 0.40 (0.36–0.43) | 0.99 |
| `all` | whole chr | 0.60 (0.57–0.62) | 0.37 (0.34–0.39) | 1.15 |
| `all` | neutral | **0.48 (0.46–0.50)** | **0.30 (0.27–0.32)** | 0.99 |

Reproduce with `bash benchmarks/synthetic_selection/run.sh` (about 8 minutes on CPU); the expected output is in `benchmarks/synthetic_selection/expected_output.txt`.

What this shows, and what it doesn't:
- Whole-genome training absorbed most of the selection. The frequency signal nearly vanished (0.89 vs a true 0.30).
- Neutral training **and** neutral calibration were both needed to recover the truth. Calibrating on the whole chromosome re-imports the selected regions.
- Scope `common` fixes only the frequency term. Its P(variant) is still trained genome-wide, so `oe_any` absorbs depletion.
- This genome is a **worst case**: functional sequence has a strong, learnable composition signature, and neutral mutability is a simple local rule that extrapolates perfectly into functional regions. In real genomes the leak may be smaller. Neutral-only training may also generalise worse, which the paper saw as lower de novo AUC for its masked model. Which effect dominates is an empirical question for real data.
