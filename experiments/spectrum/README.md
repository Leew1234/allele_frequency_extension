# Frequency-spectrum analysis (plan v2)

The v1 analysis asked whether a rare/common label improves on OVEP at telling
coding from neutral sequence. It did not. This analysis asks a different
question: can the allele frequency spectrum separate **mutation rate** from
**purifying selection**, which the observed/expected variant count mixes
together?

The idea is the one behind MAPS (Lek et al. 2016). Selection needs time, so it
removes common variants much more than new ones, and a singleton is close to
a readout of mutation alone. The share of variants that are singletons rises
where selection is acting. It also falls at highly mutable sites for a reason
that has nothing to do with selection (the same site mutates twice), so it has
to be adjusted for mutability. MAPS adjusts with the trinucleotide context.
Here the adjustment comes from the network, which reads 2,048 bp.

## Read this first: the v1 data has no singletons

The phased panel that `fetch_data.sh` downloads was filtered before phasing:
"we excluded all sites with minor allele count (MAC) < 2, as singletons are
not informative for phasing" (Byrska-Bishop et al. 2022, Methods). So:

- This analysis needs the **full call set**. `fetch_fullcallset.sh` streams it
  and keeps only a sites-only file per chromosome.
- In v1, a site whose only variant is a singleton was labeled "no variant",
  and "rare" meant 2 copies up to 1%. This does not change the v1 conclusions,
  but it should be stated when v1 is presented. Whether the published model's
  training data included singletons has not been checked; its label files
  are the place to look.
- The two call sets are filtered differently, so v1 numbers are recovered only
  approximately by pooling the classes here.

## Classes

| Class | Minor allele count | Note |
| --- | --- | --- |
| 0 none | 0 | |
| 1 singleton | 1 | |
| 2 ultra-rare | 2-5 | |
| 3 rare | 6 up to a frequency of 1% | |
| 4 common | frequency 1% or more | same boundary as v1 |

The minor allele is the second most common allele at the site, REF included,
so no ancestral allele is needed. The count boundaries are `COUNT_BINS`.

Two labeling choices differ from v1, both set in `spectrum.env`:

- `SNV_ONLY=1`: only single-nucleotide variants are labeled; every base under
  an indel, MNP or symbolic record is ignored (-1). Singleton indel calls are
  the least reliable genotypes, and coding sequence has fewer indels than
  neutral sequence, which would otherwise change the class mix for a reason
  that is not the one being measured. MAPS is SNV-only for the same reason.
- `KEEP_NON_PASS=0`: records failing the call set's VQSR filter, or with more
  than 5% missing genotypes, become ignored sites. v1 kept them because the
  original repository did.

## Model

`train.py --head spectrum`: the same U-Net with two outputs per site,
P(variant) and a four-way softmax P(class | variant). The first is the
original model's target. The softmax is trained at variant sites only.
Calibration (`calibrate.py`) fits P(variant) as before and the class
distribution through its cumulative tails, so that P(singleton | variant) and
P(common | variant) are each calibrated exactly as a binary event.

## The neutral set

The model is trained and calibrated on **mask AB**: sequence that is
unannotated in the GFF (mask A, v1's "neutral") **and** outside phastCons
conserved elements padded by 100 bp (mask B). Mask A alone still contains
conserved sequence that no annotation covers, and training on it lets that
constraint into the neutral expectation: on the synthetic fixture, where 6%
of the unannotated sequence is conserved, strictly neutral sites then scored
0.92 to 0.94 instead of 1. `SPECTRUM_CONFIGS` selects the set (`neutralAB`,
`neutralA`, `neutralB`).

## The baseline it has to beat

`kmer_baseline.py` builds the MAPS-style adjustment: for every k-mer context
(k = 3 is the trinucleotide), the variant rate and the class distribution of
variants, counted on the same neutral sites the network trains on, and
calibrated on the same held-out chromosome. Its output has the same format as
the network's, so both are scored by the same code on the same sites.

## Scores

`spectrum_score.py` reports, per sequence category:

- `oe_share.singleton`: observed singletons / the number expected given where
  variants were observed. Above 1 = more singletons than the mutability-adjusted
  neutral expectation.
- `singleton_share.excess`: observed minus expected singleton share (the
  MAPS-like difference).
- `oe_any`: the original OVEP.
- mutually exclusive tiers: `tier_cds`, `tier_conserved_noncoding` (phastCons
  outside CDS), `tier_enhancer`, `tier_neutral`, and paired contrasts between them.
- `mutability_bins`: on neutral sequence, the adjusted singleton share in each
  decile of predicted mutability. A measure that still depends on mutability
  is not flat here.
- `vs_baseline.nll_gain`: how much better the network predicts the frequency
  class of a neutral variant than the k-mer table (a proper scoring rule).

All intervals are block bootstraps over 100 kb blocks, paired across
categories and models.

## Criteria

`criteria_v2_draft.json` is a **draft and is not frozen**. Three criteria
differ from the first plan document, and Yang should agree before freezing:

- **F2** is a partial order (CDS and conserved non-coding above enhancers,
  every constrained tier above neutral), not a strict four-step chain. Whether
  an average CDS site or an average phastCons non-coding site is more
  constrained is not established; that contrast is reported, not tested.
- **F3** tests that the adjusted singleton share is flat across mutability
  bins on neutral sequence. The earlier wording ("spectrum shape does not
  predict de novo mutations") could not pass: the expected spectrum must
  depend on mutability. Only the adjusted residual can be free of it.
- **F5** compares how well the network and the trinucleotide table predict
  the frequency class of neutral variants. The earlier wording (a larger
  constraint gradient) can be won by a biased adjustment.

To freeze: copy the draft to `criteria_v2.json`, set `version` and
`frozen_on`, run `python experiments/real_data/evaluate_plan.py --criteria
experiments/spectrum/criteria_v2.json --freeze`, commit, tag, and set
`CRITERIA=` in `spectrum.env`. Do this before any real-data result exists.

## Running it

Check it first on the synthetic fixture (about 15 minutes on a CPU):

```bash
bash experiments/spectrum/dryrun/dryrun.sh /tmp/spectrum_dryrun
```

On real data:

```bash
# 0. references (no VCFs): accessibility mask, RefSeq genome + annotation, phastCons
bash experiments/real_data/fetch_data.sh $DATA refs

# 1. sites-only VCFs from the full call set (streams; CPU; hours per large chromosome)
bash experiments/spectrum/fetch_fullcallset.sh $DATA sites all
bash experiments/spectrum/fetch_fullcallset.sh $DATA tables chr22     # for label validation

# 2. settings
cp experiments/spectrum/spectrum.env.example experiments/spectrum/spectrum.env   # then edit

# 3. CPU steps
bash experiments/spectrum/run_spectrum.sh experiments/spectrum/spectrum.env sites labels masks validate kmer

# 4. GPU steps: on a server with several GPUs and no scheduler
bash experiments/spectrum/local_gpus.sh experiments/spectrum/spectrum.env train
bash experiments/spectrum/local_gpus.sh experiments/spectrum/spectrum.env predict
#    ... or on Great Lakes (all stages, with dependencies):
bash experiments/spectrum/submit_spectrum.sh experiments/spectrum/spectrum.env experiments/real_data/slurm/slurm.env

# 5. scores and report
bash experiments/spectrum/run_spectrum.sh experiments/spectrum/spectrum.env score denovo evaluate
```

Results: `WORK/results/<half>/neutralAB_vs_kmer3.{json,tsv}` (the network, with
the trinucleotide baseline on the same sites), `..._vs_kmer7`, `denovo.json`,
`validation/`, and `criteria_report.md`.

**Disk.** Predictions are five float32 columns per site: about 5 GB for
chr1 and 26 GB for the ten test chromosomes of one half. Per half there is
one such set for the network, one per k-mer baseline, and (with
`SEED_PREDICTIONS=1`) five smaller single-seed sets, roughly 170 GB per half
and 350 GB in all with the defaults. `SEED_PREDICTIONS=0` saves about 150 GB
and `KMER_K="3"` about 60 GB. One-hot DNA, labels and masks add about 25 GB.

**Time.** Training is the same size as v1 (one model per half and seed).
Streaming the full call set is CPU-bound and has not been timed.

Stop after `validate` and read `WORK/results/validation/*.json`: agreement
with bcftools is reported per class, and the singleton and ultra-rare classes
are where a counting error would show.

## Before the main run: the calibration miss (v1 criterion S1)

Every score here is a ratio to an expected count from the calibrated model,
so the 7% miss on unannotated sequence in half 1 of v1 has to be understood
first. `diagnose_calibration.py` uses files the v1 run already has:

```bash
python diagnose_calibration.py blocks --blocks $V1_WORK/blocks/half1/neutral/*.npz --category unannotated
python diagnose_calibration.py sites --labels ... --predictions ... --mask ... --dna ...
```

The first shows whether the miss is on every test chromosome or a few; the
second breaks one chromosome down by predicted probability, region and GC.

## Other tools added with this analysis

- `compare_predictions.py`: per-site agreement of two prediction files, for
  checking the retrained original model against the published predictions.
- `python spectrum.py labels|predictions`: collapse spectrum labels or
  predictions to the three v1 classes, so any v1 tool can be run on them.

## Open choices to settle with Yang before freezing

0. Which call set was the published model trained on, and did it include
   singletons? It decides how the v2 P(variant) should be compared with it.

1. All 3,202 samples, or unrelated founders only (`SAMPLES=` in
   `fetch_fullcallset.sh`)? With parent-child trios in the data, a variant
   carried by a parent and a child has count 2, not 1. The scores stay valid
   either way (the neutral expectation is learned from the same data), but
   "singleton" is cleaner with founders.
2. Sequence tiers (available now) or variant consequences (loss-of-function
   > missense > synonymous; needs an annotation step that is not built)?
3. Which neutral set? The default trains on mask AB. Running `neutralA` and
   `neutralB` as well (`SPECTRUM_CONFIGS="neutralAB neutralA neutralB"`)
   shows how much the answer depends on the definition, at three times the
   training.
