# Reproducing the original OVEP model

The comparison endpoints (P2, R1c, R3, R4, S4, G2) need the original model's
predictions. The trained original models are not available on Great Lakes, so
this folder retrains them with the original repository's own settings, on the
same 1000 Genomes data the extension uses, and writes predictions in the layout
`run_plan.sh` expects (`HALFn_ORIG_DIR`, `HALFn_ORIG_RAW_DIR`).

## What is kept identical to the original

Taken from `code_halftrainmaskmissing/base_excludeN_final_extrapolate/` and
`.../base_excludeN_final_theotherhalf_extrapolate/`, and `data/`:

- **Labels** (`make_orig_labels.py`): binary; every VCF record marks 1 with no
  FILTER, allele-count or callability check; a record whose ALT field is as long
  as REF marks POS only, any other marks POS plus len(REF) further bases
  (`select_indel.py`, `pickle_label_chr.py`). Tested against a direct port.
- **Model** (`orig_model.py`): the same 1D U-Net (`cnn_batchnorm.cnn1d`):
  Conv1D(relu) then BatchNormalization in the encoder, no batch normalization in
  the decoder, one sigmoid output per base, 1,265,921 parameters; unmasked
  binary cross-entropy; Adam with default settings.
- **Training** (`train_orig.py`): batch 16, 1,000 steps per epoch, up to 50
  epochs, early stopping on validation loss (patience 2), best model kept;
  windows drawn by choosing a training chromosome uniformly and then a start
  uniformly; windows with no label dropped (so batches can be smaller than 16);
  windows with N kept (the original's check did nothing); validation of 20,000
  batches from the same pool; Python's random seeded with 3.14; five members
  per half.
- **Chromosome roles**: half 1 trains on chr3-21 (odd) and calibrates on chr1;
  half 2 trains on chr4-22 (even) and calibrates on chr2.
- **Prediction and calibration** (`predict_orig.py`): 2,048-bp windows, 512-bp
  step, central 1,024 bp kept, the five members averaged; isotonic calibration
  of label on score, on the calibration chromosome, using only 512-bp windows
  (step 256) with at least one label and no N (`calibration.py`). Tested
  against a direct port.

## What differs, and why it does not matter for the comparison

- Keras 3 / TensorFlow 2.21 instead of TensorFlow 2.7 (the old version does not
  run on the current GPUs), `.keras` model files, and the weight
  initialization is seeded with the member index so runs repeat.
- Input arrays come from `run_plan.sh` (one-hot order A, C, G, T rather than
  A, C, T, G); the model learns either coding.
- Overlapping windows are averaged correctly and chromosome ends are covered.
  The original divided by a miscounted denominator (a monotone distortion
  removed by calibration) and skipped the last partial chunk of each chromosome.
- The isotonic fit uses 1,000 quantile bins instead of every site.

This is a retrained reproduction, not the published model, and results should
say so. Its de novo AUC and its observed/expected values on CDS and V gene
segments should be compared with the paper's as a check of the reproduction.

## Running on Great Lakes

```bash
# 1. all chromosomes downloaded and labeled (CPU)
bash experiments/real_data/fetch_data.sh $DATA all
bash experiments/real_data/run_plan.sh plan.env labels
# 2. original labels, 10 trainings, prediction and calibration (GPU on spgpu)
DRY_RUN=1 bash reproduce_original/submit_orig.sh plan.env slurm/slurm.env
bash reproduce_original/submit_orig.sh plan.env slurm/slurm.env
# 3. point plan.env and pilot.env at the outputs
HALF1_ORIG_DIR=$WORK/orig_repro/half1/calibrated
HALF1_ORIG_RAW_DIR=$WORK/orig_repro/half1/raw
HALF2_ORIG_DIR=$WORK/orig_repro/half2/calibrated
HALF2_ORIG_RAW_DIR=$WORK/orig_repro/half2/raw
```

Tested end to end on the synthetic dry-run fixture: the reproduced reference
gave observed/expected 1.005 on a held-out chromosome after calibration.
