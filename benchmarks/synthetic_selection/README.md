# Synthetic selection benchmark

Tests whether training and calibrating on observed variation lets selection leak into the "neutral" baseline, and whether neutral masks prevent it. Everything is generated from scratch; no real data is needed.

**Genome** (`generate.py`, seed 11): five chromosomes of 400 kb each.
- Neutral process: CpG sites are 8× more mutable than other sites, and their variants are more often common.
- 20% of each chromosome lies in GC-rich 2 kb "functional" blocks under purifying selection: half of variants are removed and P(common) is multiplied by 0.3.
- Because the true neutral rates are known, `oracle.npz` gives the correct expectation. Inside functional blocks the true values are `oe_any` ≈ 0.49 and `oe_common_share` ≈ 0.30.

**Pipeline** (`run.sh`): the real command-line scripts, with a chromosome-disjoint split.
- Train on chrA and chrB, validate on chrC, calibrate on chrD, score chrE.
- Factorized head, window 256, 6 × 150 steps.
- Three training scopes (`none`, `common` as a 2-seed ensemble, `all`) × two calibrations (whole chromosome, neutral mask).

```bash
bash benchmarks/synthetic_selection/run.sh [work_dir]   # ~8 min on CPU, needs TensorFlow
```

The final table is printed and saved to `work_dir/results.txt`. Compare it with `expected_output.txt`. Expect differences of roughly ±0.03 across TensorFlow versions and hardware; the ordering of configurations is what matters.

**Interpretation and limits.** This genome is a deliberate worst case. Functional sequence has a strong, learnable composition signature, and the neutral mutation rule is local, so it extrapolates perfectly into functional blocks. It shows that contamination *can* happen and that the masks *can* remove it. It does not show how large either effect is in the human genome.
