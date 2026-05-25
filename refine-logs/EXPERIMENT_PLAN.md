# Experiment Plan — Cusp Probability Map MVP

**Problem**: Paper 1 predicts a single cusp boundary latitude per crossing. We want a calibrated 2D probability map P(cusp | MLAT, MLT, SW) over the polar dial, learned from the existing 48k DMSP crossings without re-parsing 1 Hz raw SSJ.
**Method Thesis**: Stratified synthetic negatives mined from the 48k crossings table + polar Cartesian spatial encoding + isotonic-calibrated XGBoost binary classifier yields a calibrated 2D map that genuinely uses solar-wind signal (not just spatial priors).
**Date**: 2026-05-25
**Spec**: `docs/superpowers/specs/2026-05-25-cusp-probability-map-mvp-design.md`

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1: A calibrated 2D probability map can be learned from the existing crossings table alone (no raw SSJ re-parse) | Saves 2-3 days of data-engineering and unlocks paper-2 fast | AUC-ROC >= 0.85, Brier <= 0.10, reliability deviation < 0.05 in P in [0.1, 0.9] on test set; passes on both random and temporal splits | B1, B4 |
| C2: The map's predictive signal is driven by solar wind, not spatial geometry alone | Distinguishes a real space-weather product from a static climatology | Shuffled-SW control AUC drops >= 0.10 vs real model | B2 |

**Anti-claims to rule out:**
- A1: Gain comes from leaking crossing-region geometry across splits. Defense: crossing-level group split for train/val/cal/test.
- A2: Model is memorizing a static dayside-noon prior. Defense: shuffled-SW control (B2).
- A3: Calibration is trivially achieved by negative-heavy training. Defense: report raw + calibrated reliability separately.

## Paper Storyline

**Main paper must prove (when paper writing resumes):**
- C1 with both random and temporal splits
- C2 with shuffled-SW control
- 6 case-study heatmaps physically consistent

**Appendix can support:**
- Per-MLT-bin AUC
- Hemisphere-stratified AUC and Brier
- Ablations on K (negatives per positive) and spatial encoding
- Coverage histogram showing where the model is and is not supported by data

**Experiments intentionally cut (out of MVP scope):**
- Real-time OMNI ingestion pipeline
- Multi-instrument fusion (SuperDARN, IMAGE/FUV)
- Parametric band model (4-parameter posterior alternative)
- Raw 1 Hz SSJ re-parse path

## Experiment Blocks

### Block 1 (B1): Main MVP result

- **Claim tested**: C1
- **Why this block exists**: Establishes that the basic pipeline works end-to-end on the simplest split
- **Dataset / split / task**: Existing `output/cusp_crossings_with_omni.csv`; crossing-level random 8:1:1 (train / val / cal) plus 20% held-out test; binary classification of (MLAT, MLT, SW) -> cusp present
- **Compared systems**: Default model from spec (XGBoost binary:logistic, paper-1 hyperparameters, polar Cartesian (x,y), isotonic calibration if needed)
- **Metrics**: AUC-ROC, AUC-PR, Brier, reliability diagram (15 bins), per-MLT-bin AUC, hemisphere-stratified report
- **Setup details**: `n_estimators=1000`, `max_depth=8`, `lr=0.02`, `scale_pos_weight=10`, `early_stopping_rounds=50` on val logloss; isotonic fit on cal set with gating (skip if raw deviation < 0.05); 1 seed for MVP, seed=42
- **Success criterion**: AUC >= 0.85, Brier <= 0.10, reliability deviation < 0.05, all 6 case-study heatmaps pass physics sanity-check list
- **Failure interpretation**: If AUC low -> data-prep bug (check crossing_id grouping); if Brier high but AUC OK -> isotonic insufficient; if heatmaps wrong shape -> spatial encoding issue
- **Table / figure target**: Main paper Table 1 (metrics row 1), Figure 2 (case studies 1-6), Figure 3 (reliability diagram)
- **Priority**: MUST-RUN

### Block 2 (B2): Shuffled-SW control (novelty isolation)

- **Claim tested**: C2; rules out A2
- **Why this block exists**: The single most diagnostic check that the model uses SW information rather than spatial priors. Cheap, decisive.
- **Dataset / split / task**: Same as B1; in training set, row-shuffle the 74 SW feature columns (keep `x_polar`, `y_polar`, `hemi_code` intact); retrain identical model
- **Compared systems**: B1 model vs shuffled-SW model
- **Metrics**: AUC-ROC delta, per-MLT-bin AUC delta
- **Setup details**: Identical hyperparameters; seed=42 for both
- **Success criterion**: real_AUC - shuffled_AUC >= 0.10
- **Failure interpretation**: Gap < 0.10 -> model isn't using SW signal -> MVP failed -> escalate to raw-SSJ path (spec Q2 option 2)
- **Table / figure target**: Main paper Figure 4 (bar chart of AUC vs shuffled-SW)
- **Priority**: MUST-RUN

### Block 3 (B3): K and spatial-encoding ablations

- **Claim tested**: Defends design choices in spec (K=10, polar Cartesian)
- **Why this block exists**: Reviewers will ask why these specific defaults; the ablation justifies them or finds a better setting
- **Dataset / split / task**: Same as B1
- **Compared systems**:
  - K in {5, 10, 20} (encoding fixed at polar)
  - Encoding in {raw (MLAT, MLT), cyclic (sin/cos MLT + MLAT), polar Cartesian (x, y)} (K fixed at 10)
  - 5 runs total (K=10 polar shared between sweeps)
- **Metrics**: AUC, Brier, time-to-train
- **Setup details**: Identical hyperparameters otherwise; seed=42
- **Success criterion**: Polar Cartesian within 0.01 AUC of best encoding; K=10 within 0.01 AUC of best K. If not, switch to the winner before paper.
- **Failure interpretation**: If raw encoding wins by > 0.02 -> the polar geometric story is weaker than expected; consider both and explain
- **Table / figure target**: Appendix Table A1
- **Priority**: MUST-RUN (cheap, defends choices)

### Block 4 (B4): Temporal generalization

- **Claim tested**: C1 under temporal holdout
- **Why this block exists**: Random split overstates generalization because the same solar cycle leaks across splits; temporal split is what reviewers and operations need
- **Dataset / split / task**: Same data and expansion as B1, but split = train < 2008, test >= 2008 (paper 1 used the same boundary)
- **Compared systems**: Spec-default model trained on temporal split vs B1's random-split model
- **Metrics**: Same as B1
- **Setup details**: Identical hyperparameters; seed=42
- **Success criterion**: AUC drop from random to temporal split <= 0.05; Brier increase <= 0.02; case-study maps still pass physics sanity-check list
- **Failure interpretation**: Larger AUC drop -> overfitting to solar-cycle-specific patterns -> consider leave-one-year-out CV or year-balanced sampling
- **Table / figure target**: Main paper Table 1 (metrics row 2)
- **Priority**: MUST-RUN

### Block 5 (B5): Failure analysis + data-coverage map

- **Claim tested**: Honestly bounds where the model is reliable
- **Why this block exists**: Avoids reviewer attack on "you predict P over the whole dial but DMSP only samples MLT 5-19"
- **Dataset / split / task**: Joint histogram of training samples over (MLT bin, MLAT bin, IMF Bz sign); overlay model AUC per region
- **Compared systems**: N/A (descriptive)
- **Metrics**: Sample density per cell; per-cell AUC where >= 200 samples
- **Setup details**: Reuse B1 trained model and B1 test set
- **Success criterion**: Plot shows clear gap on nightside (MLT 20-4) consistent with sun-synchronous orbit geometry; AUC drops correspondingly
- **Failure interpretation**: If AUC stays high on nightside despite low density -> model is extrapolating geometric prior, not learning -> revise interpretation
- **Table / figure target**: Main paper Figure 5 (coverage + AUC heatmap)
- **Priority**: NICE-TO-HAVE for MVP; MUST-RUN before paper submission

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost (CPU-wall) | Risk |
|-----------|------|------|---------------|-----------------|------|
| M0 | Sanity: pipeline end-to-end on 1k crossings, overfit check | R001 | Model fits 1k rows to AUC > 0.99 -> proceed; else fix bug | 10 min | data-prep bugs (column names, dedup) |
| M1 | B1 baseline at default settings | R002 | AUC >= 0.85 -> proceed; else hyperparam sweep before B2 | 1 h | scale_pos_weight wrong direction |
| M2 | B2 shuffled-SW control | R003 | Gap >= 0.10 -> proceed; else MVP fail, escalate | 1 h | gap < 0.10 |
| M3 | B3 ablations (K + encoding) | R004-R008 | Pick winner if differs from defaults by > 0.02 AUC | 5 h | none |
| M4 | B4 temporal split | R009 | AUC drop <= 0.05 -> proceed; else investigate | 1 h | larger drop than expected |
| M5 | B5 coverage + failure plot, write results.md, decide on paper-writing trigger | R010 | All MUST-RUN claims supported -> trigger paper-plan skill | 2 h | none |

Total wall time on a single Casper htc node (64-core CPU): ~10 hours, comfortably one day.

## Compute and Data Budget

- **Total estimated CPU-hours**: ~10 wall hours on a 64-core htc node. XGBoost `n_jobs=-1` plus `tree_method='hist'` is the main consumer.
- **GPU-hours**: 0. XGBoost CPU is sufficient and matches paper 1's setup.
- **Data preparation needs**: Read existing `output/cusp_crossings_with_omni.csv` (~50 MB), expand to ~100 MB Parquet. No new ingest.
- **Human evaluation needs**: Manual sanity-check on 6 case-study heatmaps (10 min); manual review of coverage plot for honest bounds (10 min).
- **Biggest bottleneck**: Single-threaded REPL iteration speed during sanity stage M0. Once stable, B3's 5-run ablation sweep is the longest contiguous block at ~5 h.

## Risks and Mitigations

- **R1**: Negative-sampling synthesis allows the model to learn crossing geometry rather than cusp physics.
  - Mitigation: B2 shuffled-SW control is the explicit test. If it fails, fall back to spec Q2 option 2 (raw SSJ) before writing paper.
- **R2**: 48k crossings under-cover the (MLT, MLAT, SW) joint space, especially nightside MLT 20-4.
  - Mitigation: B5 coverage plot bounds where map is trustworthy; paper will explicitly restrict the supported domain.
- **R3**: Some crossings may include two cusp passes per orbit treated as independent rows under naive `crossing_id` from row index.
  - Mitigation: M0 sanity stage includes a quick check on `(date, satellite, hemisphere)` group counts; if duplicates exist, fold them into a composite key.
- **R4**: Column-name guesses (`pole_mlt` vs `mean_mlt`) may be wrong.
  - Mitigation: M0 first action is to dump CSV columns and confirm before writing any expansion code.
- **R5**: `scale_pos_weight=10` may interact poorly with isotonic calibration (over-correction).
  - Mitigation: Spec already gates isotonic on raw reliability check. If raw is already calibrated, skip isotonic.

## Final Checklist

- [x] Main paper tables are covered (Table 1 from B1+B4; Figures 2-4 from B1+B2; Figure 5 from B5)
- [x] Novelty is isolated (B2 shuffled-SW control)
- [x] Simplicity is defended (B3 ablations on K and encoding justify defaults)
- [x] Frontier contribution is justified or explicitly not claimed (no frontier component; XGBoost intentional)
- [x] Nice-to-have runs are separated from must-run runs (B5 nice-to-have for MVP, must-run for paper)
