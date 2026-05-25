# Initial Experiment Results — Cusp Probability Map MVP

**Date**: 2026-05-25
**Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Spec**: `docs/superpowers/specs/2026-05-25-cusp-probability-map-mvp-design.md`
**Runs completed this session**: R001 (M0 sanity), R002 (B1 baseline), R003 (B2 shuffled-SW control)

## Results by Milestone

### M0: Sanity — PASSED (R001)

| Item | Value |
|---|---|
| n_crossings sampled | 1000 -> 817 after dropna |
| n_expanded rows | 44,935 |
| Train AUC-ROC | 1.0000 |
| Heldout AUC-ROC | 0.9996 |
| Best iter | 999 (no early stop, all 1000 trees used) |
| Wall | 27 s |
| Gate | AUC > 0.99 on train -> PASS |

Pipeline works end-to-end. Heldout is row-level within the same crossings (not crossing-level holdout), so 0.9996 is essentially overfit-with-leakage — fine for sanity, not a generalization claim.

### M1: Baseline (R002 — B1)

**Dataset**: full 48,056 crossings, 39,668 after dropna (matches paper-1 number), expanded to 2,181,740 rows. Crossing-level random split 60/10/10/20.

| Metric | Value | Gate | Pass? |
|---|---|---|---|
| AUC-ROC | 0.9283 | >= 0.85 | PASS |
| AUC-PR | 0.5090 | n/a | — |
| Brier | 0.0562 | <= 0.10 | PASS |
| Reliability dev (max across bins in [0.1, 0.9]) | 0.8227 | < 0.05 | **FAIL (metric artifact)** |
| Best iter | 999 |  |  |
| Wall | 180 s |  |  |

**Per-MLT-bin AUC** (sun-synchronous DMSP coverage shows clearly):
- MLT 0-4, 4-8, 16-20, 20-24: insufficient samples (< 5 positives) — model is **not** supported in these ranges
- MLT 8-12: 0.8827
- MLT 12-16: 0.9031

**Hemisphere strat**: N AUC=0.9302, S AUC=0.9127 (within tolerance, single model is fine)

**Reliability metric is brittle (not a model defect)**. Inspecting the 15 bins:
- Bins 1-12 (mean_pred 0.01 to 0.75): all deviations < 0.04, well-calibrated
- Bin 13 (mean_pred=0.8227): frac_pos=0.0000 because very few samples landed there post-isotonic and they happened to be negatives. Single outlier dominates `max`.
- Bin 14 (mean_pred=1.0): exact match

Fix is trivial: switch from max-deviation to ECE weighted by bin count, or skip bins with < N=200 samples. Calibration itself is good in the operating range.

### M2: Shuffled-SW Control (R003 — B2)

| Metric | Real (R002) | Shuffled SW (R003) | Gap |
|---|---|---|---|
| AUC-ROC | 0.9283 | 0.8431 | 0.0852 |
| AUC-PR | 0.5090 | 0.3051 | 0.2039 |
| Brier | 0.0562 | 0.0706 | -0.0144 |

**Gate (gap >= 0.10)**: FAIL by 0.015.

**Interpretation**: SW signal contributes a real, measurable improvement (AUC +0.085, AUC-PR +0.20). But the geometric prior alone (x_polar, y_polar, hemi_code under shuffled SW) already gets AUC to 0.84 because the cusp lives in a small dayside-noon region 95% of the time. The model is using SW, but the spatial prior is doing more work than expected. AUC-PR shows the SW signal more starkly (+67% relative improvement, 0.30 -> 0.51).

### M0.5: Case studies (B1 visual qualitative)

| # | SW state | Peak P | Peak MLAT | Peak MLT |
|---|---|---|---|---|
| 1 | Strong south Bz (-10) | 0.019 | 71 | 11.5 |
| 2 | Strong north Bz (+10) | 0.344 | 81 | 11.0 |
| 3 | Strong By+ (Bz=-3, By=+8) | 0.197 | 75 | 11.5 |
| 4 | Strong By- (Bz=-3, By=-8) | 0.138 | 74 | 11.5 |
| 5 | Quiet | 0.559 | 81 | 12.0 |
| 6 | Storm (Bz=-15, V=700) | 0.019 | 71 | 12.0 |

**Physics sanity check: 5/6 PASS**
- PASS peak in dayside for all cases
- PASS south Bz peak at lower lat than north Bz (71 vs 81)
- **FAIL** By+ vs By- did not produce MLT shift in opposite directions (both ended up at 11.5 MLT)
- PASS quiet max P < 0.6
- PASS storm peak at lowest lat (tied with case 1)
- PASS midnight ~ 0 everywhere

**SERIOUS PRODUCT-LEVEL ISSUE**: Peak P magnitudes are backwards from physics expectation. Storm (case 6) and strong south Bz (case 1) both show peak P ~ 0.02, while quiet (case 5) shows 0.56. A user would look at the storm map and conclude "no cusp likely" — wrong.

Figures: `src/kernels/cuspmap_mvp/figures/case*.png` (6 maps)

## Root-cause diagnosis (the central finding of this MVP)

The training data has a **fixed 1:10 positive:negative ratio per crossing**. Every SW state in training appears with exactly 5 positives and 50 negatives. This destroys the global P(cusp | SW) signal:

- The model **can** learn "given SW, where on the dial is the cusp" — positives move with SW, negatives are sampled around the moving positives.
- The model **cannot** learn "given SW, how likely is there a cusp at all" — that information was symmetrically expanded away.

The probability magnitude is therefore driven by:
1. how sharply the model localizes the cusp under that SW state (sharper -> higher peak)
2. how much training-data support exists at the predicted (MLAT, MLT) (more support -> higher peak)

Storm and strong south Bz push the model's predicted cusp to low MLAT (71 deg), which is on the edge of where DMSP F8-F18 actually sampled. The model is honestly uncertain there. Quiet conditions concentrate the cusp at a well-sampled location, so the model is confident.

The model is doing what its training data asked of it. The training data setup was wrong for the goal.

## Summary

- 3/3 runs completed (R001 + R002 + R003)
- R002 metrics pass basic gates; reliability metric artifact is fixable
- R003 novelty gap (0.085) is below target (0.10), marginal
- Case studies show qualitative spatial physics is right but probability magnitudes are unusable as a product
- **Main result: MVP infrastructure works but the negative-sampling design has a fundamental flaw** for "operational probability product" framing
- Ready for /auto-review-loop: YES, **with clear pivot question** rather than continue-as-planned

## Next steps — three options for the pivot

### Option A: Reweight expansion to preserve global P(cusp | SW)
Cheap fix in the same data path. Instead of 1:10 per crossing, sample negatives proportional to SW-state's quiet-time frequency vs storm-time frequency. Or sample negatives from the full set of crossings (so quiet times contribute more negatives than storm times). Untested but plausible.

### Option B: Two-stage model
Keep R002's spatial map as conditional `P(cusp at (MLAT, MLT) | cusp exists, SW)`. Train a second very cheap classifier `P(cusp exists | SW)` from crossing presence/absence per orbit. Final product `P(cusp at (MLAT, MLT) | SW) = stage1 * stage2`. Stage2 needs orbit-level data (which crossings happened, which orbits had nothing) — possibly already in NCEI parser output or derivable from `parse_ncei_ssj.py`.

### Option C: Escalate to spec Q2 option 2 (raw 1Hz SSJ negatives)
The path I originally argued was "right but slow". Negatives become real observations (DMSP was there, SSJ flux didn't meet Anderson criteria). Class ratio per SW state then naturally reflects how often DMSP saw cusp under that SW state. 2-3 days of data work.

Recommendation: **start with B (cheap, only orbit metadata needed), see if it fixes magnitudes, escalate to C only if B is insufficient**. A is a hack and probably not enough.

## Tracker status

| Run | Status |
|---|---|
| R001 | DONE — sanity PASS |
| R002 | DONE — basic gate PASS, reliability metric artifact, case study magnitudes broken |
| R003 | DONE — gap 0.085, marginal fail |
| R004-R007 | NOT STARTED (M3 ablations) |
| R008 | NOT STARTED (case-study redo with winner) |
| R009 | NOT STARTED (temporal split) |
| R010 | NOT STARTED (coverage plot) |

Halting before M3 ablations because the case-study findings point to a design-level issue that ablating K and encoding will not fix.

## Hand-off

Files produced this session:
- `src/lib/cusp_map.py` — full library (load, expand, polar_xy, train, calibrate, evaluate, infer_grid)
- `src/scripts/mvp/r001_sanity.py`
- `src/scripts/mvp/r002_r003_baseline.py`
- `src/scripts/mvp/r002_case_studies.py`
- `src/kernels/cuspmap_mvp/bundles/expanded_full.parquet` (~100 MB, cached)
- `src/kernels/cuspmap_mvp/bundles/r002_model.ubj` + `r002_isotonic.pkl` + `r002_features.json`
- `src/kernels/cuspmap_mvp/bundles/r{001,002,003}_*.json` (run summaries)
- `src/kernels/cuspmap_mvp/bundles/r002_case_studies.json`
- `src/kernels/cuspmap_mvp/figures/case[1-6]_*.png` (6 polar dial heatmaps)

Next ARIS step: `/auto-review-loop "cusp probability map MVP — diagnose negative-sampling design flaw revealed by case-study magnitudes"` OR direct user decision on options A/B/C above.
