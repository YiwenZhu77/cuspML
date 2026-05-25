# Cusp Probability Map — MVP Design Spec

**Date:** 2026-05-25
**Author:** Yiwen Zhu (drafted by Claude via brainstorming skill)
**Status:** Approved, ready for implementation
**Scope:** MVP only — build & validate model. Paper writing deferred until MVP success gate passes.

## Goal

Extend the cuspML paper-1 pipeline from a point estimate of cusp boundary latitude to a calibrated 2D probability map P(cusp | MLAT, MLT, SW) over the polar dial. MVP target: prove the approach works on a single REPL kernel using only the existing 48k crossings table, before committing to the heavier raw-SSJ data path (option 2 from Q2).

## Paper-2 framing (deferred — recorded for context only)

- Primary: operational forecast product + physics discovery via probabilistic field
- Light methodology angle (sparse point obs → calibrated 2D field)
- All paper work held until MVP success gate clears.

## Key technical decisions (locked via brainstorming)

| # | Decision | Choice |
|---|---|---|
| Q1 | Paper framing | 1+2, light 3 (deferred) |
| Q2 | Data source | Existing 48k crossings table (`output/cusp_crossings_with_omni.csv`) |
| Q3 | Negative spatial strategy | Stratified: 5 near-boundary + 5 far-on-dial per positive |
| Q4 | Positive expansion + K | N=5 positives per crossing, K=10 negatives per positive → 55 rows/crossing |
| Q5 | Spatial encoding | Polar Cartesian `(x, y)`; single model w/ `hemi_code` for both hemispheres |
| Q6 | Split | crossing-level random 8:2, plus a sanity-check temporal split (train<2008, test≥2008) |
| Q7 | Calibration | Isotonic regression, gated by raw reliability check (skip if raw already on diagonal) |
| Q8 | Success criteria | Combined: quantitative (AUC, Brier, reliability) + 6 case-study maps + shuffled-SW control |

Defaults baked in:
- Hyperparameters: reuse paper-1 (`n_estimators=1000`, `max_depth=8`, `lr=0.02`, etc.), change objective to `binary:logistic`, add `scale_pos_weight=10`, `early_stopping_rounds=50`
- Hemisphere: single model with `hemi_code` feature
- Eval suite: AUC-ROC, AUC-PR, Brier, reliability diagram, per-MLT-bin AUC, hemisphere-stratified report

## Architecture

```
crossings.csv (~48k rows)
      |
      v
[expand]  per crossing -> 5 pos + 50 neg = 55 rows  -> expanded.parquet (~2.6M rows)
      |
      |--- positives: uniform in [eq_lat, pole_lat] x [eq_mlt, pole_mlt]
      |--- negatives per positive:
      |       5 near: MLAT in (eq_lat-5, eq_lat-1) or (pole_lat+1, pole_lat+5),
      |               MLT in (eq_mlt-1, pole_mlt+1)
      |       5 far:  uniform on dial (50-90 deg, 0-24 h), reject if inside
      |               crossing region +/-2 deg / +/-1 h buffer
      |
      v
[split]   crossing-level random 8:2 (group key = crossing_id)
      |   8:1:1 inside train -> train_rows / val_rows / cal_rows
      |
      v
[features] 74 SW + (x, y) + hemi_code = 77 (dedup hemi_code if already in 74)
      |     x = (90 - |MLAT|) * cos(2 pi * MLT / 24)
      |     y = (90 - |MLAT|) * sin(2 pi * MLT / 24)
      v
[XGBoost binary:logistic]  paper-1 hp + scale_pos_weight=10 + early_stopping
      |
      v
[isotonic calibration]  fit on cal_rows; skip if raw reliability already on diagonal
      |
      v
[infer]   given SW state -> evaluate on (x, y) grid -> reshape -> polar dial heatmap
```

## Data pipeline

**Input:** `/glade/work/yizhu/cuspML/output/cusp_crossings_with_omni.csv` (paper-1's working CSV). Verify before run that it contains the 74 features used in `src/nn_dse.py:55-67` plus `abs_eq_mlat`, `abs_pole_mlat`, `eq_mlt`, and a poleward MLT (`pole_mlt` or `mean_mlt` — confirm column name during implementation).

**Crossing ID:** generate as integer row index of the dropna-cleaned dataframe, stored alongside expanded rows so all 55 rows from one crossing share the same `crossing_id`. Group key for split.

**Expansion logic (per crossing):**

Positives (5 rows):
```python
mlat_pos = np.random.uniform(eq_lat, pole_lat, 5)
mlt_pos  = np.random.uniform(eq_mlt, pole_mlt, 5)  # confirm pole_mlt column
labels   = np.ones(5)
```

Per positive: 10 negatives (5 near + 5 far). Loop over the 5 positives -> 50 negative rows per crossing.

Near negatives per positive (5 rows): equator/polar side of the boundary
```python
mlat_near_eq   = np.random.uniform(eq_lat - 5,   eq_lat - 1, 2)
mlat_near_pole = np.random.uniform(pole_lat + 1, pole_lat + 5, 3)
mlat_near = np.concatenate([mlat_near_eq, mlat_near_pole])
mlt_near  = np.random.uniform(eq_mlt - 1, pole_mlt + 1, 5)
```

Far negatives per positive (5 rows): random on dial with rejection
```python
mlat_far, mlt_far = sample_far_negatives(
    n=5,
    crossing_box=(eq_lat-2, pole_lat+2, eq_mlt-1, pole_mlt+1),
    dial_box=(50, 90, 0, 24),
)
```

All negatives carry the same SW + `hemi_code` as the crossing.

**Total expansion factor:** 1 crossing -> 5 positives -> 5 * 10 = 50 negatives -> 55 rows. Class ratio per crossing = 1:10. Total expanded dataset ~48k * 55 = 2.6M rows.

**Stored at:** `output/zenodo_mvp/expanded.parquet` (~100 MB).

## Model

Single XGBoost classifier:

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=5,
    scale_pos_weight=10,
    random_state=42,
    tree_method='hist',
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric='logloss',
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

**Features (77):**
- 74 SW + IMF + running stats from `src/nn_dse.py:55-67` (`dipole_tilt`, `hemi_code`, `doy`, `imf_bx/by/bz`, `sw_v/n/pdyn`, `B_T`, `clock_angle`, `sin_clock_half`, `newell_cf`, `kan_lee_ef`, `vBs`, `by_hemi`, plus `mean15/30/60`, `std15/30/60`, `delta15/30/60`, `int60`, `_mean60` history columns)
- New: `x_polar`, `y_polar` (2 features)
- `hemi_code` already in the 74 → dedup

## Calibration

1. Split train as `train (80%) / val (10%) / cal (10%)` at the crossing level.
2. Train model with `eval_set=val`, early stopping on logloss.
3. On `cal` set, run `predict_proba` -> `raw_P`.
4. Plot reliability diagram via `sklearn.calibration.calibration_curve(y_cal, raw_P, n_bins=15)`.
5. If max abs deviation from diagonal in P in [0.1, 0.9] < 0.05 -> skip isotonic.
6. Else fit `IsotonicRegression(out_of_bounds='clip').fit(raw_P, y_cal)`; pickle as `isotonic.pkl`.
7. Inference: `calibrated_P = isotonic.transform(raw_P)`.

## Evaluation

All metrics on held-out **test set** (the 20% crossings not used in train/val/cal, expanded the same 5+50 way).

**Quantitative:**
- AUC-ROC
- AUC-PR
- Brier score
- Reliability diagram (15 bins)
- Per-MLT-bin AUC: bins [0-4, 4-8, 8-12, 12-16, 16-20, 20-24]
- Hemisphere-stratified AUC + Brier (N vs S)

**Qualitative — case study heatmaps** (north hemisphere; redo for south as mirror check):

| # | Bz | By | Bx | V | n | Description |
|---|----|----|----|----|----|---|
| 1 | -10 | 0 | 0 | 500 | 5 | Strong south Bz |
| 2 | +10 | 0 | 0 | 500 | 5 | Strong north Bz |
| 3 | -3 | +8 | 0 | 500 | 5 | Strong By+ |
| 4 | -3 | -8 | 0 | 500 | 5 | Strong By- |
| 5 | 0 | 0 | 0 | 350 | 3 | Quiet |
| 6 | -15 | 0 | 0 | 700 | 15 | Storm |

For each: derive running stats (set 15/30/60-min means to the instantaneous value as a first-pass approximation; flag as MVP simplification); compute (`B_T`, `clock_angle`, `newell_cf`, etc.) using same formulas as `src/nn_dse.py:46-52`; broadcast SW to all 1920 grid cells (40 lat x 48 MLT); predict + calibrate; plot polar heatmap.

Overlay paper-1's regression-predicted `eq_lat` as a dashed line on each map for visual sanity.

**Physics sanity-check list:**
- Peak P at MLT 10-14 for all dayside cases
- Case 1 vs 2: south-Bz peak 3-5 deg lower MLAT than north-Bz peak
- Case 3 vs 4: By+ vs By- shift peak MLT by 0.5-1.5 h in opposite directions
- Case 5: max P < 0.6
- Case 6: max P > 0.8, peak at lowest MLAT among cases
- midnight + deep polar cap P near 0

**Shuffled-SW control (critical):**
- Shuffle the 74 SW columns row-wise on the training set (decouple SW from labels)
- Keep `(x_polar, y_polar, hemi_code)` intact
- Retrain identical model
- Report control AUC vs real AUC; gate: `real - control >= 0.10`
- If gate fails -> MVP failed, escalate to raw-SSJ path (Q2 option 2)

## Success gate

All of the following must pass:

- [ ] AUC-ROC >= 0.85 on test set, calibrated
- [ ] Brier <= 0.10
- [ ] Reliability diagram in P in [0.1, 0.9] deviates from diagonal < 0.05
- [ ] All 6 case-study heatmaps pass physics sanity-check list
- [ ] Shuffled-SW control: `real_AUC - shuffled_AUC >= 0.10`
- [ ] Hemisphere-stratified AUC differs by < 0.05 between N and S (else: train two separate models)

Any failure -> stop, diagnose, decide next move (re-tune hyperparams, re-engineer negatives, escalate to raw-SSJ pipeline).

## Code organization

REPL-first, promote to lib once stable. Per `~/.claude/rules/repl.md`:

```
src/
  kernels/cuspmap_mvp/
    config.py                  # source paths, kernel id, output dir
    bundles/
      expanded.parquet         # post-expansion rows
      model.ubj                # XGBoost model
      isotonic.pkl             # calibrator
      snap_*.pkl               # REPL snapshots (auto)
    figures/
      case_<n>_<descr>.png     # 6+6 case maps (N + S)
      reliability.png
      per_mlt_auc.png
      shuffled_control.png
      hemi_strat.png
  lib/
    cusp_map.py                # promoted: expand(), polar_xy(),
                               #           train_with_calib(),
                               #           infer_grid(), plot_dial()
  cells.md                     # all REPL cells (project-level ledger)
```

Kernel id: `cuspmap_mvp` (single source combo: paper-1 CSV).

Functions to write in `src/lib/cusp_map.py` after first stable cells:
- `expand(crossings_df, n_pos=5, k_neg=10, buffer_lat=2.0, buffer_mlt=1.0, seed=42) -> pl.DataFrame`
- `polar_xy(mlat, mlt) -> (x, y)`
- `build_feature_matrix(df, sw_feature_names) -> np.ndarray`
- `train_with_calibration(X_train, y_train, X_val, y_val, X_cal, y_cal, hp_overrides=None) -> (model, isotonic_or_none)`
- `infer_grid(model, isotonic, sw_state_dict, mlat_range=(50,90,1.0), mlt_range=(0,24,0.5)) -> 2d array`
- `plot_dial(prob_map, mlat_grid, mlt_grid, title, out_path, eq_lat_overlay=None)`

## Known risks / decisions deferred

- Far-negatives occasionally land inside another crossing's region under the same SW. Probability < 1% per sample given 48k crossings spread over 27 years; ignored for MVP. Revisit if shuffled-SW control fails.
- Case-study heatmaps use instantaneous SW as proxy for 15/30/60-min means (no real time series for synthetic SW states). Flagged in figure captions; not a model defect but an inference-time approximation.
- Some crossings may include two cusp passes per orbit; `crossing_id` from row index treats them as independent rows. If `parse_ncei_ssj.py` already deduplicates this, no action. If not, group key may need to fold to (date, satellite, hemisphere) — confirm during implementation.
- `pole_mlt` column name unverified; could be `mean_mlt` instead. Confirm by reading first row of CSV before writing `expand()`.
- `hemi_code` duplication between `(x, y)` derivation source (`|MLAT|`) and existing 74 features — keep `hemi_code` once, drop dup before fit.
- Class ratio of 1:10 may not need `scale_pos_weight=10` if isotonic calibration handles the bias; sweep `scale_pos_weight in {1, 5, 10}` if calibration alone leaves AUC < 0.85.

## Out of scope (explicit)

- Real-time OMNI ingestion
- Web UI / CLI / deployment
- Raw 1Hz SSJ re-parse
- Multi-instrument fusion (SuperDARN, IMAGE/FUV)
- Parametric band model
- Paper writing (held until success gate clears)
