# Revised manuscript reproduction

This directory reproduces the current temporal manuscript from its processed crossing table. The inputs are included under `current/data`; cloning the repository is sufficient. Do not substitute the older Zenodo feature table or random-split weights.

## Commands

Use Python 3.10 and install `current/requirements.txt` in a fresh virtual environment. `current/requirements-lock.txt` records the full dependency versions used for the clean-environment validation. Run from the repository root:

```bash
python -m pip install -r reproduce/current/requirements.txt
python -m unittest discover -s reproduce/current -p test_train.py
python reproduce/reproduce.py --output-dir results/check
python reproduce/reproduce.py --train --jobs 6 --output-dir results/retrained
```

The first workflow validates the released reference results, recomputes primary-model inference, Tree SHAP and partial dependence, and generates all 10 figures. The `--train` workflow additionally fits every model and every validation fold from scratch. It never reads saved predictions during fitting. Both workflows stop on failed verification rather than printing a successful reproduction with missing outputs.

A complete training run includes the primary XGBoost model, three other targets, Ridge/GBR/MLP, cumulative windows, balanced-hemisphere training, four independent window replacements, and 34 grouped test folds. Use a compute node for the complete run. The complete run requires `--jobs 6`, matching the original BLAS thread count. MLP optimization is sensitive to the thread-dependent floating-point arithmetic: a separate 4-thread run did not reproduce the saved predictions, so the entry point rejects that setting for the full run. The 6-thread setting works on ordinary workstations and compute nodes; PBS itself is not required. Ridge is internally restricted to one BLAS thread to preserve its original float32 solve arithmetic; that setting reproduces its saved predictions exactly. Numerical agreement is checked with explicit tolerances; fonts and PDF metadata may vary across systems.

Separate stages are available for inspection:

```bash
python reproduce/current/train.py --data-dir reproduce/current/data --output-dir results/models --tasks all --jobs 6
python reproduce/current/verify.py --data-dir reproduce/current/data --run-dir results/models --output-dir results/checked --explain
python reproduce/current/figures.py --data-dir reproduce/current/data --output-dir results/reference_figures
```

The last command only renders the reference outputs; the combined entry point renders freshly trained predictions when `--train` is selected. `current/export_snapshot.py` is a maintainer-only export of existing author artifacts, not a required consumer step.

## Inputs and sample definitions

| File | Meaning |
|---|---|
| `rows.parquet` | The 39,668 ordered modeling rows, including 74 primary features, four targets and the additional 90/120-minute statistics |
| `manifest.json` | Required feature order, hyperparameters, split definition, source provenance and every distributed file's SHA-256 |
| `coverage.parquet` | The 40,813 crossings used in Figure 1, before complete-history filtering; this is not the test set |
| `calibration.parquet` | The 29,935 pre-2008 input rows used for the frozen Newell calibration |
| `dst.parquet` | Hourly Dst for deterministic storm-event grouping |
| `models/primary.ubj` | The final temporal XGBoost model; always supply columns in manifest order |
| `predictions/*.npz` | Reference predictions, observed labels and held-out row indices |
| `analysis.json` | Reference numerical summaries and interpretation arrays |
| `independent/history_features.npz` | Ordered 60/90/120-minute replacement statistics for the response-letter experiment |

`row_index` is the stable zero-based row position in `rows.parquet`. Train rows have `year < 2008`; test rows have `year >= 2008`. All primary performance figures use all 9,733 test crossings, including 9,582 northern and 151 southern crossings. SHAP and partial dependence use the same seeded subset of 2,000 test crossings; they are not calculated on a 2,000-crossing performance test set.

Targets are `abs_eq_mlat` and `abs_pole_mlat` in AACGM latitude degrees, and `eq_mlt` and `mean_mlt` in hours. Solar-wind speed is in km/s, IMF in nT, density in cm^-3, pressure in nPa, dipole tilt in degrees, and clock angle in radians. `hemi_code=1` for north and 0 for south. Derived `newell_cf` columns are the 2007 coupling proxy used as ML features; they must not be substituted for the separate 2006 empirical baseline.

The cohort and precipitation labels are retained from the manuscript analysis. This package makes processed-data modeling and figures reproducible; it does not perform new raw DMSP decoding or label selection. The full OMNI refresh changed some feature values relative to the earlier archive. The released snapshot is the exact input used for the current results, rather than a fresh download from a mutable upstream service.

## Newell baseline

The empirical comparison uses the 2006 function

`E_WAV = v * B_T * sin(clock_angle / 2)^4`

and `latitude = intercept + slope * E_WAV^(2/3)`. The frozen manuscript coefficients are `[78.40937950077127, -0.02247589222708998]`. `train.py` reconstructs them by least squares on the explicitly released calibration table. That calibration was made before the full OMNI feature refresh; its pre-2008 rows and labels are provided separately. Prediction uses the current temporal test inputs. No test labels enter calibration.

This preserves the manuscript baseline (MAE 1.7486284771 degrees) and improvement (36.5483854%). Recalibrating on the refreshed training inputs is a different calculation: MAE 1.7410880302 degrees. The code does not silently replace the frozen calibration. The older historical implementation incorrectly raised the 2007 proxy to another 2/3 power; it is not used by the current workflow.

## History-window experiments

The cumulative experiment in Figure S2 uses maximum windows 0, 15, 30, 60, 90 and 120 minutes, with 16, 34, 52, 74, 92 and 110 features. The 90/120 cases retain the shorter histories and the four 60-minute coupling features, then append raw-variable statistics. Their MAEs are 1.457030, 1.265228, 1.188568, 1.109533, 1.087175 and 1.078696 degrees.

The independent experiment in the response letter retains 74 columns. It replaces either the four coupling columns or the complete 22-column 60-minute block with its 90/120-minute counterpart. The 15/30-minute columns remain unchanged. These are distinct experiments, not points on one independent response-time curve.

History construction retains the original endpoint convention: the nearest OMNI minute to crossing start minus 600 seconds, with ties choosing the earlier record. A width `w` includes endpoints and therefore up to `w+1` one-minute records. Available values are used; at least `w//3` valid records are required. Standard deviations use `ddof=0`; integrals sum valid samples times 60 seconds. The snapshot freezes these values, including NaNs in extended windows.

## Figure map

All figures are generated by `current/figures.py` with repository-local styling. `figure_sources.json` records the input SHA-256 for each figure set.

| Manuscript figure | Output stem | Numerical source |
|---|---|---|
| 1 | `fig01_data_coverage` | Coverage table |
| 2 | `temporal_scatter` | Primary temporal predictions |
| 3 | `temporal_baseline` | Newell, MLP, Ridge, GBR and final XGBoost metrics |
| 4 | `temporal_errors` | Primary residuals and empirical error distribution |
| 5 | `temporal_importance` | Gain and Tree SHAP on 2,000 test crossings |
| 6 | `temporal_hemisphere` | Test hemisphere and AE subsets with explicit counts |
| 7 | `temporal_residuals` | Primary residuals versus prediction, coupling, tilt and AE |
| S1 | `temporal_other_targets` | Three other target predictions on all test rows |
| S2 | `temporal_windows` | Cumulative history experiment |
| S3 | `temporal_pdp` | Six variables, 25 grid points each; training 2nd to 98th percentiles |

## Expected validation MAE

| Test design | MAE (degrees) |
|---|---|
| Temporal holdout | 1.109533 |
| One grouped day split | 1.185190 |
| Five storm-grouped folds | 1.224753 ± 0.018467 |
| 23 held-out years | 1.250483 ± 0.194468 |
| Five contiguous time blocks | 1.255554 ± 0.151828 |

The ± values are population standard deviations across held-out folds. The first two designs each have one split. These tests reuse the fixed model settings; the reproduction does not conduct hyperparameter selection.

## Historical files

`legacy/` retains the previous code and metrics for provenance. Earlier `src/gen_figures*` and exploratory scripts do not define the current manuscript figures. The previous Zenodo model package is random-trained and should not be passed to this entry point.
