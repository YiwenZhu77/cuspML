# Verified Data — cuspML Paper

Last verified: 2026-03-29, via REPL full recomputation on omni_full_hist (39,668 samples)

## Dataset
- Raw crossings: 48,056
- After dropna (solar wind): 40,813
- After dropna (all 74 features + targets): **39,668**
- Train/test split: 80/20, random_state=42 → Train: 31,734, Test: 7,934
- Years: 1987–2014 (27 years, ~2.5 solar cycles)
- Satellites: DMSP F06–F18
- Hemispheres: North ~90%, South ~10%

## Table 2: Overall Performance (random split, test set)
| Target | MAE | RMSE | R² | r |
|--------|-----|------|----|---|
| \|eq MLAT\| (°) | 0.97 | 1.34 | 0.79 | 0.887 |
| \|pole MLAT\| (°) | 0.95 | 1.31 | 0.79 | 0.888 |
| eq MLT (hr) | 0.77 | 1.01 | 0.48 | 0.700 |
| mean MLT (hr) | 0.73 | 0.98 | 0.49 | 0.709 |

## Baseline Comparison (eq_MLAT MAE, random split)
| Model | MAE (°) | Source |
|-------|---------|--------|
| Bz only (linear) | 2.04 | REPL verified |
| Newell CF (linear) | 1.80 | REPL verified |
| Ridge (74 feat, α=1.0) | 1.41 | REPL verified |
| GBR-300 (depth 5) | 1.11 | REPL verified |
| XGBoost-1000 (depth 8) | 0.97 | REPL verified |
| Improvement over Newell | 46% | (1.80-0.97)/1.80 |

## NN Results (from dse_log.csv, on omni_hist ~11,700 samples)
| Model | eq_MLAT MAE (°) | time_sec |
|-------|-----------------|----------|
| MLP-gelu | 1.5304 | 10.67 |
| ResMLP-4blk-128 | 1.0178 | 36.41 |
| TabTF-d128-L2 | 1.1137 | 135.89 |

## Feature Importance (gain-based, full-data XGBoost)
- newell_cf_mean60: **~31%** (varies 31–33% due to XGBoost threading nondeterminism)
- Top 4 sixty-minute coupling features: >55% total

## SHAP (2,000 test-set samples)
- newell_cf_mean60: mean |SHAP| = 1.07°
- dipole_tilt: mean |SHAP| = 0.32°
- imf_bz_mean15: mean |SHAP| = 0.23°

## Error Distribution (eq_MLAT, random split)
- Residual mean bias: <0.03° in magnitude
- Residual std: 1.34°
- Within 1°: 63%
- Within 2°: 89%
- Within 3°: 97%

## Hemispheric Performance (test set)
| Hemisphere | n | MAE (°) | r |
|------------|---|---------|---|
| North | 7,180 | 0.93 | 0.890 |
| South | 754 | 1.29 | 0.867 |

## Temporal Generalization
- Temporal holdout (train <2008, test ≥2008): MAE = **1.11°**
- LOYO (n≥100 years only, 23 years): MAE = **1.26 ± 0.19°**, worst year (1992) ≈ 1.7°
- LOYO excluded years: 1988 (n=6), 1990 (n=23)

## Irreducible Error (multi-crossing groups)
- Definition: same calendar day + same 0.5-hr MLT bin, >1 crossing
- Groups: **9,679**
- Within-group std of |eq_MLAT|: **1.03 ± 1.01°**
- Within-group mean range: **2.35°**

## Newell CF Correlation (individual crossings)
- r ≈ -0.583 (on test set; literature binned value is 0.78–0.83)

## Timescale Ablation (eq_MLAT MAE)
| Feature set | MAE (°) |
|-------------|---------|
| Instantaneous only | 1.271 |
| + 15-min history | 1.083 |
| + 30-min history | 1.022 |
| + 60-min history | 0.987 |
| All windows (15+30+60) | 0.967 |

## AE-Stratified Performance (from previous verification, test set)
| Activity | n | MAE (°) | R² | r |
|----------|---|---------|----|----|
| Quiet (AE<100) | 2,339 | 0.96 | 0.68 | 0.83 |
| Moderate (100–300) | 2,955 | 0.92 | 0.62 | 0.79 |
| Active (300–500) | 1,521 | 0.94 | 0.60 | 0.78 |
| Storm (AE≥500) | 1,119 | 1.16 | 0.57 | 0.78 |

## Tilt Bin MAE (from gen_figures_batch2.py)
| Tilt bin | MAE (°) | n |
|----------|---------|---|
| [-35, -20) | 0.871 | 1,028 |
| [-20, -10) | 0.938 | 1,187 |
| [-10, 0) | 0.929 | 1,080 |
| [0, 10) | 1.001 | 1,381 |
| [10, 20) | 0.982 | 2,791 |
| [20, 35) | 1.145 | 467 |

## XGBoost Hyperparameters
- n_estimators: 1000
- max_depth: 8
- learning_rate: 0.02
- subsample: 0.8
- colsample_bytree: 0.7
- reg_alpha: 0.1
- reg_lambda: 1.0
- min_child_weight: 5
- random_state: 42

## Publishing
- JGR PU formula: words/500 + figures + tables
- Current: ~25.0 PU (limit 25)
- Pages: 21
- Figures: 10, Tables: 3
