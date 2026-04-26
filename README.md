# cuspML

Machine learning prediction of ionospheric cusp boundary location from solar wind parameters.

This repository contains the analysis code for:

> Zhu, Y., Michael, A. T., & Toffoletto, F. R. (2026). Predicting Ionospheric Cusp Location from Solar Wind: An XGBoost Model Trained on 27 Years of DMSP Data. *Journal of Geophysical Research: Space Physics* (under review).

## Data and trained models

The cusp crossing database (48,056 events, 1987–2014) and trained XGBoost model weights are archived at Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19340792.svg)](https://doi.org/10.5281/zenodo.19340792)

Concept DOI (always points to latest version): `10.5281/zenodo.19340792`

## Repository layout

```
src/                 Analysis source (Python)
  identify_cusp.py     DMSP SSJ cusp boundary identification (Anderson 2024 criteria)
  add_omni.py          OMNI solar wind feature matching
  add_omni_batch.py    Batch OMNI matching driver
  parse_ncei_ssj.py    NCEI DMSP SSJ binary file parser
  tree_dse.py          Tree-model design space exploration (XGBoost / GBR / Ridge)
  nn_dse.py            Neural network architecture search (MLP / ResMLP / TabTransformer)
  compare_anderson.py  Cross-check against Anderson & Bukowski (2024) results
  gen_figures_jgr.py   Generate paper figures
  gen_figures_batch2.py
  gen_figures_final.py

scripts/             Helper scripts and utilities
run_*.sh             PBS submission scripts for Derecho/Casper
```

## Reproducing the paper results

1. Download the cusp crossing database from Zenodo:
   ```
   wget https://zenodo.org/records/19780238/files/cusp_crossings_1987_2014.csv.zip
   unzip cusp_crossings_1987_2014.csv.zip
   ```
2. Apply the feature derivations described in Section 2.2 of the paper (transverse IMF magnitude, IMF clock angle, Newell coupling function, Kan-Lee electric field, half-wave rectified `vBs`, hemisphere-adjusted `By`, day-of-year, and hemisphere code).
3. Use the XGBoost hyperparameters reported in Section 2.3:
   `n_estimators=1000, max_depth=8, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5, random_state=42`.
4. With an 80/20 random split (`random_state=42`) the model reproduces MAE = 0.97° on the equatorward cusp boundary latitude. With a temporal split (train < 2008, test ≥ 2008) the temporal-holdout MAE is 1.11°.

Pre-trained models (XGBoost `.ubj` format) are also included in the Zenodo archive (`cuspML_models.zip`).

## License

Code: MIT License.
Data and trained models: CC-BY-4.0 (via Zenodo).
