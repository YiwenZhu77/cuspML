# cuspML reproduction package

Reproduces the headline results and core figures of *"Predicting Ionospheric Cusp Location
from Solar Wind: An XGBoost Model Trained on 27 Years of DMSP Data"* (JGR Space Physics).

## Data (download once, ~34 MB)
The two input files are archived at the **data DOI 10.5281/zenodo.19340792** (CC-BY-4.0). Download
them into `reproduce/data/` (or point `$CUSPML_DATA` at their location):
- `cusp_crossings_1987_2014.parquet` — the archived cusp-crossing catalog (48,056 crossings,
  1987–2014, DMSP F06–F18) with pre-computed solar-wind/IMF running-window features. This is the
  single input needed; the raw DMSP/OMNI queries are not required (and CDAWeb reprocessing means
  they no longer reproduce this catalog bit-for-bit — hence the archived dataset).
- `omni_dst_hourly_1987_2014.parquet` — hourly Dst (OMNI2/CDAWeb), used only for the
  storm-event validation split.

## Contents
- `reproduce.py` — loads the catalog, rebuilds the 39,668-crossing modeling frame, and reproduces
  the validation split ladder, baseline ladder, and core figures.
- `requirements.txt` — Python dependencies.

## Run
```
pip install -r requirements.txt
# place the two .parquet files in reproduce/data/ (from data DOI 10.5281/zenodo.19340792)
python reproduce.py
```
Prints a reproduced-vs-paper comparison table and writes `figures/*.png` + `reproduced_metrics.json`.

## What it reproduces
The primary (dependence-aware) metrics reproduce exactly: temporal-holdout MAE 1.11°, LOYO 1.26°,
day-grouped 1.20°, contiguous-block ~1.27°, storm-event 1.23°; baseline ladder Newell 1.80° →
XGBoost ~0.97°; r ≈ 0.887; SSPB ≈ +0.04%. The random-split MAE is order-sensitive at the 0.01°
level (0.96–0.97) and is de-emphasized in the paper in favor of the dependence-aware splits.

Determinism: all splits and models use `random_state=42`.
