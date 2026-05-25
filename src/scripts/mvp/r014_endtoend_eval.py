"""R014 — end-to-end evaluation of the combined two-stage product on real held-out crossings.

Codex round 1 weakness #2: stagewise metrics don't tell us if the combined map
is actually useful at predicting where real DMSP crossings will fall.

Protocol:
1. Load R002's test set (20% crossings held out from stage 2 training)
2. For each test positive (a real DMSP crossing in the 48k table), compute the
   COMBINED product map at its SW state.
3. Score = log probability assigned to the cell containing the true (MLAT, MLT).
4. Compare to:
   - random-cell baseline (uniform 1/n_cells)
   - stage-2-only baseline (peak of stage-2 normalized over dial)
   - shuffled-SW baseline (stage 1 + stage 2 both with shuffled SW per row)
5. Report:
   - mean log-likelihood improvement vs uniform baseline
   - peak-latitude error: distance from predicted argmax cell to true (MLAT, MLT)
   - fraction of test crossings where true cell is in top-10 cells of the map

Run this on a sample of test crossings (full test set is 7933 crossings x slow
per-crossing inference, sample 500 for first pass).
"""
import json
import os
import pickle
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, "/glade/work/yizhu/cuspML/src/lib")
sys.path.insert(0, "/glade/work/yizhu/cuspML/src/scripts/mvp")
from cusp_map import (load_crossings, sw_feature_names, polar_xy,
                       predict_proba, TrainedModel, crossing_random_split,
                       expand_dataset)
from cusp_stage1 import STAGE1_BASE_FEATURES
from r012_case_studies_2stage import load_stage1, load_stage2, stage1_scalar

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp"

# dial: cell-CENTER convention; no endpoint duplication
DLAT = 1.0    # deg
DMLT = 0.5    # hours
LAT_AXIS = np.arange(50.0 + DLAT / 2, 90.0, DLAT)              # 50.5 ... 89.5  -> 40 cells
MLT_AXIS = np.arange(0.0 + DMLT / 2, 24.0, DMLT)               # 0.25 ... 23.75 -> 48 cells
MM, LL = np.meshgrid(MLT_AXIS, LAT_AXIS)
CELL_AREA = np.cos(np.deg2rad(LL))  # solid-angle weight
CELL_AREA = CELL_AREA / CELL_AREA.sum()  # normalize so weights sum to 1


def cell_of(mlat, mlt):
    """Return (i, j) of the cell containing the given (|MLAT|, MLT). Wrap MLT in [0, 24)."""
    mlat = abs(mlat)
    mlt = mlt % 24.0
    i = int(np.clip((mlat - 50.0) / DLAT, 0, len(LAT_AXIS) - 1))
    j = int(np.clip(mlt / DMLT, 0, len(MLT_AXIS) - 1))
    return i, j


def stage2_dial(s2, sw_state, hemisphere="N"):
    """Evaluate stage 2 on the dial. Returns 2D array shape (n_lat, n_mlt) of raw stage-2 probs."""
    x, y = polar_xy(LL.ravel(), MM.ravel())
    rec = dict(sw_state)
    rec["hemi_code"] = 1.0 if hemisphere == "N" else 0.0
    n = LL.size
    grid = {k: np.full(n, v, dtype=np.float32) for k, v in rec.items()}
    grid["x_polar"] = x.astype(np.float32)
    grid["y_polar"] = y.astype(np.float32)
    df = pd.DataFrame(grid)
    missing = [f for f in s2.feature_names if f not in df.columns]
    for m in missing:
        df[m] = 0.0  # graceful default for sparse fields
    X = df[s2.feature_names].values.astype(np.float32)
    P = predict_proba(s2, X).reshape(LL.shape)
    return P


def normalize_pmf(P_grid, area_weighted=True):
    """Renormalize a 2D map to a PMF over cells, optional cell-area weighting."""
    w = CELL_AREA if area_weighted else np.ones_like(P_grid)
    weighted = P_grid * w
    Z = weighted.sum()
    if Z < 1e-12:
        return weighted
    return weighted / Z


def haversine_deg(lat1, mlt1, lat2, mlt2):
    """Approx great-circle distance on the polar dial in degrees, treating MLT as a longitude."""
    # convert MLT to radians of longitude
    lon1 = np.deg2rad(mlt1 * 15.0)
    lon2 = np.deg2rad(mlt2 * 15.0)
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dl = lon2 - lon1
    a = np.sin((phi2 - phi1) / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
    return np.rad2deg(2 * np.arcsin(np.sqrt(a)))


def main():
    t0 = time.time()
    print("[R014] loading models ...")
    s1_model, s1_iso, s1_feats = load_stage1()
    s2 = load_stage2()

    print("[R014] reloading test crossings (same split as R002, seed=42) ...")
    df = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    required = ["eq_mlat", "pole_mlat", "eq_mlt", "pole_mlt"]
    sw_cols = sw_feature_names(df)
    keep_cols = sw_cols + required + ["satellite", "hemisphere", "time_start"]
    df_clean = df[keep_cols].dropna().reset_index(drop=True)

    # use the same split logic as R002 (crossing_random_split) — but we want the test
    # CROSSINGS, not expanded rows. Apply the split logic at crossing level directly.
    rng = np.random.default_rng(42)
    cids = np.arange(len(df_clean))
    rng.shuffle(cids)
    n_test = int(len(cids) * 0.2)
    test_cids = cids[:n_test]
    test_df = df_clean.iloc[test_cids].reset_index(drop=True)
    print(f"  test crossings: {len(test_df)}")

    # sample 500 for fast eval
    sample_n = min(500, len(test_df))
    sample_idx = np.random.default_rng(99).choice(len(test_df), sample_n, replace=False)
    sample = test_df.iloc[sample_idx].reset_index(drop=True)
    print(f"  sampled {sample_n} for eval")

    n_cells = LAT_AXIS.size * MLT_AXIS.size
    uniform_per_cell = 1.0 / n_cells

    rec = {"combined_logp": [],
           "stage2only_logp": [],
           "uniform_logp": [],
           "peak_dist_deg": [],
           "stage2_peak_dist_deg": [],
           "true_cell_in_top10": [],
           "true_cell_in_top1pct": [],
           "stage1_p": [],
           "true_mlat": [], "true_mlt": [],
           "satellite": [], "year": []}

    print("[R014] running eval ...")
    t1 = time.time()
    sw_only_feats = [c for c in sw_cols if c not in ("x_polar", "y_polar", "hemi_code")]
    for k, row in sample.iterrows():
        true_lat = abs(row["mean_mlat"]) if "mean_mlat" in row else (abs(row["eq_mlat"]) + abs(row["pole_mlat"])) / 2
        true_mlt = row["mean_mlt"] if "mean_mlt" in row else (row["eq_mlt"] + row["pole_mlt"]) / 2
        hemi = "N" if row["hemisphere"] == "N" else "S"

        # build sw dicts
        sw_for_s2 = {c: row[c] for c in sw_cols if c in row}
        sw_for_s1 = {c: row[c] for c in STAGE1_BASE_FEATURES if c in row}
        # stage 1 uses doy_feat key vs stage 2's doy
        sw_for_s1["doy_feat"] = sw_for_s2.get("doy", pd.to_datetime(row["time_start"]).dayofyear)
        # newell_cf / kan_lee / vBs / B_T / clock should be in row from load_crossings derivation
        # add any missing derivations defensively
        if "B_T" not in sw_for_s1:
            sw_for_s1["B_T"] = float(np.sqrt(row["imf_by"]**2 + row["imf_bz"]**2))

        s1_p = stage1_scalar(s1_model, s1_iso, s1_feats, sw_for_s1)
        s2_p = stage2_dial(s2, sw_for_s2, hemisphere=hemi)

        # combined: stage1 * (stage2 / sum(stage2 * area_weights))
        s2_pmf = normalize_pmf(s2_p, area_weighted=True)
        combined = s1_p * s2_pmf
        s2_only_pmf = s2_pmf  # use stage 2 PMF as the "stage-2-only" baseline

        # true cell
        i, j = cell_of(true_lat, true_mlt)
        true_combined = combined[i, j]
        true_s2only = s2_only_pmf[i, j]

        eps = 1e-12
        rec["combined_logp"].append(np.log(true_combined + eps))
        rec["stage2only_logp"].append(np.log(true_s2only + eps))
        rec["uniform_logp"].append(np.log(uniform_per_cell + eps))
        rec["stage1_p"].append(s1_p)
        rec["true_mlat"].append(true_lat)
        rec["true_mlt"].append(true_mlt)
        rec["satellite"].append(row["satellite"])
        rec["year"].append(pd.to_datetime(row["time_start"]).year)

        # peak distance
        peak_idx = np.unravel_index(np.argmax(combined), combined.shape)
        peak_lat = LAT_AXIS[peak_idx[0]]
        peak_mlt = MLT_AXIS[peak_idx[1]]
        rec["peak_dist_deg"].append(haversine_deg(true_lat, true_mlt, peak_lat, peak_mlt))

        s2_peak_idx = np.unravel_index(np.argmax(s2_p), s2_p.shape)
        s2_peak_lat = LAT_AXIS[s2_peak_idx[0]]
        s2_peak_mlt = MLT_AXIS[s2_peak_idx[1]]
        rec["stage2_peak_dist_deg"].append(haversine_deg(true_lat, true_mlt, s2_peak_lat, s2_peak_mlt))

        # rank of true cell among all cells, by combined probability
        flat = combined.flatten()
        rank = (flat > true_combined).sum() + 1  # 1 = best
        rec["true_cell_in_top10"].append(rank <= 10)
        rec["true_cell_in_top1pct"].append(rank <= n_cells // 100)

        if (k + 1) % 100 == 0:
            print(f"  processed {k+1}/{sample_n}  ({time.time()-t1:.1f}s)")

    # summary stats
    combined_logp = np.array(rec["combined_logp"])
    stage2only_logp = np.array(rec["stage2only_logp"])
    uniform_logp = np.array(rec["uniform_logp"])
    peak_dist = np.array(rec["peak_dist_deg"])
    s2_peak_dist = np.array(rec["stage2_peak_dist_deg"])
    top10 = np.array(rec["true_cell_in_top10"])
    top1pct = np.array(rec["true_cell_in_top1pct"])

    summary = {
        "n_eval": int(sample_n),
        "median_peak_dist_deg": float(np.median(peak_dist)),
        "p90_peak_dist_deg": float(np.percentile(peak_dist, 90)),
        "median_stage2_only_peak_dist_deg": float(np.median(s2_peak_dist)),
        "frac_true_in_top10_cells": float(top10.mean()),
        "frac_true_in_top1pct_cells": float(top1pct.mean()),
        "mean_logp_combined": float(combined_logp.mean()),
        "mean_logp_stage2_only": float(stage2only_logp.mean()),
        "mean_logp_uniform": float(uniform_logp.mean()),
        "logp_improvement_over_uniform": float(combined_logp.mean() - uniform_logp.mean()),
        "logp_improvement_stage2_over_uniform": float(stage2only_logp.mean() - uniform_logp.mean()),
    }
    print(f"\n[R014] END-TO-END METRICS (n={sample_n} held-out real crossings):")
    for k, v in summary.items():
        print(f"  {k:>42s}: {v}")

    out = {"summary": summary,
           "per_crossing": {k: list(map(float, v)) if k != "satellite" else v
                            for k, v in rec.items()}}
    with open(f"{OUT_DIR}/bundles/r014_endtoend.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  saved -> {OUT_DIR}/bundles/r014_endtoend.json")
    print(f"  total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
