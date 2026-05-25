"""R011 (B stage 1) — train P(cusp observed in hour | SW(hour)).

Uses OMNI2 hourly flat file. Positives = hours containing any of the 48k DMSP
crossings. Negatives = all other valid OMNI hours in 1987-2014. Trains binary
XGBoost on 14 base + derived SW features (no history columns; OMNI2 hourly has
no sub-hour history).

After training, isotonic-calibrates on a held-out set and saves the model so
the combined stage1*stage2 product can be invoked from case-study scripts.
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
from cusp_map import load_crossings
from cusp_stage1 import (
    load_omni2_hourly, derive_features, label_hours,
    STAGE1_BASE_FEATURES, fit_stage1,
)

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles"
OMNI_PATH = "/glade/work/yizhu/cuspML/output/omni_raw/omni2_all_years.dat"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print("[R011] loading OMNI2 hourly ...")
    omni = load_omni2_hourly(OMNI_PATH, year_min=1987, year_max=2014)
    print(f"  loaded {len(omni)} hours ({omni['datetime'].min()} to {omni['datetime'].max()})")
    omni = derive_features(omni)

    # drop rows with any NaN in needed features
    feat_cols = STAGE1_BASE_FEATURES
    keep = omni[feat_cols].notna().all(axis=1)
    omni_ok = omni[keep].reset_index(drop=True)
    print(f"  after dropna on {len(feat_cols)} features: {len(omni_ok)} hours kept "
          f"({100*len(omni_ok)/len(omni):.1f}%)")

    print("[R011] loading 48k crossings to label hours ...")
    crossings = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    print(f"  loaded {len(crossings)} crossings")

    labels = label_hours(omni_ok, crossings)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    print(f"  positive hours: {n_pos} ({100*n_pos/len(labels):.2f}%)")
    print(f"  negative hours: {n_neg}")

    # split by time chunk (random by hour, no leakage concern at hourly granularity)
    rng = np.random.default_rng(42)
    idx = np.arange(len(omni_ok))
    rng.shuffle(idx)
    n_test = int(len(idx) * 0.2)
    n_val = int(len(idx) * 0.1)
    n_cal = int(len(idx) * 0.1)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    cal_idx = idx[n_test + n_val:n_test + n_val + n_cal]
    train_idx = idx[n_test + n_val + n_cal:]

    X = omni_ok[feat_cols].values.astype(np.float32)
    y = labels.astype(int)
    print(f"  split: train={len(train_idx)}  val={len(val_idx)}  cal={len(cal_idx)}  test={len(test_idx)}")
    print(f"  features ({len(feat_cols)}): {feat_cols}")

    print("[R011] fitting stage-1 XGBoost ...")
    t1 = time.time()
    model = fit_stage1(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
    print(f"  trained in {time.time()-t1:.1f}s  best_iter={model.best_iteration}")

    # isotonic calibration
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    raw_cal = model.predict_proba(X[cal_idx])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, y[cal_idx])

    raw_test = model.predict_proba(X[test_idx])[:, 1]
    cal_test = iso.transform(raw_test)
    y_test = y[test_idx]

    auc = float(roc_auc_score(y_test, cal_test))
    ap = float(average_precision_score(y_test, cal_test))
    brier = float(brier_score_loss(y_test, cal_test))
    frac_pos, mean_pred = calibration_curve(y_test, cal_test, n_bins=15)

    print(f"\n[R011] STAGE-1 TEST METRICS:")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  AUC-PR : {ap:.4f}")
    print(f"  Brier  : {brier:.4f}")
    print(f"  Reliability bins (15):")
    for m, f in zip(mean_pred, frac_pos):
        print(f"    mean_pred={m:.4f}  frac_pos={f:.4f}  |diff|={abs(m-f):.4f}")

    # save
    model.save_model(f"{OUT_DIR}/r011_stage1_model.ubj")
    with open(f"{OUT_DIR}/r011_stage1_isotonic.pkl", "wb") as f:
        pickle.dump(iso, f)
    with open(f"{OUT_DIR}/r011_stage1_features.json", "w") as f:
        json.dump(feat_cols, f)
    summary = {
        "run_id": "R011",
        "milestone": "B-stage1",
        "purpose": "P(cusp observed in hour | SW)",
        "n_hours_total": int(len(omni_ok)),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "best_iter": int(model.best_iteration),
        "test_auc": auc,
        "test_ap": ap,
        "test_brier": brier,
        "reliability_frac_pos": frac_pos.tolist(),
        "reliability_mean_pred": mean_pred.tolist(),
        "elapsed_sec": float(time.time() - t0),
    }
    with open(f"{OUT_DIR}/r011_stage1_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n  saved -> {OUT_DIR}/r011_stage1_*")
    print(f"  total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
