"""R015 — retrain stage 1 with opportunity-restricted negatives.

Addresses codex round 2 blocker: original R011 used "any OMNI hour without a
recorded crossing" as the negative pool, which conflates "no cusp" with "no
DMSP coverage". This biases stage 1 toward predicting low probability for
SW states that just happened to occur during data-gap periods.

Opportunity restriction: only include OMNI hours that fall within +/- WINDOW
hours of some DMSP crossing in the 48k table. The intuition: if at least one
satellite produced a crossing within +/- 24 h of hour H, DMSP was operational
and observing the dayside in that period. Hour H is then a valid "opportunity"
for stage 1 to score.

WINDOW = 24 hours initially (sweep in next iteration if needed).

After filtering, stage 1 trains on (positive opportunities, negative
opportunities) pairs that are physically comparable.
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
WINDOW_HOURS = 24


def opportunity_mask(omni: pd.DataFrame, crossings: pd.DataFrame,
                     window_hours: int = 24) -> np.ndarray:
    """True for OMNI hours within +/- window_hours of any crossing time."""
    cross_t = pd.to_datetime(crossings["time_start"]).astype(np.int64).values  # ns
    cross_h = cross_t // (3600 * 10**9)  # hour bins
    cross_set = set(cross_h.tolist())
    # build expansion set: for each crossing hour, mark +/- window
    expanded = set()
    for h in cross_set:
        for d in range(-window_hours, window_hours + 1):
            expanded.add(h + d)
    omni_h = omni["datetime"].astype(np.int64).values // (3600 * 10**9)
    return np.isin(omni_h, list(expanded))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print("[R015] loading OMNI2 hourly ...")
    omni = load_omni2_hourly(OMNI_PATH, year_min=1987, year_max=2014)
    print(f"  loaded {len(omni)} hours")
    omni = derive_features(omni)
    feat_cols = STAGE1_BASE_FEATURES
    keep = omni[feat_cols].notna().all(axis=1)
    omni_ok = omni[keep].reset_index(drop=True)
    print(f"  after dropna: {len(omni_ok)} hours")

    crossings = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    print(f"  loaded {len(crossings)} crossings")

    print(f"[R015] building opportunity mask (window = +/- {WINDOW_HOURS} h) ...")
    opp = opportunity_mask(omni_ok, crossings, window_hours=WINDOW_HOURS)
    print(f"  opportunity hours: {opp.sum()} ({100*opp.mean():.1f}% of {len(omni_ok)})")

    omni_opp = omni_ok[opp].reset_index(drop=True)
    labels = label_hours(omni_opp, crossings)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    print(f"  opportunity-restricted positives: {n_pos} ({100*n_pos/len(labels):.2f}%)")
    print(f"  opportunity-restricted negatives: {n_neg}")

    rng = np.random.default_rng(42)
    idx = np.arange(len(omni_opp))
    rng.shuffle(idx)
    n_test = int(len(idx) * 0.2)
    n_val = int(len(idx) * 0.1)
    n_cal = int(len(idx) * 0.1)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    cal_idx = idx[n_test + n_val:n_test + n_val + n_cal]
    train_idx = idx[n_test + n_val + n_cal:]

    X = omni_opp[feat_cols].values.astype(np.float32)
    y = labels.astype(int)
    print(f"  split: train={len(train_idx)}  val={len(val_idx)}  cal={len(cal_idx)}  test={len(test_idx)}")

    print("[R015] fitting stage-1 XGBoost (opportunity-restricted) ...")
    t1 = time.time()
    model = fit_stage1(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
    print(f"  trained in {time.time()-t1:.1f}s  best_iter={model.best_iteration}")

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

    print(f"\n[R015] STAGE-1 OPPORTUNITY-RESTRICTED TEST METRICS:")
    print(f"  AUC-ROC: {auc:.4f}  (R011 unrestricted: 0.7285)")
    print(f"  AUC-PR : {ap:.4f}  (R011 unrestricted: 0.2521)")
    print(f"  Brier  : {brier:.4f}  (R011 unrestricted: 0.0982)")

    # save
    model.save_model(f"{OUT_DIR}/r015_stage1_opp_model.ubj")
    with open(f"{OUT_DIR}/r015_stage1_opp_isotonic.pkl", "wb") as f:
        pickle.dump(iso, f)
    with open(f"{OUT_DIR}/r015_stage1_opp_features.json", "w") as f:
        json.dump(feat_cols, f)
    summary = {
        "run_id": "R015",
        "milestone": "B-stage1-fix",
        "purpose": "stage 1 with DMSP opportunity-restricted negatives",
        "window_hours": WINDOW_HOURS,
        "n_hours_eligible": int(len(omni_opp)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": float(n_pos / (n_pos + n_neg)),
        "best_iter": int(model.best_iteration),
        "test_auc": auc,
        "test_ap": ap,
        "test_brier": brier,
        "reliability_frac_pos": frac_pos.tolist(),
        "reliability_mean_pred": mean_pred.tolist(),
        "elapsed_sec": float(time.time() - t0),
    }
    with open(f"{OUT_DIR}/r015_stage1_opp_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n  saved -> {OUT_DIR}/r015_stage1_opp_*")


if __name__ == "__main__":
    main()
