"""R017 — opportunity-window sensitivity sweep for stage 1.

Codex round 3 pre-submission polish: confirm stage 1 metrics are stable across
plausible opportunity-window choices. Sweep +/-{6, 12, 24, 48} h, report AUC,
AUC-PR, Brier, positive rate, and combined-product logp on 200 held-out
crossings.
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
from cusp_map import load_crossings, sw_feature_names, predict_proba, TrainedModel
from cusp_stage1 import (
    load_omni2_hourly, derive_features, label_hours,
    STAGE1_BASE_FEATURES, fit_stage1,
)
from r012_case_studies_2stage import load_stage2
from r014_endtoend_eval import (LAT_AXIS, MLT_AXIS, MM, LL, CELL_AREA,
                                 cell_of, stage2_dial, normalize_pmf, haversine_deg)
from r015_stage1_opportunity import opportunity_mask

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles"
OMNI_PATH = "/glade/work/yizhu/cuspML/output/omni_raw/omni2_all_years.dat"


def train_for_window(omni_ok, crossings, window_h: int):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    opp = opportunity_mask(omni_ok, crossings, window_hours=window_h)
    om = omni_ok[opp].reset_index(drop=True)
    labels = label_hours(om, crossings)
    X = om[STAGE1_BASE_FEATURES].values.astype(np.float32)
    y = labels.astype(int)

    rng = np.random.default_rng(42)
    idx = np.arange(len(om)); rng.shuffle(idx)
    n_test = int(len(idx) * 0.2)
    n_val = int(len(idx) * 0.1)
    n_cal = int(len(idx) * 0.1)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    cal_idx = idx[n_test + n_val:n_test + n_val + n_cal]
    train_idx = idx[n_test + n_val + n_cal:]

    model = fit_stage1(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
    raw_cal = model.predict_proba(X[cal_idx])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, y[cal_idx])
    raw_test = model.predict_proba(X[test_idx])[:, 1]
    cal_test = iso.transform(raw_test)
    return {
        "window_h": window_h,
        "n_eligible": int(len(om)),
        "pos_rate": float(y.mean()),
        "test_auc": float(roc_auc_score(y[test_idx], cal_test)),
        "test_ap": float(average_precision_score(y[test_idx], cal_test)),
        "test_brier": float(brier_score_loss(y[test_idx], cal_test)),
    }, model, iso


def combined_logp(model, iso, s2, sample, sw_cols):
    from cusp_stage1 import STAGE1_BASE_FEATURES as S1F
    n_cells = LAT_AXIS.size * MLT_AXIS.size
    eps = 1e-12
    uniform_logp = np.log(1.0 / n_cells)
    logps = []
    for _, row in sample.iterrows():
        true_lat = abs(row["mean_mlat"]) if "mean_mlat" in row else (abs(row["eq_mlat"]) + abs(row["pole_mlat"])) / 2
        true_mlt = row["mean_mlt"] if "mean_mlt" in row else (row["eq_mlt"] + row["pole_mlt"]) / 2
        hemi = "N" if row["hemisphere"] == "N" else "S"
        sw_for_s2 = {c: row[c] for c in sw_cols if c in row}
        sw_for_s1 = {c: row[c] for c in S1F if c in row}
        sw_for_s1["doy_feat"] = sw_for_s2.get("doy", pd.to_datetime(row["time_start"]).dayofyear)
        if "B_T" not in sw_for_s1:
            sw_for_s1["B_T"] = float(np.sqrt(row["imf_by"]**2 + row["imf_bz"]**2))
        arr = np.array([[sw_for_s1.get(k, 0.0) for k in S1F]], dtype=np.float32)
        s1_p = float(iso.transform(model.predict_proba(arr)[:, 1])[0])
        s2_p = stage2_dial(s2, sw_for_s2, hemisphere=hemi)
        s2_pmf = normalize_pmf(s2_p, area_weighted=True)
        combined = s1_p * s2_pmf
        i, j = cell_of(true_lat, true_mlt)
        logps.append(np.log(combined[i, j] + eps))
    return float(np.mean(logps)), float(np.mean(logps) - uniform_logp)


def main():
    t0 = time.time()
    print("[R017] loading OMNI ...")
    omni = load_omni2_hourly(OMNI_PATH, year_min=1987, year_max=2014)
    omni = derive_features(omni)
    keep = omni[STAGE1_BASE_FEATURES].notna().all(axis=1)
    omni_ok = omni[keep].reset_index(drop=True)
    crossings = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    print(f"  loaded {len(omni_ok)} OMNI hours, {len(crossings)} crossings")

    s2 = load_stage2()
    sw_cols = sw_feature_names(crossings)
    required = ["eq_mlat", "pole_mlat", "eq_mlt", "pole_mlt"]
    keep_cols = sw_cols + required + ["satellite", "hemisphere", "time_start"]
    df_clean = crossings[keep_cols].dropna().reset_index(drop=True)
    rng = np.random.default_rng(42)
    cids = np.arange(len(df_clean)); rng.shuffle(cids)
    n_test = int(len(cids) * 0.2)
    test_df = df_clean.iloc[cids[:n_test]].reset_index(drop=True)
    sample = test_df.iloc[np.random.default_rng(99).choice(len(test_df), 200, replace=False)].reset_index(drop=True)

    results = []
    for w in [6, 12, 24, 48]:
        print(f"\n[R017] window +/- {w} h ...")
        t1 = time.time()
        metrics, model, iso = train_for_window(omni_ok, crossings, w)
        mean_logp, improvement = combined_logp(model, iso, s2, sample, sw_cols)
        metrics["mean_logp_combined"] = mean_logp
        metrics["logp_improvement_over_uniform"] = improvement
        metrics["wall_sec"] = float(time.time() - t1)
        results.append(metrics)
        print(f"  window {w}h: AUC={metrics['test_auc']:.4f}  AP={metrics['test_ap']:.4f}  "
              f"Brier={metrics['test_brier']:.4f}  pos_rate={metrics['pos_rate']:.3f}  "
              f"combined_logp_improv={improvement:+.3f} nats")

    print(f"\n[R017] SENSITIVITY TABLE")
    print(f"{'window':>8}  {'n_elig':>8}  {'pos%':>6}  {'AUC':>6}  {'AP':>6}  {'Brier':>6}  {'logp_imp':>10}")
    for r in results:
        print(f"{r['window_h']:>6}h  {r['n_eligible']:>8}  {100*r['pos_rate']:>5.2f}  "
              f"{r['test_auc']:>6.4f}  {r['test_ap']:>6.4f}  {r['test_brier']:>6.4f}  "
              f"{r['logp_improvement_over_uniform']:>+10.3f}")

    out = {"window_sweep": results, "elapsed_sec": float(time.time() - t0)}
    with open(f"{OUT_DIR}/r017_window_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved -> {OUT_DIR}/r017_window_sweep.json")
    print(f"  total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
