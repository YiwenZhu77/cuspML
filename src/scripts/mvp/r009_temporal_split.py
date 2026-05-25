"""R009 — retrain stage 2 with temporal split (train < 2008, test >= 2008).

Reuses prepare_data from R002 to expand the dataset, then applies
crossing_temporal_split instead of crossing_random_split. Trains the same
spec-default XGBoost classifier with isotonic calibration. Reports the AUC drop
relative to R002 (random split) — gate is drop <= 0.05.
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

from cusp_map import (
    load_crossings, expand_dataset, sw_feature_names,
    build_feature_matrix, fit_xgb, maybe_calibrate, evaluate,
    TrainedModel, crossing_temporal_split,
)

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles"


def main():
    t0 = time.time()
    cache = f"{OUT_DIR}/expanded_full.parquet"
    if os.path.exists(cache):
        print(f"[R009] loading cached expansion from {cache}")
        expanded = pd.read_parquet(cache)
    else:
        print("[R009] loading + expanding crossings (no cache) ...")
        df = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
        sw_cols0 = sw_feature_names(df)
        required = ["eq_mlat", "pole_mlat", "eq_mlt", "pole_mlt"]
        keep_cols = sw_cols0 + required + ["satellite", "hemisphere", "time_start"]
        df_clean = df[keep_cols].dropna().reset_index(drop=True)
        expanded = expand_dataset(df_clean, n_pos=5, k_neg=10, seed=42, verbose=True)
        expanded.to_parquet(cache, index=False)
    sw_cols = sw_feature_names(expanded)

    print("[R009] temporal split (train<2008, test>=2008) ...")
    splits = crossing_temporal_split(expanded, cutoff_year=2008,
                                      frac_val=0.125, frac_cal=0.125, seed=42)
    for k, v in splits.items():
        print(f"  {k:>5}: {len(v):>8} rows from {v['crossing_id'].nunique():>5} crossings")

    X_train, feat_names = build_feature_matrix(splits["train"], sw_cols)
    X_val, _ = build_feature_matrix(splits["val"], sw_cols)
    X_cal, _ = build_feature_matrix(splits["cal"], sw_cols)
    y_train = splits["train"]["label"].values.astype(int)
    y_val = splits["val"]["label"].values.astype(int)
    y_cal = splits["cal"]["label"].values.astype(int)

    print("[R009] fitting XGBoost (temporal train) ...")
    t1 = time.time()
    model = fit_xgb(X_train, y_train, X_val, y_val)
    print(f"  trained in {time.time()-t1:.1f}s, best_iter={model.best_iteration}")

    iso, cal_info = maybe_calibrate(model, X_cal, y_cal, deviation_threshold=0.05)
    trained = TrainedModel(model=model, isotonic=iso, feature_names=feat_names,
                           used_calibration=(iso is not None))

    metrics = evaluate(trained, splits["test"], sw_cols)
    print(f"\n[R009] TEMPORAL-SPLIT TEST METRICS:")
    print(f"  AUC-ROC:        {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:         {metrics['auc_pr']:.4f}")
    print(f"  Brier:          {metrics['brier']:.4f}")

    # compare to R002 random split
    with open(f"{OUT_DIR}/r002_baseline_results.json") as f:
        r002 = json.load(f)
    drop = r002["test_metrics"]["auc_roc"] - metrics["auc_roc"]
    print(f"\n  R002 random-split AUC: {r002['test_metrics']['auc_roc']:.4f}")
    print(f"  R009 temporal-split AUC: {metrics['auc_roc']:.4f}")
    print(f"  AUC drop: {drop:.4f}  (gate: drop <= 0.05 = {'PASS' if drop <= 0.05 else 'FAIL'})")

    out = {
        "run_id": "R009",
        "milestone": "M4",
        "purpose": "temporal split",
        "split": {k: int(v["crossing_id"].nunique()) for k, v in splits.items()},
        "best_iter": int(model.best_iteration),
        "calibration_info": cal_info,
        "test_metrics": metrics,
        "auc_drop_vs_r002": float(drop),
        "gate_pass": bool(drop <= 0.05),
        "elapsed_sec": float(time.time() - t0),
    }
    with open(f"{OUT_DIR}/r009_temporal_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  saved -> {OUT_DIR}/r009_temporal_results.json")
    print(f"  total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
