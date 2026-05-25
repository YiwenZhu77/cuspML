"""R002 (B1 baseline) + R003 (B2 shuffled-SW control) on full 48k crossings.

Crossing-level 60/10/10/20 random split. R002 trains spec-default model.
R003 row-shuffles the 74 SW columns in train (keeps x_polar, y_polar, hemi_code)
and retrains identical model. Reports AUC delta as the novelty isolation
diagnostic.

Outputs:
- bundles/r002_baseline_results.json
- bundles/r003_shuffled_sw_results.json
- bundles/r002_model.ubj
- bundles/r002_isotonic.pkl  (if calibration not skipped)
- bundles/r002_features.json
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
from cusp_map import (
    load_crossings, expand_dataset, sw_feature_names,
    build_feature_matrix, fit_xgb, maybe_calibrate, evaluate,
    TrainedModel, crossing_random_split,
)

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles"


def prepare_data():
    t0 = time.time()
    print("[prep] loading 48k crossings ...")
    df = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    print(f"  loaded {len(df)} crossings  ({time.time()-t0:.1f}s)")

    required = ["eq_mlat", "pole_mlat", "eq_mlt", "pole_mlt"]
    sw_cols = sw_feature_names(df)
    keep_cols = sw_cols + required + ["satellite", "hemisphere", "time_start"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_clean = df[keep_cols].dropna().reset_index(drop=True)
    print(f"  after dropna: {len(df_clean)} crossings ({100*len(df_clean)/len(df):.1f}% kept)")

    cache = f"{OUT_DIR}/expanded_full.parquet"
    if os.path.exists(cache):
        print(f"  loading cached expansion from {cache}")
        expanded = pd.read_parquet(cache)
    else:
        print("[prep] expanding full dataset (5 pos + 50 neg per crossing) ...")
        t1 = time.time()
        expanded = expand_dataset(df_clean, n_pos=5, k_neg=10, seed=42, verbose=True)
        print(f"  expanded in {time.time()-t1:.1f}s -> {len(expanded)} rows")
        expanded.to_parquet(cache, index=False)
        print(f"  cached -> {cache}")

    return df_clean, expanded


def run_R002(splits, sw_cols):
    print("\n" + "=" * 70)
    print("[R002] B1 baseline: spec-default model on full crossings")
    print("=" * 70)
    t0 = time.time()
    X_train, feat_names = build_feature_matrix(splits["train"], sw_cols)
    X_val, _ = build_feature_matrix(splits["val"], sw_cols)
    X_cal, _ = build_feature_matrix(splits["cal"], sw_cols)
    X_test, _ = build_feature_matrix(splits["test"], sw_cols)
    y_train = splits["train"]["label"].values.astype(int)
    y_val = splits["val"]["label"].values.astype(int)
    y_cal = splits["cal"]["label"].values.astype(int)

    print(f"  train={X_train.shape}  val={X_val.shape}  cal={X_cal.shape}  test={X_test.shape}")
    print(f"  features ({len(feat_names)}): first 6 = {feat_names[:6]} ... last 3 = {feat_names[-3:]}")

    print("[R002] fitting XGBoost ...")
    t1 = time.time()
    model = fit_xgb(X_train, y_train, X_val, y_val)
    print(f"  trained in {time.time()-t1:.1f}s, best_iter={model.best_iteration}")

    print("[R002] calibrating (isotonic if raw deviation >= 0.05) ...")
    iso, cal_info = maybe_calibrate(model, X_cal, y_cal, deviation_threshold=0.05)
    print(f"  raw_reliability_deviation = {cal_info['raw_reliability_deviation']:.4f}")
    print(f"  calibration = {cal_info['calibration']}")

    trained = TrainedModel(model=model, isotonic=iso, feature_names=feat_names,
                           used_calibration=(iso is not None))

    print("[R002] evaluating on test set ...")
    metrics = evaluate(trained, splits["test"], sw_cols)
    print(f"\n[R002] TEST METRICS:")
    print(f"  AUC-ROC:        {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:         {metrics['auc_pr']:.4f}")
    print(f"  Brier:          {metrics['brier']:.4f}")
    print(f"  Reliability dev: {metrics['reliability_deviation']:.4f}")
    print(f"  Per-MLT AUC:")
    for k, v in metrics["per_mlt_auc"].items():
        print(f"    {k:>6} : {v if v is None else f'{v:.4f}'}")
    print(f"  Hemisphere strat:")
    for h, m in metrics["hemi"].items():
        if m is None:
            print(f"    {h}: skipped (insufficient)")
        else:
            print(f"    {h}: AUC={m['auc_roc']:.4f}  Brier={m['brier']:.4f}  n={m['n_test']}")

    # gate check
    gate = {
        "auc_geq_0.85": bool(metrics["auc_roc"] >= 0.85),
        "brier_leq_0.10": bool(metrics["brier"] <= 0.10),
        "reliability_dev_lt_0.05": bool(metrics["reliability_deviation"] < 0.05),
    }
    print(f"\n[R002] MVP GATE:")
    for k, v in gate.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    # save artifacts
    model.save_model(f"{OUT_DIR}/r002_model.ubj")
    if iso is not None:
        with open(f"{OUT_DIR}/r002_isotonic.pkl", "wb") as f:
            pickle.dump(iso, f)
    with open(f"{OUT_DIR}/r002_features.json", "w") as f:
        json.dump(feat_names, f)
    summary = {
        "run_id": "R002",
        "milestone": "M1",
        "purpose": "B1 baseline",
        "n_train_rows": int(len(splits["train"])),
        "n_val_rows": int(len(splits["val"])),
        "n_cal_rows": int(len(splits["cal"])),
        "n_test_rows": int(len(splits["test"])),
        "best_iter": int(model.best_iteration),
        "calibration_info": cal_info,
        "test_metrics": metrics,
        "mvp_gate": gate,
        "elapsed_sec": float(time.time() - t0),
    }
    with open(f"{OUT_DIR}/r002_baseline_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n  saved model + metrics to {OUT_DIR}/r002_*")
    print(f"  R002 elapsed: {time.time()-t0:.1f}s")
    return metrics["auc_roc"], trained


def run_R003(splits, sw_cols, baseline_auc):
    print("\n" + "=" * 70)
    print("[R003] B2 shuffled-SW control: row-shuffle SW cols, retrain")
    print("=" * 70)
    t0 = time.time()
    # split SW cols from spatial cols
    spatial_cols = ["x_polar", "y_polar", "hemi_code"]
    sw_only = [c for c in sw_cols if c not in spatial_cols]

    # shuffle SW columns row-wise in train (decouple SW from labels)
    train_shuf = splits["train"].copy().reset_index(drop=True)
    rng = np.random.default_rng(123)
    perm = rng.permutation(len(train_shuf))
    train_shuf.loc[:, sw_only] = train_shuf.loc[:, sw_only].values[perm]
    print(f"  shuffled {len(sw_only)} SW columns in train (kept spatial intact)")

    # also shuffle val (so early stopping signal is consistent) but NOT cal/test
    val_shuf = splits["val"].copy().reset_index(drop=True)
    perm_val = np.random.default_rng(124).permutation(len(val_shuf))
    val_shuf.loc[:, sw_only] = val_shuf.loc[:, sw_only].values[perm_val]

    X_train, feat_names = build_feature_matrix(train_shuf, sw_cols)
    X_val, _ = build_feature_matrix(val_shuf, sw_cols)
    X_cal, _ = build_feature_matrix(splits["cal"], sw_cols)
    y_train = train_shuf["label"].values.astype(int)
    y_val = val_shuf["label"].values.astype(int)
    y_cal = splits["cal"]["label"].values.astype(int)

    print("[R003] fitting XGBoost on shuffled-SW train ...")
    t1 = time.time()
    model = fit_xgb(X_train, y_train, X_val, y_val)
    print(f"  trained in {time.time()-t1:.1f}s, best_iter={model.best_iteration}")

    iso, cal_info = maybe_calibrate(model, X_cal, y_cal, deviation_threshold=0.05)
    trained = TrainedModel(model=model, isotonic=iso, feature_names=feat_names,
                           used_calibration=(iso is not None))

    metrics = evaluate(trained, splits["test"], sw_cols)
    print(f"\n[R003] TEST METRICS (shuffled SW):")
    print(f"  AUC-ROC:        {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:         {metrics['auc_pr']:.4f}")
    print(f"  Brier:          {metrics['brier']:.4f}")

    gap = baseline_auc - metrics["auc_roc"]
    gate = bool(gap >= 0.10)
    print(f"\n[R003] NOVELTY CONTROL:")
    print(f"  Real AUC      : {baseline_auc:.4f}")
    print(f"  Shuffled AUC  : {metrics['auc_roc']:.4f}")
    print(f"  Gap           : {gap:.4f}")
    print(f"  Gate gap >= 0.10:  {'PASS' if gate else 'FAIL'}")

    summary = {
        "run_id": "R003",
        "milestone": "M2",
        "purpose": "B2 shuffled-SW control",
        "best_iter": int(model.best_iteration),
        "calibration_info": cal_info,
        "test_metrics": metrics,
        "baseline_auc": float(baseline_auc),
        "shuffled_auc": float(metrics["auc_roc"]),
        "auc_gap": float(gap),
        "gate_pass": gate,
        "elapsed_sec": float(time.time() - t0),
    }
    with open(f"{OUT_DIR}/r003_shuffled_sw_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  R003 elapsed: {time.time()-t0:.1f}s")
    return gate


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    df_clean, expanded = prepare_data()
    sw_cols = sw_feature_names(expanded)
    print(f"\n[prep] expanded rows: {len(expanded)}")
    print(f"  positives: {(expanded['label']==1).sum()}, negatives: {(expanded['label']==0).sum()}")
    print(f"  SW features: {len(sw_cols)}; total features (+x,y): {len(sw_cols) + 2 - sum(1 for c in ['hemi_code'] if c in sw_cols)}")

    print("\n[prep] crossing-level split 60/10/10/20 ...")
    splits = crossing_random_split(expanded, frac_test=0.2, frac_val=0.1, frac_cal=0.1, seed=42)
    for k, v in splits.items():
        n_cross = v["crossing_id"].nunique()
        print(f"  {k:>5}: {len(v):>8} rows from {n_cross:>5} crossings")

    baseline_auc, trained = run_R002(splits, sw_cols)
    novelty_pass = run_R003(splits, sw_cols, baseline_auc)

    print(f"\n[summary] total wall: {time.time()-t0:.1f}s")
    print(f"[summary] R002 AUC: {baseline_auc:.4f}")
    print(f"[summary] R003 novelty gate: {'PASS' if novelty_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
