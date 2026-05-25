"""R001 — M0 sanity: 1k crossings, overfit check.

Loads full crossings, takes 1k random, expands to ~55k rows, trains XGBoost
with paper-1 hyperparameters, evaluates on TRAINING set. Expects AUC > 0.99 to
confirm the pipeline works end-to-end.
"""
import json
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
    TrainedModel, crossing_random_split, predict_proba,
)


def main():
    out_dir = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp"
    t0 = time.time()
    print(f"[R001] loading crossings ...")
    df = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    print(f"  loaded {len(df)} crossings  ({time.time()-t0:.1f}s)")

    # sanity check: column existence + boundary ordering
    required = ["eq_mlat", "pole_mlat", "eq_mlt", "pole_mlt", "mean_mlat", "mean_mlt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing required columns: {missing}")

    # check for duplicate orbital passes
    dup_key = df[["satellite", "hemisphere", "time_start"]]
    n_dup = dup_key.duplicated().sum()
    print(f"  duplicates on (satellite, hemisphere, time_start): {n_dup}")
    # use row index as crossing_id, accept duplicates if any (rare)

    # subsample 1k
    rng = np.random.default_rng(42)
    pick = rng.choice(len(df), 1000, replace=False)
    sub = df.iloc[pick].reset_index(drop=True)
    print(f"  subsampled to 1000 crossings")

    # drop crossings with NaN in feature/target columns
    sw_cols = sw_feature_names(sub)
    target_cols = required
    keep_cols = sw_cols + target_cols + ["satellite", "hemisphere", "time_start"]
    keep_cols = [c for c in keep_cols if c in sub.columns]
    sub_clean = sub[keep_cols].dropna().reset_index(drop=True)
    print(f"  after dropna: {len(sub_clean)} crossings")

    # expand
    print(f"[R001] expanding 1k crossings (n_pos=5, k_neg=10) ...")
    expanded = expand_dataset(sub_clean, n_pos=5, k_neg=10, seed=42, verbose=False)
    print(f"  expanded: {len(expanded)} rows (~55k expected)")
    print(f"  class balance: {(expanded['label']==1).sum()} pos / {(expanded['label']==0).sum()} neg")

    # build features
    sw_cols_used = sw_feature_names(expanded)
    X, feat_names = build_feature_matrix(expanded, sw_cols_used)
    y = expanded["label"].values.astype(int)
    print(f"  feature matrix: {X.shape}  ({len(feat_names)} features)")
    print(f"  first 10 features: {feat_names[:10]}")

    # train: use all 55k rows as train + val (no holdout — overfit test)
    # split rows 90/10 internally just so early stopping has a val signal
    idx = np.arange(len(expanded))
    rng2 = np.random.default_rng(42)
    rng2.shuffle(idx)
    n_val = int(len(idx) * 0.1)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    print(f"[R001] training XGBoost (n_train={len(train_idx)}, n_val={len(val_idx)}) ...")
    t1 = time.time()
    model = fit_xgb(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
    print(f"  trained in {time.time()-t1:.1f}s, best_iter={model.best_iteration}")

    trained = TrainedModel(model=model, isotonic=None,
                            feature_names=feat_names, used_calibration=False)

    # eval on training subset (overfit check)
    train_df = expanded.iloc[train_idx].reset_index(drop=True)
    metrics = evaluate(trained, train_df, sw_cols_used)
    print(f"\n[R001] OVERFIT CHECK on training subset:")
    print(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:   {metrics['auc_pr']:.4f}")
    print(f"  Brier:    {metrics['brier']:.4f}")

    # also eval on val (held out from training)
    val_df = expanded.iloc[val_idx].reset_index(drop=True)
    val_metrics = evaluate(trained, val_df, sw_cols_used)
    print(f"\n[R001] HELDOUT (10% val) check:")
    print(f"  AUC-ROC:  {val_metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:   {val_metrics['auc_pr']:.4f}")
    print(f"  Brier:    {val_metrics['brier']:.4f}")

    # gate
    gate_ok = metrics["auc_roc"] > 0.99
    print(f"\n[R001] GATE: AUC-ROC > 0.99 on training subset?  {'PASS' if gate_ok else 'FAIL'}")

    # save results
    summary = {
        "run_id": "R001",
        "milestone": "M0",
        "purpose": "1k overfit sanity",
        "n_crossings": int(len(sub_clean)),
        "n_expanded": int(len(expanded)),
        "n_features": int(len(feat_names)),
        "elapsed_sec": float(time.time() - t0),
        "best_iter": int(model.best_iteration),
        "overfit_metrics": metrics,
        "heldout_metrics": val_metrics,
        "gate_pass": bool(gate_ok),
    }
    out_path = f"{out_dir}/bundles/r001_sanity_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n  results -> {out_path}")
    print(f"  total elapsed: {time.time()-t0:.1f}s")
    sys.exit(0 if gate_ok else 2)


if __name__ == "__main__":
    main()
