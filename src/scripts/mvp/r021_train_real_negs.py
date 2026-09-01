"""R021 (pilot) — train stage 2 on REAL negatives from R020 spectra.

Load pilot spectra parquet (F10 1993 + 1994), match each row to OMNI 1-min
features, stratified-sample at 1 pos : K neg, train XGBoost stage 2 with
paper-1 hyperparameters. Eval on a held-out slice.
"""
import argparse
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
from cusp_map import polar_xy, fit_xgb, maybe_calibrate, TrainedModel
from omni_1min import load_omni_1min, compute_history, derive_paper1_features

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles"
PILOT_DIR = "/glade/work/yizhu/cuspML/output/pilot_spectra"
OMNI_MIN_TEMPLATE = "/glade/work/yizhu/cuspML/output/omni_raw/omni_min{year}.asc"


SW_FEATURE_COLS = [
    # base 16
    "dipole_tilt", "hemi_code", "doy",
    "imf_bx", "imf_by", "imf_bz",
    "sw_v", "sw_n", "sw_pdyn",
    "B_T", "clock_angle", "sin_clock_half",
    "newell_cf", "kan_lee_ef", "vBs", "by_hemi",
]
# 6 base history vars (excluding redundant by_hemi/clock features that aren't in OMNI raw)
# We compute history for imf_bx, by, bz, sw_v, sw_n, sw_pdyn at {15, 30, 60} min for mean/std/delta
HIST_COLS = [
    f"{v}_{stat}{w}"
    for v in ["imf_bx", "imf_by", "imf_bz", "sw_v", "sw_n", "sw_pdyn"]
    for stat in ["mean", "std", "delta"]
    for w in [15, 30, 60]
]
# 4 derived history
DERIVED_HIST = ["newell_cf_mean60", "newell_cf_int60", "vBs_mean60", "vBs_int60"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k-neg", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--years", nargs="+", type=int, default=[1993, 1994])
    p.add_argument("--sat", default="F10")
    p.add_argument("--max-pos", type=int, default=0, help="cap positives (0=unlimited)")
    args = p.parse_args()

    t0 = time.time()
    print(f"[R021] loading pilot spectra for {args.sat} years {args.years} ...")
    dfs = []
    for y in args.years:
        path = f"{PILOT_DIR}/pilot_spectra_{args.sat}_{y}.parquet"
        if not os.path.exists(path):
            print(f"  MISSING {path}, skip")
            continue
        df = pd.read_parquet(path)
        df["year"] = y
        dfs.append(df)
    spectra = pd.concat(dfs, ignore_index=True)
    print(f"  total spectra: {len(spectra)}  (pos {int(spectra['cusp_mask'].sum())}, "
          f"neg {int(len(spectra) - spectra['cusp_mask'].sum())})")

    # stratified sample 1 pos : K neg, with negatives stratified to match
    # positive lat distribution (avoids "orbital duty cycle" prior contamination).
    rng = np.random.default_rng(args.seed)
    pos = spectra[spectra["cusp_mask"] == 1].reset_index(drop=True)
    neg = spectra[spectra["cusp_mask"] == 0].reset_index(drop=True)
    if args.max_pos and len(pos) > args.max_pos:
        idx = rng.choice(len(pos), args.max_pos, replace=False)
        pos = pos.iloc[idx].reset_index(drop=True)

    # bin lat at 2-degree resolution; for each bin, sample K * n_pos_in_bin negatives
    lat_bins = np.arange(50, 91, 2.0)  # 50, 52, ..., 90
    pos_bin = np.digitize(pos["abs_mlat"].values, lat_bins)
    neg_bin = np.digitize(neg["abs_mlat"].values, lat_bins)
    neg_keep_idx = []
    for b in np.unique(pos_bin):
        n_pos_b = int((pos_bin == b).sum())
        n_target = n_pos_b * args.k_neg
        avail = np.where(neg_bin == b)[0]
        if len(avail) == 0:
            continue
        take = min(n_target, len(avail))
        chosen = rng.choice(avail, take, replace=False)
        neg_keep_idx.extend(chosen.tolist())
    neg = neg.iloc[neg_keep_idx].reset_index(drop=True)
    sample = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print(f"  lat-stratified sample: {len(sample)} rows ({len(pos)} pos / {len(neg)} neg)")
    print(f"  pos lat: mean={pos['abs_mlat'].mean():.1f}, std={pos['abs_mlat'].std():.1f}")
    print(f"  neg lat: mean={neg['abs_mlat'].mean():.1f}, std={neg['abs_mlat'].std():.1f}")

    # load OMNI 1-min for the years involved
    print("[R021] loading + processing 1-min OMNI ...")
    omni_parts = []
    for y in args.years:
        omni_path = OMNI_MIN_TEMPLATE.format(year=y)
        if not os.path.exists(omni_path):
            print(f"  WARN missing {omni_path}, downloading ...")
            import urllib.request
            url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_min{y}.asc"
            urllib.request.urlretrieve(url, omni_path)
        om = load_omni_1min(omni_path)
        om = compute_history(om)
        om = derive_paper1_features(om)
        omni_parts.append(om)
    omni = pd.concat(omni_parts, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    print(f"  OMNI rows: {len(omni)}")

    # nearest-neighbor match (within +/- 5 min)
    print("[R021] matching OMNI to spectra ...")
    t1 = time.time()
    sample = sample.sort_values("time").reset_index(drop=True)
    omni_t = omni["datetime"].values.astype("datetime64[s]").astype(np.int64)
    sample_t = pd.to_datetime(sample["time"]).values.astype("datetime64[s]").astype(np.int64)
    idx = np.searchsorted(omni_t, sample_t)
    idx = np.clip(idx, 0, len(omni_t) - 1)
    idx_left = np.clip(idx - 1, 0, len(omni_t) - 1)
    # pick whichever is closer
    d_right = np.abs(omni_t[idx] - sample_t)
    d_left = np.abs(omni_t[idx_left] - sample_t)
    use_left = d_left < d_right
    pick = np.where(use_left, idx_left, idx)
    dt_sec = np.where(use_left, d_left, d_right)

    omni_sub = omni.iloc[pick].reset_index(drop=True)
    sample["omni_dt_sec"] = dt_sec
    for c in SW_FEATURE_COLS + HIST_COLS + DERIVED_HIST:
        if c in omni_sub.columns:
            sample[c] = omni_sub[c].values
    sample["hemi_code"] = np.where(sample["hemisphere"] == "N", 1.0, 0.0)
    # recompute hemi-dependent by_hemi (OMNI's was assumed N)
    sample["by_hemi"] = sample["imf_by"] * np.where(sample["hemisphere"] == "N", 1.0, -1.0)
    # dipole tilt at sample time (was 0 in OMNI loader)
    sys.path.insert(0, "/glade/work/yizhu/cuspML/src")
    from identify_cusp import dipole_tilt_angle
    sample_dt = pd.to_datetime(sample["time"]).dt
    sample["dipole_tilt"] = [dipole_tilt_angle(t.to_pydatetime()) for t in pd.to_datetime(sample["time"])]
    print(f"  match done in {time.time()-t1:.1f}s, median dt = {np.median(dt_sec):.0f}s")

    # spatial encoding
    x, y = polar_xy(sample["abs_mlat"].values, sample["mlt"].values)
    sample["x_polar"] = x; sample["y_polar"] = y

    feats = SW_FEATURE_COLS + HIST_COLS + DERIVED_HIST + ["x_polar", "y_polar"]
    feats = [f for f in feats if f in sample.columns]
    # dedupe
    feats = list(dict.fromkeys(feats))
    print(f"  features: {len(feats)}")

    # drop rows where ANY base SW feature is missing (these are not recoverable)
    # but fill NaN in history/derived features with 0 (XGBoost handles partial features fine)
    base_required = ["imf_bx", "imf_by", "imf_bz", "sw_v", "sw_n", "sw_pdyn"]
    before = len(sample)
    sample = sample.dropna(subset=base_required).reset_index(drop=True)
    n_after_base = len(sample)
    print(f"  dropna(base only): {before} -> {n_after_base} rows (kept {100*n_after_base/before:.1f}%)")
    hist_to_fill = [c for c in feats if c not in base_required]
    sample[hist_to_fill] = sample[hist_to_fill].fillna(0.0)
    # final NaN check
    bad = sample[feats].isna().any(axis=1)
    if bad.any():
        print(f"  WARN: {bad.sum()} rows still have NaN after fillna, dropping")
        sample = sample[~bad].reset_index(drop=True)
    print(f"  final: {len(sample)} rows")

    X = sample[feats].values.astype(np.float32)
    y_lab = sample["cusp_mask"].values.astype(int)
    print(f"  final pos/neg: {int(y_lab.sum())} / {int(len(y_lab) - y_lab.sum())}")

    # 60/10/10/20 split — group by hour-bin to avoid leakage
    sample["hour_bin"] = pd.to_datetime(sample["time"]).dt.floor("h").astype(np.int64)
    hour_ids = sample["hour_bin"].values
    unique_hours = np.unique(hour_ids)
    rng.shuffle(unique_hours)
    n_test = int(len(unique_hours) * 0.2)
    n_val = int(len(unique_hours) * 0.1)
    n_cal = int(len(unique_hours) * 0.1)
    test_h = set(unique_hours[:n_test])
    val_h = set(unique_hours[n_test:n_test + n_val])
    cal_h = set(unique_hours[n_test + n_val:n_test + n_val + n_cal])
    train_h = set(unique_hours[n_test + n_val + n_cal:])

    train_mask = np.isin(hour_ids, list(train_h))
    val_mask = np.isin(hour_ids, list(val_h))
    cal_mask = np.isin(hour_ids, list(cal_h))
    test_mask = np.isin(hour_ids, list(test_h))
    print(f"  splits (rows): train={train_mask.sum()}  val={val_mask.sum()}  cal={cal_mask.sum()}  test={test_mask.sum()}")

    print("[R021] training XGBoost ...")
    t2 = time.time()
    # scale_pos_weight = (n_neg/n_pos) in training set
    n_pos_tr = max(1, int(y_lab[train_mask].sum()))
    n_neg_tr = max(1, int((1 - y_lab[train_mask]).sum()))
    spw = n_neg_tr / n_pos_tr
    print(f"  scale_pos_weight = {spw:.2f}")
    model = fit_xgb(X[train_mask], y_lab[train_mask], X[val_mask], y_lab[val_mask],
                    hp_overrides={"scale_pos_weight": spw})
    print(f"  trained in {time.time()-t2:.1f}s, best_iter={model.best_iteration}")

    iso, cal_info = maybe_calibrate(model, X[cal_mask], y_lab[cal_mask], deviation_threshold=0.05)

    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    raw_test = model.predict_proba(X[test_mask])[:, 1]
    p_test = iso.transform(raw_test) if iso is not None else raw_test
    y_test = y_lab[test_mask]
    auc = float(roc_auc_score(y_test, p_test))
    ap = float(average_precision_score(y_test, p_test))
    brier = float(brier_score_loss(y_test, p_test))
    print(f"\n[R021] TEST METRICS (real negatives, K={args.k_neg}):")
    print(f"  AUC-ROC: {auc:.4f}  AUC-PR: {ap:.4f}  Brier: {brier:.4f}")
    print(f"  Compare R002 (synthetic): AUC 0.9283  Brier 0.0562 (different test set, not directly comparable)")

    tag = f"K{args.k_neg}_{args.sat}_{'_'.join(map(str, args.years))}"
    out_path = f"{OUT_DIR}/r021_real_negs_{tag}.json"
    summary = {
        "run_id": "R021", "sat": args.sat, "years": args.years,
        "k_neg": args.k_neg,
        "n_total_pilot_spectra": int(len(spectra)),
        "n_pos_used": int(len(pos)), "n_neg_sampled": int(len(neg)),
        "n_features": len(feats),
        "test_metrics": {"auc_roc": auc, "auc_pr": ap, "brier": brier},
        "best_iter": int(model.best_iteration),
        "scale_pos_weight": float(spw),
        "calibration_info": cal_info,
        "elapsed_sec": float(time.time() - t0),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    model.save_model(f"{OUT_DIR}/r021_real_negs_{tag}_model.ubj")
    if iso is not None:
        with open(f"{OUT_DIR}/r021_real_negs_{tag}_isotonic.pkl", "wb") as f:
            pickle.dump(iso, f)
    with open(f"{OUT_DIR}/r021_real_negs_{tag}_features.json", "w") as f:
        json.dump(feats, f)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
