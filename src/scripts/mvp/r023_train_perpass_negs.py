"""R023 — train stage 2 on per-pass near-temporal real negatives.

Per the diagnosis from R021 failure: real negatives sampled uniformly across
years lose the "same SW state, multiple spatial samples" structure that lets
R002 learn SW->spatial mapping. Fix: for each crossing in the 48k table,
emit a positive spectrum from inside the cusp window AND K near-temporal
non-cusp spectra from the SAME orbital pass (5 min window before + after
cusp). All rows for one crossing share approximately the same SW state,
matching R002's data structure.

Loads R020 emitted spectra (full 1Hz parquet) and groups by orbital pass.
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
from cusp_map import polar_xy, fit_xgb, maybe_calibrate, TrainedModel, load_crossings
from omni_1min import load_omni_1min, compute_history, derive_paper1_features
from r021_train_real_negs import SW_FEATURE_COLS, HIST_COLS, DERIVED_HIST

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles"
PILOT_DIR = "/glade/work/yizhu/cuspML/output/pilot_spectra"
OMNI_MIN_TEMPLATE = "/glade/work/yizhu/cuspML/output/omni_raw/omni_min{year}.asc"

# per-positive: how many 1Hz near-negatives to take from the surrounding orbital pass
N_POS_PER_CROSSING = 5
K_NEG_PER_POS = 10
NEAR_WINDOW_SEC = 300  # within +/-5 min of cusp window
PAD_SEC = 30           # buffer: drop negatives within +/-30s of cusp window boundary


def build_perpass_dataset(spectra: pd.DataFrame, n_pos=N_POS_PER_CROSSING,
                          k_neg=K_NEG_PER_POS, rng=None):
    """Group 1Hz spectra into orbital cusp events and sample per-pass."""
    rng = rng or np.random.default_rng(42)
    spectra = spectra.sort_values("time").reset_index(drop=True)
    # detect contiguous cusp runs (orbital cusp windows)
    is_cusp = spectra["cusp_mask"].values == 1
    # gap > 30s between consecutive cusps -> new event
    t_int = spectra["time"].astype(np.int64).values // 10**9  # seconds
    starts, ends = [], []
    i = 0
    while i < len(is_cusp):
        if is_cusp[i]:
            j = i
            while j + 1 < len(is_cusp):
                if is_cusp[j + 1] and (t_int[j + 1] - t_int[j]) <= 30:
                    j += 1
                else:
                    break
            starts.append(i); ends.append(j)
            i = j + 1
        else:
            i += 1
    print(f"  detected {len(starts)} cusp events")

    rows_out = []
    for evi, (i0, i1) in enumerate(zip(starts, ends)):
        t_start = t_int[i0]; t_end = t_int[i1]
        # positives: random n_pos within [i0, i1]
        n_p = min(n_pos, i1 - i0 + 1)
        pos_idxs = rng.choice(np.arange(i0, i1 + 1), n_p, replace=False)

        # near negatives: spectra within +/-NEAR_WINDOW_SEC of cusp window, excluding pad zone
        early = (t_int >= t_start - NEAR_WINDOW_SEC) & (t_int < t_start - PAD_SEC) & (~is_cusp)
        late = (t_int > t_end + PAD_SEC) & (t_int <= t_end + NEAR_WINDOW_SEC) & (~is_cusp)
        neg_pool = np.where(early | late)[0]
        if len(neg_pool) == 0:
            continue
        n_neg = min(n_p * k_neg, len(neg_pool))
        neg_idxs = rng.choice(neg_pool, n_neg, replace=False)

        for idx in pos_idxs:
            rec = spectra.iloc[idx].to_dict()
            rec["event_id"] = evi
            rec["label"] = 1
            rows_out.append(rec)
        for idx in neg_idxs:
            rec = spectra.iloc[idx].to_dict()
            rec["event_id"] = evi
            rec["label"] = 0
            rows_out.append(rec)
    print(f"  per-pass dataset: {len(rows_out)} rows from {len(starts)} events")
    return pd.DataFrame(rows_out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sat", default="F10")
    p.add_argument("--years", nargs="+", type=int, default=[1993, 1994])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    t0 = time.time()
    print(f"[R023] loading pilot spectra ...")
    dfs = []
    for y in args.years:
        path = f"{PILOT_DIR}/pilot_spectra_{args.sat}_{y}.parquet"
        df = pd.read_parquet(path)
        df["year"] = y
        dfs.append(df)
    spectra = pd.concat(dfs, ignore_index=True)
    print(f"  total spectra: {len(spectra)}")

    rng = np.random.default_rng(args.seed)
    sample = build_perpass_dataset(spectra, rng=rng)
    print(f"  pos: {int((sample['label']==1).sum())}, neg: {int((sample['label']==0).sum())}")

    # OMNI match
    print("[R023] OMNI 1-min features ...")
    om_parts = []
    for y in args.years:
        om = load_omni_1min(OMNI_MIN_TEMPLATE.format(year=y))
        om = compute_history(om)
        om = derive_paper1_features(om)
        om_parts.append(om)
    omni = pd.concat(om_parts, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    om_t = omni["datetime"].values.astype("datetime64[s]").astype(np.int64)
    s_t = pd.to_datetime(sample["time"]).values.astype("datetime64[s]").astype(np.int64)
    idx = np.searchsorted(om_t, s_t); idx = np.clip(idx, 0, len(om_t) - 1)
    idx_left = np.clip(idx - 1, 0, len(om_t) - 1)
    d_right = np.abs(om_t[idx] - s_t); d_left = np.abs(om_t[idx_left] - s_t)
    pick = np.where(d_left < d_right, idx_left, idx)
    om_sub = omni.iloc[pick].reset_index(drop=True)
    for c in SW_FEATURE_COLS + HIST_COLS + DERIVED_HIST:
        if c in om_sub.columns:
            sample[c] = om_sub[c].values
    sample["hemi_code"] = np.where(sample["hemisphere"] == "N", 1.0, 0.0)
    sample["by_hemi"] = sample["imf_by"] * np.where(sample["hemisphere"] == "N", 1.0, -1.0)
    sys.path.insert(0, "/glade/work/yizhu/cuspML/src")
    from identify_cusp import dipole_tilt_angle
    sample["dipole_tilt"] = [dipole_tilt_angle(t.to_pydatetime()) for t in pd.to_datetime(sample["time"])]

    x, y = polar_xy(sample["abs_mlat"].values, sample["mlt"].values)
    sample["x_polar"] = x; sample["y_polar"] = y

    feats = SW_FEATURE_COLS + HIST_COLS + DERIVED_HIST + ["x_polar", "y_polar"]
    feats = list(dict.fromkeys([f for f in feats if f in sample.columns]))

    base_required = ["imf_bx", "imf_by", "imf_bz", "sw_v", "sw_n", "sw_pdyn"]
    before = len(sample)
    sample = sample.dropna(subset=base_required).reset_index(drop=True)
    print(f"  dropna(base): {before} -> {len(sample)}")
    sample[feats] = sample[feats].fillna(0.0)

    # Split by event_id (groups all rows from one cusp event together)
    event_ids = sample["event_id"].unique()
    rng.shuffle(event_ids)
    n_test = int(len(event_ids) * 0.2)
    n_val = int(len(event_ids) * 0.1)
    n_cal = int(len(event_ids) * 0.1)
    test_e = set(event_ids[:n_test]); val_e = set(event_ids[n_test:n_test+n_val])
    cal_e = set(event_ids[n_test+n_val:n_test+n_val+n_cal])
    train_e = set(event_ids[n_test+n_val+n_cal:])

    masks = {k: sample["event_id"].isin(s).values for k, s in
             [("train", train_e), ("val", val_e), ("cal", cal_e), ("test", test_e)]}
    for k, m in masks.items():
        print(f"  {k}: {m.sum()} rows from {sample.loc[m, 'event_id'].nunique()} events")

    X = sample[feats].values.astype(np.float32)
    y_lab = sample["label"].values.astype(int)
    n_pos_tr = max(1, int(y_lab[masks["train"]].sum()))
    n_neg_tr = max(1, int((1 - y_lab[masks["train"]]).sum()))
    spw = n_neg_tr / n_pos_tr
    print(f"  scale_pos_weight = {spw:.2f}")
    model = fit_xgb(X[masks["train"]], y_lab[masks["train"]],
                    X[masks["val"]], y_lab[masks["val"]],
                    hp_overrides={"scale_pos_weight": spw})

    iso, cal_info = maybe_calibrate(model, X[masks["cal"]], y_lab[masks["cal"]], deviation_threshold=0.05)

    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    raw = model.predict_proba(X[masks["test"]])[:, 1]
    p_test = iso.transform(raw) if iso is not None else raw
    yt = y_lab[masks["test"]]
    print(f"\n[R023] TEST:")
    print(f"  AUC-ROC {roc_auc_score(yt, p_test):.4f}  AUC-PR {average_precision_score(yt, p_test):.4f}  Brier {brier_score_loss(yt, p_test):.4f}")

    tag = f"perpass_{args.sat}_{'_'.join(map(str, args.years))}"
    model.save_model(f"{OUT_DIR}/r023_{tag}_model.ubj")
    if iso is not None:
        with open(f"{OUT_DIR}/r023_{tag}_isotonic.pkl", "wb") as f: pickle.dump(iso, f)
    with open(f"{OUT_DIR}/r023_{tag}_features.json", "w") as f: json.dump(feats, f)
    summary = {
        "run_id": "R023", "sat": args.sat, "years": args.years,
        "n_pos": int((y_lab == 1).sum()), "n_neg": int((y_lab == 0).sum()),
        "test_auc": float(roc_auc_score(yt, p_test)),
        "test_ap": float(average_precision_score(yt, p_test)),
        "test_brier": float(brier_score_loss(yt, p_test)),
        "calibration_info": cal_info,
        "elapsed_sec": float(time.time() - t0),
    }
    with open(f"{OUT_DIR}/r023_{tag}_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  saved -> {OUT_DIR}/r023_{tag}_*  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
