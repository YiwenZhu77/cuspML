"""R025 — hybrid: real per-pass near negatives + synthetic dial-random far negatives.

Per crossing event:
  - 5 real positives (1Hz cusp spectra)
  - 5 real per-pass NEAR negatives (1Hz non-cusp from same orbital pass, +/-5min)
  - 5 SYNTHETIC far negatives at random (|MLAT|, MLT) on the dial (same SW
    as the event)

Combines R002's coverage with R023's real boundary precision.
"""
import argparse, json, os, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, "/glade/work/yizhu/cuspML/src/lib")
sys.path.insert(0, "/glade/work/yizhu/cuspML/src/scripts/mvp")
from cusp_map import polar_xy, fit_xgb, maybe_calibrate, TrainedModel, load_crossings
from cusp_map import _sample_far_negatives
from omni_1min import load_omni_1min, compute_history, derive_paper1_features
from r021_train_real_negs import SW_FEATURE_COLS, HIST_COLS, DERIVED_HIST
from r023_train_perpass_negs import build_perpass_dataset, NEAR_WINDOW_SEC, PAD_SEC

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles"
PILOT_DIR = "/glade/work/yizhu/cuspML/output/pilot_spectra"
OMNI_MIN_TEMPLATE = "/glade/work/yizhu/cuspML/output/omni_raw/omni_min{year}.asc"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sat", default="F10")
    p.add_argument("--years", nargs="+", type=int, default=[1993, 1994])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    print(f"[R025] loading pilot spectra ...")
    dfs = [pd.read_parquet(f"{PILOT_DIR}/pilot_spectra_{args.sat}_{y}.parquet").assign(year=y)
           for y in args.years]
    spectra = pd.concat(dfs, ignore_index=True)

    # Real near pos+neg via R023 builder
    real_part = build_perpass_dataset(spectra, n_pos=5, k_neg=5, rng=rng)
    print(f"  real near part: {len(real_part)} rows ({int((real_part['label']==1).sum())} pos)")

    # For each event in real_part, generate 5 synthetic far negatives at random
    # (|MLAT|, MLT) sharing the event's SW state (we'll use the FIRST positive's
    # time as the event's representative SW timestamp).
    event_times = {}
    for _, r in real_part[real_part["label"] == 1].iterrows():
        event_times.setdefault(int(r["event_id"]), r["time"])
    print(f"  generating synth-far negs for {len(event_times)} events ...")
    synth_rows = []
    for eid, t_event in event_times.items():
        # pick positive's spatial range to define exclude box
        ev_rows = real_part[real_part["event_id"] == eid]
        ev_pos = ev_rows[ev_rows["label"] == 1]
        eq_lat = float(ev_pos["abs_mlat"].min())
        pole_lat = float(ev_pos["abs_mlat"].max())
        eq_mlt = float(ev_pos["mlt"].min())
        pole_mlt = float(ev_pos["mlt"].max())
        far_lats, far_mlts = _sample_far_negatives(
            5, exclude_box=(max(50, eq_lat - 2), min(90, pole_lat + 2),
                             max(0, eq_mlt - 1), min(24, pole_mlt + 1)),
            dial_box=(50, 90, 5, 19), rng=rng)
        for lat, mlt in zip(far_lats, far_mlts):
            synth_rows.append({"time": t_event, "satellite": ev_pos.iloc[0]["satellite"],
                                "hemisphere": ev_pos.iloc[0]["hemisphere"],
                                "abs_mlat": lat, "mlt": mlt, "cusp_mask": 0,
                                "year": ev_pos.iloc[0]["year"], "event_id": eid, "label": 0})
    synth_df = pd.DataFrame(synth_rows)
    print(f"  synth far: {len(synth_df)} rows")
    sample = pd.concat([real_part, synth_df], ignore_index=True)
    print(f"  combined: {len(sample)} ({int((sample['label']==1).sum())} pos / {int((sample['label']==0).sum())} neg)")

    # OMNI 1-min match for all rows (real + synth)
    print("[R025] OMNI 1-min match ...")
    om_parts = []
    for y in args.years:
        om = load_omni_1min(OMNI_MIN_TEMPLATE.format(year=y))
        om = compute_history(om); om = derive_paper1_features(om)
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

    feats = list(dict.fromkeys([f for f in SW_FEATURE_COLS + HIST_COLS + DERIVED_HIST + ["x_polar", "y_polar"] if f in sample.columns]))

    base_req = ["imf_bx", "imf_by", "imf_bz", "sw_v", "sw_n", "sw_pdyn"]
    before = len(sample)
    sample = sample.dropna(subset=base_req).reset_index(drop=True)
    print(f"  dropna(base): {before} -> {len(sample)}")
    sample[feats] = sample[feats].fillna(0.0)

    # group split by event_id
    event_ids = sample["event_id"].unique()
    rng.shuffle(event_ids)
    n_test = int(len(event_ids) * 0.2)
    n_val = int(len(event_ids) * 0.1)
    n_cal = int(len(event_ids) * 0.1)
    test_e = set(event_ids[:n_test])
    val_e = set(event_ids[n_test:n_test+n_val])
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
    auc = float(roc_auc_score(yt, p_test))
    ap = float(average_precision_score(yt, p_test))
    brier = float(brier_score_loss(yt, p_test))
    print(f"\n[R025 hybrid] TEST: AUC {auc:.4f}  AP {ap:.4f}  Brier {brier:.4f}")

    tag = f"hybrid_{args.sat}_{'_'.join(map(str, args.years))}"
    model.save_model(f"{OUT_DIR}/r025_{tag}_model.ubj")
    if iso is not None:
        with open(f"{OUT_DIR}/r025_{tag}_isotonic.pkl", "wb") as f: pickle.dump(iso, f)
    with open(f"{OUT_DIR}/r025_{tag}_features.json", "w") as f: json.dump(feats, f)
    with open(f"{OUT_DIR}/r025_{tag}_results.json", "w") as f:
        json.dump({"test_auc": auc, "test_ap": ap, "test_brier": brier,
                   "elapsed_sec": float(time.time() - t0)}, f, indent=2)
    print(f"  saved, elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
