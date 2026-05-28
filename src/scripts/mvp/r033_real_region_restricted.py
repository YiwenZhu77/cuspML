"""R033 — stage 2 trained on REAL DMSP 1Hz pos+neg, region-restricted.

The justified approach (post lit-verification): DMSP coverage is sufficient
for the physically-relevant cusp region (MLT 5-19, |MLAT| 60-86). So train
stage 2 on REAL DMSP 1Hz spectra:
  - positive = cusp_mask 1 spectra
  - negative = cusp_mask 0 spectra (real DMSP non-cusp observations)
both restricted to MLT [5,19] x |MLAT| [60,86]. No synthetic negatives.

Earlier pilots (R021/R023) failed because they evaluated on the FULL dial
(0-24 MLT, 50-90 MLAT) including nightside where DMSP has no data and the
model extrapolated wildly. Here we restrict BOTH training and the inference/
eval grid to the DMSP-covered cusp region.

Lat-stratified negative sampling (R021 lesson): sample negatives to match the
positive |MLAT| distribution, avoiding the orbital-duty-cycle prior that makes
the model think 'low lat = cusp'.
"""
import argparse, json, os, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, "/glade/work/yizhu/cuspML/src/lib")
sys.path.insert(0, "/glade/work/yizhu/cuspML/src/scripts/mvp")
from cusp_map import (load_crossings, sw_feature_names, polar_xy, predict_proba,
                       fit_xgb, maybe_calibrate, TrainedModel)
from cusp_stage1 import STAGE1_BASE_FEATURES
from omni_1min import load_omni_1min, compute_history, derive_paper1_features
from r021_train_real_negs import SW_FEATURE_COLS, HIST_COLS, DERIVED_HIST
from r016_combined_v2 import load_stage1_v2, stage1_scalar_v2

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles"
PILOT_DIR = "/glade/work/yizhu/cuspML/output/pilot_spectra"
OMNI_MIN_TEMPLATE = "/glade/work/yizhu/cuspML/output/omni_raw/omni_min{year}.asc"

MLT_LO, MLT_HI = 5.0, 19.0
LAT_LO, LAT_HI = 60.0, 86.0
DLAT, DMLT = 1.0, 0.5
R_LAT_AXIS = np.arange(LAT_LO + DLAT/2, LAT_HI, DLAT)
R_MLT_AXIS = np.arange(MLT_LO + DMLT/2, MLT_HI, DMLT)


def cell_of_restricted(mlat, mlt):
    mlat = abs(mlat); mlt = mlt % 24.0
    i = int(np.clip((mlat - LAT_LO) / DLAT, 0, len(R_LAT_AXIS) - 1))
    j = int(np.clip((mlt - MLT_LO) / DMLT, 0, len(R_MLT_AXIS) - 1))
    return i, j


def haversine_deg(lat1, mlt1, lat2, mlt2):
    lon1 = np.deg2rad(mlt1 * 15.0); lon2 = np.deg2rad(mlt2 * 15.0)
    phi1 = np.deg2rad(lat1); phi2 = np.deg2rad(lat2); dl = lon2 - lon1
    a = np.sin((phi2-phi1)/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return np.rad2deg(2*np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def stage2_dial_restricted(trained, sw_state, hemisphere="N"):
    MM, LL = np.meshgrid(R_MLT_AXIS, R_LAT_AXIS)
    x, y = polar_xy(LL.ravel(), MM.ravel())
    rec = dict(sw_state); rec["hemi_code"] = 1.0 if hemisphere == "N" else 0.0
    n = LL.size
    grid = {k: np.full(n, v, dtype=np.float32) for k, v in rec.items()}
    grid["x_polar"] = x.astype(np.float32); grid["y_polar"] = y.astype(np.float32)
    df = pd.DataFrame(grid)
    for f in trained.feature_names:
        if f not in df.columns: df[f] = 0.0
    X = df[trained.feature_names].values.astype(np.float32)
    return predict_proba(trained, X).reshape(LL.shape)


def normalize_pmf_restricted(P):
    MM, LL = np.meshgrid(R_MLT_AXIS, R_LAT_AXIS)
    w = np.cos(np.deg2rad(LL)); w = w / w.sum()
    weighted = P * w; Z = weighted.sum()
    return weighted / Z if Z > 1e-12 else weighted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sat-years", nargs="+", required=True)
    p.add_argument("--k-neg", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    t0 = time.time()

    sy_list = [tuple(s.split(":")) for s in args.sat_years]
    print(f"[R033] loading {len(sy_list)} sat-years, region MLT[{MLT_LO},{MLT_HI}] |MLAT|[{LAT_LO},{LAT_HI}]")
    parts = []; years_needed = set()
    for sat, yr in sy_list:
        path = f"{PILOT_DIR}/pilot_spectra_{sat}_{yr}.parquet"
        if not os.path.exists(path):
            print(f"  MISSING {path}"); continue
        df = pd.read_parquet(path)
        df = df[(df["mlt"]>=MLT_LO)&(df["mlt"]<=MLT_HI)&(df["abs_mlat"]>=LAT_LO)&(df["abs_mlat"]<=LAT_HI)].copy()
        df["year"] = int(yr); parts.append(df); years_needed.add(int(yr))
    spec = pd.concat(parts, ignore_index=True)
    n_pos_tot = int((spec["cusp_mask"]==1).sum()); n_neg_tot = int((spec["cusp_mask"]==0).sum())
    print(f"  in-region: {len(spec)} (pos {n_pos_tot}, neg {n_neg_tot}, 1:{n_neg_tot/max(1,n_pos_tot):.0f})")

    rng = np.random.default_rng(args.seed)
    pos = spec[spec["cusp_mask"]==1].reset_index(drop=True)
    neg = spec[spec["cusp_mask"]==0].reset_index(drop=True)
    lat_bins = np.arange(LAT_LO, LAT_HI+0.1, 2.0)
    pos_bin = np.digitize(pos["abs_mlat"].values, lat_bins)
    neg_bin = np.digitize(neg["abs_mlat"].values, lat_bins)
    keep = []
    for b in np.unique(pos_bin):
        nt_ = int((pos_bin==b).sum())*args.k_neg
        av = np.where(neg_bin==b)[0]
        if len(av): keep.extend(rng.choice(av, min(nt_, len(av)), replace=False).tolist())
    neg = neg.iloc[keep].reset_index(drop=True)
    sample = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print(f"  lat-stratified: {len(sample)} ({len(pos)} pos / {len(neg)} neg); pos lat {pos['abs_mlat'].mean():.1f}, neg lat {neg['abs_mlat'].mean():.1f}")

    print(f"[R033] OMNI 1-min for {sorted(years_needed)}")
    om_parts = []
    for y in sorted(years_needed):
        op = OMNI_MIN_TEMPLATE.format(year=y)
        if not os.path.exists(op): print(f"  MISSING OMNI {y}"); continue
        om = load_omni_1min(op); om = compute_history(om); om = derive_paper1_features(om)
        om_parts.append(om)
    omni = pd.concat(om_parts, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    om_t = omni["datetime"].values.astype("datetime64[s]").astype(np.int64)
    s_t = pd.to_datetime(sample["time"]).values.astype("datetime64[s]").astype(np.int64)
    idx = np.clip(np.searchsorted(om_t, s_t), 0, len(om_t)-1); idxl = np.clip(idx-1, 0, len(om_t)-1)
    use_l = np.abs(om_t[idxl]-s_t) < np.abs(om_t[idx]-s_t); pick = np.where(use_l, idxl, idx)
    om_sub = omni.iloc[pick].reset_index(drop=True)
    for c in SW_FEATURE_COLS + HIST_COLS + DERIVED_HIST:
        if c in om_sub.columns: sample[c] = om_sub[c].values
    sample["hemi_code"] = np.where(sample["hemisphere"]=="N", 1.0, 0.0)
    sample["by_hemi"] = sample["imf_by"] * np.where(sample["hemisphere"]=="N", 1.0, -1.0)
    sys.path.insert(0, "/glade/work/yizhu/cuspML/src")
    from identify_cusp import dipole_tilt_angle
    sample["dipole_tilt"] = [dipole_tilt_angle(t.to_pydatetime()) for t in pd.to_datetime(sample["time"])]
    x, y = polar_xy(sample["abs_mlat"].values, sample["mlt"].values)
    sample["x_polar"] = x; sample["y_polar"] = y

    feats = list(dict.fromkeys([f for f in SW_FEATURE_COLS+HIST_COLS+DERIVED_HIST+["x_polar","y_polar"] if f in sample.columns]))
    base_req = ["imf_bx","imf_by","imf_bz","sw_v","sw_n","sw_pdyn"]
    before = len(sample); sample = sample.dropna(subset=base_req).reset_index(drop=True)
    sample[feats] = sample[feats].fillna(0.0)
    print(f"  OMNI dropna: {before} -> {len(sample)}")

    sample["hour_bin"] = pd.to_datetime(sample["time"]).dt.floor("h").astype(np.int64)
    hids = sample["hour_bin"].unique(); rng.shuffle(hids)
    n = len(hids); nt=int(n*0.2); nv=int(n*0.1); nc=int(n*0.1)
    test_h=set(hids[:nt]); val_h=set(hids[nt:nt+nv]); cal_h=set(hids[nt+nv:nt+nv+nc]); train_h=set(hids[nt+nv+nc:])
    m = {k: sample["hour_bin"].isin(s).values for k,s in [("train",train_h),("val",val_h),("cal",cal_h),("test",test_h)]}
    X = sample[feats].values.astype(np.float32); ylab = sample["cusp_mask"].values.astype(int)
    for k in m: print(f"  {k}: {m[k].sum()} ({int(ylab[m[k]].sum())} pos)")

    spw = max(1, int((1-ylab[m["train"]]).sum())) / max(1, int(ylab[m["train"]].sum()))
    print(f"[R033] train XGBoost spw={spw:.2f}")
    t1 = time.time()
    model = fit_xgb(X[m["train"]], ylab[m["train"]], X[m["val"]], ylab[m["val"]], hp_overrides={"scale_pos_weight": spw})
    print(f"  trained {time.time()-t1:.1f}s iter={model.best_iteration}")
    iso, cal_info = maybe_calibrate(model, X[m["cal"]], ylab[m["cal"]], deviation_threshold=0.05)
    trained = TrainedModel(model=model, isotonic=iso, feature_names=feats, used_calibration=(iso is not None))

    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    raw = model.predict_proba(X[m["test"]])[:,1]; pt = iso.transform(raw) if iso else raw; yt = ylab[m["test"]]
    auc=float(roc_auc_score(yt,pt)); ap=float(average_precision_score(yt,pt)); brier=float(brier_score_loss(yt,pt))
    print(f"\n[R033] STAGE2 TEST (real, region): AUC {auc:.4f} AP {ap:.4f} Brier {brier:.4f}")

    print("[R033] end-to-end on held-out real crossings (restricted grid)")
    s1m, s1iso, s1f = load_stage1_v2()
    df_x = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    sw_x = sw_feature_names(df_x)
    keep_x = sw_x + ["eq_mlat","pole_mlat","eq_mlt","pole_mlt","mean_mlat","mean_mlt","satellite","hemisphere","time_start"]
    df_x = df_x[keep_x].dropna().reset_index(drop=True)
    df_x["year"] = pd.to_datetime(df_x["time_start"]).dt.year
    sat_set=set(s[0] for s in sy_list); yr_set=set(int(s[1]) for s in sy_list)
    tx = df_x[df_x["satellite"].isin(sat_set)&df_x["year"].isin(yr_set)].reset_index(drop=True)
    tx = tx[(tx["mean_mlt"]>=MLT_LO)&(tx["mean_mlt"]<=MLT_HI)&(tx["mean_mlat"].abs()>=LAT_LO)&(tx["mean_mlat"].abs()<=LAT_HI)].reset_index(drop=True)
    samp = tx.sample(n=min(500,len(tx)), random_state=99).reset_index(drop=True)
    print(f"  eval crossings: {len(samp)}")

    def ev(s2t, swc):
        ds, lp, t10, t1p = [], [], [], []
        nc_ = len(R_LAT_AXIS)*len(R_MLT_AXIS)
        for _, row in samp.iterrows():
            tl=abs(row["mean_mlat"]); tm=row["mean_mlt"]; hemi="N" if row["hemisphere"]=="N" else "S"
            sw={c:row[c] for c in swc if c in row}
            sw1={c:row[c] for c in STAGE1_BASE_FEATURES if c in row}
            sw1["doy_feat"]=sw.get("doy", pd.to_datetime(row["time_start"]).dayofyear)
            if "B_T" not in sw1: sw1["B_T"]=float(np.sqrt(row["imf_by"]**2+row["imf_bz"]**2))
            s1p=stage1_scalar_v2(s1m,s1iso,s1f,sw1)
            P=stage2_dial_restricted(s2t, sw, hemisphere=hemi); pmf=normalize_pmf_restricted(P); comb=s1p*pmf
            i,j=cell_of_restricted(tl,tm); pk=np.unravel_index(np.argmax(comb), comb.shape)
            ds.append(haversine_deg(tl,tm,R_LAT_AXIS[pk[0]],R_MLT_AXIS[pk[1]]))
            lp.append(np.log(comb[i,j]+1e-12))
            rank=(comb.flatten()>comb[i,j]).sum()+1; t10.append(rank<=10); t1p.append(rank<=max(1,nc_//100))
        return float(np.median(ds)), float(np.percentile(ds,90)), float(np.mean(t10)), float(np.mean(t1p)), float(np.mean(lp))

    rr = ev(trained, feats)
    print(f"  REAL-neg: median {rr[0]:.2f}deg p90 {rr[1]:.2f} top10 {rr[2]:.1%} top1% {rr[3]:.1%} logp {rr[4]:.3f}")
    from r012_case_studies_2stage import load_stage2
    sr = ev(load_stage2(), sw_x)
    print(f"  SYNTH R002: median {sr[0]:.2f}deg p90 {sr[1]:.2f} top10 {sr[2]:.1%} top1% {sr[3]:.1%} logp {sr[4]:.3f}")

    tag = f"real_region_{'_'.join(args.sat_years).replace(':','')}"
    model.save_model(f"{OUT_DIR}/r033_{tag}_model.ubj")
    if iso: pickle.dump(iso, open(f"{OUT_DIR}/r033_{tag}_iso.pkl","wb"))
    json.dump(feats, open(f"{OUT_DIR}/r033_{tag}_features.json","w"))
    json.dump({"sat_years":args.sat_years,"region":{"mlt":[MLT_LO,MLT_HI],"lat":[LAT_LO,LAT_HI]},
               "n_in_region":len(spec),"n_pos":n_pos_tot,"n_neg":n_neg_tot,"n_train":int(m["train"].sum()),
               "stage2_test":{"auc":auc,"ap":ap,"brier":brier},
               "endtoend_real":{"median":rr[0],"p90":rr[1],"top10":rr[2],"top1pct":rr[3],"logp":rr[4]},
               "endtoend_synth":{"median":sr[0],"p90":sr[1],"top10":sr[2],"top1pct":sr[3],"logp":sr[4]},
               "elapsed_sec":float(time.time()-t0)},
              open(f"{OUT_DIR}/r033_{tag}_results.json","w"), indent=2)
    print(f"\n[R033] verdict: real-neg {'BEATS' if rr[0]<sr[0] else 'does NOT beat'} synth ({rr[0]:.2f} vs {sr[0]:.2f} deg)")
    print(f"  saved, {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
