"""R024 — R022 redux, comparing R002 (synth) vs R023 (per-pass real-neg)."""
import json
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, "/glade/work/yizhu/cuspML/src/lib")
sys.path.insert(0, "/glade/work/yizhu/cuspML/src/scripts/mvp")
from cusp_map import load_crossings, sw_feature_names, polar_xy, predict_proba, TrainedModel
from cusp_stage1 import STAGE1_BASE_FEATURES
from omni_1min import load_omni_1min, compute_history, derive_paper1_features
from r014_endtoend_eval import (LAT_AXIS, MLT_AXIS, MM, LL,
                                 cell_of, stage2_dial, normalize_pmf, haversine_deg)
from r016_combined_v2 import load_stage1_v2, stage1_scalar_v2
from r012_case_studies_2stage import load_stage2
from r022_compare_real_vs_synth import stage2_dial_custom

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles"


def load_r023(tag, prefix="r023"):
    from xgboost import XGBClassifier
    m = XGBClassifier()
    m.load_model(f"{OUT_DIR}/{prefix}_{tag}_model.ubj")
    iso = None
    iso_path = f"{OUT_DIR}/{prefix}_{tag}_isotonic.pkl"
    if os.path.exists(iso_path):
        with open(iso_path, "rb") as f: iso = pickle.load(f)
    with open(f"{OUT_DIR}/{prefix}_{tag}_features.json") as f: feats = json.load(f)
    return TrainedModel(model=m, isotonic=iso, feature_names=feats,
                        used_calibration=(iso is not None))


def main():
    s1_model, s1_iso, s1_feats = load_stage1_v2()
    s2_synth = load_stage2()
    s2_real = load_r023("hybrid_F10_1993_1994", prefix="r025")  # switch to R025 hybrid

    df = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    sw_cols = sw_feature_names(df)
    keep = sw_cols + ["eq_mlat", "pole_mlat", "eq_mlt", "pole_mlt", "satellite", "hemisphere", "time_start"]
    df_clean = df[keep].dropna().reset_index(drop=True)
    f10 = df_clean[df_clean["satellite"] == "F10"].copy()
    f10["year"] = pd.to_datetime(f10["time_start"]).dt.year
    f10_pilot = f10[f10["year"].isin([1993, 1994])].reset_index(drop=True)
    sample = f10_pilot.sample(n=min(200, len(f10_pilot)), random_state=99).reset_index(drop=True)
    print(f"  loaded models; evaluating on {len(sample)} F10 1993-94 crossings")

    n_cells = LAT_AXIS.size * MLT_AXIS.size
    eps = 1e-12
    uniform_logp = np.log(1.0 / n_cells)
    rec = {"synth_dist": [], "real_dist": [], "synth_lp": [], "real_lp": [],
           "synth_t10": [], "real_t10": [], "synth_t1p": [], "real_t1p": []}

    for _, row in sample.iterrows():
        true_lat = abs(row["mean_mlat"]) if "mean_mlat" in row else (abs(row["eq_mlat"]) + abs(row["pole_mlat"])) / 2
        true_mlt = row["mean_mlt"] if "mean_mlt" in row else (row["eq_mlt"] + row["pole_mlt"]) / 2
        hemi = "N" if row["hemisphere"] == "N" else "S"
        sw = {c: row[c] for c in sw_cols if c in row}
        sw_s1 = {c: row[c] for c in STAGE1_BASE_FEATURES if c in row}
        sw_s1["doy_feat"] = sw.get("doy", pd.to_datetime(row["time_start"]).dayofyear)
        if "B_T" not in sw_s1:
            sw_s1["B_T"] = float(np.sqrt(row["imf_by"]**2 + row["imf_bz"]**2))
        s1_p = stage1_scalar_v2(s1_model, s1_iso, s1_feats, sw_s1)

        for tag, trained, d_k, l_k, t10_k, t1p_k in [
            ("synth", s2_synth, "synth_dist", "synth_lp", "synth_t10", "synth_t1p"),
            ("real",  s2_real,  "real_dist",  "real_lp",  "real_t10",  "real_t1p"),
        ]:
            s2_p = stage2_dial_custom(trained, sw, hemisphere=hemi)
            pmf = normalize_pmf(s2_p, area_weighted=True)
            combined = s1_p * pmf
            i, j = cell_of(true_lat, true_mlt)
            rec[l_k].append(np.log(combined[i, j] + eps))
            peak = np.unravel_index(np.argmax(combined), combined.shape)
            rec[d_k].append(haversine_deg(true_lat, true_mlt, LAT_AXIS[peak[0]], MLT_AXIS[peak[1]]))
            flat = combined.flatten()
            rank = (flat > combined[i, j]).sum() + 1
            rec[t10_k].append(rank <= 10)
            rec[t1p_k].append(rank <= n_cells // 100)

    print(f"\n[R024] R002(synth) vs R023(per-pass real) on 200 F10 1993-94 crossings:")
    print(f"{'metric':>34s}  {'synth':>10s}  {'real':>10s}  delta")
    for k_s, k_r, k in [
        ("synth_dist", "real_dist", "median_peak_dist_deg"),
        ("synth_dist", "real_dist", "p90_peak_dist_deg"),
        ("synth_t10", "real_t10", "frac_top10"),
        ("synth_t1p", "real_t1p", "frac_top1pct"),
        ("synth_lp", "real_lp", "mean_logp_true"),
    ]:
        if "p90" in k:
            s = np.percentile(rec[k_s], 90); r = np.percentile(rec[k_r], 90)
        elif "median" in k:
            s = np.median(rec[k_s]); r = np.median(rec[k_r])
        else:
            s = np.mean(rec[k_s]); r = np.mean(rec[k_r])
        print(f"  {k:>34s}: {s:>10.4f}  {r:>10.4f}  {r-s:>+.4f}")
    out = {
        "synth": {"median_dist": float(np.median(rec["synth_dist"])),
                  "p90_dist": float(np.percentile(rec["synth_dist"], 90)),
                  "frac_top10": float(np.mean(rec["synth_t10"])),
                  "frac_top1pct": float(np.mean(rec["synth_t1p"])),
                  "mean_logp": float(np.mean(rec["synth_lp"]))},
        "real":  {"median_dist": float(np.median(rec["real_dist"])),
                  "p90_dist": float(np.percentile(rec["real_dist"], 90)),
                  "frac_top10": float(np.mean(rec["real_t10"])),
                  "frac_top1pct": float(np.mean(rec["real_t1p"])),
                  "mean_logp": float(np.mean(rec["real_lp"]))},
    }
    with open(f"{OUT_DIR}/r024_compare_perpass.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
