"""R022 v2 — diagnostic + fix. Compare R002 vs R021 with each model fed its
NATIVE feature distribution.

Hypothesis: R021's bad R022 performance is feature-distribution mismatch
(trained on omni_min*.asc-derived features, evaluated on paper-1 add_omni
features). Fix: feed R021 features computed from omni_min*.asc at each
crossing's timestamp.

R002 keeps using paper-1 features (its native).
"""
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
from r022_compare_real_vs_synth import load_r021, stage2_dial_custom

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles"
OMNI_MIN_TEMPLATE = "/glade/work/yizhu/cuspML/output/omni_raw/omni_min{year}.asc"


def main():
    s1_model, s1_iso, s1_feats = load_stage1_v2()
    s2_synth = load_stage2()
    s2_real = load_r021("K10_F10_1993_1994")

    # crossings
    df = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    sw_cols = sw_feature_names(df)
    required = ["eq_mlat", "pole_mlat", "eq_mlt", "pole_mlt"]
    keep = sw_cols + required + ["satellite", "hemisphere", "time_start"]
    df_clean = df[keep].dropna().reset_index(drop=True)
    f10 = df_clean[df_clean["satellite"] == "F10"].copy()
    f10["year"] = pd.to_datetime(f10["time_start"]).dt.year
    f10_pilot = f10[f10["year"].isin([1993, 1994])].reset_index(drop=True)
    sample = f10_pilot.sample(n=200, random_state=99).reset_index(drop=True)

    # load omni_min features
    print("[diag] loading omni_min for 1993+1994 + computing history ...")
    om_parts = []
    for y in [1993, 1994]:
        om = load_omni_1min(OMNI_MIN_TEMPLATE.format(year=y))
        om = compute_history(om)
        om = derive_paper1_features(om)
        om_parts.append(om)
    om_all = pd.concat(om_parts, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    om_t = om_all["datetime"].values.astype("datetime64[s]").astype(np.int64)
    # match each crossing to nearest omni minute
    cross_t = pd.to_datetime(sample["time_start"]).values.astype("datetime64[s]").astype(np.int64)
    idx = np.searchsorted(om_t, cross_t)
    idx = np.clip(idx, 0, len(om_t) - 1)
    idx_left = np.clip(idx - 1, 0, len(om_t) - 1)
    d_right = np.abs(om_t[idx] - cross_t)
    d_left = np.abs(om_t[idx_left] - cross_t)
    pick = np.where(d_left < d_right, idx_left, idx)
    om_match = om_all.iloc[pick].reset_index(drop=True)

    # fix hemisphere-dependent features for omni_min match
    om_match["hemi_code"] = np.where(sample["hemisphere"] == "N", 1.0, 0.0)
    om_match["by_hemi"] = om_match["imf_by"] * np.where(sample["hemisphere"] == "N", 1.0, -1.0)
    sys.path.insert(0, "/glade/work/yizhu/cuspML/src")
    from identify_cusp import dipole_tilt_angle
    om_match["dipole_tilt"] = [dipole_tilt_angle(t.to_pydatetime()) for t in pd.to_datetime(sample["time_start"])]

    n_cells = LAT_AXIS.size * MLT_AXIS.size
    eps = 1e-12

    rec = {"synth_dist": [], "real_paper1_dist": [], "real_ominmin_dist": [],
           "synth_top10": [], "real_paper1_top10": [], "real_ominmin_top10": []}

    for i, row in sample.iterrows():
        true_lat = abs(row["mean_mlat"]) if "mean_mlat" in row else (abs(row["eq_mlat"]) + abs(row["pole_mlat"])) / 2
        true_mlt = row["mean_mlt"] if "mean_mlt" in row else (row["eq_mlt"] + row["pole_mlt"]) / 2
        hemi = "N" if row["hemisphere"] == "N" else "S"

        sw_paper1 = {c: row[c] for c in sw_cols if c in row}
        sw_ominmin = {c: om_match.iloc[i][c] for c in s2_real.feature_names
                      if c in om_match.columns and c not in ("x_polar", "y_polar")
                      and pd.notna(om_match.iloc[i][c])}
        for c in s2_real.feature_names:
            if c not in sw_ominmin and c not in ("x_polar", "y_polar"):
                sw_ominmin[c] = 0.0  # match training-time fillna(0)

        for tag, model, sw_dict, dist_k, top10_k in [
            ("synth(paper1 SW)", s2_synth, sw_paper1, "synth_dist", "synth_top10"),
            ("real(paper1 SW)", s2_real, sw_paper1, "real_paper1_dist", "real_paper1_top10"),
            ("real(omni_min SW)", s2_real, sw_ominmin, "real_ominmin_dist", "real_ominmin_top10"),
        ]:
            P = stage2_dial_custom(model, sw_dict, hemisphere=hemi)
            pmf = normalize_pmf(P, area_weighted=True)
            peak_idx = np.unravel_index(np.argmax(pmf), pmf.shape)
            d = haversine_deg(true_lat, true_mlt, LAT_AXIS[peak_idx[0]], MLT_AXIS[peak_idx[1]])
            rec[dist_k].append(d)
            cell_i, cell_j = cell_of(true_lat, true_mlt)
            flat = pmf.flatten()
            rank = (flat > pmf[cell_i, cell_j]).sum() + 1
            rec[top10_k].append(rank <= 10)

    print(f"\n[diag] DIAGNOSTIC RESULTS on {len(sample)} F10 1993-94 crossings\n")
    print(f"{'config':>22s}  {'median dist':>13s}  {'p90 dist':>10s}  {'top10 %':>8s}")
    for tag, dist_k, top10_k in [
        ("synth (paper1 SW)", "synth_dist", "synth_top10"),
        ("real  (paper1 SW)", "real_paper1_dist", "real_paper1_top10"),
        ("real  (omni_min SW)", "real_ominmin_dist", "real_ominmin_top10"),
    ]:
        med = np.median(rec[dist_k])
        p90 = np.percentile(rec[dist_k], 90)
        t10 = np.mean(rec[top10_k])
        print(f"  {tag:>22s}  {med:>13.4f}  {p90:>10.4f}  {100*t10:>7.1f}%")

    out = {tag: {"median_dist": float(np.median(rec[d])),
                  "p90_dist": float(np.percentile(rec[d], 90)),
                  "frac_top10": float(np.mean(rec[t]))}
           for tag, d, t in [
               ("synth_paper1SW", "synth_dist", "synth_top10"),
               ("real_paper1SW", "real_paper1_dist", "real_paper1_top10"),
               ("real_omniminSW", "real_ominmin_dist", "real_ominmin_top10"),
           ]}
    with open(f"{OUT_DIR}/r022_v2_diag.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved -> {OUT_DIR}/r022_v2_diag.json")


if __name__ == "__main__":
    main()
