"""R016 — final combined product using R015 (opportunity-restricted) stage 1.

Replaces R013/R014's stage-1 model with R015's opportunity-restricted version.
Runs both:
  - 6 case studies (replaces R013)
  - End-to-end metric on 500 held-out crossings (replaces R014)
"""
import json
import os
import pickle
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/glade/work/yizhu/cuspML/src/lib")
sys.path.insert(0, "/glade/work/yizhu/cuspML/src/scripts/mvp")
from cusp_map import polar_xy, predict_proba, TrainedModel, load_crossings, sw_feature_names
from cusp_stage1 import STAGE1_BASE_FEATURES
from r002_case_studies import CASES, build_sw_state
from r012_case_studies_2stage import load_stage2, stage2_grid, plot_dial
from r013_normalized_2stage import combined_renorm
from r014_endtoend_eval import (LAT_AXIS, MLT_AXIS, MM, LL, CELL_AREA, DLAT, DMLT,
                                 cell_of, stage2_dial, normalize_pmf, haversine_deg)

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp"


def load_stage1_v2():
    """Load R015 stage 1 instead of R011."""
    from xgboost import XGBClassifier
    m = XGBClassifier()
    m.load_model(f"{OUT_DIR}/bundles/r015_stage1_opp_model.ubj")
    with open(f"{OUT_DIR}/bundles/r015_stage1_opp_isotonic.pkl", "rb") as f:
        iso = pickle.load(f)
    with open(f"{OUT_DIR}/bundles/r015_stage1_opp_features.json") as f:
        feats = json.load(f)
    return m, iso, feats


def stage1_scalar_v2(s1_model, s1_iso, s1_feats, sw_state):
    rec = dict(sw_state)
    rec["doy_feat"] = sw_state.get("doy", 80)
    arr = np.array([[rec[k] for k in s1_feats]], dtype=np.float32)
    raw = s1_model.predict_proba(arr)[:, 1]
    return float(s1_iso.transform(raw)[0])


def main():
    t0 = time.time()
    s1_model, s1_iso, s1_feats = load_stage1_v2()
    s2 = load_stage2()
    figs_dir = f"{OUT_DIR}/figures"
    os.makedirs(figs_dir, exist_ok=True)

    # ==== A: 6 case-study heatmaps ====
    print("[R016 A] case studies with R015 stage 1 ...")
    all_maps = {}
    for c in CASES:
        sw = build_sw_state(c, hemisphere="N")
        s1_p = stage1_scalar_v2(s1_model, s1_iso, s1_feats, sw)
        mlat_axis = np.arange(50.0, 90.0 + 1e-9, 1.0)
        mlt_axis = np.arange(0.0, 24.0 + 1e-9, 0.5)
        # use r012's stage2_grid (legacy inclusive endpoints; minor 1/1920 bias acceptable for visualization)
        from r012_case_studies_2stage import stage2_grid as s2_grid_legacy
        _, _, s2_p = s2_grid_legacy(s2, sw, hemisphere="N")
        combined = combined_renorm(s1_p, s2_p, mlat_axis, mlt_axis)
        all_maps[c["name"]] = {"s1": s1_p, "s2": s2_p, "combined": combined,
                                "mlat": mlat_axis, "mlt": mlt_axis, "title": c["title"]}
        pi = np.unravel_index(np.argmax(combined), combined.shape)
        print(f"  {c['name']:>30s}  s1={s1_p:.4f}  combined_peak={combined.max():.5f} "
              f"at lat={mlat_axis[pi[0]]:.0f}, MLT={mlt_axis[pi[1]]:.1f}")

    vmax = max(d["combined"].max() for d in all_maps.values())
    print(f"\n  shared vmax: {vmax:.5f}")
    for name, d in all_maps.items():
        plot_dial(d["mlat"], d["mlt"], d["combined"],
                  f"{d['title']}  (R016: opp-restricted s1, s1={d['s1']:.3f})",
                  f"{figs_dir}/2stage_v2_{name}.png", vmax=vmax)

    s = {name: {"s1": float(d["s1"]),
                "combined_peak": float(d["combined"].max()),
                "peak_lat": float(d["mlat"][np.unravel_index(np.argmax(d["combined"]),
                                                              d["combined"].shape)[0]]),
                "peak_mlt": float(d["mlt"][np.unravel_index(np.argmax(d["combined"]),
                                                             d["combined"].shape)[1]]),
                "midnight_mean": float(d["combined"][:,
                    (d["mlt"] < 4) | (d["mlt"] > 20)].mean())}
         for name, d in all_maps.items()}

    checks = {
        "storm_peak_higher_than_quiet": s["case6_storm"]["combined_peak"] > s["case5_quiet"]["combined_peak"],
        "south_Bz_peak_higher_than_north_Bz": s["case1_strong_south_Bz"]["combined_peak"] > s["case2_strong_north_Bz"]["combined_peak"],
        "south_Bz_lat_lower_than_north_Bz": s["case1_strong_south_Bz"]["peak_lat"] < s["case2_strong_north_Bz"]["peak_lat"],
        "quiet_below_active_in_stage1": s["case5_quiet"]["s1"] < s["case1_strong_south_Bz"]["s1"],
        "strong_above_quiet_by_5x": (s["case1_strong_south_Bz"]["combined_peak"] > 5 * s["case5_quiet"]["combined_peak"]
                                     if s["case5_quiet"]["combined_peak"] > 0
                                     else s["case1_strong_south_Bz"]["combined_peak"] > 0),
        "midnight_low": all(s[k]["midnight_mean"] < 0.05 * max(s[kk]["combined_peak"] for kk in s) for k in s),
    }
    print(f"\n[R016 A physics sanity]")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    # ==== B: end-to-end on 500 held-out crossings ====
    print("\n[R016 B] end-to-end metric on 500 real crossings ...")
    df = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    required = ["eq_mlat", "pole_mlat", "eq_mlt", "pole_mlt"]
    sw_cols = sw_feature_names(df)
    keep_cols = sw_cols + required + ["satellite", "hemisphere", "time_start"]
    df_clean = df[keep_cols].dropna().reset_index(drop=True)

    rng = np.random.default_rng(42)
    cids = np.arange(len(df_clean)); rng.shuffle(cids)
    n_test = int(len(cids) * 0.2)
    test_df = df_clean.iloc[cids[:n_test]].reset_index(drop=True)
    sample_n = min(500, len(test_df))
    sample_idx = np.random.default_rng(99).choice(len(test_df), sample_n, replace=False)
    sample = test_df.iloc[sample_idx].reset_index(drop=True)

    n_cells = LAT_AXIS.size * MLT_AXIS.size
    uniform_per_cell = 1.0 / n_cells
    rec = {"combined_logp": [], "stage2only_logp": [], "uniform_logp": [],
           "peak_dist_deg": [], "stage2_peak_dist_deg": [],
           "true_cell_in_top10": [], "true_cell_in_top1pct": [],
           "stage1_p": []}

    t1 = time.time()
    for k, row in sample.iterrows():
        true_lat = abs(row["mean_mlat"]) if "mean_mlat" in row else (abs(row["eq_mlat"]) + abs(row["pole_mlat"])) / 2
        true_mlt = row["mean_mlt"] if "mean_mlt" in row else (row["eq_mlt"] + row["pole_mlt"]) / 2
        hemi = "N" if row["hemisphere"] == "N" else "S"
        sw_for_s2 = {c: row[c] for c in sw_cols if c in row}
        sw_for_s1 = {c: row[c] for c in STAGE1_BASE_FEATURES if c in row}
        sw_for_s1["doy_feat"] = sw_for_s2.get("doy", pd.to_datetime(row["time_start"]).dayofyear)
        if "B_T" not in sw_for_s1:
            sw_for_s1["B_T"] = float(np.sqrt(row["imf_by"]**2 + row["imf_bz"]**2))
        s1_p = stage1_scalar_v2(s1_model, s1_iso, s1_feats, sw_for_s1)
        s2_p = stage2_dial(s2, sw_for_s2, hemisphere=hemi)
        s2_pmf = normalize_pmf(s2_p, area_weighted=True)
        combined = s1_p * s2_pmf
        i, j = cell_of(true_lat, true_mlt)
        eps = 1e-12
        rec["combined_logp"].append(np.log(combined[i, j] + eps))
        rec["stage2only_logp"].append(np.log(s2_pmf[i, j] + eps))
        rec["uniform_logp"].append(np.log(uniform_per_cell + eps))
        rec["stage1_p"].append(s1_p)
        peak_idx = np.unravel_index(np.argmax(combined), combined.shape)
        rec["peak_dist_deg"].append(haversine_deg(true_lat, true_mlt,
                                                   LAT_AXIS[peak_idx[0]], MLT_AXIS[peak_idx[1]]))
        s2_peak_idx = np.unravel_index(np.argmax(s2_p), s2_p.shape)
        rec["stage2_peak_dist_deg"].append(haversine_deg(true_lat, true_mlt,
                                                          LAT_AXIS[s2_peak_idx[0]], MLT_AXIS[s2_peak_idx[1]]))
        flat = combined.flatten()
        rank = (flat > combined[i, j]).sum() + 1
        rec["true_cell_in_top10"].append(rank <= 10)
        rec["true_cell_in_top1pct"].append(rank <= n_cells // 100)
    print(f"  scored {sample_n} crossings in {time.time()-t1:.1f}s")

    summary = {
        "n_eval": sample_n,
        "median_peak_dist_deg": float(np.median(rec["peak_dist_deg"])),
        "p90_peak_dist_deg": float(np.percentile(rec["peak_dist_deg"], 90)),
        "median_stage2_only_peak_dist_deg": float(np.median(rec["stage2_peak_dist_deg"])),
        "frac_true_in_top10_cells": float(np.mean(rec["true_cell_in_top10"])),
        "frac_true_in_top1pct_cells": float(np.mean(rec["true_cell_in_top1pct"])),
        "mean_logp_combined": float(np.mean(rec["combined_logp"])),
        "mean_logp_stage2_only": float(np.mean(rec["stage2only_logp"])),
        "mean_logp_uniform": float(np.mean(rec["uniform_logp"])),
        "logp_improvement_over_uniform": float(np.mean(rec["combined_logp"]) - np.mean(rec["uniform_logp"])),
        "logp_improvement_stage2_over_uniform": float(np.mean(rec["stage2only_logp"]) - np.mean(rec["uniform_logp"])),
    }
    print(f"\n[R016 B end-to-end (R015 stage 1)]:")
    for k, v in summary.items():
        print(f"  {k:>42s}: {v}")

    # ==== save combined R016 results ====
    out = {
        "case_studies": {"summary": s, "physics_checks": checks,
                          "n_pass": sum(checks.values()), "n_total": len(checks)},
        "end_to_end": summary,
        "elapsed_sec": float(time.time() - t0),
    }
    with open(f"{OUT_DIR}/bundles/r016_final_combined.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  R016 total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
