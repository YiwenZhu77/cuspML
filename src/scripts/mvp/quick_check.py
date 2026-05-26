"""Quick visual check: 4 real held-out crossings, plot combined map + mark truth.

Picks 4 crossings from different SW conditions, runs the two-stage model, plots
the predicted probability map on the polar dial, overlays a red dot at the true
(MLAT, MLT) so we can see at a glance whether predictions land near truth.
"""
import json
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/glade/work/yizhu/cuspML/src/lib")
sys.path.insert(0, "/glade/work/yizhu/cuspML/src/scripts/mvp")
from cusp_map import load_crossings, sw_feature_names, polar_xy, predict_proba
from cusp_stage1 import STAGE1_BASE_FEATURES
from r014_endtoend_eval import (LAT_AXIS, MLT_AXIS, MM, LL,
                                 cell_of, stage2_dial, normalize_pmf, haversine_deg)
from r016_combined_v2 import load_stage1_v2, stage1_scalar_v2
from r012_case_studies_2stage import load_stage2

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp"


def pick_diverse_crossings(test_df, n=4, seed=7):
    """Pick crossings spanning different IMF Bz regimes."""
    rng = np.random.default_rng(seed)
    picks = []
    # 1 strong south Bz, 1 strong north Bz, 1 quiet-ish, 1 By-dominated
    df = test_df.copy()
    df["bz_abs"] = df["imf_bz"].abs()
    df["by_abs"] = df["imf_by"].abs()
    # strong south Bz (Bz < -3, by_abs small)
    cands = df[(df["imf_bz"] < -3) & (df["by_abs"] < 4)]
    if len(cands): picks.append(cands.sample(1, random_state=rng.integers(1e9)).iloc[0])
    cands = df[(df["imf_bz"] > 3) & (df["by_abs"] < 4)]
    if len(cands): picks.append(cands.sample(1, random_state=rng.integers(1e9)).iloc[0])
    cands = df[(df["bz_abs"] < 1.5) & (df["by_abs"] < 1.5) & (df["sw_v"] < 400)]
    if len(cands): picks.append(cands.sample(1, random_state=rng.integers(1e9)).iloc[0])
    cands = df[(df["by_abs"] > 4) & (df["bz_abs"] < 3)]
    if len(cands): picks.append(cands.sample(1, random_state=rng.integers(1e9)).iloc[0])
    return picks[:n]


def main():
    s1_model, s1_iso, s1_feats = load_stage1_v2()
    s2 = load_stage2()
    print(f"  loaded R015 stage 1 + R002 stage 2")

    df = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    sw_cols = sw_feature_names(df)
    required = ["eq_mlat", "pole_mlat", "eq_mlt", "pole_mlt"]
    keep_cols = sw_cols + required + ["satellite", "hemisphere", "time_start"]
    df_clean = df[keep_cols].dropna().reset_index(drop=True)
    rng = np.random.default_rng(42)
    cids = np.arange(len(df_clean)); rng.shuffle(cids)
    n_test = int(len(cids) * 0.2)
    test_df = df_clean.iloc[cids[:n_test]].reset_index(drop=True)
    print(f"  test pool: {len(test_df)} crossings")

    picks = pick_diverse_crossings(test_df, n=4, seed=7)
    print(f"  picked {len(picks)} diverse crossings")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), subplot_kw=dict(projection="polar"))
    cbar_ref = None
    for ax, row in zip(axes, picks):
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

        peak_idx = np.unravel_index(np.argmax(combined), combined.shape)
        peak_lat = LAT_AXIS[peak_idx[0]]
        peak_mlt = MLT_AXIS[peak_idx[1]]
        dist = haversine_deg(true_lat, true_mlt, peak_lat, peak_mlt)

        # polar plot
        theta = 2 * np.pi * MLT_AXIS / 24.0
        r = 90.0 - LAT_AXIS
        TT, RR = np.meshgrid(theta, r)
        cf = ax.pcolormesh(TT, RR, combined, cmap="viridis", shading="auto")
        # true location
        ax.plot(2 * np.pi * true_mlt / 24.0, 90.0 - true_lat,
                marker="o", color="red", markersize=12, mfc="none", mew=2.5,
                label="true")
        # predicted peak
        ax.plot(2 * np.pi * peak_mlt / 24.0, 90.0 - peak_lat,
                marker="x", color="white", markersize=12, mew=2.5, label="pred peak")
        ax.set_theta_zero_location("S")
        ax.set_theta_direction(1)
        ax.set_ylim(0, 40)
        ax.set_yticks([10, 20, 30, 40])
        ax.set_yticklabels(["80", "70", "60", "50"])
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(["00", "06", "12", "18"])
        time_str = pd.to_datetime(row["time_start"]).strftime("%Y-%m-%d %H:%M")
        title = (f"{row['satellite']} {hemi}H {time_str}\n"
                 f"Bz={row['imf_bz']:+.1f}, By={row['imf_by']:+.1f}, V={row['sw_v']:.0f}\n"
                 f"true {true_lat:.1f}deg / {true_mlt:.1f}h ; pred {peak_lat:.1f}deg / {peak_mlt:.1f}h\n"
                 f"2D dist = {dist:.1f}deg ; s1 P = {s1_p:.3f}")
        ax.set_title(title, fontsize=9, pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.05), fontsize=8)
        cbar_ref = cf

    fig.suptitle("Quick visual check — 4 real held-out crossings, combined probability map (R016 two-stage)",
                 fontsize=12, y=1.02)
    fig.colorbar(cbar_ref, ax=axes, shrink=0.7, pad=0.05, label="P(cusp in cell)")
    out_path = f"{OUT_DIR}/figures/quick_check_4_real.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
