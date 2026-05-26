"""R019 — same as R018 but with REAL 1-min OMNI inputs to stage 2.

Eliminates the R018 history-feature mismatch: pulls 1-min OMNI for the storm
window, computes mean15/30/60, std15/30/60, delta15/30/60 the same way
add_omni.py does for training crossings. Stage 2 then sees a matched input
distribution.
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
from cusp_map import load_crossings, sw_feature_names, predict_proba
from cusp_stage1 import STAGE1_BASE_FEATURES
from omni_1min import load_omni_1min, compute_history, derive_paper1_features
from r014_endtoend_eval import (LAT_AXIS, MLT_AXIS, MM, LL,
                                 stage2_dial, normalize_pmf, haversine_deg)
from r016_combined_v2 import load_stage1_v2, stage1_scalar_v2
from r012_case_studies_2stage import load_stage2

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp"
OMNI_MIN_PATH = "/glade/work/yizhu/cuspML/output/omni_raw/omni_min2011.asc"

EVENT_START = pd.Timestamp("2011-08-05 00:00")
EVENT_END = pd.Timestamp("2011-08-07 00:00")
FRAME_HOURS = [0, 8, 16, 24, 32, 40]
FRAME_WINDOW_H = 4


def main():
    s1_model, s1_iso, s1_feats = load_stage1_v2()
    s2 = load_stage2()
    print("  loaded R015 stage 1 + R002 stage 2")

    print(f"  loading 1-min OMNI 2011 from {OMNI_MIN_PATH} ...")
    om = load_omni_1min(OMNI_MIN_PATH)
    print(f"  {len(om)} 1-min rows in 2011")
    # restrict to storm window +- 2h buffer for history
    om = om[(om["datetime"] >= EVENT_START - pd.Timedelta("2h"))
            & (om["datetime"] <= EVENT_END + pd.Timedelta("2h"))].reset_index(drop=True)
    print(f"  restricted to event +-buffer: {len(om)} rows")
    print("  computing history features ...")
    om = compute_history(om)
    print("  deriving paper-1 features ...")
    om = derive_paper1_features(om)

    crossings = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    ct = pd.to_datetime(crossings["time_start"])
    event_mask = (ct >= EVENT_START) & (ct <= EVENT_END)
    event_xings = crossings[event_mask].reset_index(drop=True)
    event_xings["t"] = pd.to_datetime(event_xings["time_start"])
    print(f"  {len(event_xings)} crossings in event window")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), subplot_kw=dict(projection="polar"))
    sw_cols_full = sw_feature_names(crossings)
    last_cf = None
    frame_summaries = []

    for ax, hours_offset in zip(axes.flat, FRAME_HOURS):
        center_t = EVENT_START + pd.Timedelta(hours=hours_offset)
        # nearest 1-min row
        idx = (om["datetime"] - center_t).abs().idxmin()
        row = om.iloc[idx]

        # build sw dict for stage 2 (now with REAL history stats)
        sw_for_s2 = {}
        for c in sw_cols_full:
            if c in row.index and pd.notna(row[c]):
                sw_for_s2[c] = float(row[c])
        # backfill any stage-2 feature missing from OMNI 1-min (e.g. by_hemi already set,
        # dipole_tilt approximated 0)
        for fname in s2.feature_names:
            if fname not in sw_for_s2 and fname not in ("x_polar", "y_polar"):
                sw_for_s2[fname] = 0.0

        # stage 1
        sw_for_s1 = {k: float(row[k]) for k in STAGE1_BASE_FEATURES if k in row.index}
        sw_for_s1["doy_feat"] = float(row["doy"])
        s1_p = stage1_scalar_v2(s1_model, s1_iso, s1_feats, sw_for_s1)

        s2_p = stage2_dial(s2, sw_for_s2, hemisphere="N")
        s2_pmf = normalize_pmf(s2_p, area_weighted=True)
        combined = s1_p * s2_pmf

        theta = 2 * np.pi * MLT_AXIS / 24.0
        r = 90.0 - LAT_AXIS
        TT, RR = np.meshgrid(theta, r)
        last_cf = ax.pcolormesh(TT, RR, combined, cmap="viridis", shading="auto")
        ax.set_theta_zero_location("S")
        ax.set_theta_direction(1)
        ax.set_ylim(0, 40)
        ax.set_yticks([10, 20, 30, 40])
        ax.set_yticklabels(["80", "70", "60", "50"])
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(["00", "06", "12", "18"])

        win = (event_xings["t"] >= center_t - pd.Timedelta(hours=FRAME_WINDOW_H)) & \
              (event_xings["t"] <= center_t + pd.Timedelta(hours=FRAME_WINDOW_H))
        nearby = event_xings[win]
        per_xing_dist = []
        n_nh = 0
        for _, x in nearby.iterrows():
            if x["hemisphere"] != "N":
                continue
            n_nh += 1
            lat = abs(x["mean_mlat"]); mlt = x["mean_mlt"]
            ax.plot(2 * np.pi * mlt / 24.0, 90.0 - lat,
                    marker="o", color="red", markersize=8, mfc="none", mew=1.8)
            peak_idx = np.unravel_index(np.argmax(combined), combined.shape)
            d = haversine_deg(lat, mlt, LAT_AXIS[peak_idx[0]], MLT_AXIS[peak_idx[1]])
            per_xing_dist.append(d)

        # also show std60(bz) so reader can see whether history features add info
        std60_bz = row.get("imf_bz_std60", np.nan)
        title = (f"{center_t.strftime('%m-%d %H:%M')} UT\n"
                 f"Bz={row['imf_bz']:+.1f}  By={row['imf_by']:+.1f}  V={row['sw_v']:.0f}\n"
                 f"Bz_std60={std60_bz:.2f}  s1={s1_p:.3f}  n N-xings={n_nh}")
        if per_xing_dist:
            title += f"\nmed peak-dist {np.median(per_xing_dist):.1f}deg"
        ax.set_title(title, fontsize=9, pad=12)

        frame_summaries.append({
            "frame_t": str(center_t),
            "Bz": float(row["imf_bz"]),
            "By": float(row["imf_by"]),
            "V": float(row["sw_v"]),
            "Bz_std60": float(std60_bz) if pd.notna(std60_bz) else None,
            "stage1_P": float(s1_p),
            "combined_peak": float(combined.max()),
            "n_crossings": int(n_nh),
            "median_peak_dist_deg": float(np.median(per_xing_dist)) if per_xing_dist else None,
        })

    fig.suptitle(f"R019: 2011-08-05 storm with REAL 1-min OMNI + history stats\n"
                 f"red circles = actual DMSP N-hemi crossings in 8-h window",
                 fontsize=13, y=1.00)
    fig.colorbar(last_cf, ax=axes, shrink=0.6, pad=0.05, label="P(cusp in cell)")
    out_png = f"{OUT_DIR}/figures/r019_storm_2011_1min.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved -> {out_png}")
    print(f"\n  R019 frame summary:")
    for s in frame_summaries:
        print(f"    {s['frame_t']}  Bz={s['Bz']:+.1f}  Bz_std60={s['Bz_std60']}  "
              f"s1={s['stage1_P']:.3f}  n={s['n_crossings']}  med_dist={s['median_peak_dist_deg']}")

    with open(f"{OUT_DIR}/bundles/r019_storm_1min.json", "w") as f:
        json.dump({"event": "2011-08-05",
                    "input": "1-min OMNI + real history stats",
                    "frames": frame_summaries}, f, indent=2)


if __name__ == "__main__":
    main()
