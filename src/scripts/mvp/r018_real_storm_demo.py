"""R018 — real-event demo: predict cusp probability map over a real storm event,
overlay actual DMSP crossings as ground truth.

Picks the 2011-08-05 G4 geomagnetic storm window (5-7 Aug 2011). Pulls real
OMNI hourly SW for that window. Runs combined two-stage model at each hour.
Plots 6 frames (every 8 h, total 48 h) as a 2x3 panel. On each frame, overlays
ALL DMSP crossings from the 48k table whose time falls within +/-4 h of that
frame's center hour.

Goal: visually demonstrate the product on real data, not synthetic.
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
from cusp_stage1 import (load_omni2_hourly, derive_features, STAGE1_BASE_FEATURES)
from r014_endtoend_eval import (LAT_AXIS, MLT_AXIS, MM, LL,
                                 stage2_dial, normalize_pmf, haversine_deg)
from r016_combined_v2 import load_stage1_v2, stage1_scalar_v2
from r012_case_studies_2stage import load_stage2

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp"
OMNI_PATH = "/glade/work/yizhu/cuspML/output/omni_raw/omni2_all_years.dat"

EVENT_START = pd.Timestamp("2011-08-05 00:00")
EVENT_END = pd.Timestamp("2011-08-07 00:00")
FRAME_HOURS = [0, 8, 16, 24, 32, 40]
FRAME_WINDOW_H = 4  # overlay crossings within +/- this many hours of frame center


def main():
    s1_model, s1_iso, s1_feats = load_stage1_v2()
    s2 = load_stage2()
    print("  loaded R015 stage 1 + R002 stage 2")

    print(f"  loading OMNI hourly for storm window {EVENT_START} to {EVENT_END} ...")
    omni = load_omni2_hourly(OMNI_PATH, year_min=2011, year_max=2011)
    omni = derive_features(omni)
    omni = omni[(omni["datetime"] >= EVENT_START - pd.Timedelta("12h"))
                & (omni["datetime"] <= EVENT_END + pd.Timedelta("12h"))].reset_index(drop=True)
    print(f"  {len(omni)} OMNI hours in window")
    # need 14 stage-1 base features clean (doy_feat, imf_*, sw_*, B_T, clock_*, newell_cf, kan_lee_ef, vBs)
    omni_clean = omni.dropna(subset=STAGE1_BASE_FEATURES).reset_index(drop=True)
    print(f"  {len(omni_clean)} after dropna")

    print("  loading crossings + filtering to event window ...")
    crossings = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    ct = pd.to_datetime(crossings["time_start"])
    event_mask = (ct >= EVENT_START) & (ct <= EVENT_END)
    event_xings = crossings[event_mask].reset_index(drop=True)
    event_xings["t"] = pd.to_datetime(event_xings["time_start"])
    print(f"  {len(event_xings)} DMSP crossings during event window")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), subplot_kw=dict(projection="polar"))
    sw_cols_full = sw_feature_names(crossings)  # for stage 2; 74 cols
    last_cf = None

    frame_summaries = []
    for ax, hours_offset in zip(axes.flat, FRAME_HOURS):
        center_t = EVENT_START + pd.Timedelta(hours=hours_offset)
        # find nearest OMNI hour
        nearest_idx = (omni_clean["datetime"] - center_t).abs().idxmin()
        omni_row = omni_clean.iloc[nearest_idx]
        # Build SW dict for stage 1
        sw_for_s1 = {k: float(omni_row[k]) for k in STAGE1_BASE_FEATURES if k in omni_row}
        s1_p = stage1_scalar_v2(s1_model, s1_iso, s1_feats, sw_for_s1)

        # Build SW dict for stage 2: need all 74 stage-2 features
        # Map from OMNI hourly columns to stage-2 feature names. History stats not available
        # at hourly cadence -> set to instantaneous value (MVP approximation).
        sw_for_s2 = {}
        # base features (already present)
        for k in ("imf_bx", "imf_by", "imf_bz", "sw_v", "sw_n", "sw_pdyn",
                  "B_T", "clock_angle", "sin_clock_half", "newell_cf",
                  "kan_lee_ef", "vBs"):
            sw_for_s2[k] = float(omni_row[k])
        sw_for_s2["dipole_tilt"] = 0.0  # not in OMNI hourly; rough proxy
        sw_for_s2["doy"] = float(omni_row["doy_feat"])
        sw_for_s2["by_hemi"] = sw_for_s2["imf_by"]  # treating as N hemi for plot
        # history features: set mean = instantaneous, std/delta = 0
        for v, k in [(sw_for_s2["imf_bx"], "imf_bx"),
                     (sw_for_s2["imf_by"], "imf_by"),
                     (sw_for_s2["imf_bz"], "imf_bz"),
                     (sw_for_s2["sw_v"], "sw_v"),
                     (sw_for_s2["sw_n"], "sw_n"),
                     (sw_for_s2["sw_pdyn"], "sw_pdyn")]:
            for w in (15, 30, 60):
                sw_for_s2[f"{k}_mean{w}"] = v
                sw_for_s2[f"{k}_std{w}"] = 0.0
                sw_for_s2[f"{k}_delta{w}"] = 0.0
        sw_for_s2["newell_cf_mean60"] = sw_for_s2["newell_cf"]
        sw_for_s2["newell_cf_int60"] = sw_for_s2["newell_cf"] * 60
        sw_for_s2["vBs_mean60"] = sw_for_s2["vBs"]
        sw_for_s2["vBs_int60"] = sw_for_s2["vBs"] * 60

        s2_p = stage2_dial(s2, sw_for_s2, hemisphere="N")
        s2_pmf = normalize_pmf(s2_p, area_weighted=True)
        combined = s1_p * s2_pmf

        # plot
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

        # overlay crossings within +/- FRAME_WINDOW_H of center
        win = (event_xings["t"] >= center_t - pd.Timedelta(hours=FRAME_WINDOW_H)) & \
              (event_xings["t"] <= center_t + pd.Timedelta(hours=FRAME_WINDOW_H))
        nearby = event_xings[win]
        n_xings = 0
        n_xings_nh = 0
        per_xing_dist = []
        for _, x in nearby.iterrows():
            if x["hemisphere"] != "N":
                continue
            n_xings += 1; n_xings_nh += 1
            lat = abs(x["mean_mlat"]); mlt = x["mean_mlt"]
            ax.plot(2 * np.pi * mlt / 24.0, 90.0 - lat,
                    marker="o", color="red", markersize=8, mfc="none", mew=1.8)
            # measure peak distance for stats
            peak_idx = np.unravel_index(np.argmax(combined), combined.shape)
            d = haversine_deg(lat, mlt, LAT_AXIS[peak_idx[0]], MLT_AXIS[peak_idx[1]])
            per_xing_dist.append(d)

        title = (f"{center_t.strftime('%m-%d %H:%M')} UT\n"
                 f"Bz={sw_for_s2['imf_bz']:+.1f} By={sw_for_s2['imf_by']:+.1f} V={sw_for_s2['sw_v']:.0f}\n"
                 f"s1={s1_p:.3f}  N-hemi crossings within +/-{FRAME_WINDOW_H}h: {n_xings_nh}")
        if per_xing_dist:
            title += f"\nmed peak-dist {np.median(per_xing_dist):.1f}deg"
        ax.set_title(title, fontsize=9, pad=12)

        frame_summaries.append({
            "frame_t": str(center_t),
            "Bz": float(sw_for_s2["imf_bz"]),
            "By": float(sw_for_s2["imf_by"]),
            "V": float(sw_for_s2["sw_v"]),
            "stage1_P": s1_p,
            "combined_peak": float(combined.max()),
            "n_crossings_in_window": int(n_xings_nh),
            "median_peak_dist_deg": float(np.median(per_xing_dist)) if per_xing_dist else None,
        })

    fig.suptitle(f"R018: Cusp probability map during 2011-08-05 storm, N hemisphere\n"
                 f"red circles = actual DMSP N-hemi crossings (this 8-h window)",
                 fontsize=13, y=1.00)
    fig.colorbar(last_cf, ax=axes, shrink=0.6, pad=0.05, label="P(cusp in cell)")
    out_png = f"{OUT_DIR}/figures/r018_storm_2011_demo.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved -> {out_png}")
    print(f"  total event crossings: {len(event_xings)} (N+S)")
    print(f"\n  frame-by-frame summary:")
    for s in frame_summaries:
        print(f"    {s['frame_t']}  Bz={s['Bz']:+.1f}  V={s['V']:.0f}  "
              f"s1={s['stage1_P']:.3f}  n={s['n_crossings_in_window']}  "
              f"med_dist={s['median_peak_dist_deg']}")

    with open(f"{OUT_DIR}/bundles/r018_storm_demo.json", "w") as f:
        json.dump({"event": "2011-08-05",
                    "frames": frame_summaries,
                    "n_event_crossings_total": int(len(event_xings))},
                   f, indent=2)


if __name__ == "__main__":
    main()
