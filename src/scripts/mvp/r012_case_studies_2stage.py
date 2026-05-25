"""R012 — combine stage1 P(cusp | SW) * stage2 P(MLAT,MLT | cusp,SW), regen 6 case-study heatmaps.

Loads R011 (stage 1) and R002 (stage 2). For each of the 6 case SW states:
  - stage 1 -> scalar P(cusp at all | SW)
  - stage 2 -> 2D heatmap P(MLAT, MLT | cusp, SW)
  - combined = scalar * map  (already a calibrated joint probability)
Then re-runs the physics sanity check on the combined product.
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
from cusp_map import polar_xy, predict_proba, TrainedModel
from cusp_stage1 import STAGE1_BASE_FEATURES
from r002_case_studies import CASES, build_sw_state

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp"


def load_stage1():
    from xgboost import XGBClassifier
    m = XGBClassifier()
    m.load_model(f"{OUT_DIR}/bundles/r011_stage1_model.ubj")
    with open(f"{OUT_DIR}/bundles/r011_stage1_isotonic.pkl", "rb") as f:
        iso = pickle.load(f)
    with open(f"{OUT_DIR}/bundles/r011_stage1_features.json") as f:
        feats = json.load(f)
    return m, iso, feats


def load_stage2():
    from xgboost import XGBClassifier
    m = XGBClassifier()
    m.load_model(f"{OUT_DIR}/bundles/r002_model.ubj")
    iso_path = f"{OUT_DIR}/bundles/r002_isotonic.pkl"
    iso = None
    if os.path.exists(iso_path):
        with open(iso_path, "rb") as f:
            iso = pickle.load(f)
    with open(f"{OUT_DIR}/bundles/r002_features.json") as f:
        feats = json.load(f)
    return TrainedModel(model=m, isotonic=iso, feature_names=feats,
                        used_calibration=(iso is not None))


def stage1_scalar(s1_model, s1_iso, s1_feats, sw_state):
    """Map a stage-2 sw_state dict to stage-1 hourly features + predict scalar P."""
    rec = dict(sw_state)
    rec["doy_feat"] = sw_state.get("doy", 80)
    arr = np.array([[rec[k] for k in s1_feats]], dtype=np.float32)
    raw = s1_model.predict_proba(arr)[:, 1]
    return float(s1_iso.transform(raw)[0])


def stage2_grid(trained, sw_state, hemisphere="N",
                mlat_range=(50, 90, 1.0), mlt_range=(0, 24, 0.5)):
    """Same as r002_case_studies.predict_grid but local to avoid circular import."""
    lat_lo, lat_hi, dlat = mlat_range
    mlt_lo, mlt_hi, dmlt = mlt_range
    mlat_axis = np.arange(lat_lo, lat_hi + 1e-9, dlat)
    mlt_axis = np.arange(mlt_lo, mlt_hi + 1e-9, dmlt)
    MM, LL = np.meshgrid(mlt_axis, mlat_axis)
    n = MM.size
    x, y = polar_xy(LL.ravel(), MM.ravel())
    rec = dict(sw_state)
    rec["hemi_code"] = 1.0 if hemisphere == "N" else 0.0
    grid = {k: np.full(n, v, dtype=np.float32) for k, v in rec.items()}
    grid["x_polar"] = x.astype(np.float32)
    grid["y_polar"] = y.astype(np.float32)
    df = pd.DataFrame(grid)
    X = df[trained.feature_names].values.astype(np.float32)
    P = predict_proba(trained, X).reshape(LL.shape)
    return mlat_axis, mlt_axis, P


def plot_dial(mlat_axis, mlt_axis, P, title, out_path, vmax=None):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    theta = 2 * np.pi * mlt_axis / 24.0
    r = 90.0 - mlat_axis
    TT, RR = np.meshgrid(theta, r)
    cf = ax.pcolormesh(TT, RR, P, cmap="viridis", vmin=0, vmax=vmax or 1, shading="auto")
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 40)
    ax.set_yticks([10, 20, 30, 40])
    ax.set_yticklabels(["80", "70", "60", "50"])
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["00 MLT", "06", "12 (noon)", "18"])
    ax.set_title(title, pad=20)
    cbar = fig.colorbar(cf, ax=ax, shrink=0.7, pad=0.10)
    cbar.set_label("P(cusp observed)")
    fig.text(0.5, 0.02, "north hemisphere; stage1 x stage2 combined",
             ha="center", fontsize=8, color="gray")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    s1_model, s1_iso, s1_feats = load_stage1()
    s2 = load_stage2()
    print(f"  loaded stage1 ({len(s1_feats)} feats) and stage2 ({len(s2.feature_names)} feats)")

    figs_dir = f"{OUT_DIR}/figures"
    os.makedirs(figs_dir, exist_ok=True)
    summary = {}
    # first pass: compute all maps to find shared vmax
    all_maps = {}
    for c in CASES:
        sw = build_sw_state(c, hemisphere="N")
        s1_p = stage1_scalar(s1_model, s1_iso, s1_feats, sw)
        mlat_axis, mlt_axis, s2_p = stage2_grid(s2, sw, hemisphere="N")
        combined = s1_p * s2_p
        all_maps[c["name"]] = {"s1": s1_p, "s2": s2_p, "combined": combined,
                                "mlat": mlat_axis, "mlt": mlt_axis,
                                "title": c["title"]}
        print(f"  {c['name']:>30s}  stage1={s1_p:.4f}  stage2_peak={s2_p.max():.3f}  "
              f"combined_peak={combined.max():.4f}")

    # shared vmax for visual comparability
    vmax_combined = max(d["combined"].max() for d in all_maps.values())
    print(f"\n  shared vmax for combined plots: {vmax_combined:.4f}")

    for name, d in all_maps.items():
        plot_dial(d["mlat"], d["mlt"], d["combined"],
                  f"{d['title']}  (combined, stage1={all_maps[name]['s1']:.3f})",
                  f"{figs_dir}/2stage_{name}.png", vmax=vmax_combined)
        summary[name] = {
            "stage1_p": float(d["s1"]),
            "stage2_peak_p": float(d["s2"].max()),
            "combined_peak_p": float(d["combined"].max()),
            "combined_peak_lat": float(d["mlat"][np.argmax(d["combined"].max(axis=1))]),
            "combined_peak_mlt": float(d["mlt"][np.argmax(d["combined"].max(axis=0))]),
            "combined_midnight_mean": float(
                d["combined"][:, (d["mlt"] < 4) | (d["mlt"] > 20)].mean()),
        }

    # physics sanity on combined product
    s = summary
    checks = {
        "storm_peak_higher_than_quiet": s["case6_storm"]["combined_peak_p"] > s["case5_quiet"]["combined_peak_p"],
        "south_Bz_peak_higher_than_north_Bz": s["case1_strong_south_Bz"]["combined_peak_p"] > s["case2_strong_north_Bz"]["combined_peak_p"],
        "strong_driving_above_0.1": s["case1_strong_south_Bz"]["combined_peak_p"] > 0.1 and s["case6_storm"]["combined_peak_p"] > 0.1,
        "quiet_below_strong_driving_in_stage1": s["case5_quiet"]["stage1_p"] < s["case1_strong_south_Bz"]["stage1_p"],
        "south_Bz_lower_lat_than_north_Bz": s["case1_strong_south_Bz"]["combined_peak_lat"] < s["case2_strong_north_Bz"]["combined_peak_lat"],
        "midnight_low": all(s[k]["combined_midnight_mean"] < 0.05 for k in s),
    }
    print(f"\n[physics sanity 2-stage]")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    out = {"summary": summary, "physics_checks": checks,
           "n_pass": sum(checks.values()), "n_total": len(checks)}
    with open(f"{OUT_DIR}/bundles/r012_2stage_case_studies.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  figures -> {figs_dir}/2stage_*.png")
    print(f"  sanity: {out['n_pass']}/{out['n_total']}")


if __name__ == "__main__":
    main()
