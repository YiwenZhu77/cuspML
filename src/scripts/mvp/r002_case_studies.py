"""Render 6 case-study probability maps from the R002 model.

For each SW state in the spec's case table, broadcast to a (|MLAT|, MLT) grid,
predict P, plot as a polar dial heatmap (north hemisphere). Also compute
quantitative summary statistics per case (peak P, peak location, midnight P).
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
from cusp_map import polar_xy, predict_proba, TrainedModel

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp"


CASES = [
    {"name": "case1_strong_south_Bz",   "Bz": -10, "By":  0, "Bx": 0, "V": 500, "n":  5, "title": "Strong south Bz (-10 nT)"},
    {"name": "case2_strong_north_Bz",   "Bz":  10, "By":  0, "Bx": 0, "V": 500, "n":  5, "title": "Strong north Bz (+10 nT)"},
    {"name": "case3_strong_By_pos",     "Bz":  -3, "By":  8, "Bx": 0, "V": 500, "n":  5, "title": "Strong By+ (Bz=-3, By=+8)"},
    {"name": "case4_strong_By_neg",     "Bz":  -3, "By": -8, "Bx": 0, "V": 500, "n":  5, "title": "Strong By- (Bz=-3, By=-8)"},
    {"name": "case5_quiet",             "Bz":   0, "By":  0, "Bx": 0, "V": 350, "n":  3, "title": "Quiet (Bz=0, V=350)"},
    {"name": "case6_storm",             "Bz": -15, "By":  0, "Bx": 0, "V": 700, "n": 15, "title": "Storm (Bz=-15, V=700)"},
]


def build_sw_state(c, hemisphere="N", doy=80, dipole_tilt=0.0):
    """Construct a 74-feature SW state for a synthetic case.
    History stats are set to the instantaneous value as a first-pass approximation.
    Flagged in figure caption as MVP simplification.
    """
    Bx, By, Bz = c["Bx"], c["By"], c["Bz"]
    V, n = c["V"], c["n"]
    pdyn = 1.6726e-6 * n * V * V * 1e-3  # nPa, approx
    B_T = np.sqrt(By ** 2 + Bz ** 2)
    clock = np.arctan2(By, Bz)
    sin_half = np.sin(clock / 2)
    newell = (V ** (4 / 3)) * (B_T ** (2 / 3)) * (abs(sin_half) ** (8 / 3))
    kan_lee = V * B_T * (sin_half ** 2)
    vBs = V * (-Bz if Bz < 0 else 0.0)
    by_hemi = By * (1 if hemisphere == "N" else -1)
    hemi_code = 1.0 if hemisphere == "N" else 0.0

    base = {
        "dipole_tilt": dipole_tilt, "hemi_code": hemi_code, "doy": doy,
        "imf_bx": Bx, "imf_by": By, "imf_bz": Bz,
        "sw_v": V, "sw_n": n, "sw_pdyn": pdyn,
        "B_T": B_T, "clock_angle": clock, "sin_clock_half": sin_half,
        "newell_cf": newell, "kan_lee_ef": kan_lee, "vBs": vBs, "by_hemi": by_hemi,
    }
    # history features: instantaneous value, std=0, delta=0
    hist = {}
    for v, k in [(Bx, "imf_bx"), (By, "imf_by"), (Bz, "imf_bz"),
                 (V, "sw_v"), (n, "sw_n"), (pdyn, "sw_pdyn")]:
        for win in (15, 30, 60):
            hist[f"{k}_mean{win}"] = v
            hist[f"{k}_std{win}"] = 0.0
            hist[f"{k}_delta{win}"] = 0.0
    hist["newell_cf_mean60"] = newell
    hist["newell_cf_int60"] = newell * 60  # rough integral
    hist["vBs_mean60"] = vBs
    hist["vBs_int60"] = vBs * 60
    base.update(hist)
    return base


def predict_grid(trained, sw_state, hemisphere="N",
                 mlat_range=(50, 90, 1.0), mlt_range=(0, 24, 0.5)):
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
    missing = [f for f in trained.feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"sw_state missing features: {missing[:10]}")
    X = df[trained.feature_names].values.astype(np.float32)
    P = predict_proba(trained, X).reshape(LL.shape)
    return mlat_axis, mlt_axis, P


def plot_dial(mlat_axis, mlt_axis, P, title, out_path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    theta = 2 * np.pi * mlt_axis / 24.0
    r = 90.0 - mlat_axis
    TT, RR = np.meshgrid(theta, r)
    cf = ax.pcolormesh(TT, RR, P, cmap="viridis", vmin=0, vmax=1, shading="auto")
    ax.set_theta_zero_location("S")  # 0 MLT (midnight) at bottom
    ax.set_theta_direction(1)        # MLT increases counter-clockwise (matches AACGM convention)
    ax.set_ylim(0, 40)
    ax.set_yticks([10, 20, 30, 40])
    ax.set_yticklabels(["80", "70", "60", "50"])
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["00 MLT", "06", "12 (noon)", "18"])
    ax.set_title(title, pad=20)
    cbar = fig.colorbar(cf, ax=ax, shrink=0.7, pad=0.10)
    cbar.set_label("P(cusp)")
    fig.text(0.5, 0.02, "north hemisphere; instantaneous SW (no real history)",
             ha="center", fontsize=8, color="gray")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize(mlat_axis, mlt_axis, P):
    idx = np.unravel_index(np.argmax(P), P.shape)
    peak_lat = mlat_axis[idx[0]]
    peak_mlt = mlt_axis[idx[1]]
    peak_p = P[idx]
    # midnight average
    mlt_mid_mask = (mlt_axis < 4) | (mlt_axis > 20)
    midnight_p = P[:, mlt_mid_mask].mean()
    # dayside (MLT 9-15) average at MLAT 70-80
    dayside_mask = (mlt_axis >= 9) & (mlt_axis <= 15)
    daylat_mask = (mlat_axis >= 70) & (mlat_axis <= 80)
    dayside_p = P[np.ix_(daylat_mask, dayside_mask)].mean()
    return {"peak_p": float(peak_p), "peak_lat": float(peak_lat), "peak_mlt": float(peak_mlt),
            "midnight_mean_p": float(midnight_p), "dayside_mean_p": float(dayside_p)}


def main():
    # load trained model + isotonic + features
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(f"{OUT_DIR}/bundles/r002_model.ubj")
    iso_path = f"{OUT_DIR}/bundles/r002_isotonic.pkl"
    iso = None
    if os.path.exists(iso_path):
        with open(iso_path, "rb") as f:
            iso = pickle.load(f)
    with open(f"{OUT_DIR}/bundles/r002_features.json") as f:
        feat_names = json.load(f)
    trained = TrainedModel(model=model, isotonic=iso, feature_names=feat_names,
                           used_calibration=(iso is not None))
    print(f"  loaded R002 model: {len(feat_names)} features, isotonic={'on' if iso else 'off'}")

    figs_dir = f"{OUT_DIR}/figures"
    os.makedirs(figs_dir, exist_ok=True)
    summary = {}
    for c in CASES:
        sw = build_sw_state(c, hemisphere="N")
        mlat_axis, mlt_axis, P = predict_grid(trained, sw, hemisphere="N")
        plot_dial(mlat_axis, mlt_axis, P, c["title"], f"{figs_dir}/{c['name']}.png")
        summary[c["name"]] = summarize(mlat_axis, mlt_axis, P)
        print(f"  {c['name']:>30s}  peak P={summary[c['name']]['peak_p']:.3f} "
              f"at lat={summary[c['name']]['peak_lat']:.0f}, MLT={summary[c['name']]['peak_mlt']:.1f} "
              f"| midnight={summary[c['name']]['midnight_mean_p']:.3f}")

    # physics sanity check
    s = summary
    checks = {
        "peak_in_dayside_all_cases": all(
            8 <= s[c["name"]]["peak_mlt"] <= 16 for c in CASES if "quiet" not in c["name"]),
        "south_Bz_lower_lat_than_north_Bz": s["case1_strong_south_Bz"]["peak_lat"] < s["case2_strong_north_Bz"]["peak_lat"],
        "by_shifts_mlt_in_opposite_directions": abs(s["case3_strong_By_pos"]["peak_mlt"] - s["case4_strong_By_neg"]["peak_mlt"]) >= 0.5,
        "quiet_max_p_below_0.6": s["case5_quiet"]["peak_p"] < 0.6,
        "storm_lowest_peak_lat": s["case6_storm"]["peak_lat"] <= min(
            s[c["name"]]["peak_lat"] for c in CASES if c["name"] != "case6_storm"),
        "midnight_low_all_cases": all(s[c["name"]]["midnight_mean_p"] < 0.1 for c in CASES),
    }
    print(f"\n[physics sanity check]")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    out = {"summary": summary, "physics_checks": checks,
           "n_pass": sum(checks.values()), "n_total": len(checks)}
    with open(f"{OUT_DIR}/bundles/r002_case_studies.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  figures -> {figs_dir}/case*.png")
    print(f"  case-study sanity passes: {out['n_pass']}/{out['n_total']}")


if __name__ == "__main__":
    main()
