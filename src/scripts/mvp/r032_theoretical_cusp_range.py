"""R032 — DMSP coverage vs theoretical cusp position range.

Verify whether DMSP's observed (MLT, MLAT) distribution covers all positions
where theory says cusp can be, across the full SW state range we observed.

Approach: for each of our 48k crossings, take its SW state. Use published
empirical cusp position formulas to predict where cusp SHOULD be at that SW.
Then plot theory-predicted distribution vs observed DMSP distribution.

Cusp position formulas used (literature-derived, not from DMSP cusp catalog
itself):
- cusp invariant latitude (Newell 2006, fit equation 5 approximation):
    cusp_lat = ~78.5 - 4.5 * (newell_cf / 1e4)^(1/3)
  (parameters approximate; what matters is the FUNCTIONAL FORM)
- cusp MLT shift (Cowley 1981 + Newell 2007 By dependence):
    cusp_mlt_shift = -0.5 * sign(by_hemi) * |By| / 5.0
    cusp_mlt = 12.0 + cusp_mlt_shift
- Dipole tilt shift (Newell 1989): cusp_lat -= 0.05 * tilt_deg

These are coarse empirical scalings — not exact, but they bracket where cusp
CAN be at extreme SW. If DMSP-observed locations cover the predicted range,
DMSP coverage is sufficient.
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/glade/work/yizhu/cuspML/src/lib")
from cusp_map import load_crossings

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp"


def theory_cusp_position(row):
    """Predicted cusp (MLAT, MLT) for a given SW state, from empirical formulas."""
    cf = max(0.0, float(row.get("newell_cf", 0.0)))
    by = float(row.get("imf_by", 0.0))
    by_hemi = float(row.get("by_hemi", by))
    tilt = float(row.get("dipole_tilt", 0.0))

    # Newell 2006 cusp lat fit (coarse): cusp moves equatorward with stronger CF
    # CF typically in [0, 30000] units; cube-root for the nonlinear saturation
    cusp_lat = 78.5 - 4.5 * (cf / 1e4) ** (1 / 3)
    # dipole tilt: -0.05 deg per deg tilt (Newell 1989)
    cusp_lat -= 0.05 * tilt

    # MLT shift: Cowley 1981, by_hemi positive in N -> dawnward shift
    cusp_mlt_shift = -0.5 * np.sign(by_hemi) * min(abs(by), 15.0) / 5.0
    cusp_mlt = 12.0 + cusp_mlt_shift

    return cusp_lat, cusp_mlt


def main():
    df = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    print(f"loaded {len(df)} crossings")
    obs_lat = df["mean_mlat"].abs().values
    obs_mlt = df["mean_mlt"].values

    # theory prediction per crossing
    print("computing theoretical cusp position per crossing ...")
    th_lat = np.zeros(len(df))
    th_mlt = np.zeros(len(df))
    for i, row in df.iterrows():
        lat, mlt = theory_cusp_position(row)
        th_lat[i] = lat
        th_mlt[i] = mlt

    # SW state extremes — also predict cusp at corners of SW range
    sw_extremes = pd.DataFrame([
        {"label": "strong south Bz storm", "imf_bz": -25, "imf_by": 0,  "sw_v": 800, "newell_cf": 30000, "dipole_tilt": 0, "by_hemi": 0},
        {"label": "strong south Bz quiet", "imf_bz": -10, "imf_by": 0,  "sw_v": 400, "newell_cf": 8000,  "dipole_tilt": 0, "by_hemi": 0},
        {"label": "strong north Bz",       "imf_bz": +15, "imf_by": 0,  "sw_v": 400, "newell_cf": 200,   "dipole_tilt": 0, "by_hemi": 0},
        {"label": "strong By+ N hemi",     "imf_bz": -5,  "imf_by": +12,"sw_v": 500, "newell_cf": 5000,  "dipole_tilt": 0, "by_hemi": +12},
        {"label": "strong By- N hemi",     "imf_bz": -5,  "imf_by": -12,"sw_v": 500, "newell_cf": 5000,  "dipole_tilt": 0, "by_hemi": -12},
        {"label": "strong By+ S hemi",     "imf_bz": -5,  "imf_by": +12,"sw_v": 500, "newell_cf": 5000,  "dipole_tilt": 0, "by_hemi": -12},
        {"label": "summer tilt",           "imf_bz": -3,  "imf_by": 0,  "sw_v": 400, "newell_cf": 3000,  "dipole_tilt": +34, "by_hemi": 0},
        {"label": "winter tilt",           "imf_bz": -3,  "imf_by": 0,  "sw_v": 400, "newell_cf": 3000,  "dipole_tilt": -34, "by_hemi": 0},
        {"label": "extreme storm",         "imf_bz": -35, "imf_by": 0,  "sw_v":1000, "newell_cf": 60000, "dipole_tilt": 0, "by_hemi": 0},
    ])
    ext_th = sw_extremes.apply(lambda r: pd.Series(theory_cusp_position(r), index=["lat", "mlt"]), axis=1)
    sw_extremes = pd.concat([sw_extremes, ext_th], axis=1)
    print(f"\nSW corner cases — theoretical cusp position:")
    print(sw_extremes[["label", "imf_bz", "imf_by", "newell_cf", "dipole_tilt", "lat", "mlt"]].to_string(index=False))

    # ---- compare ----
    print(f"\nObserved DMSP crossings:")
    print(f"  lat 5/50/95%: {np.percentile(obs_lat,5):.1f} / {np.percentile(obs_lat,50):.1f} / {np.percentile(obs_lat,95):.1f}")
    print(f"  mlt 5/50/95%: {np.percentile(obs_mlt,5):.1f} / {np.percentile(obs_mlt,50):.1f} / {np.percentile(obs_mlt,95):.1f}")

    print(f"\nTheory prediction from same 48k SW states:")
    print(f"  lat 5/50/95%: {np.percentile(th_lat,5):.1f} / {np.percentile(th_lat,50):.1f} / {np.percentile(th_lat,95):.1f}")
    print(f"  mlt 5/50/95%: {np.percentile(th_mlt,5):.1f} / {np.percentile(th_mlt,50):.1f} / {np.percentile(th_mlt,95):.1f}")

    # ---- plot 2-panel comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # MLAT distribution
    ax = axes[0]
    ax.hist(obs_lat, bins=np.arange(50, 91, 1), alpha=0.5, color="firebrick",
            label=f"DMSP observed (n={len(df)})", density=True)
    ax.hist(th_lat, bins=np.arange(50, 91, 1), alpha=0.5, color="steelblue",
            label="Theory prediction (Newell 2006 + Newell 1989 tilt)", density=True)
    for _, r in sw_extremes.iterrows():
        ax.axvline(r["lat"], color="green", alpha=0.4, linestyle=":", lw=1)
    ax.set_xlabel("|MLAT| (deg)")
    ax.set_ylabel("density")
    ax.set_title("|MLAT| distribution: DMSP vs theory\ngreen dashed = SW corner cases")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(50, 90)

    # MLT distribution
    ax = axes[1]
    ax.hist(obs_mlt, bins=np.arange(0, 24.1, 0.5), alpha=0.5, color="firebrick",
            label="DMSP observed", density=True)
    ax.hist(th_mlt, bins=np.arange(0, 24.1, 0.5), alpha=0.5, color="steelblue",
            label="Theory prediction (Cowley 1981 By shift)", density=True)
    for _, r in sw_extremes.iterrows():
        ax.axvline(r["mlt"], color="green", alpha=0.4, linestyle=":", lw=1)
    ax.set_xlabel("MLT (hr)")
    ax.set_ylabel("density")
    ax.set_title("MLT distribution: DMSP vs theory")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])

    fig.suptitle("R032: DMSP cusp coverage vs theoretical possible-cusp range\n"
                 "If DMSP covers theory-predicted range, DMSP coverage is sufficient",
                 fontsize=11)
    fig.tight_layout()
    out_png = f"{OUT_DIR}/figures/r032_theory_vs_dmsp_coverage.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out_png}")

    # ---- coverage summary ----
    th_lat_range = (np.percentile(th_lat, 1), np.percentile(th_lat, 99))
    th_mlt_range = (np.percentile(th_mlt, 1), np.percentile(th_mlt, 99))
    obs_lat_in_theory = ((obs_lat >= th_lat_range[0] - 1) & (obs_lat <= th_lat_range[1] + 1)).mean()
    obs_mlt_in_theory = ((obs_mlt >= th_mlt_range[0] - 0.5) & (obs_mlt <= th_mlt_range[1] + 0.5)).mean()
    # theory-predicted positions outside what DMSP actually observed:
    obs_lat_range = (np.percentile(obs_lat, 1), np.percentile(obs_lat, 99))
    obs_mlt_range = (np.percentile(obs_mlt, 1), np.percentile(obs_mlt, 99))
    theory_outside_obs_lat = ((th_lat < obs_lat_range[0]) | (th_lat > obs_lat_range[1])).mean()
    theory_outside_obs_mlt = ((th_mlt < obs_mlt_range[0]) | (th_mlt > obs_mlt_range[1])).mean()
    print(f"\nCoverage check:")
    print(f"  Theory MLAT 1-99% range: [{th_lat_range[0]:.1f}, {th_lat_range[1]:.1f}]")
    print(f"  Theory MLT  1-99% range: [{th_mlt_range[0]:.1f}, {th_mlt_range[1]:.1f}]")
    print(f"  DMSP MLAT 1-99% range:   [{obs_lat_range[0]:.1f}, {obs_lat_range[1]:.1f}]")
    print(f"  DMSP MLT  1-99% range:   [{obs_mlt_range[0]:.1f}, {obs_mlt_range[1]:.1f}]")
    print(f"  Theory cusps OUTSIDE DMSP MLAT range: {100*theory_outside_obs_lat:.1f}%")
    print(f"  Theory cusps OUTSIDE DMSP MLT range:  {100*theory_outside_obs_mlt:.1f}%")
    if theory_outside_obs_lat < 0.05 and theory_outside_obs_mlt < 0.05:
        print(f"  CONCLUSION: <5% of theory-predicted cusps fall outside DMSP-observed range → DMSP coverage IS SUFFICIENT")
    else:
        print(f"  CONCLUSION: DMSP misses {max(theory_outside_obs_lat, theory_outside_obs_mlt)*100:.1f}% of theory-predicted positions")

    out = {
        "n_crossings": int(len(df)),
        "observed_lat_p1_p50_p99": [float(np.percentile(obs_lat, p)) for p in [1, 50, 99]],
        "observed_mlt_p1_p50_p99": [float(np.percentile(obs_mlt, p)) for p in [1, 50, 99]],
        "theory_lat_p1_p50_p99": [float(np.percentile(th_lat, p)) for p in [1, 50, 99]],
        "theory_mlt_p1_p50_p99": [float(np.percentile(th_mlt, p)) for p in [1, 50, 99]],
        "frac_theory_outside_obs_lat": float(theory_outside_obs_lat),
        "frac_theory_outside_obs_mlt": float(theory_outside_obs_mlt),
        "sw_corner_cases": sw_extremes.to_dict(orient="records"),
    }
    with open(f"{OUT_DIR}/bundles/r032_theory_vs_dmsp.json", "w") as f:
        json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
