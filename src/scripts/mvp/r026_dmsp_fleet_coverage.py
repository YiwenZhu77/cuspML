"""R026 — DMSP F06-F18 fleet (lat, MLT) coverage analysis.

Two questions:
1. Where do the 48k crossings actually live on the dial?
2. After accounting for per-sat sun-synchronous MLT plane, what fraction of
   the polar dial is uncovered vs covered?

Plots:
  - per-satellite (lat, MLT) scatter, color-coded by satellite
  - combined density heatmap on the polar dial
  - "support mask": cells with at least 5 crossings vs not
"""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/glade/work/yizhu/cuspML/src/lib")
from cusp_map import load_crossings

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp"


def main():
    df = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    print(f"  loaded {len(df)} crossings")
    df["abs_mlat"] = df["mean_mlat"].abs()
    df["sat_year"] = df["satellite"] + "_" + pd.to_datetime(df["time_start"]).dt.year.astype(str)
    sats = sorted(df["satellite"].unique())
    print(f"  satellites: {sats}")

    # ---- per-sat MLT distribution ----
    print("\n[per-sat MLT range]")
    print(f"{'sat':>6}  {'n_xings':>8}  {'mlt_med':>8}  {'mlt 5-95%':>14}  {'lat_med':>8}  {'lat 5-95%':>14}  {'years':>14}")
    for s in sats:
        sub = df[df["satellite"] == s]
        years = pd.to_datetime(sub["time_start"]).dt.year
        mlt_p = np.percentile(sub["mean_mlt"], [5, 50, 95])
        lat_p = np.percentile(sub["abs_mlat"], [5, 50, 95])
        print(f"{s:>6}  {len(sub):>8}  {mlt_p[1]:>8.1f}  {mlt_p[0]:>5.1f}-{mlt_p[2]:>4.1f}  "
              f"{lat_p[1]:>8.1f}  {lat_p[0]:>5.1f}-{lat_p[2]:>4.1f}  {years.min()}-{years.max()}")

    # ---- combined (lat, MLT) density heatmap on polar dial ----
    fig = plt.figure(figsize=(15, 6.5))
    # left panel: density on dial
    ax1 = fig.add_subplot(1, 3, 1, projection="polar")
    mlt_edges = np.arange(0, 24.1, 0.5)
    lat_edges = np.arange(50, 90.1, 1.0)
    H_all, _, _ = np.histogram2d(df["mean_mlt"].values, df["abs_mlat"].values,
                                  bins=[mlt_edges, lat_edges])
    theta = 2 * np.pi * (mlt_edges[:-1] + 0.25) / 24.0
    r = 90.0 - (lat_edges[:-1] + 0.5)
    TT, RR = np.meshgrid(theta, r)
    pcm = ax1.pcolormesh(TT, RR, H_all.T, cmap="viridis", shading="auto")
    ax1.set_theta_zero_location("S")
    ax1.set_theta_direction(1)
    ax1.set_ylim(0, 40)
    ax1.set_yticks([10, 20, 30, 40])
    ax1.set_yticklabels(["80", "70", "60", "50"])
    ax1.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax1.set_xticklabels(["00", "06", "12", "18"])
    ax1.set_title("All 48k crossings: density per cell\n(N hemisphere folded)", fontsize=10, pad=10)
    fig.colorbar(pcm, ax=ax1, shrink=0.7, pad=0.08, label="n crossings")

    # middle panel: support mask
    ax2 = fig.add_subplot(1, 3, 2, projection="polar")
    supp = H_all.T >= 5
    pcm2 = ax2.pcolormesh(TT, RR, supp.astype(float), cmap="Greys", shading="auto", vmin=0, vmax=1)
    ax2.set_theta_zero_location("S")
    ax2.set_theta_direction(1)
    ax2.set_ylim(0, 40)
    ax2.set_yticks([10, 20, 30, 40])
    ax2.set_yticklabels(["80", "70", "60", "50"])
    ax2.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax2.set_xticklabels(["00", "06", "12", "18"])
    n_supp = supp.sum(); n_tot = supp.size
    ax2.set_title(f"Support mask (>= 5 crossings)\n{n_supp}/{n_tot} cells = {100*n_supp/n_tot:.1f}%",
                  fontsize=10, pad=10)

    # right panel: per-satellite scatter (MLT vs lat)
    ax3 = fig.add_subplot(1, 3, 3)
    colors = plt.cm.tab20(np.linspace(0, 1, len(sats)))
    for s, c in zip(sats, colors):
        sub = df[df["satellite"] == s]
        ax3.scatter(sub["mean_mlt"], sub["abs_mlat"], s=2, alpha=0.3, c=[c], label=f"{s} (n={len(sub)})")
    ax3.set_xlabel("MLT (hr)")
    ax3.set_ylabel("|MLAT| (deg)")
    ax3.set_xlim(0, 24)
    ax3.set_ylim(50, 90)
    ax3.set_xticks([0, 6, 12, 18, 24])
    ax3.legend(loc="upper right", fontsize=6, markerscale=4, framealpha=0.9, ncol=2)
    ax3.set_title("Per-satellite (MLT, |MLAT|) of crossings", fontsize=10)
    ax3.grid(alpha=0.3)

    fig.suptitle("DMSP fleet F06-F18 cusp crossing coverage (1987-2014, n=48,056)", fontsize=12, y=1.00)
    fig.tight_layout()
    out_png = f"{OUT_DIR}/figures/r026_fleet_coverage.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved -> {out_png}")

    # ---- write summary JSON ----
    cell_summary = {
        "n_cells_total": int(supp.size),
        "n_cells_supported_5plus": int(supp.sum()),
        "frac_supported": float(supp.mean()),
        "mlt_supported_bins": [f"{mlt_edges[j]:.1f}-{mlt_edges[j+1]:.1f}" for j in range(supp.shape[1])
                                if supp[:, j].any()],
    }
    per_sat = {}
    for s in sats:
        sub = df[df["satellite"] == s]
        per_sat[s] = {
            "n": int(len(sub)),
            "mlt_range": [float(np.percentile(sub["mean_mlt"], 5)),
                          float(np.percentile(sub["mean_mlt"], 95))],
            "lat_range": [float(np.percentile(sub["abs_mlat"], 5)),
                          float(np.percentile(sub["abs_mlat"], 95))],
            "years_min": int(pd.to_datetime(sub["time_start"]).dt.year.min()),
            "years_max": int(pd.to_datetime(sub["time_start"]).dt.year.max()),
        }
    out = {"cell_summary": cell_summary, "per_satellite": per_sat,
            "lat_bin_edges": lat_edges.tolist(), "mlt_bin_edges": mlt_edges.tolist()}
    with open(f"{OUT_DIR}/bundles/r026_fleet_coverage.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
