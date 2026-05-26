"""R031 — 真 vs 合成负样本覆盖 + cusp positives 三联图。

左: 48k cusp crossings positive 位置密度 (cusp 物理位置)
中: F10 1993-94 真 DMSP 非 cusp 1Hz spectra 密度 (实际 DMSP 飞过哪里)
右: R002 合成负样本密度 (per crossing 5 near + 5 far × 48k crossings)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, "/glade/work/yizhu/cuspML/src/lib")
from cusp_map import load_crossings

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp"
PILOT_DIR = "/glade/work/yizhu/cuspML/output/pilot_spectra"

LAT_EDGES = np.arange(50, 90.01, 1.0)
MLT_EDGES = np.arange(0, 24.01, 0.5)


def polar_plot(ax, H, title, vmin=None, vmax=None, cmap="viridis"):
    theta = 2 * np.pi * (MLT_EDGES[:-1] + 0.25) / 24.0
    r = 90.0 - (LAT_EDGES[:-1] + 0.5)
    TT, RR = np.meshgrid(theta, r)
    H_safe = np.where(H > 0, H, np.nan)
    pcm = ax.pcolormesh(TT, RR, H_safe,
                         cmap=cmap, shading="auto",
                         norm=LogNorm(vmin=max(1, vmin or 1), vmax=vmax or H.max()))
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 40)
    ax.set_yticks([10, 20, 30, 40])
    ax.set_yticklabels(["80", "70", "60", "50"])
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["00", "06", "12", "18"])
    ax.set_title(title, fontsize=10, pad=14)
    return pcm


def main():
    # ---- 1. cusp positives from 48k crossings ----
    print("loading 48k crossings ...")
    df = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    pos_mlt = df["mean_mlt"].values
    pos_lat = df["mean_mlat"].abs().values
    H_pos, _, _ = np.histogram2d(pos_mlt, pos_lat, bins=[MLT_EDGES, LAT_EDGES])
    H_pos = H_pos.T  # (lat, mlt)
    print(f"  48k positives: {len(df)} rows, max cell {int(H_pos.max())}, n cells with data {(H_pos > 0).sum()}/{H_pos.size} ({100*(H_pos>0).sum()/H_pos.size:.1f}%)")

    # ---- 2. real DMSP non-cusp 1Hz spectra (F10 1993-94 from R020 parquet) ----
    print("\nloading F10 1993-94 real 1Hz spectra ...")
    parts = []
    for y in [1993, 1994]:
        path = f"{PILOT_DIR}/pilot_spectra_F10_{y}.parquet"
        if os.path.exists(path):
            parts.append(pd.read_parquet(path))
    if parts:
        spec = pd.concat(parts, ignore_index=True)
        spec_neg = spec[spec["cusp_mask"] == 0]
        spec_pos = spec[spec["cusp_mask"] == 1]
        print(f"  total spectra: {len(spec)} (pos {len(spec_pos)}, neg {len(spec_neg)})")
        H_real_neg, _, _ = np.histogram2d(spec_neg["mlt"].values, spec_neg["abs_mlat"].values,
                                          bins=[MLT_EDGES, LAT_EDGES])
        H_real_neg = H_real_neg.T
        print(f"  real neg cells with data: {(H_real_neg > 0).sum()}/{H_real_neg.size} ({100*(H_real_neg>0).sum()/H_real_neg.size:.1f}%)")
    else:
        H_real_neg = np.zeros_like(H_pos)
        print("  MISSING — no F10 pilot parquet")

    # ---- 3. R002 synth negatives (regenerate to get spatial dist) ----
    print("\nregenerating R002 synth negatives for spatial dist (sample 5000 crossings) ...")
    from cusp_map import expand_crossing
    rng = np.random.default_rng(42)
    sub = df.sample(n=5000, random_state=42).reset_index(drop=True)
    neg_lats = []; neg_mlts = []
    for _, row in sub.iterrows():
        try:
            exp_df = expand_crossing(row, n_pos=5, k_neg=10, rng=rng)
        except Exception:
            continue
        neg = exp_df[exp_df["label"] == 0]
        neg_lats.extend(neg["abs_mlat"].tolist())
        neg_mlts.extend(neg["mlt"].tolist())
    H_synth_neg, _, _ = np.histogram2d(neg_mlts, neg_lats, bins=[MLT_EDGES, LAT_EDGES])
    H_synth_neg = H_synth_neg.T
    print(f"  synth neg: {len(neg_lats)} samples, cells with data {(H_synth_neg > 0).sum()}/{H_synth_neg.size} ({100*(H_synth_neg>0).sum()/H_synth_neg.size:.1f}%)")
    # scale to full 48k equivalent
    H_synth_neg_full = H_synth_neg * (len(df) / len(sub))

    # ---- plot ----
    fig = plt.figure(figsize=(17, 6.5))

    ax1 = fig.add_subplot(1, 3, 1, projection="polar")
    pcm1 = polar_plot(ax1, H_pos,
                      f"48k cusp positives (real)\ncells with data: {100*(H_pos>0).sum()/H_pos.size:.1f}%",
                      cmap="Reds")
    fig.colorbar(pcm1, ax=ax1, shrink=0.65, pad=0.10, label="n crossings")

    ax2 = fig.add_subplot(1, 3, 2, projection="polar")
    pcm2 = polar_plot(ax2, H_real_neg,
                      f"F10 1993-94 real DMSP non-cusp 1Hz (R020 pilot)\ncells with data: {100*(H_real_neg>0).sum()/H_real_neg.size:.1f}%",
                      cmap="Greens")
    fig.colorbar(pcm2, ax=ax2, shrink=0.65, pad=0.10, label="n 1Hz spectra")

    ax3 = fig.add_subplot(1, 3, 3, projection="polar")
    pcm3 = polar_plot(ax3, H_synth_neg_full,
                      f"R002 合成负样本 (5 near + 5 random / pos, all 48k)\ncells with data: {100*(H_synth_neg>0).sum()/H_synth_neg.size:.1f}%",
                      cmap="Blues")
    fig.colorbar(pcm3, ax=ax3, shrink=0.65, pad=0.10, label="n synth neg (estimated)")

    fig.suptitle("R031: cusp positives vs real DMSP non-cusp coverage vs R002 synth negatives\n"
                 "对比为什么我们用合成负样本",
                 fontsize=11)
    fig.tight_layout()
    out_png = f"{OUT_DIR}/figures/r031_coverage_comparison.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out_png}")


if __name__ == "__main__":
    main()
