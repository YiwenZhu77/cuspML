"""R028 (thesis chapter, deliverable 3) — per-bin error analysis on full test set.

End-to-end peak distance + true-cell logp for ALL 7933 held-out crossings.
Stratified by:
  - MLT bin (6 bins: 0-4, 4-8, 8-12, 12-16, 16-20, 20-24)
  - |MLAT| bin (4 bins: 50-65, 65-75, 75-83, 83-90)
  - Hemisphere
  - IMF Bz sign (+/-)
  - AE level (4 bins: <100, 100-300, 300-500, >=500)
  - Storm time (|Bz|>=10 OR V>=600 OR AE>=300)

Plus coverage density per cell (from training crossings) to enable
error-vs-coverage correlation analysis.

Output:
- src/kernels/cuspmap_mvp/figures/thesis_per_bin_error.png (panels)
- src/kernels/cuspmap_mvp/figures/thesis_coverage_vs_error.png
- src/kernels/cuspmap_mvp/bundles/r028_per_bin.json
"""
import json, os, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/glade/work/yizhu/cuspML/src/lib")
sys.path.insert(0, "/glade/work/yizhu/cuspML/src/scripts/mvp")
from cusp_map import load_crossings, sw_feature_names, polar_xy, predict_proba, TrainedModel
from cusp_stage1 import STAGE1_BASE_FEATURES
from r014_endtoend_eval import (LAT_AXIS, MLT_AXIS, MM, LL, CELL_AREA,
                                 cell_of, stage2_dial, normalize_pmf, haversine_deg)
from r016_combined_v2 import load_stage1_v2, stage1_scalar_v2
from r012_case_studies_2stage import load_stage2

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp"


def main():
    t0 = time.time()
    s1_model, s1_iso, s1_feats = load_stage1_v2()
    s2 = load_stage2()
    print("loaded R015 stage 1 + R002 stage 2")

    df = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    sw_cols = sw_feature_names(df)
    keep = sw_cols + ["eq_mlat", "pole_mlat", "eq_mlt", "pole_mlt",
                      "satellite", "hemisphere", "time_start", "ae_index"]
    keep = [c for c in keep if c in df.columns]
    df_clean = df[keep].dropna(subset=sw_cols + ["eq_mlat", "pole_mlat", "eq_mlt", "pole_mlt"]).reset_index(drop=True)
    rng = np.random.default_rng(42)
    cids = np.arange(len(df_clean)); rng.shuffle(cids)
    n_test = int(len(cids) * 0.2)
    test_df = df_clean.iloc[cids[:n_test]].reset_index(drop=True)
    train_df = df_clean.iloc[cids[n_test:]].reset_index(drop=True)
    print(f"test {len(test_df)}, train {len(train_df)}")

    # ---- coverage density from training set ----
    train_mlt = train_df["mean_mlt"].values if "mean_mlt" in train_df.columns else (train_df["eq_mlt"] + train_df["pole_mlt"]).values / 2
    train_lat = train_df["mean_mlat"].abs().values if "mean_mlat" in train_df.columns else (train_df["eq_mlat"].abs() + train_df["pole_mlat"].abs()).values / 2
    cov, _, _ = np.histogram2d(train_mlt, train_lat,
                                bins=[np.arange(0, 24.01, 0.5), np.arange(50, 90.01, 1.0)])
    cov = cov.T  # (lat, mlt)
    print(f"coverage hist: shape {cov.shape}, max {cov.max():.0f}, n_supported_cells (>=5) {(cov >= 5).sum()}/{cov.size}")

    # ---- end-to-end eval over ALL test crossings ----
    print(f"running end-to-end on {len(test_df)} test crossings ...")
    rec = []
    t1 = time.time()
    for i, row in test_df.iterrows():
        true_lat = abs(row["mean_mlat"]) if "mean_mlat" in row else (abs(row["eq_mlat"]) + abs(row["pole_mlat"])) / 2
        true_mlt = row["mean_mlt"] if "mean_mlt" in row else (row["eq_mlt"] + row["pole_mlt"]) / 2
        hemi = "N" if row["hemisphere"] == "N" else "S"
        sw = {c: row[c] for c in sw_cols if c in row}
        sw_s1 = {c: row[c] for c in STAGE1_BASE_FEATURES if c in row}
        sw_s1["doy_feat"] = sw.get("doy", pd.to_datetime(row["time_start"]).dayofyear)
        if "B_T" not in sw_s1:
            sw_s1["B_T"] = float(np.sqrt(row["imf_by"] ** 2 + row["imf_bz"] ** 2))
        s1_p = stage1_scalar_v2(s1_model, s1_iso, s1_feats, sw_s1)
        s2_p = stage2_dial(s2, sw, hemisphere=hemi)
        pmf = normalize_pmf(s2_p, area_weighted=True)
        comb = s1_p * pmf
        ci, cj = cell_of(true_lat, true_mlt)
        true_p = comb[ci, cj]
        peak = np.unravel_index(np.argmax(comb), comb.shape)
        pk_lat = LAT_AXIS[peak[0]]; pk_mlt = MLT_AXIS[peak[1]]
        d = haversine_deg(true_lat, true_mlt, pk_lat, pk_mlt)
        ae = float(row.get("ae_index", np.nan))
        rec.append({
            "true_lat": true_lat, "true_mlt": true_mlt, "hemi": hemi,
            "imf_bz": float(row["imf_bz"]), "sw_v": float(row["sw_v"]),
            "ae": ae,
            "peak_dist": d, "true_logp": float(np.log(true_p + 1e-12)),
            "s1_p": s1_p, "peak_lat": float(pk_lat), "peak_mlt": float(pk_mlt),
        })
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(test_df)}  ({time.time()-t1:.1f}s)")
    print(f"eval done in {time.time()-t1:.1f}s")
    R = pd.DataFrame(rec)

    # ---- stratified summary ----
    def stratify(df_eval, key, bins=None, labels=None):
        if bins is not None:
            df_eval = df_eval.assign(_b=pd.cut(df_eval[key], bins, labels=labels, include_lowest=True))
            grp = df_eval.groupby("_b", observed=True)
        else:
            grp = df_eval.groupby(key, observed=True)
        out = grp.agg(n=("peak_dist", "size"),
                      median_dist=("peak_dist", "median"),
                      p90_dist=("peak_dist", lambda x: np.percentile(x, 90)),
                      mean_logp=("true_logp", "mean"))
        return out

    print("\n[Per-MLT bin]")
    mlt_stat = stratify(R, "true_mlt", bins=[0, 4, 8, 12, 16, 20, 24],
                         labels=["0-4", "4-8", "8-12", "12-16", "16-20", "20-24"])
    print(mlt_stat.to_string())

    print("\n[Per-MLAT bin]")
    lat_stat = stratify(R, "true_lat", bins=[50, 65, 75, 83, 90],
                         labels=["50-65", "65-75", "75-83", "83-90"])
    print(lat_stat.to_string())

    print("\n[Hemisphere]")
    hemi_stat = stratify(R, "hemi")
    print(hemi_stat.to_string())

    print("\n[Bz sign]")
    R["bz_sign"] = np.where(R["imf_bz"] >= 0, "north", "south")
    bz_stat = stratify(R, "bz_sign")
    print(bz_stat.to_string())

    print("\n[AE level]")
    ae_stat = stratify(R, "ae", bins=[-1, 100, 300, 500, 9999],
                        labels=["<100", "100-300", "300-500", ">=500"])
    print(ae_stat.to_string())

    R["storm"] = ((R["imf_bz"].abs() >= 10) | (R["sw_v"] >= 600) | (R["ae"] >= 300)).fillna(False)
    print("\n[Storm flag]")
    print(stratify(R, "storm").to_string())

    # ---- coverage vs error scatter at cell level ----
    print("\nbuilding per-cell error map ...")
    err_per_cell = np.full(cov.shape, np.nan)
    cnt_per_cell = np.zeros(cov.shape, dtype=int)
    for _, r in R.iterrows():
        i_lat = int(np.clip((r["true_lat"] - 50) / 1.0, 0, len(LAT_AXIS) - 1))
        j_mlt = int(np.clip(r["true_mlt"] / 0.5, 0, len(MLT_AXIS) - 1))
        # store median per cell — first pass: just average
        if np.isnan(err_per_cell[i_lat, j_mlt]):
            err_per_cell[i_lat, j_mlt] = r["peak_dist"]; cnt_per_cell[i_lat, j_mlt] = 1
        else:
            err_per_cell[i_lat, j_mlt] = (err_per_cell[i_lat, j_mlt] * cnt_per_cell[i_lat, j_mlt] + r["peak_dist"]) / (cnt_per_cell[i_lat, j_mlt] + 1)
            cnt_per_cell[i_lat, j_mlt] += 1
    # filter cells with >=5 eval samples
    err_per_cell[cnt_per_cell < 5] = np.nan
    cov_flat = cov[~np.isnan(err_per_cell)]
    err_flat = err_per_cell[~np.isnan(err_per_cell)]
    if len(cov_flat) > 10:
        # Spearman correlation between coverage and error
        from scipy.stats import spearmanr
        rho, p = spearmanr(cov_flat, err_flat)
        print(f"\nSpearman corr (coverage, peak_dist): rho = {rho:.3f}, p = {p:.2e}, n_cells = {len(cov_flat)}")
    else:
        rho = None; p = None

    # ---- plots ----
    figs_dir = f"{OUT_DIR}/figures"
    os.makedirs(figs_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (df_stat, title, ylab) in zip(axes.flat, [
        (mlt_stat, "Per-MLT bin", "median peak dist (deg)"),
        (lat_stat, "Per-|MLAT| bin", "median peak dist (deg)"),
        (hemi_stat, "Hemisphere", "median peak dist (deg)"),
        (bz_stat, "IMF Bz sign", "median peak dist (deg)"),
        (ae_stat, "AE level (nT)", "median peak dist (deg)"),
        (stratify(R, "storm"), "Storm flag", "median peak dist (deg)"),
    ]):
        df_stat["median_dist"].plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(f"{title} (n={int(df_stat['n'].sum())})", fontsize=10)
        ax.set_ylabel(ylab, fontsize=9)
        ax.tick_params(axis="x", rotation=0, labelsize=9)
        ax.grid(alpha=0.3, axis="y")
        for i, (n, m) in enumerate(zip(df_stat["n"], df_stat["median_dist"])):
            ax.text(i, m + 0.2, f"n={int(n)}", ha="center", fontsize=7)
    fig.suptitle("R028: per-bin median peak distance (n=7933 held-out crossings)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{figs_dir}/thesis_per_bin_error.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {figs_dir}/thesis_per_bin_error.png")

    # coverage vs error scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    if len(cov_flat) > 10:
        ax.loglog(cov_flat, err_flat, "o", alpha=0.4, ms=3)
        ax.set_xlabel("training coverage (n crossings in cell)")
        ax.set_ylabel("median peak dist (deg)")
        ax.set_title(f"Coverage vs error (Spearman rho={rho:.3f}, p={p:.1e})")
        ax.grid(True, which="both", alpha=0.3)
    ax = axes[1]
    # err heatmap on dial
    valid_mask = ~np.isnan(err_per_cell)
    theta = 2 * np.pi * (MLT_AXIS) / 24.0
    r_arr = 90.0 - LAT_AXIS
    TT, RR = np.meshgrid(theta, r_arr)
    ax = plt.subplot(1, 2, 2, projection="polar")
    pcm = ax.pcolormesh(TT, RR, err_per_cell, cmap="hot_r", shading="auto", vmin=0, vmax=15)
    ax.set_theta_zero_location("S"); ax.set_theta_direction(1)
    ax.set_ylim(0, 40); ax.set_yticks([10, 20, 30, 40])
    ax.set_yticklabels(["80", "70", "60", "50"])
    ax.set_xticks(np.deg2rad([0, 90, 180, 270])); ax.set_xticklabels(["00", "06", "12", "18"])
    ax.set_title("Per-cell median peak dist (deg)")
    fig.colorbar(pcm, ax=ax, shrink=0.7, pad=0.10, label="dist (deg)")
    fig.tight_layout()
    fig.savefig(f"{figs_dir}/thesis_coverage_vs_error.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {figs_dir}/thesis_coverage_vs_error.png")

    summary = {
        "n_eval": len(R),
        "overall_median_dist_deg": float(R["peak_dist"].median()),
        "overall_p90_dist_deg": float(np.percentile(R["peak_dist"], 90)),
        "mean_true_logp": float(R["true_logp"].mean()),
        "per_MLT": mlt_stat.to_dict(orient="index"),
        "per_MLAT": lat_stat.to_dict(orient="index"),
        "per_hemi": hemi_stat.to_dict(orient="index"),
        "per_bz_sign": bz_stat.to_dict(orient="index"),
        "per_AE": ae_stat.to_dict(orient="index"),
        "per_storm": stratify(R, "storm").to_dict(orient="index"),
        "coverage_vs_err_spearman_rho": float(rho) if rho is not None else None,
        "coverage_vs_err_spearman_p": float(p) if p is not None else None,
        "n_cells_evaluated": int(len(cov_flat)),
    }
    with open(f"{OUT_DIR}/bundles/r028_per_bin.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    R.to_parquet(f"{OUT_DIR}/bundles/r028_eval_per_crossing.parquet")
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
