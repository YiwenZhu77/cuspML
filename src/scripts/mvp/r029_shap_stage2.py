"""R029 (thesis chapter, deliverable 2) — SHAP analysis on stage 2 spatial classifier.

Computes mean(|SHAP|) for all 76 stage-2 features over a 5000-row subset of
the expanded training rows. Identifies top predictors. Generates:
  - Top-20 feature importance bar chart
  - PDP for top 5 features
  - SHAP summary plot (top 15)
  - Hemisphere-stratified SHAP for top 10 features
"""
import json, os, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
from xgboost import XGBClassifier

sys.path.insert(0, "/glade/work/yizhu/cuspML/src/lib")
sys.path.insert(0, "/glade/work/yizhu/cuspML/src/scripts/mvp")
from cusp_map import load_crossings, sw_feature_names, build_feature_matrix, expand_dataset

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp"


def main():
    t0 = time.time()
    print("loading stage 2 model + expanded data ...")
    model = XGBClassifier()
    model.load_model(f"{OUT_DIR}/bundles/r002_model.ubj")
    with open(f"{OUT_DIR}/bundles/r002_features.json") as f:
        feat_names = json.load(f)
    print(f"features: {len(feat_names)}")

    parquet = f"{OUT_DIR}/bundles/expanded_full.parquet"
    if os.path.exists(parquet):
        expanded = pd.read_parquet(parquet)
        print(f"loaded expanded {len(expanded)} rows from cache")
    else:
        print("regenerating expanded ...")
        df = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
        sw = sw_feature_names(df)
        keep = sw + ["eq_mlat", "pole_mlat", "eq_mlt", "pole_mlt",
                     "satellite", "hemisphere", "time_start"]
        df_clean = df[keep].dropna().reset_index(drop=True)
        expanded = expand_dataset(df_clean, n_pos=5, k_neg=10, seed=42, verbose=False)

    sw_cols = sw_feature_names(expanded)
    rng = np.random.default_rng(42)
    # use 5000 rows for SHAP (~10 min wall on XGBoost depth 8)
    idx = rng.choice(len(expanded), 5000, replace=False)
    sub = expanded.iloc[idx].reset_index(drop=True)
    X, _ = build_feature_matrix(sub, sw_cols)
    print(f"SHAP background: {X.shape}")

    print("computing SHAP values (TreeExplainer) ...")
    t1 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    print(f"SHAP done in {time.time()-t1:.1f}s")

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    rank = np.argsort(-mean_abs_shap)
    print("\nTop 20 features by mean(|SHAP|):")
    for r in rank[:20]:
        print(f"  {r:3d}  {feat_names[r]:<25s}  {mean_abs_shap[r]:.4f}")

    figs_dir = f"{OUT_DIR}/figures"

    # bar chart top 20
    fig, ax = plt.subplots(figsize=(9, 7))
    top20_idx = rank[:20]
    ax.barh(range(20), mean_abs_shap[top20_idx][::-1], color="steelblue", edgecolor="black")
    ax.set_yticks(range(20))
    ax.set_yticklabels([feat_names[i] for i in top20_idx[::-1]], fontsize=9)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("R029: stage 2 feature importance (SHAP, n=5000)", fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(f"{figs_dir}/thesis_shap_top20.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {figs_dir}/thesis_shap_top20.png")

    # SHAP summary (default beeswarm-like)
    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_vals, X, feature_names=feat_names, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(f"{figs_dir}/thesis_shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {figs_dir}/thesis_shap_summary.png")

    # PDP for top 5
    from sklearn.inspection import PartialDependenceDisplay
    top5_idx = rank[:5].tolist()
    top5_names = [feat_names[i] for i in top5_idx]
    print(f"\ncomputing PDP for top 5: {top5_names}")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, fi, fn in zip(axes, top5_idx, top5_names):
        # quick manual PDP: sweep feature over 30 percentiles, fix others to row median
        med = np.median(X, axis=0).astype(np.float32)
        x_vals = np.percentile(X[:, fi], np.linspace(2, 98, 30))
        ys = []
        for v in x_vals:
            X_eval = np.tile(med, (1, 1))
            X_eval = np.repeat(X_eval, 30, axis=0)[:30]
            for k in range(30):
                X_eval[k, fi] = v
            ys.append(model.predict_proba(X_eval[:1])[:, 1].mean())
        ax.plot(x_vals, ys, "-", lw=2, color="darkblue")
        ax.set_xlabel(fn, fontsize=9)
        ax.set_ylabel("P(cusp)")
        ax.grid(alpha=0.3)
        ax.set_title(f"PDP: {fn}", fontsize=9)
    fig.suptitle("R029: partial dependence for top 5 features", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{figs_dir}/thesis_pdp_top5.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {figs_dir}/thesis_pdp_top5.png")

    # hemisphere-stratified SHAP for top 10
    hemi_n = sub["hemi_code"].values == 1.0
    print(f"\nhemisphere split: N {hemi_n.sum()}, S {(~hemi_n).sum()}")
    top10 = rank[:10]
    mean_shap_N = np.abs(shap_vals[hemi_n]).mean(axis=0)
    mean_shap_S = np.abs(shap_vals[~hemi_n]).mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(10)
    w = 0.4
    ax.bar(xs - w/2, mean_shap_N[top10], w, label=f"N (n={hemi_n.sum()})", color="steelblue")
    ax.bar(xs + w/2, mean_shap_S[top10], w, label=f"S (n={(~hemi_n).sum()})", color="firebrick")
    ax.set_xticks(xs)
    ax.set_xticklabels([feat_names[i] for i in top10], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("mean(|SHAP|)")
    ax.set_title("R029: hemisphere-stratified SHAP, top 10 features")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{figs_dir}/thesis_shap_hemi.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {figs_dir}/thesis_shap_hemi.png")

    summary = {
        "n_background": 5000,
        "top20": [(feat_names[i], float(mean_abs_shap[i])) for i in rank[:20]],
        "hemisphere_split": {"N": int(hemi_n.sum()), "S": int((~hemi_n).sum())},
        "elapsed_sec": float(time.time() - t0),
    }
    with open(f"{OUT_DIR}/bundles/r029_shap.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nelapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
