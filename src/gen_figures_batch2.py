#!/usr/bin/env python3
"""Generate Figures 6-10 for JGR Space Physics paper on ML cusp prediction."""

import json
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import partial_dependence
from xgboost import XGBRegressor
from scipy import stats

warnings.filterwarnings("ignore")

OUT_DIR = "/glade/work/yizhu/cuspML/figures"

# ── Load & derive features ──────────────────────────────────────────────
def load_data():
    files = sorted(glob.glob("/glade/work/yizhu/cuspML/output/omni_full_hist/cusp_crossings_*.json"))
    records = []
    for f in files:
        with open(f) as fp:
            records.extend(json.load(fp))
    df = pd.DataFrame(records)
    # drop rows with NaN in key fields
    df = df.dropna(subset=["eq_mlat", "pole_mlat", "imf_bz", "sw_v", "sw_n", "sw_pdyn"])
    return df


def derive_features(df):
    df = df.copy()
    # abs targets
    df["abs_eq_mlat"] = df["eq_mlat"].abs()
    df["abs_pole_mlat"] = df["pole_mlat"].abs()
    # hemisphere code
    df["hemi_code"] = (df["hemisphere"] == "N").astype(int)
    # day of year
    df["doy"] = pd.to_datetime(df["date"]).dt.dayofyear
    # B_T
    df["B_T"] = np.sqrt(df["imf_by"]**2 + df["imf_bz"]**2)
    # clock angle
    df["clock_angle"] = np.degrees(np.arctan2(df["imf_by"], df["imf_bz"]))
    # sin(clock/2)
    ca_rad = np.radians(df["clock_angle"])
    df["sin_clock_half"] = np.sin(ca_rad / 2.0)
    # Newell coupling function (instantaneous)
    v = df["sw_v"].values
    bt = df["B_T"].values
    ca = np.radians(df["clock_angle"].values)
    df["newell_cf"] = (v ** (4.0 / 3.0)) * (bt ** (2.0 / 3.0)) * (np.abs(np.sin(ca / 2.0)) ** (8.0 / 3.0))
    # Kan-Lee EF
    df["kan_lee_ef"] = v * df["B_T"] * np.sin(ca / 2.0) ** 2 * 1e-3
    # vBs
    df["vBs"] = np.where(df["imf_bz"] < 0, -df["sw_v"] * df["imf_bz"] * 1e-3, 0.0)
    # By * hemi
    df["by_hemi"] = df["imf_by"] * np.where(df["hemi_code"] == 1, 1.0, -1.0)
    return df


# ── Feature lists ────────────────────────────────────────────────────────
BASE_FEATS = [
    "dipole_tilt", "hemi_code", "doy",
    "imf_bx", "imf_by", "imf_bz",
    "sw_v", "sw_n", "sw_pdyn",
    "B_T", "clock_angle", "sin_clock_half",
    "newell_cf", "kan_lee_ef", "vBs", "by_hemi",
]

# history suffix patterns
HIST_VARS = ["imf_bx", "imf_by", "imf_bz", "sw_v", "sw_n", "sw_pdyn"]
HIST_SUFFIXES_15 = [f"{v}_{s}15" for v in HIST_VARS for s in ("mean", "std", "delta")]
HIST_SUFFIXES_30 = [f"{v}_{s}30" for v in HIST_VARS for s in ("mean", "std", "delta")]
HIST_SUFFIXES_60 = [f"{v}_{s}60" for v in HIST_VARS for s in ("mean", "std", "delta")]
EXTRA_60 = ["newell_cf_int60", "newell_cf_mean60", "vBs_int60", "vBs_mean60"]

ALL_HIST = HIST_SUFFIXES_15 + HIST_SUFFIXES_30 + HIST_SUFFIXES_60 + EXTRA_60

TARGETS = ["abs_eq_mlat", "abs_pole_mlat"]

XGB_PARAMS = dict(
    n_estimators=1000, max_depth=8, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
    random_state=42, n_jobs=-1,
)


def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost via MultiOutputRegressor and return model + predictions."""
    base = XGBRegressor(**XGB_PARAMS)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred


# ── MAIN ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = load_data()
df = derive_features(df)
print(f"Total samples: {len(df)}")

# Drop rows with any NaN in full feature set
all_feats = BASE_FEATS + ALL_HIST
df_clean = df.dropna(subset=all_feats + TARGETS).reset_index(drop=True)
print(f"Clean samples (no NaN): {len(df_clean)}")

X_all = df_clean[all_feats]
y_all = df_clean[TARGETS]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
idx_train, idx_test = X_train.index, X_test.index

# Full model (used for figs 7, 8, 9, 10)
print("Training full model...")
full_model, full_pred = train_model(X_train, y_train, X_test, y_test)
full_mae_eq = mean_absolute_error(y_test["abs_eq_mlat"], full_pred[:, 0])
print(f"Full model eq_MLAT MAE: {full_mae_eq:.3f}")

# GBR-300 baseline (same 74 features, same train/test split)
print("Training GBR-300 baseline...")
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
gbr_model = GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42)
gbr_model.fit(X_train, y_train["abs_eq_mlat"])
gbr_pred_eq = gbr_model.predict(X_test)
gbr_mae_eq = mean_absolute_error(y_test["abs_eq_mlat"], gbr_pred_eq)
print(f"GBR-300 eq_MLAT MAE: {gbr_mae_eq:.3f}")

# Ridge baseline (same 74 features, standardized)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_sc, y_train["abs_eq_mlat"])
ridge_pred_eq = ridge_model.predict(X_test_sc)
ridge_mae_eq = mean_absolute_error(y_test["abs_eq_mlat"], ridge_pred_eq)
print(f"Ridge eq_MLAT MAE: {ridge_mae_eq:.3f}")

# =====================================================================
# Fig 6: Time Window Comparison
# =====================================================================
print("\n=== Fig 6: Time Window Comparison ===")
feat_sets = {
    "Instantaneous\nonly": BASE_FEATS,
    "+ 15-min\nhistory": BASE_FEATS + HIST_SUFFIXES_15,
    "+ 30-min\nhistory": BASE_FEATS + HIST_SUFFIXES_30,
    "+ 60-min\nhistory": BASE_FEATS + HIST_SUFFIXES_60 + EXTRA_60,
    "All windows\n(15+30+60)": BASE_FEATS + ALL_HIST,
}

mae_results = {}
for label, feats in feat_sets.items():
    Xtr = df_clean.loc[idx_train, feats]
    Xte = df_clean.loc[idx_test, feats]
    _, yp = train_model(Xtr, y_train, Xte, y_test)
    mae_eq = mean_absolute_error(y_test["abs_eq_mlat"], yp[:, 0])
    mae_results[label] = mae_eq
    print(f"  {label.replace(chr(10), ' ')}: MAE = {mae_eq:.3f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
plt.style.use("default")
labels = list(mae_results.keys())
values = list(mae_results.values())
colors = ["#8c8c8c", "#6baed6", "#3182bd", "#08519c", "#b30000"]
bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6, width=0.6)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{v:.3f}°", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Mean Absolute Error (°MLAT)", fontsize=12)
ax.set_title("Effect of Historical Time Windows on eq-MLAT Prediction", fontsize=13)
ax.set_ylim(0, max(values) * 1.18)
ax.tick_params(axis="both", labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig06_time_window_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved fig06_time_window_comparison.png")

# =====================================================================
# Fig 7: Partial Dependence Plots
# =====================================================================
print("\n=== Fig 7: Partial Dependence ===")
pd_vars = ["dipole_tilt", "newell_cf_mean60", "imf_bz", "sw_pdyn", "imf_by", "vBs_mean60"]
pd_labels = ["Dipole Tilt (°)",
             "Newell CF $d\\Phi_{MP}/dt$\n(60-min mean)",
             "IMF $B_z$ (nT)",
             "$P_{dyn}$ (nPa)",
             "IMF $B_y$ (nT)",
             "$vB_s$ (60-min mean;\n$v \\cdot \\max(-B_z,0) \\times 10^{-3}$)"]

# Get the eq_MLAT estimator (index 0)
eq_estimator = full_model.estimators_[0]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for idx, (var, label) in enumerate(zip(pd_vars, pd_labels)):
    ax = axes.flat[idx]
    feat_idx = all_feats.index(var)
    result = partial_dependence(eq_estimator, X_train.values, features=[feat_idx],
                                kind="average", grid_resolution=80)
    ax.plot(result["grid_values"][0], result["average"][0], color="#08519c", lw=2)
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Partial Dependence\n(°MLAT)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, ls="--")
    ax.text(0.02, 0.95, f"({chr(97+idx)})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

fig.suptitle("Partial Dependence of Equatorward Boundary MLAT", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig07_partial_dependence.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved fig07_partial_dependence.png")

# =====================================================================
# Fig 8: Per-tilt-bin MAE
# =====================================================================
print("\n=== Fig 8: Tilt Bin MAE ===")
tilt_bins = [(-35, -20), (-20, -10), (-10, 0), (0, 10), (10, 20), (20, 35)]
tilt_test = df_clean.loc[idx_test, "dipole_tilt"].values
y_true_eq = y_test["abs_eq_mlat"].values
y_pred_eq = full_pred[:, 0]

bin_labels, bin_maes, bin_counts = [], [], []
for lo, hi in tilt_bins:
    mask = (tilt_test >= lo) & (tilt_test < hi)
    n = mask.sum()
    if n > 0:
        mae = mean_absolute_error(y_true_eq[mask], y_pred_eq[mask])
    else:
        mae = 0
    lab = f"[{lo}, {hi})"
    bin_labels.append(lab)
    bin_maes.append(mae)
    bin_counts.append(n)
    print(f"  {lab}: MAE={mae:.3f}, n={n}")

fig, ax = plt.subplots(figsize=(8, 4.5))
colors_tilt = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(tilt_bins)))
bars = ax.bar(bin_labels, bin_maes, color=colors_tilt, edgecolor="black", linewidth=0.6, width=0.6)
for bar, mae_v, cnt in zip(bars, bin_maes, bin_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{mae_v:.2f}°\n(n={cnt})", ha="center", va="bottom", fontsize=10)
ax.set_xlabel("Dipole Tilt Bin (°)", fontsize=12)
ax.set_ylabel("Mean Absolute Error (°MLAT)", fontsize=12)
ax.set_title("Equatorward Boundary Prediction Error by Dipole Tilt", fontsize=13)
ax.set_ylim(0, max(bin_maes) * 1.35)
ax.tick_params(labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig08_tilt_bin_mae.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved fig08_tilt_bin_mae.png")

# =====================================================================
# Fig 9: Model Architecture Comparison
# =====================================================================
print("\n=== Fig 9: Model Comparison ===")
model_names = [
    "Linear\n(Newell CF)",
    "Ridge\n74 features",
    "GBR\n300-d5",
    "XGBoost\n1000-d8",
    "MLP\n(GeLU)",
    "ResMLP\n4blk-128",
    "TabTF\nd128-L2",
]
# NN values from dse_log on omni_hist (held-out test set)
model_maes = [1.80, ridge_mae_eq, gbr_mae_eq, full_mae_eq, 1.5304, 1.0178, 1.1137]

fig, ax = plt.subplots(figsize=(9, 4.5))
colors_model = ["#bdbdbd", "#d9b38c", "#74c476", "#b30000", "#9ecae1", "#6baed6", "#3182bd"]
bars = ax.bar(model_names, model_maes, color=colors_model, edgecolor="black", linewidth=0.6, width=0.55)
for bar, v in zip(bars, model_maes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{v:.2f}°", ha="center", va="bottom", fontsize=11, fontweight="bold")
# highlight the XGBoost bar
bars[3].set_edgecolor("#b30000")
bars[3].set_linewidth(2.0)
ax.set_ylabel("Mean Absolute Error (°MLAT)", fontsize=12)
ax.set_title("Equatorward Boundary Prediction: Model Comparison", fontsize=13)
ax.set_ylim(0, max(model_maes) * 1.18)
ax.tick_params(axis="both", labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig09_model_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved fig09_model_comparison.png")

# =====================================================================
# Fig 10: Hemispheric Asymmetry
# =====================================================================
print("\n=== Fig 10: Hemispheric Asymmetry ===")
hemi_test = df_clean.loc[idx_test, "hemi_code"].values  # 1=N, 0=S

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, sharex=True)
for i, (hcode, hname) in enumerate([(1, "Northern Hemisphere"), (0, "Southern Hemisphere")]):
    ax = axes[i]
    mask = hemi_test == hcode
    yt = y_true_eq[mask]
    yp = y_pred_eq[mask]
    mae = mean_absolute_error(yt, yp)
    r, _ = stats.pearsonr(yt, yp)

    ax.scatter(yt, yp, s=8, alpha=0.35, color="#08519c", rasterized=True)
    lims = [min(yt.min(), yp.min()) - 1, max(yt.max(), yp.max()) + 1]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed |eq MLAT| (°)", fontsize=12)
    if i == 0:
        ax.set_ylabel("Predicted |eq MLAT| (°)", fontsize=12)
    ax.set_title(hname, fontsize=13)
    ax.text(0.05, 0.92, f"MAE = {mae:.2f}°\n$r$ = {r:.3f}\n$n$ = {mask.sum()}",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    ax.set_aspect("equal")
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.2, ls="--")

fig.suptitle("Hemispheric Performance: Equatorward Cusp Boundary", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig10_hemisphere.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved fig10_hemisphere.png")

print("\n✓ All figures saved to", OUT_DIR)
