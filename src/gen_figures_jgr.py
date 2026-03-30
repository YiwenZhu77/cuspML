#!/usr/bin/env python3
"""Generate 7 publication-quality figures for JGR: ML & Computation."""

import json, glob, os, warnings, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore")

# ── JGR:MLC style (matplotlib defaults, no seaborn) ──────────────────────────
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
})
OUTDIR = '/glade/work/yizhu/cuspML/figures_jgr'
COLORS = plt.cm.tab10.colors
os.makedirs(OUTDIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
def load_data():
    files = sorted(glob.glob("/glade/work/yizhu/cuspML/output/omni_full_hist/cusp_crossings_*.json"))
    records = []
    for f in files:
        with open(f) as fp:
            records.extend(json.load(fp))
    df = pd.DataFrame(records)
    df = df.dropna(subset=["eq_mlat", "pole_mlat", "imf_bz", "sw_v", "sw_n", "sw_pdyn"])
    return df

def derive_features(df):
    df = df.copy()
    df["abs_eq_mlat"] = df["eq_mlat"].abs()
    df["abs_pole_mlat"] = df["pole_mlat"].abs()
    df["hemi_code"] = (df["hemisphere"] == "N").astype(float)
    df["doy"] = pd.to_datetime(df["time_start"]).dt.dayofyear
    df["year"] = pd.to_datetime(df["time_start"]).dt.year
    df["B_T"] = np.sqrt(df["imf_by"]**2 + df["imf_bz"]**2)
    df["clock_angle"] = np.arctan2(df["imf_by"], df["imf_bz"])
    df["sin_clock_half"] = np.sin(df["clock_angle"] / 2)
    df["newell_cf"] = (df["sw_v"]**(4/3)) * (df["B_T"]**(2/3)) * (np.abs(df["sin_clock_half"])**(8/3))
    df["kan_lee_ef"] = df["sw_v"] * df["B_T"] * (df["sin_clock_half"]**2)
    df["vBs"] = df["sw_v"] * np.where(df["imf_bz"] < 0, -df["imf_bz"], 0)
    df["by_hemi"] = df["imf_by"] * np.where(df["hemisphere"] == "N", 1, -1)
    return df

def get_features_targets(df):
    base = ['dipole_tilt','hemi_code','doy','imf_bx','imf_by','imf_bz',
            'sw_v','sw_n','sw_pdyn','B_T','clock_angle','sin_clock_half',
            'newell_cf','kan_lee_ef','vBs','by_hemi']
    hist = sorted([c for c in df.columns if any(s in c for s in
        ['mean15','mean30','mean60','std15','std30','std60','delta15','delta30','delta60','int60'])
        and c not in base])
    features = base + hist
    targets = ['abs_eq_mlat', 'abs_pole_mlat', 'eq_mlt', 'mean_mlt']
    all_cols = list(dict.fromkeys(features + targets + ['ae_index','year','satellite','hemisphere',
                                         'eq_mlt','mean_mlt','eq_mlat','pole_mlat']))
    df_clean = df[[c for c in all_cols if c in df.columns]].dropna()
    feats = [c for c in features if c in df_clean.columns]
    return df_clean, feats, targets

print("Loading data...")
df_raw = load_data()
df = derive_features(df_raw)
df_clean, feats, targets = get_features_targets(df)
X = df_clean[feats].values.astype(np.float32)
y = df_clean[targets].values.astype(np.float32)
print(f"Features: {len(feats)}, Samples: {len(X)}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
idx_tr, idx_te = train_test_split(np.arange(len(df_clean)), test_size=0.2, random_state=42)
df_te = df_clean.iloc[idx_te].copy()

print("Training XGBoost...")
m = MultiOutputRegressor(XGBRegressor(
    n_estimators=1000, max_depth=8, learning_rate=0.02, subsample=0.8,
    colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
    random_state=42, n_jobs=8, verbosity=0))
m.fit(X_tr, y_tr)
p_te = m.predict(X_te)
print("XGBoost trained.")

TGT_NAMES = ['eq MLAT', 'pole MLAT', 'eq MLT', 'mean MLT']
TGT_UNITS = ['\u00b0', '\u00b0', 'hr', 'hr']
for i, (tn, tu) in enumerate(zip(TGT_NAMES, TGT_UNITS)):
    mae = mean_absolute_error(y_te[:, i], p_te[:, i])
    r2 = r2_score(y_te[:, i], p_te[:, i])
    print(f"  {tn}: MAE={mae:.3f}{tu}, R2={r2:.3f}")

xgb_mae_eq = mean_absolute_error(y_te[:, 0], p_te[:, 0])

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Data Overview (7x3.5 in)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 1: Data Overview...")
fig = plt.figure(figsize=(7, 3.5))

# (a) Stacked bar: crossings per year by satellite
ax = fig.add_subplot(121)
sats = sorted(df_clean['satellite'].unique())
years = sorted(df_clean['year'].unique())
cmap20 = plt.cm.tab20
sat_colors = [cmap20(i / max(len(sats)-1, 1)) for i in range(len(sats))]

bottom = np.zeros(len(years))
for i_s, sat in enumerate(sats):
    counts = np.array([((df_clean['year'] == yr) & (df_clean['satellite'] == sat)).sum() for yr in years])
    ax.bar(years, counts, bottom=bottom, color=sat_colors[i_s], label=sat, width=0.8, edgecolor='none')
    bottom += counts

ax.set_xlabel("Year")
ax.set_ylabel("Number of Crossings")
ax.legend(ncol=3, fontsize=6, loc='upper left', frameon=True, framealpha=0.9, edgecolor='gray')
ax.text(0.02, 0.97, r"$\bf{(a)}$", transform=ax.transAxes, fontsize=11, va='top')

# (b) Polar plot: MLT angle (noon at top), |MLAT| radius (90 center, 60 edge)
ax2 = fig.add_subplot(122, polar=True)

mlt_all = df_clean['eq_mlt'].values
mlat_all = df_clean['abs_eq_mlat'].values

# MLT -> theta: noon (12) at top
theta_all = (mlt_all / 24.0) * 2 * np.pi
r_all = 90 - mlat_all

# 2D histogram for density
theta_bins = np.linspace(0, 2*np.pi, 49)
r_bins = np.linspace(0, 30, 31)
H, te, re = np.histogram2d(theta_all, r_all, bins=[theta_bins, r_bins])
H = H.T
# Mask zeros
H_masked = np.ma.masked_where(H == 0, H)
T, R = np.meshgrid(te, re)
im = ax2.pcolormesh(T, R, H_masked, cmap='jet', rasterized=True)

ax2.set_theta_zero_location('N')  # noon at top
ax2.set_theta_direction(-1)       # clockwise
ax2.set_ylim(0, 30)
ax2.set_yticks([5, 10, 15, 20, 25])
ax2.set_yticklabels(['85\u00b0', '80\u00b0', '75\u00b0', '70\u00b0', '65\u00b0'], fontsize=7)
ax2.set_xticks(np.array([0, 6, 12, 18]) / 24.0 * 2 * np.pi)
ax2.set_xticklabels(['12', '18', '00', '06'], fontsize=8)
ax2.grid(True, alpha=0.3)

cbar = plt.colorbar(im, ax=ax2, pad=0.1, shrink=0.8)
cbar.set_label('Count', fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax2.text(-0.05, 1.08, r"$\bf{(b)}$", transform=ax2.transAxes, fontsize=11, va='top')

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig01_data_overview.png')
fig.savefig(f'{OUTDIR}/fig01_data_overview.pdf')
plt.close(fig)
print("  -> fig01 saved.")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Observed vs Predicted (7x6, 2x2)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 2: Observed vs Predicted...")
fig, axes = plt.subplots(2, 2, figsize=(7, 6))
panel_labels = [r'$\bf{(a)}$ Equatorward MLAT', r'$\bf{(b)}$ Poleward MLAT',
                r'$\bf{(c)}$ Equatorward MLT', r'$\bf{(d)}$ Mean MLT']

hb_list = []
for i, ax in enumerate(axes.flat):
    obs = y_te[:, i]
    pred = p_te[:, i]
    hb = ax.hexbin(obs, pred, gridsize=40, cmap='jet', mincnt=1, rasterized=True)
    hb_list.append(hb)
    lims = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
    margin = (lims[1] - lims[0]) * 0.02
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    r = np.corrcoef(obs, pred)[0, 1]
    mae_i = mean_absolute_error(obs, pred)
    unit = TGT_UNITS[i]
    ax.text(0.05, 0.95, f"r = {r:.3f}\nMAE = {mae_i:.2f}{unit}",
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))
    ax.text(0.05, 0.05, panel_labels[i], transform=ax.transAxes, va='bottom', fontsize=9)
    ax.set_xlabel(f"Observed ({TGT_UNITS[i]})")
    ax.set_ylabel(f"Predicted ({TGT_UNITS[i]})")

fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.35)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(hb_list[-1], cax=cbar_ax, label='Count')

fig.savefig(f'{OUTDIR}/fig02_obs_vs_pred.png')
fig.savefig(f'{OUTDIR}/fig02_obs_vs_pred.pdf')
plt.close(fig)
print("  -> fig02 saved.")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Error Analysis (7x3, 2 panels)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 3: Error Analysis...")
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

residuals = y_te[:, 0] - p_te[:, 0]
abs_err = np.abs(residuals)

# (a) Residual histogram + gaussian overlay
ax = axes[0]
n_hist, bins, patches = ax.hist(residuals, bins=50, density=True, color=COLORS[0],
                                 edgecolor='white', linewidth=0.3, alpha=0.85)
mu, std = residuals.mean(), residuals.std()
x_fit = np.linspace(residuals.min(), residuals.max(), 200)
ax.plot(x_fit, stats.norm.pdf(x_fit, mu, std), 'r-', linewidth=1.5, label='Gaussian fit')
ax.text(0.97, 0.95, f"$\\mu$ = {mu:.3f}\u00b0\n$\\sigma$ = {std:.3f}\u00b0",
        transform=ax.transAxes, va='top', ha='right', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))
ax.set_xlabel("Residual (\u00b0)")
ax.set_ylabel("Density")
ax.legend(fontsize=8, frameon=True, framealpha=0.8, edgecolor='gray')
ax.text(0.02, 0.97, r"$\bf{(a)}$", transform=ax.transAxes, fontsize=11, va='top')

# (b) CDF of |error|
ax = axes[1]
sorted_err = np.sort(abs_err)
cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
ax.step(sorted_err, cdf * 100, where='post', color=COLORS[0], linewidth=1.5)
for thr in [1, 2, 3]:
    pct = (abs_err <= thr).mean() * 100
    ax.axvline(thr, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.annotate(f"{pct:.1f}%", xy=(thr, pct), xytext=(thr + 0.3, pct - 8),
                fontsize=7, color='#333333',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
ax.set_xlabel("|Error| (\u00b0)")
ax.set_ylabel("Cumulative %")
ax.set_xlim(0, min(sorted_err.max(), 10))
ax.set_ylim(0, 100)
ax.text(0.02, 0.97, r"$\bf{(b)}$", transform=ax.transAxes, fontsize=11, va='top')

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig03_error_analysis.png')
fig.savefig(f'{OUTDIR}/fig03_error_analysis.pdf')
plt.close(fig)
print("  -> fig03 saved.")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Feature Importance + SHAP (7x4, 2 panels)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 4: Feature Importance + SHAP...")

# XGBoost gain importance (average across 4 targets)
importances = np.zeros(len(feats))
for est in m.estimators_:
    imp = est.get_booster().get_score(importance_type='gain')
    for k, v in imp.items():
        idx = int(k.replace('f', ''))
        importances[idx] += v
importances /= len(m.estimators_)
total_imp = importances.sum()

top15_idx = np.argsort(importances)[::-1][:15]
top15_feats = [feats[i] for i in top15_idx]
top15_imp = importances[top15_idx]
top15_pct = top15_imp / total_imp * 100

fig, axes = plt.subplots(1, 2, figsize=(7, 4))

# (a) Top 15 horizontal bars, single color (tab10[0] blue)
ax = axes[0]
y_pos = np.arange(15)[::-1]
ax.barh(y_pos, top15_pct, color=COLORS[0], edgecolor='none', height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(top15_feats, fontsize=7)
ax.set_xlabel("Relative Importance (%)")
# Annotate percentage at bar end
for i_b in range(15):
    ax.text(top15_pct[i_b] + 0.3, y_pos[i_b], f"{top15_pct[i_b]:.1f}%",
            va='center', fontsize=7)
ax.set_xlim(0, top15_pct[0] * 1.2)
ax.text(0.02, 0.97, r"$\bf{(a)}$", transform=ax.transAxes, fontsize=11, va='top')

# (b) SHAP beeswarm
ax2 = axes[1]
print("  Computing SHAP values (subsample 2000)...")
explainer = shap.TreeExplainer(m.estimators_[0])
X_te_shap = X_te[:2000]
shap_values = explainer.shap_values(X_te_shap)

plt.sca(ax2)
shap.summary_plot(shap_values, X_te_shap, feature_names=feats, max_display=15,
                  show=False, plot_size=None)
ax2.text(0.02, 0.97, r"$\bf{(b)}$", transform=ax2.transAxes, fontsize=11, va='top')
ax2.set_xlabel("SHAP value")

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig04_importance_shap.png')
fig.savefig(f'{OUTDIR}/fig04_importance_shap.pdf')
plt.close('all')
print("  -> fig04 saved.")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Partial Dependence (7x4.5, 2x3)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 5: Partial Dependence...")
pd_vars = ['dipole_tilt', 'newell_cf_mean60', 'imf_bz', 'sw_pdyn', 'imf_by', 'vBs_mean60']
pd_labels = ["Dipole Tilt (\u00b0)", "Newell CF 60-min Mean", "IMF $B_z$ (nT)",
             "$P_{dyn}$ (nPa)", "IMF $B_y$ (nT)", "$v B_s$ 60-min Mean"]
panel_letters = [r'$\bf{(a)}$', r'$\bf{(b)}$', r'$\bf{(c)}$', r'$\bf{(d)}$', r'$\bf{(e)}$', r'$\bf{(f)}$']

fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))

X_median = np.median(X_tr, axis=0)
eq_model = m.estimators_[0]  # eq_MLAT sub-model

for i, (var, label, letter) in enumerate(zip(pd_vars, pd_labels, panel_letters)):
    ax = axes.flat[i]
    fi = feats.index(var)
    vals = X_tr[:, fi]
    grid = np.linspace(np.percentile(vals, 2), np.percentile(vals, 98), 100)
    X_synth = np.tile(X_median, (len(grid), 1))
    X_synth[:, fi] = grid
    preds = eq_model.predict(X_synth)

    ax.plot(grid, preds, color=COLORS[0], linewidth=1.5)
    # Thin gray rug at bottom
    rug_sample = vals[np.random.RandomState(42).choice(len(vals), size=min(500, len(vals)), replace=False)]
    ax.plot(rug_sample, np.full_like(rug_sample, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else preds.min()),
            '|', color='gray', alpha=0.3, markersize=3, markeredgewidth=0.5)
    # Re-draw rug after proper ylim
    ax.set_xlabel(label, fontsize=9)
    if i % 3 == 0:
        ax.set_ylabel("Predicted |eq MLAT| (\u00b0)", fontsize=9)
    ax.text(0.03, 0.95, letter, transform=ax.transAxes, fontsize=10, va='top')

# Fix rug positions after all limits are set
for i, var in enumerate(pd_vars):
    ax = axes.flat[i]
    fi = feats.index(var)
    vals = X_tr[:, fi]
    rug_sample = vals[np.random.RandomState(42).choice(len(vals), size=min(500, len(vals)), replace=False)]
    ylims = ax.get_ylim()
    rug_y = ylims[0] + (ylims[1] - ylims[0]) * 0.02
    # Clear old rug, just add new
    ax.plot(rug_sample, np.full_like(rug_sample, rug_y),
            '|', color='gray', alpha=0.3, markersize=3, markeredgewidth=0.5)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig05_partial_dependence.png')
fig.savefig(f'{OUTDIR}/fig05_partial_dependence.pdf')
plt.close(fig)
print("  -> fig05 saved.")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Model Comparison (3.5x4)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 6: Model Comparison...")

# Compute baseline MAEs
lr_all = LinearRegression().fit(X_tr, y_tr[:, 0])
lr_all_mae = mean_absolute_error(y_te[:, 0], lr_all.predict(X_te))

def single_feat_mae(feat_name):
    fi = feats.index(feat_name)
    lr = LinearRegression().fit(X_tr[:, fi:fi+1], y_tr[:, 0])
    return mean_absolute_error(y_te[:, 0], lr.predict(X_te[:, fi:fi+1]))

# Newell CF nonlinear: E_WAV^(2/3)
fi_ncf = feats.index('newell_cf')
X_ncf = X_tr[:, fi_ncf:fi_ncf+1]**(2/3)
X_ncf_te = X_te[:, fi_ncf:fi_ncf+1]**(2/3)
lr_ewav = LinearRegression().fit(X_ncf, y_tr[:, 0])
ewav_mae = mean_absolute_error(y_te[:, 0], lr_ewav.predict(X_ncf_te))

ncf_mae = single_feat_mae('newell_cf')
kle_mae = single_feat_mae('kan_lee_ef')
vbs_mae = single_feat_mae('vBs')
bz_mae = single_feat_mae('imf_bz')

models_comp = [
    ('XGBoost', xgb_mae_eq),
    ('ResMLP', 1.018),
    ('TabTF', 1.102),
    (f'Linear ({len(feats)} feat)', lr_all_mae),
    ("NL $E_{{WAV}}^{{2/3}}$", ewav_mae),
    ('Lin Newell CF', ncf_mae),
    ('Lin Kan-Lee', kle_mae),
    ('Lin $vB_s$', vbs_mae),
    ('Lin $B_z$', bz_mae),
]
models_comp.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(3.5, 4))
n_mod = len(models_comp)
y_pos = np.arange(n_mod)[::-1]
maes = [v for _, v in models_comp]
names = [n for n, _ in models_comp]

# Color gradient: dark blue (best) to light gray (worst)
from matplotlib.colors import LinearSegmentedColormap
cmap_bar = LinearSegmentedColormap.from_list('blue_gray', ['#08519c', '#c6dbef'])
colors_m = [cmap_bar(i / (n_mod - 1)) for i in range(n_mod)]

ax.barh(y_pos, maes, color=colors_m, edgecolor='none', height=0.65)
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("MAE (\u00b0)")

for i_m, (name, mae_v) in enumerate(models_comp):
    ax.text(mae_v + 0.02, y_pos[i_m], f"{mae_v:.3f}", va='center', fontsize=7)

# Vertical dashed red line at XGBoost MAE
ax.axvline(xgb_mae_eq, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
ax.set_xlim(0, max(maes) * 1.15)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig06_model_comparison.png')
fig.savefig(f'{OUTDIR}/fig06_model_comparison.pdf')
plt.close(fig)
print("  -> fig06 saved.")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Activity + Temporal (7x3.5, 2 panels)
# ═════════════════════════════════════════════════════════════════════════════
print("Figure 7: Activity + Temporal...")
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

# (a) MAE by AE bin
ax = axes[0]
ae_te = df_te['ae_index'].values
residual_te = np.abs(y_te[:, 0] - p_te[:, 0])

ae_bins = [(0, 100, '<100'), (100, 300, '100-300'), (300, 500, '300-500'), (500, 1e6, '>500')]
ae_maes = []
ae_counts = []
ae_labels = []
blue_grad = ['#c6dbef', '#6baed6', '#2171b5', '#08306b']

for lo, hi, label in ae_bins:
    mask = (ae_te >= lo) & (ae_te < hi)
    if mask.sum() > 0:
        ae_maes.append(residual_te[mask].mean())
        ae_counts.append(mask.sum())
    else:
        ae_maes.append(0)
        ae_counts.append(0)
    ae_labels.append(label)

bars = ax.bar(range(len(ae_labels)), ae_maes, color=blue_grad, edgecolor='none', width=0.65)
ax.set_xticks(range(len(ae_labels)))
ax.set_xticklabels(ae_labels)
ax.set_xlabel("AE Index Bin (nT)")
ax.set_ylabel("MAE (\u00b0)")
for i_b, (bar, cnt) in enumerate(zip(bars, ae_counts)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"n={cnt}", ha='center', va='bottom', fontsize=7)
ax.text(0.02, 0.97, r"$\bf{(a)}$", transform=ax.transAxes, fontsize=11, va='top')

# (b) LOYO: scatter+line by year, color by solar cycle
ax = axes[1]
years_all = sorted(df_clean['year'].unique())
year_arr = df_clean['year'].values

loyo_maes = []
loyo_years = []
print("  Computing LOYO...")
for yr in years_all:
    te_mask = year_arr == yr
    tr_mask = ~te_mask
    if te_mask.sum() < 20:
        continue
    X_lo_tr = X[tr_mask]
    y_lo_tr = y[tr_mask, 0]
    X_lo_te = X[te_mask]
    y_lo_te = y[te_mask, 0]
    m_lo = XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                         reg_lambda=1.0, min_child_weight=5, random_state=42,
                         n_jobs=8, verbosity=0)
    m_lo.fit(X_lo_tr, y_lo_tr)
    p_lo = m_lo.predict(X_lo_te)
    loyo_maes.append(mean_absolute_error(y_lo_te, p_lo))
    loyo_years.append(yr)
    print(f"    LOYO {yr}: MAE={loyo_maes[-1]:.3f}, n={te_mask.sum()}")

loyo_years = np.array(loyo_years)
loyo_maes = np.array(loyo_maes)
overall_loyo = loyo_maes.mean()

def sc_color(yr):
    if yr <= 1996:
        return '#e6550d'  # SC22 orange
    elif yr <= 2008:
        return '#3182bd'  # SC23 blue
    else:
        return '#31a354'  # SC24 green

sc_colors = [sc_color(yr) for yr in loyo_years]

# Light background shading for SC regions
for yr_start, yr_end, color in [(1986, 1996.5, '#e6550d'),
                                 (1996.5, 2008.5, '#3182bd'),
                                 (2008.5, 2016, '#31a354')]:
    ax.axvspan(yr_start, yr_end, alpha=0.08, color=color, edgecolor='none')

ax.plot(loyo_years, loyo_maes, '-', color='#666666', linewidth=0.8, zorder=1)
ax.scatter(loyo_years, loyo_maes, c=sc_colors, s=30, zorder=2, edgecolors='white', linewidths=0.3)
ax.axhline(overall_loyo, color='#333333', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(loyo_years[-1] + 0.5, overall_loyo, f"mean={overall_loyo:.3f}\u00b0", fontsize=7, va='center')

ax.set_xlabel("Year")
ax.set_ylabel("LOYO MAE (\u00b0)")
ax.text(0.02, 0.97, r"$\bf{(b)}$", transform=ax.transAxes, fontsize=11, va='top')

from matplotlib.lines import Line2D
sc_legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e6550d', markersize=6, label='SC 22'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3182bd', markersize=6, label='SC 23'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#31a354', markersize=6, label='SC 24'),
]
ax.legend(handles=sc_legend, fontsize=8, loc='upper right', frameon=True, framealpha=0.8, edgecolor='gray')

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig07_activity_temporal.png')
fig.savefig(f'{OUTDIR}/fig07_activity_temporal.pdf')
plt.close(fig)
print("  -> fig07 saved.")

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n=== All 7 JGR:MLC figures generated ===")
for i in range(1, 8):
    pngs = glob.glob(f'{OUTDIR}/fig0{i}_*.png')
    if pngs:
        fpath = pngs[0]
        size = os.path.getsize(fpath)
        print(f"  {os.path.basename(fpath)}: {size/1024:.0f} KB")
print("Done!")
