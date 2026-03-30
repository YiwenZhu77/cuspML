#!/usr/bin/env python3
"""Generate 7 publication-quality figures for JGR Space Physics."""

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

plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'mathtext.fontset': 'stix',
})
OUTDIR = '/glade/work/yizhu/cuspML/figures_final'
os.makedirs(OUTDIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════
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
TGT_UNITS = ['°', '°', 'hr', 'hr']
for i, (tn, tu) in enumerate(zip(TGT_NAMES, TGT_UNITS)):
    mae = mean_absolute_error(y_te[:, i], p_te[:, i])
    r2 = r2_score(y_te[:, i], p_te[:, i])
    print(f"  {tn}: MAE={mae:.3f}{tu}, R²={r2:.3f}")

xgb_mae_eq = mean_absolute_error(y_te[:, 0], p_te[:, 0])

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Data Overview
# ═══════════════════════════════════════════════════════════════════════
print("Figure 1: Data Overview...")
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

# (a) Stacked bar by year & satellite
ax = axes[0]
sats = sorted(df_clean['satellite'].unique())
years = sorted(df_clean['year'].unique())
cmap = plt.cm.tab20
colors = [cmap(i / 13) for i in range(len(sats))]

bottom = np.zeros(len(years))
for i_s, sat in enumerate(sats):
    counts = [((df_clean['year'] == yr) & (df_clean['satellite'] == sat)).sum() for yr in years]
    ax.bar(years, counts, bottom=bottom, color=colors[i_s], label=sat, width=0.8, edgecolor='none')
    bottom += counts

ax.set_xlabel("Year")
ax.set_ylabel("Number of Crossings")
ax.legend(ncol=3, fontsize=6, loc='upper left', frameon=True, framealpha=0.9, edgecolor='none')
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, fontweight='bold', va='top', fontsize=9)

# (b) Polar plot
ax2 = fig.add_subplot(122, polar=True)
axes[1].set_visible(False)

mlt_n = df_clean.loc[df_clean['hemisphere'] == 'N', 'eq_mlt'].values
mlat_n = df_clean.loc[df_clean['hemisphere'] == 'N', 'abs_eq_mlat'].values
mlt_s = df_clean.loc[df_clean['hemisphere'] == 'S', 'eq_mlt'].values
mlat_s = df_clean.loc[df_clean['hemisphere'] == 'S', 'abs_eq_mlat'].values

# MLT to angle: noon (12) at top => angle = (MLT/24)*2pi - pi/2 ... no
# Standard: noon at top => theta = pi/2 - (MLT/24)*2*pi ... let's use:
# theta = (pi/2) - (MLT * 2*pi / 24)  so noon=top, midnight=bottom
theta_n = (np.pi / 2) - (mlt_n * 2 * np.pi / 24)
theta_s = (np.pi / 2) - (mlt_s * 2 * np.pi / 24)
# radius: 90 at center, 60 at edge => r = 90 - mlat
r_n = 90 - mlat_n
r_s = 90 - mlat_s

ax2.scatter(theta_n, r_n, s=1, alpha=0.05, c='#1f77b4', label='North', rasterized=True)
ax2.scatter(theta_s, r_s, s=1, alpha=0.05, c='#ff7f0e', label='South', rasterized=True)
ax2.set_ylim(0, 30)
ax2.set_yticks([5, 10, 15, 20, 25])
ax2.set_yticklabels(['85°', '80°', '75°', '70°', '65°'], fontsize=7)
# MLT labels
ax2.set_xticks([(np.pi/2) - (h * 2 * np.pi / 24) for h in [0, 6, 12, 18]])
ax2.set_xticklabels(['00', '18', '12', '06'], fontsize=8)
ax2.legend(loc='lower left', fontsize=7, markerscale=8, frameon=True, framealpha=0.9, edgecolor='none',
           bbox_to_anchor=(-0.1, -0.15))
ax2.text(-0.05, 1.08, "(b)", transform=ax2.transAxes, fontweight='bold', va='top', fontsize=9)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig01_data_overview.png')
plt.close(fig)
print("  -> fig01_data_overview.png saved.")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Observed vs Predicted
# ═══════════════════════════════════════════════════════════════════════
print("Figure 2: Observed vs Predicted...")
fig, axes = plt.subplots(2, 2, figsize=(7, 6))
panel_labels = ['(a) eq MLAT', '(b) pole MLAT', '(c) eq MLT', '(d) mean MLT']

hb_list = []
for i, ax in enumerate(axes.flat):
    obs = y_te[:, i]
    pred = p_te[:, i]
    hb = ax.hexbin(obs, pred, gridsize=50, cmap='cividis', mincnt=1, rasterized=True)
    hb_list.append(hb)
    lims = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    r = np.corrcoef(obs, pred)[0, 1]
    mae = mean_absolute_error(obs, pred)
    unit = TGT_UNITS[i]
    ax.text(0.05, 0.95, f"r = {r:.3f}\nMAE = {mae:.3f}{unit}",
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    ax.text(0.05, 0.05, panel_labels[i], transform=ax.transAxes, va='bottom', fontsize=9, fontweight='bold')
    ax.set_xlabel(f"Observed ({TGT_UNITS[i]})")
    ax.set_ylabel(f"Predicted ({TGT_UNITS[i]})")

fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.35)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(hb_list[-1], cax=cbar_ax, label='Count')

fig.savefig(f'{OUTDIR}/fig02_obs_vs_pred.png')
plt.close(fig)
print("  -> fig02_obs_vs_pred.png saved.")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Error Analysis
# ═══════════════════════════════════════════════════════════════════════
print("Figure 3: Error Analysis...")
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

residuals = y_te[:, 0] - p_te[:, 0]
abs_err = np.abs(residuals)

# (a) Histogram + normal fit
ax = axes[0]
n_hist, bins, patches = ax.hist(residuals, bins=50, density=True, color='#4292c6',
                                 edgecolor='white', linewidth=0.3, alpha=0.85)
mu, std = residuals.mean(), residuals.std()
x_fit = np.linspace(residuals.min(), residuals.max(), 200)
ax.plot(x_fit, stats.norm.pdf(x_fit, mu, std), 'r-', linewidth=1.5, label='Normal fit')
ax.text(0.97, 0.95, f"$\\mu$ = {mu:.3f}°\n$\\sigma$ = {std:.3f}°",
        transform=ax.transAxes, va='top', ha='right', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
ax.set_xlabel("Residual (°)")
ax.set_ylabel("Density")
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, fontweight='bold', va='top', fontsize=9)

# (b) CDF of |error|
ax = axes[1]
sorted_err = np.sort(abs_err)
cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
ax.step(sorted_err, cdf * 100, where='post', color='#2171b5', linewidth=1.5)
for thr in [1, 2, 3]:
    pct = (abs_err <= thr).mean() * 100
    ax.axvline(thr, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.annotate(f"{pct:.1f}%", xy=(thr, pct), xytext=(thr + 0.2, pct - 8),
                fontsize=7, color='#333333',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
ax.set_xlabel("|Error| (°)")
ax.set_ylabel("Cumulative %")
ax.set_xlim(0, sorted_err.max())
ax.set_ylim(0, 100)
ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, fontweight='bold', va='top', fontsize=9)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig03_error_analysis.png')
plt.close(fig)
print("  -> fig03_error_analysis.png saved.")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: Feature Importance + SHAP
# ═══════════════════════════════════════════════════════════════════════
print("Figure 4: Feature Importance + SHAP...")

# XGBoost gain importance (average across 4 targets)
importances = np.zeros(len(feats))
for est in m.estimators_:
    imp = est.get_booster().get_score(importance_type='gain')
    for k, v in imp.items():
        idx = int(k.replace('f', ''))
        importances[idx] += v
importances /= len(m.estimators_)

top15_idx = np.argsort(importances)[::-1][:15]
top15_feats = [feats[i] for i in top15_idx]
top15_imp = importances[top15_idx]

# Color by category
def feat_color(name):
    if 'newell_cf' in name or 'kan_lee' in name or 'vBs' in name:
        return '#2171B5'  # coupling
    elif 'imf_b' in name:
        return '#CB181D'  # raw IMF
    elif 'sw_' in name:
        return '#238B45'  # solar wind
    elif name in ('dipole_tilt', 'doy', 'hemi_code', 'by_hemi'):
        return '#D94801'  # geometry
    else:
        return '#666666'

fig, axes = plt.subplots(1, 2, figsize=(7, 4))

# (a) Bar chart
ax = axes[0]
y_pos = np.arange(15)[::-1]
colors_bar = [feat_color(f) for f in top15_feats]
ax.barh(y_pos, top15_imp, color=colors_bar, edgecolor='none', height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(top15_feats, fontsize=7)
ax.set_xlabel("Gain Importance")
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, fontweight='bold', va='top', fontsize=9)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2171B5', label='Coupling'),
    Patch(facecolor='#CB181D', label='Raw IMF'),
    Patch(facecolor='#238B45', label='Solar Wind'),
    Patch(facecolor='#D94801', label='Geometry'),
]
ax.legend(handles=legend_elements, fontsize=6, loc='lower right', frameon=True, framealpha=0.9, edgecolor='none')

# (b) SHAP beeswarm
ax2 = axes[1]
print("  Computing SHAP values (subsample 2000)...")
explainer = shap.TreeExplainer(m.estimators_[0])
X_te_shap = X_te[:2000]
shap_values = explainer.shap_values(X_te_shap)

plt.sca(ax2)
shap.summary_plot(shap_values, X_te_shap, feature_names=feats, max_display=15,
                  show=False, plot_size=None)
ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes, fontweight='bold', va='top', fontsize=9)
ax2.set_xlabel("SHAP value")

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig04_importance_shap.png')
plt.close('all')
print("  -> fig04_importance_shap.png saved.")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: Partial Dependence
# ═══════════════════════════════════════════════════════════════════════
print("Figure 5: Partial Dependence...")
pd_vars = ['dipole_tilt', 'newell_cf_mean60', 'imf_bz', 'sw_pdyn', 'imf_by', 'vBs_mean60']
pd_labels = ["Dipole Tilt (°)", "Newell CF (60-min mean)", "IMF Bz (nT)",
             "Sw Pdyn (nPa)", "IMF By (nT)", "vBs (60-min mean)"]
panel_letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))

# Manual PD: vary one feature, hold others at median
X_median = np.median(X_tr, axis=0)
eq_model = m.estimators_[0]  # eq_MLAT sub-model

for i, (var, label, letter) in enumerate(zip(pd_vars, pd_labels, panel_letters)):
    ax = axes.flat[i]
    fi = feats.index(var)
    grid = np.linspace(np.percentile(X_tr[:, fi], 2), np.percentile(X_tr[:, fi], 98), 100)
    X_synth = np.tile(X_median, (len(grid), 1))
    X_synth[:, fi] = grid
    preds = eq_model.predict(X_synth)
    ax.plot(grid, preds, color='#2171b5', linewidth=1.5)
    ax.set_xlabel(label, fontsize=8)
    if i % 3 == 0:
        ax.set_ylabel("Predicted |eq MLAT| (°)")
    ax.text(0.03, 0.95, letter, transform=ax.transAxes, fontweight='bold', va='top', fontsize=8)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig05_partial_dependence.png')
plt.close(fig)
print("  -> fig05_partial_dependence.png saved.")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6: Model Comparison
# ═══════════════════════════════════════════════════════════════════════
print("Figure 6: Model Comparison...")

# Compute baseline MAEs
# Linear on all features
lr_all = LinearRegression().fit(X_tr, y_tr[:, 0])
lr_all_mae = mean_absolute_error(y_te[:, 0], lr_all.predict(X_te))

# Single-feature linear models
def single_feat_mae(feat_name):
    fi = feats.index(feat_name)
    lr = LinearRegression().fit(X_tr[:, fi:fi+1], y_tr[:, 0])
    return mean_absolute_error(y_te[:, 0], lr.predict(X_te[:, fi:fi+1]))

# E_WAV^(2/3) = Newell CF based nonlinear
fi_ncf = feats.index('newell_cf')
X_ncf = X_tr[:, fi_ncf:fi_ncf+1]**(2/3)
X_ncf_te = X_te[:, fi_ncf:fi_ncf+1]**(2/3)
lr_ewav = LinearRegression().fit(X_ncf, y_tr[:, 0])
ewav_mae = mean_absolute_error(y_te[:, 0], lr_ewav.predict(X_ncf_te))

ncf_mae = single_feat_mae('newell_cf')
kle_mae = single_feat_mae('kan_lee_ef')
vbs_mae = single_feat_mae('vBs')
bz_mae = single_feat_mae('imf_bz')

models = [
    ('XGBoost', xgb_mae_eq),
    ('ResMLP-4blk-128', 1.018),
    ('TabTF-d64-L3', 1.102),
    (f'Linear {len(feats)}-feat', lr_all_mae),
    ("Nonlinear E$_{WAV}^{2/3}$", ewav_mae),
    ('Linear Newell CF', ncf_mae),
    ('Linear Kan-Lee', kle_mae),
    ('Linear vBs', vbs_mae),
    ('Linear Bz', bz_mae),
]
# Sort by MAE (best first)
models.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(3.5, 4))
n_mod = len(models)
y_pos = np.arange(n_mod)[::-1]
maes = [v for _, v in models]
names = [n for n, _ in models]

# Color gradient: dark blue (best) to light gray (worst)
from matplotlib.colors import LinearSegmentedColormap
cmap_bar = LinearSegmentedColormap.from_list('blue_gray', ['#08519c', '#c6dbef'])
colors_m = [cmap_bar(i / (n_mod - 1)) for i in range(n_mod)]

ax.barh(y_pos, maes, color=colors_m, edgecolor='none', height=0.65)
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("MAE (°)")

# Annotate values
for i, (name, mae) in enumerate(models):
    ax.text(mae + 0.02, y_pos[i], f"{mae:.3f}", va='center', fontsize=7)

# Vertical dashed line at XGBoost MAE
ax.axvline(xgb_mae_eq, color='#333333', linestyle='--', linewidth=0.8, alpha=0.6)
ax.set_xlim(0, max(maes) * 1.15)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig06_model_comparison.png')
plt.close(fig)
print("  -> fig06_model_comparison.png saved.")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7: Activity + Temporal
# ═══════════════════════════════════════════════════════════════════════
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
ax.set_ylabel("MAE (°)")
for i, (bar, cnt) in enumerate(zip(bars, ae_counts)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"n={cnt}", ha='center', va='bottom', fontsize=7)
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, fontweight='bold', va='top', fontsize=9)

# (b) LOYO (Leave-One-Year-Out) MAE
ax = axes[1]
years_all = sorted(df_clean['year'].unique())
year_arr = df_clean['year'].values

loyo_maes = []
loyo_years = []
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

# Solar cycle coloring
def sc_color(yr):
    if yr <= 1996:
        return '#e6550d'  # SC22
    elif yr <= 2008:
        return '#3182bd'  # SC23
    else:
        return '#31a354'  # SC24

sc_colors = [sc_color(yr) for yr in loyo_years]

# Shade solar cycle regions
ax.axhspan(overall_loyo - 0.01, overall_loyo + 0.01, alpha=0)  # dummy for ylim
for yr_start, yr_end, color, alpha_val in [(1986, 1996.5, '#e6550d', 0.08),
                                            (1996.5, 2008.5, '#3182bd', 0.08),
                                            (2008.5, 2020, '#31a354', 0.08)]:
    ax.axvspan(yr_start, yr_end, alpha=alpha_val, color=color, edgecolor='none')

ax.plot(loyo_years, loyo_maes, '-', color='#666666', linewidth=0.8, zorder=1)
ax.scatter(loyo_years, loyo_maes, c=sc_colors, s=25, zorder=2, edgecolors='white', linewidths=0.3)
ax.axhline(overall_loyo, color='#333333', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(loyo_years[-1] + 0.5, overall_loyo, f"mean={overall_loyo:.3f}", fontsize=7, va='center')

ax.set_xlabel("Year")
ax.set_ylabel("LOYO MAE (°)")
ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, fontweight='bold', va='top', fontsize=9)

# SC legend
from matplotlib.lines import Line2D
sc_legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e6550d', markersize=6, label='SC22'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3182bd', markersize=6, label='SC23'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#31a354', markersize=6, label='SC24'),
]
ax.legend(handles=sc_legend, fontsize=7, loc='upper right', frameon=True, framealpha=0.9, edgecolor='none')

fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig07_activity_temporal.png')
plt.close(fig)
print("  -> fig07_activity_temporal.png saved.")

print("\n=== All 7 figures generated ===")
for i in range(1, 8):
    fpath = glob.glob(f'{OUTDIR}/fig0{i}_*.png')[0]
    size = os.path.getsize(fpath)
    print(f"  {os.path.basename(fpath)}: {size/1024:.0f} KB")
