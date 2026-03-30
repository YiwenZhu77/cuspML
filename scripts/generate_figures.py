#!/usr/bin/env python
"""Generate publication-quality figures for JGR Space Physics cusp ML paper."""

import json
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

FIGDIR = '/glade/work/yizhu/cuspML/figures'

# ── Load all data ──────────────────────────────────────────────────────────
print("Loading data...")
files = sorted(glob.glob('/glade/work/yizhu/cuspML/output/omni_full_hist/cusp_crossings_*.json'))
records = []
for f in files:
    records.extend(json.load(open(f)))
df = pd.DataFrame(records)
df['time_start'] = pd.to_datetime(df['time_start'])
df['year'] = df['time_start'].dt.year
print(f"Loaded {len(df)} crossings from {len(files)} files")

# Drop rows with NaN in critical columns
critical_cols = ['dipole_tilt', 'imf_bx', 'imf_by', 'imf_bz', 'sw_v', 'sw_n', 'sw_pdyn',
                 'eq_mlat', 'pole_mlat', 'eq_mlt', 'mean_mlt']
df = df.dropna(subset=critical_cols).reset_index(drop=True)
print(f"After dropping NaN: {len(df)} crossings")

# ── Derive features ───────────────────────────────────────────────────────
df['abs_eq_mlat'] = df['eq_mlat'].abs()
df['abs_pole_mlat'] = df['pole_mlat'].abs()
df['hemi_code'] = (df['hemisphere'] == 'N').astype(int)
df['doy'] = df['time_start'].dt.dayofyear
df['B_T'] = np.sqrt(df['imf_by']**2 + df['imf_bz']**2)
df['clock_angle'] = np.arctan2(df['imf_by'], df['imf_bz'])
df['sin_clock_half'] = np.sin(df['clock_angle'] / 2)
df['newell_cf'] = (df['sw_v']**(4/3)) * (df['B_T']**(2/3)) * (np.abs(df['sin_clock_half'])**(8/3))
df['kan_lee_ef'] = df['sw_v'] * df['B_T'] * df['sin_clock_half']**2
df['vBs'] = df['sw_v'] * np.maximum(-df['imf_bz'], 0)
df['by_hemi'] = df['imf_by'] * np.where(df['hemisphere'] == 'N', 1, -1)

# Identify history feature columns
history_cols = [c for c in df.columns if any(c.endswith(s) for s in
    ['_mean15','_std15','_delta15','_mean30','_std30','_delta30',
     '_mean60','_std60','_delta60','_int60'])]

base_features = ['dipole_tilt', 'hemi_code', 'doy', 'imf_bx', 'imf_by', 'imf_bz',
                 'sw_v', 'sw_n', 'sw_pdyn', 'B_T', 'clock_angle', 'sin_clock_half',
                 'newell_cf', 'kan_lee_ef', 'vBs', 'by_hemi']
all_features = base_features + history_cols

targets = ['abs_eq_mlat', 'abs_pole_mlat', 'eq_mlt', 'mean_mlt']
target_labels = ['|eq MLAT| (°)', '|pole MLAT| (°)', 'eq MLT (h)', 'mean MLT (h)']

# Drop rows with NaN in any feature or target
df_model = df.dropna(subset=all_features + targets).reset_index(drop=True)
print(f"Model dataset: {len(df_model)} crossings, {len(all_features)} features")

X = df_model[all_features].values
Y = df_model[targets].values

# ── Train XGBoost ──────────────────────────────────────────────────────────
print("Training XGBoost model (80/20 train/test split, random_state=42)...")
idx_tr, idx_te = train_test_split(np.arange(len(df_model)), test_size=0.2, random_state=42)
X_tr, X_te = X[idx_tr], X[idx_te]
Y_tr, Y_te = Y[idx_tr], Y[idx_te]
df_train = df_model.iloc[idx_tr].copy()
df_test  = df_model.iloc[idx_te].reset_index(drop=True)
xgb_params = dict(
    n_estimators=1000, max_depth=8, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
    n_jobs=-1, random_state=42,
)
model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
model.fit(X_tr, Y_tr)
Y_pred = model.predict(X_te)   # test-set predictions only
Y      = Y_te                  # align Y to test set
df_model = df_test             # align df_model to test set
print("Training complete.")
for i, lbl in enumerate(target_labels):
    print(f"  {lbl}: MAE={mean_absolute_error(Y[:,i], Y_pred[:,i]):.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# Fig 1: Data Coverage Map
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig 1...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                         gridspec_kw={'width_ratios': [1.2, 1]})

# ── Top panel: histogram by year and satellite ──
ax = axes[0]
sats = sorted(df['satellite'].unique())
# Use a categorical colormap
cmap_sat = plt.cm.tab20
sat_colors = {s: cmap_sat(i / max(len(sats)-1, 1)) for i, s in enumerate(sats)}

years = np.arange(df['year'].min(), df['year'].max() + 1)
bottom = np.zeros(len(years))
for sat in sats:
    counts = df[df['satellite'] == sat].groupby('year').size().reindex(years, fill_value=0).values
    ax.bar(years, counts, bottom=bottom, label=sat, color=sat_colors[sat], width=0.8, edgecolor='none')
    bottom += counts

ax.set_xlabel('Year')
ax.set_ylabel('Number of Crossings')
ax.set_title('(a) Cusp Crossings by Year and Satellite')
ax.legend(ncol=3, fontsize=7, loc='upper left', framealpha=0.9)
ax.set_xlim(1986.5, 2014.5)

# ── Bottom panel: polar scatter ──
ax2 = fig.add_subplot(122, projection='polar')
axes[1].remove()

theta = df['eq_mlt'].values * (2 * np.pi / 24)  # MLT to radians (0 MLT = 0 rad)
# Rotate so noon (12 MLT) is at top
theta_plot = theta - np.pi/2  # put 6 MLT at top; we want 12 at top
theta_plot = (df['eq_mlt'].values / 24) * 2 * np.pi  # 0-24h -> 0-2pi
# Convention: 12 MLT at top means theta=0 corresponds to 12 MLT
theta_plot = ((df['eq_mlt'].values - 12) / 24) * 2 * np.pi + np.pi/2
# Simpler: just use standard polar with 12h at top
theta_plot = (df['eq_mlt'].values / 24) * 2 * np.pi

r = 90 - df['eq_mlat'].abs().values  # radius: 0 at pole, 30 at 60°

sc = ax2.scatter(theta_plot, r, c=df['hemi_code'].values, cmap='coolwarm',
                 s=1, alpha=0.3, rasterized=True)
ax2.set_theta_zero_location('S')  # 0 MLT at bottom
ax2.set_theta_direction(-1)  # clockwise
ax2.set_rlim(0, 30)
ax2.set_rticks([5, 10, 15, 20, 25, 30])
ax2.set_yticklabels(['85°', '80°', '75°', '70°', '65°', '60°'], fontsize=8)
# MLT labels
ax2.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
ax2.set_xticklabels([f'{h}' for h in range(24)], fontsize=7)
ax2.set_title('(b) Crossing Locations (MLAT-MLT)', pad=20)

# Legend for hemispheres
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                          markersize=5, label='North'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='indianred',
                          markersize=5, label='South')]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=9,
           bbox_to_anchor=(1.15, -0.05))

plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig01_data_coverage.png')
plt.close()
print("  Saved fig01_data_coverage.png")

# ═══════════════════════════════════════════════════════════════════════════
# Fig 2: Observed vs Predicted Scatter (4 panels)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig 2...")
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for idx, (ax, tgt, label) in enumerate(zip(axes.ravel(), targets, target_labels)):
    obs = Y[:, idx]
    pred = Y_pred[:, idx]
    r_val = np.corrcoef(obs, pred)[0, 1]
    mae = mean_absolute_error(obs, pred)

    # 2D histogram for density
    h, xedges, yedges = np.histogram2d(obs, pred, bins=100)
    # Map each point to its density
    ix = np.clip(np.digitize(obs, xedges) - 1, 0, 99)
    iy = np.clip(np.digitize(pred, yedges) - 1, 0, 99)
    density = h[ix, iy]

    sort_idx = np.argsort(density)
    ax.scatter(obs[sort_idx], pred[sort_idx], c=density[sort_idx],
               cmap='viridis', s=3, alpha=0.6, rasterized=True, edgecolors='none')

    lims = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.7, label='1:1 line')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f'Observed {label}')
    ax.set_ylabel(f'Predicted {label}')
    ax.set_title(f'({chr(97+idx)}) {label}')
    ax.text(0.05, 0.92, f'r = {r_val:.3f}\nMAE = {mae:.2f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig02_scatter_4panel.png')
plt.close()
print("  Saved fig02_scatter_4panel.png")

# ═══════════════════════════════════════════════════════════════════════════
# Fig 3: Baseline Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig 3...")
y_eq = df_model['abs_eq_mlat'].values  # test set targets

# Baseline models: fit on training set, evaluate on test set
baselines = {}
for name, feat_col in [('Bz only', 'imf_bz'), ('vBs', 'vBs'),
                        ('Kan-Lee EF', 'kan_lee_ef'), ('Newell CF', 'newell_cf')]:
    Xb_tr = df_train[feat_col].values.reshape(-1, 1)
    y_tr_b = df_train['abs_eq_mlat'].values
    mask_tr = ~np.isnan(Xb_tr.ravel()) & ~np.isnan(y_tr_b)
    lr = LinearRegression().fit(Xb_tr[mask_tr], y_tr_b[mask_tr])
    Xb_te = df_model[feat_col].values.reshape(-1, 1)
    mask_te = ~np.isnan(Xb_te.ravel()) & ~np.isnan(y_eq)
    pred_b = lr.predict(Xb_te[mask_te])
    baselines[name] = mean_absolute_error(y_eq[mask_te], pred_b)

# XGBoost MAE for eq_MLAT
baselines['XGBoost (ours)'] = mean_absolute_error(y_eq, Y_pred[:, 0])

# Sort by MAE descending (worst at top)
sorted_baselines = sorted(baselines.items(), key=lambda x: x[1], reverse=True)
names = [x[0] for x in sorted_baselines]
maes = [x[1] for x in sorted_baselines]

fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#888888'] * (len(names) - 1)
# Find XGBoost position and color it differently
for i, n in enumerate(names):
    if 'XGBoost' in n:
        colors.insert(i, '#2171b5')
    else:
        pass
# Rebuild properly
colors = []
for n in names:
    if 'XGBoost' in n:
        colors.append('#2171b5')
    else:
        colors.append('#969696')

bars = ax.barh(range(len(names)), maes, color=colors, edgecolor='white', height=0.6)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel('Mean Absolute Error (°)')
ax.set_title('Equatorward MLAT Prediction: Model Comparison')

# Add value labels
for bar, mae_val in zip(bars, maes):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f'{mae_val:.2f}°', va='center', fontsize=11)

ax.set_xlim(0, max(maes) * 1.15)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig03_baseline_comparison.png')
plt.close()
print("  Saved fig03_baseline_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# Fig 4: Error Distribution + CDF
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig 4...")
residuals = y_eq - Y_pred[:, 0]
abs_errors = np.abs(residuals)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: histogram of residuals
ax1.hist(residuals, bins=100, color='steelblue', edgecolor='none', alpha=0.85, density=True)
ax1.axvline(0, color='k', ls='--', lw=1)
ax1.set_xlabel('Prediction Error (°)')
ax1.set_ylabel('Probability Density')
ax1.set_title('(a) Error Distribution (eq MLAT)')
ax1.text(0.95, 0.92, f'Mean = {residuals.mean():.3f}°\nStd = {residuals.std():.2f}°',
         transform=ax1.transAxes, fontsize=11, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Right: CDF of |error|
sorted_err = np.sort(abs_errors)
cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)
ax2.plot(sorted_err, cdf * 100, color='steelblue', lw=2)
ax2.set_xlabel('|Error| (°)')
ax2.set_ylabel('Cumulative Percentage (%)')
ax2.set_title('(b) CDF of |Error| (eq MLAT)')
ax2.set_xlim(0, min(10, sorted_err.max()))
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)

# Mark thresholds
for thresh, color in [(1, '#2ca02c'), (2, '#ff7f0e'), (3, '#d62728')]:
    pct = (abs_errors <= thresh).mean() * 100
    ax2.axvline(thresh, color=color, ls='--', lw=1.5, alpha=0.8)
    ax2.text(thresh + 0.1, pct - 5, f'{thresh}°: {pct:.1f}%',
             fontsize=10, color=color, fontweight='bold')

plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig04_error_distribution.png')
plt.close()
print("  Saved fig04_error_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════
# Fig 5: Feature Importance (top 20)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Fig 5...")
# Get eq_MLAT model (first estimator in MultiOutputRegressor)
eq_model = model.estimators_[0]
importances = eq_model.feature_importances_
feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False).head(20)

# Color-code by category
def get_category(feat):
    coupling = ['newell_cf', 'kan_lee_ef', 'vBs', 'B_T', 'clock_angle', 'sin_clock_half']
    raw_imf = ['imf_bx', 'imf_by', 'imf_bz']
    sw = ['sw_v', 'sw_n', 'sw_pdyn']
    geometry = ['dipole_tilt', 'hemi_code', 'doy', 'by_hemi']
    history_bases = ['_mean15', '_std15', '_delta15', '_mean30', '_std30', '_delta30',
                     '_mean60', '_std60', '_delta60', '_int60']

    if any(feat.endswith(h) for h in history_bases):
        return 'history'
    if feat in coupling:
        return 'coupling'
    if feat in raw_imf:
        return 'raw_imf'
    if feat in sw:
        return 'solar_wind'
    if feat in geometry:
        return 'geometry'
    return 'other'

cat_colors = {
    'coupling': '#2171b5',
    'raw_imf': '#cb181d',
    'solar_wind': '#238b45',
    'geometry': '#e6550d',
    'history': '#7b4173',
}
cat_labels = {
    'coupling': 'Coupling Functions',
    'raw_imf': 'Raw IMF',
    'solar_wind': 'Solar Wind',
    'geometry': 'Geometry',
    'history': 'History Features',
}

fig, ax = plt.subplots(figsize=(8, 7))
feat_imp_sorted = feat_imp.sort_values(ascending=True)  # ascending for horizontal bar
colors = [cat_colors.get(get_category(f), 'gray') for f in feat_imp_sorted.index]

ax.barh(range(len(feat_imp_sorted)), feat_imp_sorted.values, color=colors,
        edgecolor='white', height=0.7)
ax.set_yticks(range(len(feat_imp_sorted)))
ax.set_yticklabels(feat_imp_sorted.index, fontsize=10)
ax.set_xlabel('Feature Importance (Gain)')
ax.set_title('Top 20 Features for eq MLAT Prediction')

# Legend
legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor=c,
                          markersize=10, label=cat_labels[k])
                   for k, c in cat_colors.items()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig05_feature_importance.png')
plt.close()
print("  Saved fig05_feature_importance.png")

print("\nAll figures saved to", FIGDIR)
