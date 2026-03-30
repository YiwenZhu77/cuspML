#!/usr/bin/env python3
"""Tree-based model DSE: GBR, XGBoost (GPU), LightGBM (GPU) with hyperparameter search."""
import json, glob, os, time, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

def load_data():
    data_dir = os.environ.get('OMNI_DIR', 'output/omni_hist')
    files = sorted(glob.glob(f'{data_dir}/cusp_crossings_*.json'))
    all_c = []
    for f in files:
        with open(f) as fh:
            all_c.extend(json.load(fh))
    df = pd.DataFrame(all_c)

    df['abs_eq_mlat'] = df['eq_mlat'].abs()
    df['abs_pole_mlat'] = df['pole_mlat'].abs()
    df['hemi_code'] = (df['hemisphere'] == 'N').astype(float)
    df['doy'] = pd.to_datetime(df['time_start']).dt.dayofyear
    df['B_T'] = np.sqrt(df['imf_by']**2 + df['imf_bz']**2)
    df['clock_angle'] = np.arctan2(df['imf_by'], df['imf_bz'])
    df['sin_clock_half'] = np.sin(df['clock_angle'] / 2)
    df['newell_cf'] = (df['sw_v']**(4/3)) * (df['B_T']**(2/3)) * (np.abs(df['sin_clock_half'])**(8/3))
    df['kan_lee_ef'] = df['sw_v'] * df['B_T'] * (df['sin_clock_half']**2)
    df['vBs'] = df['sw_v'] * np.where(df['imf_bz'] < 0, -df['imf_bz'], 0)
    df['by_hemi'] = df['imf_by'] * np.where(df['hemisphere'] == 'N', 1, -1)

    base = ['dipole_tilt','hemi_code','doy','imf_bx','imf_by','imf_bz',
            'sw_v','sw_n','sw_pdyn','B_T','clock_angle','sin_clock_half',
            'newell_cf','kan_lee_ef','vBs','by_hemi']
    hist = sorted([c for c in df.columns if any(s in c for s in
        ['mean15','mean30','mean60','std15','std30','std60','delta15','delta30','delta60','int60','_mean60'])
        and c not in base])
    features = base + hist
    targets = ['abs_eq_mlat', 'abs_pole_mlat', 'eq_mlt', 'mean_mlt']

    df_clean = df[[c for c in features + targets if c in df.columns]].dropna()
    feats = [c for c in features if c in df_clean.columns]
    X = df_clean[feats].values.astype(np.float32)
    y = df_clean[targets].values.astype(np.float32)
    print(f"Features: {len(feats)}, Samples: {len(X)}")
    return X, y, feats

def main():
    TGT_NAMES = ['eq_MLAT', 'pole_MLAT', 'eq_MLT', 'mean_MLT']
    TGT_UNITS = ['°', '°', 'hr', 'hr']

    X, y, feats = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_tr)}, Test: {len(X_te)}")

    has_gpu = os.environ.get('USE_GPU', '0') == '1'
    n_jobs = int(os.environ.get('N_JOBS', '8'))
    xgb_device = 'cuda' if has_gpu else 'cpu'
    lgb_device = 'gpu' if has_gpu else 'cpu'
    print(f"XGB device: {xgb_device}, LGB device: {lgb_device}, n_jobs: {n_jobs}")

    configs = [
        # GBR (CPU only, but n_jobs not supported for GBR itself)
        ('GBR-300-d5', lambda: MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42), n_jobs=n_jobs)),
        ('GBR-500-d6', lambda: MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42), n_jobs=n_jobs)),
        ('GBR-800-d7', lambda: MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=800, max_depth=7, learning_rate=0.03, subsample=0.8, min_samples_leaf=10, random_state=42), n_jobs=n_jobs)),

        # XGBoost
        ('XGB-300-d5', lambda: MultiOutputRegressor(xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1, device=xgb_device, random_state=42, verbosity=0, n_jobs=n_jobs))),
        ('XGB-500-d6', lambda: MultiOutputRegressor(xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            device=xgb_device, random_state=42, verbosity=0, n_jobs=n_jobs))),
        ('XGB-800-d7', lambda: MultiOutputRegressor(xgb.XGBRegressor(
            n_estimators=800, max_depth=7, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, device=xgb_device, random_state=42, verbosity=0, n_jobs=n_jobs))),
        ('XGB-1000-d8', lambda: MultiOutputRegressor(xgb.XGBRegressor(
            n_estimators=1000, max_depth=8, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5, device=xgb_device, random_state=42, verbosity=0, n_jobs=n_jobs))),
        ('XGB-1500-d6-fine', lambda: MultiOutputRegressor(xgb.XGBRegressor(
            n_estimators=1500, max_depth=6, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.05, reg_lambda=0.5, min_child_weight=3, device=xgb_device, random_state=42, verbosity=0, n_jobs=n_jobs))),

        # LightGBM
        ('LGBM-300', lambda: MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1, device=lgb_device, random_state=42, verbose=-1, n_jobs=n_jobs))),
        ('LGBM-500', lambda: MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            device=lgb_device, random_state=42, verbose=-1, n_jobs=n_jobs))),
        ('LGBM-800', lambda: MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=800, max_depth=7, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, device=lgb_device, random_state=42, verbose=-1, n_jobs=n_jobs))),
        ('LGBM-1000', lambda: MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=1000, max_depth=8, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5, device=lgb_device, random_state=42, verbose=-1, n_jobs=n_jobs))),
        ('LGBM-1500-fine', lambda: MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=1500, max_depth=6, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.05, reg_lambda=0.5, min_child_weight=3, num_leaves=63,
            device=lgb_device, random_state=42, verbose=-1, n_jobs=n_jobs))),
    ]

    results = []
    for name, model_fn in configs:
        t0 = time.time()
        m = model_fn()
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        elapsed = time.time() - t0

        row = {'name': name, 'time': elapsed}
        parts = []
        for i, (tn, tu) in enumerate(zip(TGT_NAMES, TGT_UNITS)):
            mae = mean_absolute_error(y_te[:, i], p[:, i])
            r2 = r2_score(y_te[:, i], p[:, i])
            row[f'mae_{tn}'] = mae
            row[f'r2_{tn}'] = r2
            parts.append(f"{tn}={mae:.3f}{tu}")
        row['mlat_avg'] = (row['mae_eq_MLAT'] + row['mae_pole_MLAT']) / 2
        results.append(row)
        print(f"{name:20s} | {' '.join(parts)} | avg_MLAT={row['mlat_avg']:.3f}° | {elapsed:.0f}s")

    # Save
    df_res = pd.DataFrame(results).sort_values('mlat_avg')
    os.makedirs('dse_results', exist_ok=True)
    df_res.to_csv('dse_results/tree_dse_log.csv', index=False)

    print("\n" + "=" * 80)
    print("RANKING by avg MLAT MAE:")
    print("=" * 80)
    for _, r in df_res.iterrows():
        print(f"  {r['name']:20s} | MLAT_avg={r['mlat_avg']:.3f}° | eq={r['mae_eq_MLAT']:.3f} pole={r['mae_pole_MLAT']:.3f} | MLT={r['mae_mean_MLT']:.3f}hr | R²_eq={r['r2_eq_MLAT']:.4f} | {r['time']:.0f}s")

    best = df_res.iloc[0]
    print(f"\n🏆 BEST: {best['name']} — avg MLAT MAE={best['mlat_avg']:.3f}°")

if __name__ == '__main__':
    main()
