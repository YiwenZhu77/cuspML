import json, glob, numpy as np, pandas as pd, time
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

files = sorted(glob.glob('output/omni_full_hist/cusp_crossings_*.json'))
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
    ['mean15','mean30','mean60','std15','std30','std60','delta15','delta30','delta60','int60'])
    and c not in base])
features = base + hist
targets = ['abs_eq_mlat', 'abs_pole_mlat', 'eq_mlt', 'mean_mlt']
df_clean = df[[c for c in features + targets if c in df.columns]].dropna()
feats = [c for c in features if c in df_clean.columns]
X = df_clean[feats].values.astype(np.float32)
y = df_clean[targets].values.astype(np.float32)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data: {len(X_tr)} train, {len(X_te)} test, {len(feats)} features")

params = dict(n_estimators=1000, max_depth=8, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, random_state=42, verbosity=0)

# CPU
t0 = time.time()
m_cpu = MultiOutputRegressor(xgb.XGBRegressor(**params, tree_method='hist', n_jobs=8))
m_cpu.fit(X_tr, y_tr)
cpu_train = time.time() - t0
p_cpu = m_cpu.predict(X_te)
cpu_mae = mean_absolute_error(y_te[:,0], p_cpu[:,0])

t0 = time.time()
for _ in range(100):
    m_cpu.predict(X_te)
cpu_infer = (time.time() - t0) / 100
print(f"CPU (8 cores): train={cpu_train:.1f}s, infer={cpu_infer:.3f}s/batch, MAE={cpu_mae:.4f}")

# GPU
t0 = time.time()
m_gpu = MultiOutputRegressor(xgb.XGBRegressor(**params, tree_method='hist', device='cuda'))
m_gpu.fit(X_tr, y_tr)
gpu_train = time.time() - t0
p_gpu = m_gpu.predict(X_te)
gpu_mae = mean_absolute_error(y_te[:,0], p_gpu[:,0])

t0 = time.time()
for _ in range(100):
    m_gpu.predict(X_te)
gpu_infer = (time.time() - t0) / 100
print(f"GPU (V100):    train={gpu_train:.1f}s, infer={gpu_infer:.3f}s/batch, MAE={gpu_mae:.4f}")

print(f"\nSpeedup: train {cpu_train/gpu_train:.1f}x, infer {cpu_infer/gpu_infer:.1f}x")
