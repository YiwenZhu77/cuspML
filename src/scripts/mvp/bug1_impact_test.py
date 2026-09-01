"""Quick test: does the OMNI density/pressure fill-value contamination (BUG1) change the headline MAE?

Compares XGBoost eq-MLAT MAE (random + temporal) with the poisoned sw_n/sw_pdyn window features
AS-IS vs with the physically impossible values NaN-ed out (XGBoost handles NaN natively). If the two
agree within noise, BUG1 does not affect the paper's numbers and only the dataset needs cleaning.
RUN: python src/scripts/mvp/bug1_impact_test.py  (Casper htc)
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/glade/work/yizhu/cuspML/src/scripts/mvp")
from r046_matched_splits import load, feats, XGB, mae
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

df = load()
f74 = feats(df, 60)
need = f74 + ['abs_eq_mlat', 'ae_index', 'hemisphere', 'date', 'year', 'newell_cf']
df = df[df[[c for c in need if c in df.columns]].notna().all(axis=1)].sort_values('t').reset_index(drop=True)
y = df['abs_eq_mlat'].values.astype(np.float32)
yr = df['year'].values
Xo = df[f74].copy()
Xc = Xo.copy()
for c in Xc.columns:
    if 'sw_n' in c:
        Xc.loc[Xc[c].abs() > 200, c] = np.nan
    elif 'sw_pdyn' in c:
        Xc.loc[Xc[c].abs() > 60, c] = np.nan
print("cells NaN-ed:", int(np.isnan(Xc.values).sum() - np.isnan(Xo.values).sum()), flush=True)

itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
trm, tem = yr < 2008, yr >= 2008
for name, Xd in [("ORIGINAL(poisoned)", Xo.values.astype(np.float32)),
                 ("BUG1-FIXED(NaN)", Xc.values.astype(np.float32))]:
    r = mae(y[ite], XGBRegressor(**XGB).fit(Xd[itr], y[itr]).predict(Xd[ite]))
    t = mae(y[tem], XGBRegressor(**XGB).fit(Xd[trm], y[trm]).predict(Xd[tem]))
    print(f"{name:22s} random={r:.4f}  temporal={t:.4f}", flush=True)
print("DONE", flush=True)
