#!/usr/bin/env python
"""R052: operational GAP robustness for the real-time claim.
Defense soft spot: the model needs complete 60-min input windows, but real-time L1 streams
have gaps, and this was never tested. Here we simulate missing inputs at prediction time and
measure how MAE degrades, under (a) XGBoost native NaN handling and (b) mean imputation, and
whether training WITH random gap injection (so the model learns to tolerate gaps) helps.
Not a substitute for real raw-L1 validation, but demonstrates graceful degradation + a
concrete gap-handling strategy.

INPUT : output/omni_full_hist_90120/cusp_crossings_*.json (via r044 load/feats)
OUTPUT: src/kernels/cuspmap_mvp/bundles/r052_gap.json (+ printed table)
RUN   : conda py3.10 ; OMP_NUM_THREADS=6 python r052_gap_robustness.py
"""
import json, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/glade/work/yizhu/cuspML/src/scripts/mvp")
import numpy as np
from r044_fixes import load, feats, XGB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
OUT="/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r052_gap.json"
def mae(a,b): return float(mean_absolute_error(a,b))
def maskgaps(X, p, rng):
    if p<=0: return X
    M=rng.random(X.shape)<p; Xg=X.copy(); Xg[M]=np.nan; return Xg
def main():
    df=load(); cols=feats(df,120)
    X=df[cols].values.astype(float); y=df["abs_eq_mlat"].values.astype(float)
    comp=~np.isnan(X).any(axis=1); X,y=X[comp],y[comp]
    itr,ite=train_test_split(np.arange(len(X)),test_size=0.2,random_state=42)
    Xtr,Xte,ytr,yte=X[itr],X[ite],y[itr],y[ite]
    means=np.nanmean(Xtr,axis=0)
    rng=np.random.default_rng(0)
    fracs=[0.0,0.05,0.10,0.20,0.30]
    # (1) baseline model trained on complete data
    m0=XGBRegressor(**{**XGB,"n_jobs":6}); m0.fit(Xtr,ytr)
    # (2) gap-robust model: train with random gap injection (10-30% per row)
    Xtr_g=Xtr.copy(); pr=rng.uniform(0.0,0.3,size=len(Xtr))[:,None]; Mtr=rng.random(Xtr.shape)<pr; Xtr_g[Mtr]=np.nan
    m1=XGBRegressor(**{**XGB,"n_jobs":6}); m1.fit(Xtr_g,ytr)
    R={"clean_MAE":mae(yte,m0.predict(Xte)),"fracs":fracs,"native_nan":[],"mean_impute":[],"gaptrained_native":[]}
    for p in fracs:
        Xg=maskgaps(Xte,p,rng)
        R["native_nan"].append(mae(yte,m0.predict(Xg)))
        Xi=Xg.copy(); inds=np.where(np.isnan(Xi)); Xi[inds]=np.take(means,inds[1])
        R["mean_impute"].append(mae(yte,m0.predict(Xi)))
        R["gaptrained_native"].append(mae(yte,m1.predict(Xg)))
        print(f"gap {int(p*100):>2}% | native-NaN {R['native_nan'][-1]:.3f} | mean-impute {R['mean_impute'][-1]:.3f} | gap-trained {R['gaptrained_native'][-1]:.3f}",flush=True)
    json.dump(R,open(OUT,"w"),indent=1); print("saved",OUT)
if __name__=="__main__": main()
