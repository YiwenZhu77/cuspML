#!/usr/bin/env python
"""R051: dependence-aware validation for ALL FOUR targets (not just eq-MLAT).
Defense soft spot: the paper's temporal/day-grouped/LOYO validation focused on the
equatorward MLAT; the other three targets (poleward MLAT, eq MLT, mean MLT) were only
reported on the random split. Here we run every target through random / temporal-holdout /
day-grouped / LOYO so no target rests on the optimistic random split alone.

INPUT : output/omni_full_hist_90120/cusp_crossings_*.json (via r044 load/feats)
OUTPUT: src/kernels/cuspmap_mvp/bundles/r051_alltarget.json (+ printed table)
RUN   : conda py3.10 ; OMP_NUM_THREADS=6 python r051_alltarget_validation.py
"""
import json, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/glade/work/yizhu/cuspML/src/scripts/mvp")
import numpy as np, pandas as pd
from r044_fixes import load, feats, TARGETS, XGB
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
OUT="/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r051_alltarget.json"
def mae(a,b): return float(mean_absolute_error(a,b))
def fit(Xtr,ytr,Xte,yte):
    m=XGBRegressor(**{**XGB,"n_jobs":6}); m.fit(Xtr,ytr); return mae(yte,m.predict(Xte))
def main():
    df=load(); cols=feats(df,120)
    X=df[cols].values.astype(float); year=df["year"].values
    day=pd.to_datetime(df["time_start"]).dt.floor("D").astype(str).values
    comp=~np.isnan(X).any(axis=1)
    X,year,day=X[comp],year[comp],day[comp]
    Y={t:df[t].values[comp] for t in TARGETS}
    idx=np.arange(len(X)); print(f"rows {len(X)}",flush=True)
    itr_r,ite_r=train_test_split(idx,test_size=0.2,random_state=42)
    tr_t,te_t=idx[year<=2007],idx[year>=2008]
    gtr,gte=next(GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42).split(X,groups=day))
    years=sorted(set(year)); R={t:{} for t in TARGETS}
    for t in TARGETS:
        y=Y[t]
        R[t]["random"]=fit(X[itr_r],y[itr_r],X[ite_r],y[ite_r])
        R[t]["temporal"]=fit(X[tr_t],y[tr_t],X[te_t],y[te_t])
        R[t]["day_grouped"]=fit(X[gtr],y[gtr],X[gte],y[gte])
        loyo=[]
        for yr in years:
            te=idx[year==yr];
            if len(te)<100: continue
            tr=idx[year!=yr]; loyo.append(fit(X[tr],y[tr],X[te],y[te]))
        R[t]["loyo_mean"]=float(np.mean(loyo)); R[t]["loyo_std"]=float(np.std(loyo)); R[t]["loyo_nyears"]=len(loyo)
        print(f"{t:14} rand {R[t]['random']:.3f} | temporal {R[t]['temporal']:.3f} | day-grp {R[t]['day_grouped']:.3f} | LOYO {R[t]['loyo_mean']:.3f}±{R[t]['loyo_std']:.3f}",flush=True)
    json.dump(R,open(OUT,"w"),indent=1); print("saved",OUT)
if __name__=="__main__": main()
