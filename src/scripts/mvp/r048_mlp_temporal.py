#!/usr/bin/env python
"""R048: tuned MLP vs XGBoost on the TEMPORAL holdout (and day-grouped), closing the
external-review point that the NN comparison was only on the optimistic random split.
Uses the best MLP config from r044 (256-128, alpha 1e-3). Output: r048_mlp_temporal.json
"""
import json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
IN="/glade/work/yizhu/cuspML/output/omni_full_hist_90120"
OUT="/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r048_mlp_temporal.json"
XGB=dict(n_estimators=1000,max_depth=8,learning_rate=0.02,subsample=0.8,colsample_bytree=0.7,
         reg_alpha=0.1,reg_lambda=1.0,min_child_weight=5,random_state=42,n_jobs=32,verbosity=0)
BASE=['dipole_tilt','hemi_code','doy','imf_bx','imf_by','imf_bz','sw_v','sw_n','sw_pdyn',
      'B_T','clock_angle','sin_clock_half','newell_cf','kan_lee_ef','vBs','by_hemi']
WIN=['mean15','std15','delta15','mean30','std30','delta30','mean60','std60','delta60','int60']
mae=lambda a,b: float(mean_absolute_error(a,b))
def load():
    recs=[]
    for f in sorted(glob.glob(f"{IN}/cusp_crossings_*.json")): recs.extend(json.load(open(f)))
    df=pd.DataFrame(recs).dropna(subset=["eq_mlat","imf_bz","sw_v","sw_n","sw_pdyn"])
    df["abs_eq_mlat"]=df["eq_mlat"].abs(); df["hemi_code"]=(df["hemisphere"]=="N").astype(float)
    df["doy"]=pd.to_datetime(df["time_start"]).dt.dayofyear; df["year"]=pd.to_datetime(df["time_start"]).dt.year
    df["B_T"]=np.sqrt(df["imf_by"]**2+df["imf_bz"]**2)
    df["clock_angle"]=np.arctan2(df["imf_by"],df["imf_bz"]); df["sin_clock_half"]=np.sin(df["clock_angle"]/2)
    df["newell_cf"]=(df["sw_v"]**(4/3))*(df["B_T"]**(2/3))*(np.abs(df["sin_clock_half"])**(8/3))
    df["kan_lee_ef"]=df["sw_v"]*df["B_T"]*(df["sin_clock_half"]**2)
    df["vBs"]=df["sw_v"]*np.where(df["imf_bz"]<0,-df["imf_bz"],0)
    df["by_hemi"]=df["imf_by"]*np.where(df["hemisphere"]=="N",1,-1)
    return df
def mlp(): return MLPRegressor(hidden_layer_sizes=(256,128),activation='relu',alpha=1e-3,max_iter=500,early_stopping=True,n_iter_no_change=15,random_state=42)
df=load(); feats=BASE+sorted([c for c in df.columns if any(s in c for s in WIN) and c not in BASE])
need=feats+['abs_eq_mlat','date','year']; df=df[df[[c for c in need if c in df.columns]].notna().all(axis=1)].reset_index(drop=True)
X=df[feats].values.astype(np.float32); y=df['abs_eq_mlat'].values.astype(np.float32); yr=df['year'].values
R={'n_rows':int(len(df))}; print("rows",len(df))
# temporal
tr=yr<2008; te=yr>=2008; sc=StandardScaler().fit(X[tr])
m=mlp().fit(sc.transform(X[tr]),y[tr]); xg=XGBRegressor(**XGB).fit(X[tr],y[tr])
R['temporal']=dict(MLP=round(mae(y[te],m.predict(sc.transform(X[te]))),4),XGB=round(mae(y[te],xg.predict(X[te])),4))
print("temporal",R['temporal'])
# day-grouped
groups=pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').values
gtr,gte=next(GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42).split(X,y,groups))
sc2=StandardScaler().fit(X[gtr]); m2=mlp().fit(sc2.transform(X[gtr]),y[gtr]); xg2=XGBRegressor(**XGB).fit(X[gtr],y[gtr])
R['day_grouped']=dict(MLP=round(mae(y[gte],m2.predict(sc2.transform(X[gte]))),4),XGB=round(mae(y[gte],xg2.predict(X[gte])),4))
print("day_grouped",R['day_grouped'])
json.dump(R,open(OUT,'w'),indent=1); print("saved",OUT)
