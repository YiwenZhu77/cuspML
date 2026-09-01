#!/usr/bin/env python
"""R047: round-3 external-review robustness additions.

 (#7) Leave-one-satellite-out CV: hold out each DMSP satellite (F06-F18) in turn and test,
      to check the model is not exploiting platform-specific label/calibration structure.
 (#5) LOYO sensitivity: mean over the 23 years with >=100 crossings vs all 25 years with data.
 (#8) Day-grouped split dispersion: repeat GroupShuffleSplit over 8 seeds, report mean+-std.

Input : output/omni_full_hist_90120/cusp_crossings_*.json (restricted to the 39,668 set)
Output: src/kernels/cuspmap_mvp/bundles/r047_robustness.json
"""
import json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

IN="/glade/work/yizhu/cuspML/output/omni_full_hist_90120"
OUT="/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r047_robustness.json"
XGB=dict(n_estimators=1000,max_depth=8,learning_rate=0.02,subsample=0.8,colsample_bytree=0.7,
         reg_alpha=0.1,reg_lambda=1.0,min_child_weight=5,random_state=42,n_jobs=32,verbosity=0)
BASE=['dipole_tilt','hemi_code','doy','imf_bx','imf_by','imf_bz','sw_v','sw_n','sw_pdyn',
      'B_T','clock_angle','sin_clock_half','newell_cf','kan_lee_ef','vBs','by_hemi']
WIN=['mean15','std15','delta15','mean30','std30','delta30','mean60','std60','delta60','int60']
mae=lambda a,b: float(mean_absolute_error(a,b))

def load():
    recs=[]
    for f in sorted(glob.glob(f"{IN}/cusp_crossings_*.json")): recs.extend(json.load(open(f)))
    df=pd.DataFrame(recs).dropna(subset=["eq_mlat","pole_mlat","imf_bz","sw_v","sw_n","sw_pdyn"])
    df["abs_eq_mlat"]=df["eq_mlat"].abs()
    df["hemi_code"]=(df["hemisphere"]=="N").astype(float)
    df["doy"]=pd.to_datetime(df["time_start"]).dt.dayofyear; df["year"]=pd.to_datetime(df["time_start"]).dt.year
    df["B_T"]=np.sqrt(df["imf_by"]**2+df["imf_bz"]**2)
    df["clock_angle"]=np.arctan2(df["imf_by"],df["imf_bz"]); df["sin_clock_half"]=np.sin(df["clock_angle"]/2)
    df["newell_cf"]=(df["sw_v"]**(4/3))*(df["B_T"]**(2/3))*(np.abs(df["sin_clock_half"])**(8/3))
    df["kan_lee_ef"]=df["sw_v"]*df["B_T"]*(df["sin_clock_half"]**2)
    df["vBs"]=df["sw_v"]*np.where(df["imf_bz"]<0,-df["imf_bz"],0)
    df["by_hemi"]=df["imf_by"]*np.where(df["hemisphere"]=="N",1,-1)
    return df

def main():
    df=load()
    feats=BASE+sorted([c for c in df.columns if any(s in c for s in WIN) and c not in BASE])
    need=feats+['abs_eq_mlat','ae_index','hemisphere','date','year','satellite']
    df=df[df[[c for c in need if c in df.columns]].notna().all(axis=1)].reset_index(drop=True)
    X=df[feats].values.astype(np.float32); y=df['abs_eq_mlat'].values.astype(np.float32)
    R={'n_rows':int(len(df))}; print("rows",len(df))

    # (#7) leave-one-satellite-out
    sats=sorted(df['satellite'].unique()); loso={}
    for st in sats:
        tr=df['satellite'].values!=st; te=~tr
        if te.sum()<50: continue
        m=XGBRegressor(**XGB).fit(X[tr],y[tr]); loso[st]=dict(MAE=round(mae(y[te],m.predict(X[te])),4),n=int(te.sum()))
        print("LOSO",st,loso[st])
    vals=[v['MAE'] for v in loso.values()]
    R['leave_one_sat_out']=dict(per_sat=loso,mean=round(float(np.mean(vals)),4),std=round(float(np.std(vals)),4),
                                min=round(min(vals),4),max=round(max(vals),4))
    print("LOSO mean",R['leave_one_sat_out']['mean'],"std",R['leave_one_sat_out']['std'])

    # (#5) LOYO all-years vs >=100
    yr=df['year'].values; all_maes={};
    for Y in sorted(set(yr)):
        tr=yr!=Y; te=yr==Y
        m=XGBRegressor(**XGB).fit(X[tr],y[tr]); all_maes[int(Y)]=dict(MAE=round(mae(y[te],m.predict(X[te])),4),n=int(te.sum()))
    ge100=[v['MAE'] for v in all_maes.values() if v['n']>=100]
    allv=[v['MAE'] for v in all_maes.values()]
    R['LOYO_sensitivity']=dict(mean_ge100=round(float(np.mean(ge100)),4),std_ge100=round(float(np.std(ge100)),4),n_ge100=len(ge100),
                               mean_all=round(float(np.mean(allv)),4),std_all=round(float(np.std(allv)),4),n_all=len(allv),
                               per_year=all_maes)
    print("LOYO >=100",R['LOYO_sensitivity']['mean_ge100'],"all",R['LOYO_sensitivity']['mean_all'])

    # (#8) day-grouped over seeds
    groups=pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').values; dg=[]
    for seed in range(8):
        gss=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=seed)
        gtr,gte=next(gss.split(X,y,groups))
        m=XGBRegressor(**XGB).fit(X[gtr],y[gtr]); dg.append(mae(y[gte],m.predict(X[gte])))
    R['day_grouped_seeds']=dict(mean=round(float(np.mean(dg)),4),std=round(float(np.std(dg)),4),
                                min=round(min(dg),4),max=round(max(dg),4),n_seeds=8)
    print("day-grouped seeds",R['day_grouped_seeds'])
    json.dump(R,open(OUT,'w'),indent=1); print("saved",OUT)

if __name__=="__main__": main()
