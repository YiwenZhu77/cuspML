#!/usr/bin/env python
"""R046: matched-split baseline ladder + contiguous-block split + bootstrap significance.

Addresses round-2 external review:
 (#1/#7) Evaluate EVERY baseline (Newell nonlinear, Ridge-74, GBR-300, XGBoost) on the
         SAME split, for the random, temporal-holdout, AND a contiguous-time-block split,
         so percentage improvements are computed on matched train/test partitions.
 (#3)    Add a contiguous-block split (5 equal contiguous time chunks, leave-one-block-out)
         as a stronger interval/storm-independence control than the day-grouped split.
 (#2)    Bootstrap the equatorward-MLAT test MAE for the 60/90/120-min models to test
         whether the sub-0.03 deg gains beyond 60 min are statistically distinguishable.

Input : output/omni_full_hist_90120/cusp_crossings_*.json (restricted to the 39,668 set)
Output: src/kernels/cuspmap_mvp/bundles/r046_matched.json
"""
import json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

IN="/glade/work/yizhu/cuspML/output/omni_full_hist_90120"
OUT="/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r046_matched.json"
XGB=dict(n_estimators=1000,max_depth=8,learning_rate=0.02,subsample=0.8,colsample_bytree=0.7,
         reg_alpha=0.1,reg_lambda=1.0,min_child_weight=5,random_state=42,n_jobs=32,verbosity=0)
BASE=['dipole_tilt','hemi_code','doy','imf_bx','imf_by','imf_bz','sw_v','sw_n','sw_pdyn',
      'B_T','clock_angle','sin_clock_half','newell_cf','kan_lee_ef','vBs','by_hemi']
WIN_SUFF={15:['mean15','std15','delta15'],30:['mean30','std30','delta30'],
          60:['mean60','std60','delta60','int60'],90:['mean90','std90','delta90'],120:['mean120','std120','delta120']}
mae=lambda a,b: float(mean_absolute_error(a,b))

def load():
    recs=[]
    for f in sorted(glob.glob(f"{IN}/cusp_crossings_*.json")): recs.extend(json.load(open(f)))
    df=pd.DataFrame(recs).dropna(subset=["eq_mlat","pole_mlat","imf_bz","sw_v","sw_n","sw_pdyn"])
    df["abs_eq_mlat"]=df["eq_mlat"].abs(); df["abs_pole_mlat"]=df["pole_mlat"].abs()
    df["hemi_code"]=(df["hemisphere"]=="N").astype(float)
    df["t"]=pd.to_datetime(df["time_start"])
    df["doy"]=df["t"].dt.dayofyear; df["year"]=df["t"].dt.year
    df["B_T"]=np.sqrt(df["imf_by"]**2+df["imf_bz"]**2)
    df["clock_angle"]=np.arctan2(df["imf_by"],df["imf_bz"]); df["sin_clock_half"]=np.sin(df["clock_angle"]/2)
    df["newell_cf"]=(df["sw_v"]**(4/3))*(df["B_T"]**(2/3))*(np.abs(df["sin_clock_half"])**(8/3))
    df["kan_lee_ef"]=df["sw_v"]*df["B_T"]*(df["sin_clock_half"]**2)
    df["vBs"]=df["sw_v"]*np.where(df["imf_bz"]<0,-df["imf_bz"],0)
    df["by_hemi"]=df["imf_by"]*np.where(df["hemisphere"]=="N",1,-1)
    return df

def feats(df,mw):
    suff=sum((WIN_SUFF[w] for w in [15,30,60,90,120] if w<=mw),[])
    return BASE+sorted([c for c in df.columns if suff and any(s in c for s in suff) and c not in BASE])

def newell_base(ytr, cf_tr, cf_te):
    # Lambda = a * newell_cf^(2/3) + b, least squares on train
    x=(cf_tr**(2/3)); A=np.vstack([x,np.ones_like(x)]).T
    a,b=np.linalg.lstsq(A,ytr,rcond=None)[0]
    return a*(cf_te**(2/3))+b

def ladder(Xtr,Xte,ytr,yte,cf_tr,cf_te):
    r={}
    r['Newell']=mae(yte,newell_base(ytr,cf_tr,cf_te))
    sc=StandardScaler().fit(Xtr); rg=Ridge(alpha=1.0).fit(sc.transform(Xtr),ytr)
    r['Ridge74']=mae(yte,rg.predict(sc.transform(Xte)))
    gb=GradientBoostingRegressor(n_estimators=300,max_depth=5,random_state=42).fit(Xtr,ytr)
    r['GBR300']=mae(yte,gb.predict(Xte))
    xg=XGBRegressor(**XGB).fit(Xtr,ytr)
    r['XGBoost']=mae(yte,xg.predict(Xte))
    r['pct_reduction_vs_Newell']=round(100*(r['Newell']-r['XGBoost'])/r['Newell'],1)
    return {k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()}

def main():
    df=load()
    f74=feats(df,60); need=f74+['abs_eq_mlat','ae_index','hemisphere','date','year','newell_cf']
    df=df[df[[c for c in need if c in df.columns]].notna().all(axis=1)].sort_values('t').reset_index(drop=True)
    N=len(df); print("rows",N)
    X=df[f74].values.astype(np.float32); y=df['abs_eq_mlat'].values.astype(np.float32)
    cf=df['newell_cf'].values.astype(float); yr=df['year'].values
    R={'n_rows':int(N)}

    # --- random split ---
    itr,ite=train_test_split(np.arange(N),test_size=0.2,random_state=42)
    R['random']=ladder(X[itr],X[ite],y[itr],y[ite],cf[itr],cf[ite]); print("random",R['random'])
    # --- temporal holdout ---
    tr=yr<2008; te=yr>=2008
    R['temporal']=ladder(X[tr],X[te],y[tr],y[te],cf[tr],cf[te]); print("temporal",R['temporal'])
    # --- contiguous-block: 5 equal contiguous time blocks, leave-one-block-out ---
    blocks=np.floor(np.arange(N)/N*5).astype(int)  # data already time-sorted
    xgb_block=[]
    for k in range(5):
        tr_k=blocks!=k; te_k=blocks==k
        xg=XGBRegressor(**XGB).fit(X[tr_k],y[tr_k]); xgb_block.append(mae(y[te_k],xg.predict(X[te_k])))
    # full ladder on one representative block split (hold out last block = latest period)
    trL=blocks!=4; teL=blocks==4
    R['contig_block_holdout_last']=ladder(X[trL],X[teL],y[trL],y[teL],cf[trL],cf[teL])
    R['contig_block_xgb_LOBO']=dict(MAE_mean=round(float(np.mean(xgb_block)),4),
                                    MAE_std=round(float(np.std(xgb_block)),4),
                                    per_block=[round(x,4) for x in xgb_block])
    print("contig LOBO XGB",R['contig_block_xgb_LOBO'])
    print("contig last-block ladder",R['contig_block_holdout_last'])

    # --- bootstrap: 60 vs 90 vs 120 test MAE on random split ---
    preds={}
    for mw in [60,90,120]:
        F=[c for c in feats(df,mw) if c in df.columns]; Xm=df[F].values.astype(np.float32)
        xg=XGBRegressor(**XGB).fit(Xm[itr],y[itr]); preds[mw]=xg.predict(Xm[ite])
    yte=y[ite]; rng=np.random.default_rng(42); nb=2000; m=len(ite)
    d60_90=[]; d60_120=[]; mae60=[]
    for _ in range(nb):
        bi=rng.integers(0,m,m)
        e60=mean_absolute_error(yte[bi],preds[60][bi]); e90=mean_absolute_error(yte[bi],preds[90][bi]); e120=mean_absolute_error(yte[bi],preds[120][bi])
        mae60.append(e60); d60_90.append(e60-e90); d60_120.append(e60-e120)
    def ci(a): return [round(float(np.percentile(a,2.5)),4),round(float(np.percentile(a,97.5)),4)]
    R['bootstrap']=dict(nboot=nb,
        MAE60_CI=ci(mae60),
        delta_60_minus_90_CI=ci(d60_90), delta_60_minus_90_mean=round(float(np.mean(d60_90)),4),
        delta_60_minus_120_CI=ci(d60_120), delta_60_minus_120_mean=round(float(np.mean(d60_120)),4))
    print("bootstrap 60-90 dMAE mean/CI:",R['bootstrap']['delta_60_minus_90_mean'],R['bootstrap']['delta_60_minus_90_CI'])
    json.dump(R,open(OUT,'w'),indent=1); print("saved",OUT)

if __name__=="__main__": main()
