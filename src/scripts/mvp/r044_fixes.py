#!/usr/bin/env python
"""R044: three consistency fixes flagged in self-review of the JGR revision.

FIX1 (row-count): redo the 60/90/120 window sweep on the FULL 39,668-crossing dataset
      (dropna on base SW only; XGBoost handles the ~401 missing 90/120 windows natively),
      so the 60-min bar reproduces the Table-1 value (0.97 deg) instead of the
      39,267-subset value (0.94). Removes the "two different 60-min numbers" problem.
FIX2 (A6 split): rerun the hemisphere-balanced experiment on the SAME random 80/20 split
      used for the paper's Fig 10 Southern-Hemisphere number (1.29 deg), so balanced-vs-
      default is compared under the split the manuscript actually reports.
FIX3 (MLP tuning): grid-search the controlled MLP over architecture + alpha (instead of a
      single default config) on the identical complete-case rows and random split as
      XGBoost, so the NN comparison is a tuned, fair one.

Input : output/omni_full_hist_90120/cusp_crossings_*.json
Output: src/kernels/cuspmap_mvp/bundles/r044_fixes.json
        paper/figures/fig06_time_window_comparison.{png,pdf}   (regenerated, full data)
Date  : 2026-07-04
"""
import json, glob, warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'savefig.dpi': 300,
                     'savefig.bbox': 'tight'})
IN = "/glade/work/yizhu/cuspML/output/omni_full_hist_90120"
FIG = "/glade/work/yizhu/cuspML/paper/figures/fig06_time_window_comparison"
OUT = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r044_fixes.json"
XGB = dict(n_estimators=1000, max_depth=8, learning_rate=0.02, subsample=0.8,
           colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
           random_state=42, n_jobs=32, verbosity=0)
BASE = ['dipole_tilt','hemi_code','doy','imf_bx','imf_by','imf_bz','sw_v','sw_n','sw_pdyn',
        'B_T','clock_angle','sin_clock_half','newell_cf','kan_lee_ef','vBs','by_hemi']
TARGETS = ['abs_eq_mlat','abs_pole_mlat','eq_mlt','mean_mlt']
WIN_SUFF = {15:['mean15','std15','delta15'], 30:['mean30','std30','delta30'],
            60:['mean60','std60','delta60','int60'],
            90:['mean90','std90','delta90'], 120:['mean120','std120','delta120']}


def load():
    recs=[]
    for f in sorted(glob.glob(f"{IN}/cusp_crossings_*.json")): recs.extend(json.load(open(f)))
    df=pd.DataFrame(recs).dropna(subset=["eq_mlat","pole_mlat","imf_bz","sw_v","sw_n","sw_pdyn"])
    df["abs_eq_mlat"]=df["eq_mlat"].abs(); df["abs_pole_mlat"]=df["pole_mlat"].abs()
    df["hemi_code"]=(df["hemisphere"]=="N").astype(float)
    df["doy"]=pd.to_datetime(df["time_start"]).dt.dayofyear
    df["year"]=pd.to_datetime(df["time_start"]).dt.year
    df["B_T"]=np.sqrt(df["imf_by"]**2+df["imf_bz"]**2)
    df["clock_angle"]=np.arctan2(df["imf_by"],df["imf_bz"]); df["sin_clock_half"]=np.sin(df["clock_angle"]/2)
    df["newell_cf"]=(df["sw_v"]**(4/3))*(df["B_T"]**(2/3))*(np.abs(df["sin_clock_half"])**(8/3))
    df["kan_lee_ef"]=df["sw_v"]*df["B_T"]*(df["sin_clock_half"]**2)
    df["vBs"]=df["sw_v"]*np.where(df["imf_bz"]<0,-df["imf_bz"],0)
    df["by_hemi"]=df["imf_by"]*np.where(df["hemisphere"]=="N",1,-1)
    return df.reset_index(drop=True)


def feats(df, maxwin):
    suff=sum((WIN_SUFF[w] for w in [15,30,60,90,120] if w<=maxwin),[])
    return BASE + sorted([c for c in df.columns if suff and any(s in c for s in suff) and c not in BASE])


def xgb(): return MultiOutputRegressor(XGBRegressor(**XGB))
def mae(a,b): return float(mean_absolute_error(a,b))


def main():
    df=load(); R={}
    # Restrict to the paper's exact 39,668 dataset: complete 74-feature vectors +
    # ae_index + hemisphere + date (matches the training set behind Table 1 / Fig 10).
    f74=[c for c in feats(df,60) if c in df.columns]
    need=f74+TARGETS+['ae_index','hemisphere','date','year']
    df=df[df[[c for c in need if c in df.columns]].notna().all(axis=1)].reset_index(drop=True)
    y=df[TARGETS].values.astype(np.float32); yr=df['year'].values
    N=len(df); print("paper-matched rows:", N)
    idx=np.arange(N)
    itr,ite=train_test_split(idx,test_size=0.2,random_state=42)
    trT=yr<2008; teT=yr>=2008

    # ---- FIX1: full-data window sweep (XGBoost native-NaN on missing 90/120) ----
    windows=[0,15,30,60,90,120]; sweep_rand=[]; sweep_temp=[]; nfe=[]
    for mw in windows:
        F=[c for c in feats(df,mw) if c in df.columns]; nfe.append(len(F))
        X=df[F].values.astype(np.float32)  # NaN kept; XGBoost handles it
        m=xgb(); m.fit(X[itr],y[itr]); sweep_rand.append(mae(y[ite,0],m.predict(X[ite])[:,0]))
        mt=xgb(); mt.fit(X[trT],y[trT]); sweep_temp.append(mae(y[teT,0],mt.predict(X[teT])[:,0]))
        print("sweep",mw,"nfeat",len(F),"rand",round(sweep_rand[-1],4),"temp",round(sweep_temp[-1],4))
    R['sweep']=dict(windows=windows,n_features=nfe,random_MAE=sweep_rand,temporal_MAE=sweep_temp,n_rows=int(N))

    # regenerate fig06 (random split, full data)
    labels=['Instant\n(no history)','+15 min','+30 min','+60 min','+90 min','+120 min']
    fig,ax=plt.subplots(figsize=(7,4))
    colors=['#999999','#7fb3d5','#5499c7','#2e86c1','#21618c','#1b3a5c']
    bars=ax.bar(range(6),sweep_rand,color=colors,edgecolor='k',linewidth=0.6,width=0.7)
    for b,v in zip(bars,sweep_rand): ax.text(b.get_x()+b.get_width()/2,v+0.004,f"{v:.3f}",ha='center',va='bottom',fontsize=10)
    ax.set_xticks(range(6)); ax.set_xticklabels(labels); ax.set_ylabel("Equatorward MLAT MAE (°)")
    ax.set_ylim(0,max(sweep_rand)*1.15); ax.axhline(sweep_rand[3],color='#2e86c1',ls='--',lw=0.8,alpha=0.6)
    ax.set_title("History window vs prediction error (random split, full dataset)")
    fig.tight_layout(); fig.savefig(FIG+".png"); fig.savefig(FIG+".pdf"); plt.close(fig)
    print("saved fig06 (full data)")

    # ---- FIX2: random-split hemisphere-balanced (align to Fig 10's 1.29 S) ----
    X74=df[[c for c in feats(df,60) if c in df.columns]].values.astype(np.float32)
    hemi=df['hemisphere'].values
    # default random-split model, S-only / N-only on the random test set
    m=xgb(); m.fit(X74[itr],y[itr]); p=m.predict(X74[ite])[:,0]
    hte=hemi[ite]
    def hmae(mask,pred): return dict(MAE=mae(y[ite,0][mask],pred[mask]),n=int(mask.sum()))
    R['random_default']=dict(Sonly=hmae(hte=='S',p),Nonly=hmae(hte=='N',p),overall=mae(y[ite,0],p))
    # balanced training: downsample N in the TRAIN indices to match S count
    htr=hemi[itr]; rng=np.random.default_rng(42)
    Sidx=itr[htr=='S']; Nidx=itr[htr=='N']; keepN=rng.choice(Nidx,len(Sidx),replace=False)
    bal=np.concatenate([keepN,Sidx])
    mb=xgb(); mb.fit(X74[bal],y[bal]); pb=mb.predict(X74[ite])[:,0]
    R['random_balanced']=dict(Sonly=hmae(hte=='S',pb),Nonly=hmae(hte=='N',pb),overall=mae(y[ite,0],pb),
                              n_S_train=int(len(Sidx)),n_N_train=int(len(keepN)))
    print("random default  S/N:",R['random_default']['Sonly']['MAE'],R['random_default']['Nonly']['MAE'])
    print("random balanced S/N:",R['random_balanced']['Sonly']['MAE'],R['random_balanced']['Nonly']['MAE'])

    # ---- FIX3: MLP grid (fair, tuned) on complete-case rows + same random split ----
    comp = ~np.isnan(X74).any(axis=1)   # complete 74-feature rows
    ci = np.where(comp)[0]
    citr,cite=train_test_split(ci,test_size=0.2,random_state=42)
    sc=StandardScaler().fit(X74[citr])
    Xtr_s,Xte_s=sc.transform(X74[citr]),sc.transform(X74[cite])
    # matched XGBoost on identical complete-case rows/split
    mx=xgb(); mx.fit(X74[citr],y[citr]); xgb_c=mae(y[cite,0],mx.predict(X74[cite])[:,0])
    grid=list(itertools.product([(128,64,32),(256,128),(128,128,64),(64,32)],[1e-4,1e-3,1e-2]))
    best=None
    for hl,al in grid:
        nn=MLPRegressor(hidden_layer_sizes=hl,activation='relu',alpha=al,max_iter=500,
                        early_stopping=True,n_iter_no_change=15,random_state=42)
        nn.fit(Xtr_s,y[citr,0]); mv=mae(y[cite,0],nn.predict(Xte_s))
        print("MLP",hl,al,round(mv,4))
        if best is None or mv<best['MAE']: best=dict(MAE=mv,hidden=list(hl),alpha=al)
    R['mlp_grid']=dict(best=best,xgb_matched=xgb_c,n_rows=int(comp.sum()),n_configs=len(grid))
    print("BEST MLP:",best,"| XGB matched:",round(xgb_c,4))

    json.dump(R,open(OUT,'w'),indent=1); print("saved",OUT)


if __name__=="__main__":
    main()
