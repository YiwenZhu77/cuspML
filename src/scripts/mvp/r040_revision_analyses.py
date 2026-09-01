#!/usr/bin/env python
"""R040: JGR revision analyses for the 1D paper (reviewer responses).
A2 storm/day-grouped split (R1-M1) | A3 controlled NN vs XGBoost same data (R1-M4)
A4 residual diagnostics fig (R1-m3) | A5 SSPB metric Morley2018 (R1-m7)
A6 hemisphere S-only + balanced (R2-M5,R1-m2) | temporal-holdout + LOYO primary metrics.
Date: 2026-07-04
"""
import json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def load():
    recs=[]
    for f in sorted(glob.glob("/glade/work/yizhu/cuspML/output/omni_full_hist/cusp_crossings_*.json")):
        recs.extend(json.load(open(f)))
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
    return df

df=load()
base=['dipole_tilt','hemi_code','doy','imf_bx','imf_by','imf_bz','sw_v','sw_n','sw_pdyn','B_T','clock_angle','sin_clock_half','newell_cf','kan_lee_ef','vBs','by_hemi']
hist=sorted([c for c in df.columns if any(s in c for s in ['mean15','mean30','mean60','std15','std30','std60','delta15','delta30','delta60','int60']) and c not in base])
feats=base+hist; targets=['abs_eq_mlat','abs_pole_mlat','eq_mlt','mean_mlt']
keep=list(dict.fromkeys(feats+targets+['ae_index','year','satellite','hemisphere','date']))
dfc=df[[c for c in keep if c in df.columns]].dropna().reset_index(drop=True)
feats=[c for c in feats if c in dfc.columns]
X=dfc[feats].values.astype(np.float32); y=dfc[targets].values.astype(np.float32)
print(f"n={len(X)} feats={len(feats)}")
XGB=dict(n_estimators=1000,max_depth=8,learning_rate=0.02,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=1.0,min_child_weight=5,random_state=42,n_jobs=32,verbosity=0)
def xgb(): return MultiOutputRegressor(XGBRegressor(**XGB))
def sspb(obs,pred):  # Morley 2018 symmetric signed percentage bias, on positive lat
    q=np.log(pred/obs); m=np.median(q)
    return float(100*np.sign(m)*(np.exp(abs(m))-1))
def metrics(obs,pred):
    return dict(MAE=float(mean_absolute_error(obs,pred)),RMSE=float(np.sqrt(mean_squared_error(obs,pred))),
                r=float(np.corrcoef(obs,pred)[0,1]),SSPB=sspb(obs,pred))
R={}

# --- random split baseline (eq MLAT = target 0) ---
Xtr,Xte,ytr,yte,itr,ite=train_test_split(X,y,np.arange(len(X)),test_size=0.2,random_state=42)
m=xgb(); m.fit(Xtr,ytr); pr=m.predict(Xte)
R['random_split']=metrics(yte[:,0],pr[:,0])
print("random:",R['random_split'])

# --- A2: day-grouped (storm/interval-independent) split ---
groups=pd.to_datetime(dfc['date']).dt.strftime('%Y-%m-%d').values
gss=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
gtr,gte=next(gss.split(X,y,groups))
mg=xgb(); mg.fit(X[gtr],y[gtr]); pg=mg.predict(X[gte])
R['day_grouped_split']=metrics(y[gte,0],pg[:,0]); R['day_grouped_split']['n_train_days']=int(len(set(groups[gtr]))); R['day_grouped_split']['n_test_days']=int(len(set(groups[gte])))
print("day-grouped:",R['day_grouped_split'])

# --- temporal holdout (train pre-2008, test 2008-2014) ---
yr=dfc['year'].values; tr=yr<2008; te=yr>=2008
mt=xgb(); mt.fit(X[tr],y[tr]); pt=mt.predict(X[te])
R['temporal_holdout']=metrics(y[te,0],pt[:,0]); R['temporal_holdout']['n_train']=int(tr.sum()); R['temporal_holdout']['n_test']=int(te.sum())
print("temporal:",R['temporal_holdout'])

# --- LOYO ---
loyo=[]
for Y in sorted(set(yr)):
    tr2=yr!=Y; te2=yr==Y
    if te2.sum()<50: continue
    ml=xgb(); ml.fit(X[tr2],y[tr2]); pl=ml.predict(X[te2])
    loyo.append(mean_absolute_error(y[te2,0],pl[:,0]))
R['LOYO']=dict(MAE=float(np.mean(loyo)),MAE_std=float(np.std(loyo)),n_years=len(loyo))
print("LOYO:",R['LOYO'])

# --- A3: controlled NN vs XGBoost on SAME full data + same random split ---
sc=StandardScaler().fit(Xtr)
nn=MLPRegressor(hidden_layer_sizes=(128,64,32),activation='relu',alpha=1e-3,max_iter=300,early_stopping=True,random_state=42)
nn.fit(sc.transform(Xtr),ytr[:,0]); pnn=nn.predict(sc.transform(Xte))
R['controlled_NN']=metrics(yte[:,0],pnn); R['controlled_NN']['note']='same 39668 data + same split as XGBoost'
R['controlled_XGB_eqmlat']=R['random_split']
print("NN (full,controlled):",R['controlled_NN'])

# --- A6: hemisphere S-only (temporal holdout test) + balanced training ---
hemi_te=dfc['hemisphere'].values[te]
for h in ['N','S']:
    mh=hemi_te==h
    if mh.sum()>20: R[f'temporal_{h}only']=metrics(y[te,0][mh],pt[:,0][mh]); R[f'temporal_{h}only']['n']=int(mh.sum())
# balanced: downsample N in training to match S count, retrain
tri=np.where(tr)[0]; hemi_tr=dfc['hemisphere'].values[tri]
nN=(hemi_tr=='N').sum(); nS=(hemi_tr=='S').sum()
rng=np.random.default_rng(42)
Nidx=tri[hemi_tr=='N']; keepN=rng.choice(Nidx,min(nS,nN)*1,replace=False) if nS<nN else Nidx
bal=np.concatenate([keepN,tri[hemi_tr=='S']])
mb=xgb(); mb.fit(X[bal],y[bal]); pb=mb.predict(X[te])
R['balanced_train']=dict(overall=metrics(y[te,0],pb[:,0]),
    Nonly=metrics(y[te,0][hemi_te=='N'],pb[:,0][hemi_te=='N']),
    Sonly=metrics(y[te,0][hemi_te=='S'],pb[:,0][hemi_te=='S']),
    n_train=int(len(bal)),n_N_train=int(len(keepN)),n_S_train=int(nS))
print("balanced:",{k:(v.get('MAE') if isinstance(v,dict) else v) for k,v in R['balanced_train'].items()})

# --- A4: residual diagnostics figure (temporal-holdout test) ---
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
resid=pt[:,0]-y[te,0]  # pred - obs, eq MLAT
dte=dfc.iloc[np.where(te)[0]]
fig,ax=plt.subplots(2,2,figsize=(9,7))
def panel(a,x,xl):
    a.scatter(x,resid,s=3,alpha=0.15,c='steelblue'); a.axhline(0,c='k',lw=0.8)
    bins=np.linspace(np.nanpercentile(x,1),np.nanpercentile(x,99),12); bc=(bins[:-1]+bins[1:])/2
    md=[np.median(resid[(x>=bins[i])&(x<bins[i+1])]) for i in range(len(bc))]
    a.plot(bc,md,'r-o',ms=4,lw=1.5); a.set_xlabel(xl,fontsize=11); a.set_ylabel('residual (pred-obs) [deg]',fontsize=10); a.set_ylim(-6,6)
panel(ax[0,0],pt[:,0],'predicted eq MLAT [deg]')
panel(ax[0,1],dte['newell_cf'].values,'Newell CF (SW driving)')
panel(ax[1,0],dte['dipole_tilt'].values,'dipole tilt [deg]')
panel(ax[1,1],dte['ae_index'].values,'AE index [nT]')
fig.suptitle('Residual diagnostics (temporal holdout): eq-MLAT prediction error vs drivers',fontsize=12)
fig.tight_layout(); fig.savefig('/glade/work/yizhu/cuspML/paper/figures/figR1_residual_diagnostics.png',dpi=200,bbox_inches='tight')
fig.savefig('/glade/work/yizhu/cuspML/paper/figures/figR1_residual_diagnostics.pdf',bbox_inches='tight')
print("saved residual diagnostics fig")

json.dump(R,open('/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r040_revision_metrics.json','w'),indent=1)
print("saved r040_revision_metrics.json")
