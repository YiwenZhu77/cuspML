#!/usr/bin/env python
"""R053: robustness of the model/observational error decomposition.
Defense soft spot: the irreducible-floor estimate sigma_wg (within-group label scatter for
crossings sharing the same day and 0.5-hr MLT bin) and the implied model error
sqrt(RMSE^2 - sigma_wg^2) depend on the group definition and an independence assumption.
Here we recompute sigma_wg and the implied floor under several group definitions (MLT bin
width in {0.25,0.5,1.0} hr; time key in {same UT day, same 3-hr block}) to show the ~1-deg
floor is not an artifact of one binning choice.

INPUT : output/omni_full_hist_90120/cusp_crossings_*.json (via r044 load/feats)
OUTPUT: src/kernels/cuspmap_mvp/bundles/r053_errordecomp.json (+ printed table)
RUN   : conda py3.10 ; OMP_NUM_THREADS=6 python r053_errordecomp_robust.py
"""
import json, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/glade/work/yizhu/cuspML/src/scripts/mvp")
import numpy as np, pandas as pd
from r044_fixes import load, feats, XGB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
OUT="/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r053_errordecomp.json"
def main():
    df=load(); cols=feats(df,120)
    X=df[cols].values.astype(float); y=df["abs_eq_mlat"].values.astype(float)
    comp=~np.isnan(X).any(axis=1)
    dfc=df[comp].reset_index(drop=True); X,y=X[comp],y[comp]
    itr,ite=train_test_split(np.arange(len(X)),test_size=0.2,random_state=42)
    m=XGBRegressor(**{**XGB,"n_jobs":6}); m.fit(X[itr],y[itr])
    rmse_rand=float(np.sqrt(mean_squared_error(y[ite],m.predict(X[ite]))))
    t=pd.to_datetime(dfc["time_start"]); mlt=dfc["eq_mlt"].values
    R={"rmse_random":rmse_rand,"groupings":[]}
    print(f"random-split RMSE = {rmse_rand:.3f}",flush=True)
    for tkey,tname in [(t.dt.floor("D").astype(str).values,"day"),(t.dt.floor("3h").astype(str).values,"3hr")]:
        for w in (0.25,0.5,1.0):
            mb=np.floor(mlt/w).astype(int)
            key=pd.Series([f"{a}_{b}" for a,b in zip(tkey,mb)])
            g=pd.DataFrame({"k":key,"y":y}).groupby("k")["y"]
            sizes=g.size(); multi=sizes[sizes>=2].index
            sig=float(pd.DataFrame({"k":key,"y":y}).groupby("k")["y"].std().reindex(multi).mean())
            ngrp=len(multi); ncross=int(sizes[sizes>=2].sum())
            floor=float(np.sqrt(max(rmse_rand**2-sig**2,0)))
            R["groupings"].append(dict(time=tname,mlt_bin_hr=w,sigma_wg=sig,n_groups=ngrp,
                                       n_crossings=ncross,implied_model_floor=floor))
            print(f"  time={tname:3} mlt_bin={w}hr: sigma_wg={sig:.3f} (n_grp={ngrp}) -> implied model floor={floor:.3f}",flush=True)
    sigs=[gg["sigma_wg"] for gg in R["groupings"]]
    R["sigma_wg_range"]=[float(min(sigs)),float(max(sigs))]
    print(f"sigma_wg spans {min(sigs):.3f}-{max(sigs):.3f} deg across group definitions",flush=True)
    json.dump(R,open(OUT,"w"),indent=1); print("saved",OUT)
if __name__=="__main__": main()
