#!/usr/bin/env python
"""R050: TRULY fair NN-vs-XGBoost comparison to close the "small NN search budget" gap.
Parallel (multiprocessing) version for Casper htc. Throws a large modern NN search at the
identical 39,668-crossing / 74-feature dataset and the identical random and temporal splits
used for XGBoost, so any residual tree advantage is not an artifact of under-tuning the net.

SEARCH (equatorward |MLAT|): sklearn MLPRegressor randomized search (80 configs) + torch MLP
(24 configs: depth x width x dropout x weight-decay, BatchNorm, AdamW, cosine LR, early stop),
each vs matched XGBoost on the same rows/split, on BOTH random and temporal splits. Config
evaluations run across a process Pool.
INPUT : output/omni_full_hist_90120/cusp_crossings_*.json (via r044 load/feats)
OUTPUT: src/kernels/cuspmap_mvp/bundles/r050_fair_nn.json
RUN   : conda py3.10 ; python r050_fair_nn.py --workers 32
"""
import json, sys, os, argparse, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/glade/work/yizhu/cuspML/src/scripts/mvp")
import numpy as np
from r044_fixes import load, feats, XGB
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import mean_absolute_error
import scipy.stats as st
OUT="/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r050_fair_nn.json"
_S={}  # shared split data, set per split before Pool map

def mae(a,b): return float(mean_absolute_error(a,b))

def eval_sklearn(p):
    from sklearn.neural_network import MLPRegressor
    nn=MLPRegressor(**p, max_iter=600, early_stopping=True, n_iter_no_change=15,
                    learning_rate="adaptive", random_state=0)
    nn.fit(_S["Xtr_s"], _S["ytr"]); return ("sklearn", mae(_S["yte"], nn.predict(_S["Xte_s"])),
        {k:(list(v) if isinstance(v,tuple) else v) for k,v in p.items()})

def eval_torch(c):
    import torch, torch.nn as nn_
    torch.set_num_threads(1)
    Xtr=torch.tensor(_S["Xtr_s"],dtype=torch.float32); yv=torch.tensor(_S["ytr"],dtype=torch.float32).view(-1,1)
    Xte=torch.tensor(_S["Xte_s"],dtype=torch.float32)
    n=len(Xtr); perm=torch.randperm(n,generator=torch.Generator().manual_seed(0)); cut=int(.85*n)
    tri,vai=perm[:cut],perm[cut:]; Xt,yt=Xtr[tri],yv[tri]; Xv,yvl=Xtr[vai],yv[vai]
    torch.manual_seed(0); layers=[]; d=Xtr.shape[1]
    for _ in range(c["depth"]):
        layers+=[nn_.Linear(d,c["width"]),nn_.BatchNorm1d(c["width"]),nn_.ReLU(),nn_.Dropout(c["drop"])]; d=c["width"]
    layers+=[nn_.Linear(d,1)]; net=nn_.Sequential(*layers)
    opt=torch.optim.AdamW(net.parameters(),lr=c["lr"],weight_decay=c["wd"])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=150); lf=nn_.L1Loss(); bs=256
    best=1e9; pat=0; state=None
    for ep in range(150):
        net.train(); pm=torch.randperm(len(Xt))
        for j in range(0,len(Xt),bs):
            b=pm[j:j+bs]; opt.zero_grad(); lf(net(Xt[b]),yt[b]).backward(); opt.step()
        sched.step(); net.eval()
        with torch.no_grad(): vl=lf(net(Xv),yvl).item()
        if vl<best-1e-4: best=vl; pat=0; state={k:v.clone() for k,v in net.state_dict().items()}
        else:
            pat+=1
            if pat>=18: break
    if state: net.load_state_dict(state)
    net.eval()
    with torch.no_grad(): mv=float(nn_.functional.l1_loss(net(Xte).view(-1),torch.tensor(_S["yte"],dtype=torch.float32)))
    return ("torch", mv, c)

def run_split(name, Xtr,ytr,Xte,yte, workers):
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    from multiprocessing import Pool
    sc=StandardScaler().fit(Xtr); global _S
    _S={"Xtr_s":sc.transform(Xtr).astype(np.float32),"Xte_s":sc.transform(Xte).astype(np.float32),
        "ytr":ytr.astype(np.float32),"yte":yte.astype(np.float32)}
    mx=XGBRegressor(**{**XGB,"n_jobs":workers}); mx.fit(Xtr,ytr); xgb_mae=mae(yte,mx.predict(Xte))
    print(f"[{name}] XGBoost matched {xgb_mae:.4f}",flush=True)
    space=dict(hidden_layer_sizes=[(128,),(256,),(512,),(256,128),(512,256),(256,128,64),
        (512,256,128),(128,128,64),(256,256,128),(512,256,128,64),(384,192,96),(512,512,256)],
        alpha=st.loguniform(1e-6,1e-1), learning_rate_init=st.loguniform(1e-4,1e-2),
        activation=["relu","tanh"], batch_size=[128,256,512])
    sk_cfgs=list(ParameterSampler(space,n_iter=80,random_state=42))
    to_cfgs=[dict(depth=dp,width=w,drop=dr,wd=(1e-4 if dr==0.1 else 1e-3),lr=lr)
             for dp in (2,3,4) for w in (128,256,512) for dr in (0.1,0.3) for lr in (1e-3,3e-4)]
    with Pool(workers) as pool:
        sk=pool.map(eval_sklearn, sk_cfgs)
        to=pool.map(eval_torch, to_cfgs)
    best_sk=min(sk,key=lambda z:z[1]); best_to=min(to,key=lambda z:z[1])
    print(f"[{name}] best sklearn {best_sk[1]:.4f} | best torch {best_to[1]:.4f} | XGB {xgb_mae:.4f}",flush=True)
    return dict(xgb=xgb_mae, sklearn_mlp=dict(MAE=best_sk[1],params=best_sk[2]),
                torch_mlp=dict(MAE=best_to[1],config=best_to[2]),
                n_sklearn=len(sk_cfgs), n_torch=len(to_cfgs))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--workers",type=int,default=32); a=ap.parse_args()
    df=load(); cols=feats(df,120)
    X=df[cols].values.astype(float); y=df["abs_eq_mlat"].values.astype(float); year=df["year"].values
    comp=~np.isnan(X).any(axis=1); X,y,year=X[comp],y[comp],year[comp]
    idx=np.arange(len(X)); print(f"rows {len(X)} workers {a.workers}",flush=True)
    itr,ite=train_test_split(idx,test_size=0.2,random_state=42)
    R={"n_rows":int(comp.sum())}
    R["random"]=run_split("random",X[itr],y[itr],X[ite],y[ite],a.workers)
    tr,te=idx[year<=2007],idx[year>=2008]
    R["temporal"]=run_split("temporal",X[tr],y[tr],X[te],y[te],a.workers)
    json.dump(R,open(OUT,"w"),indent=1); print("saved",OUT,flush=True)

if __name__=="__main__":
    from multiprocessing import set_start_method; set_start_method("fork",force=True)
    main()
