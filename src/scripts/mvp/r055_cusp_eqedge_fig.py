#!/usr/bin/env python
"""R055: clean 2-curve comparison — cuspML equatorward-MLAT (mapped to 4000 km) vs the OpOF
simulation's OWN cusp EMISSION equatorward boundary (from the O+ launch region at 4000 km),
per storm, time-resolved. No Zhang, no myfinder centre. Both are equatorward boundaries at
the same altitude (4000 km), so the comparison is apples-to-apples (removes the earlier
centre-vs-boundary and 840-vs-4000 km offsets).

SIM CUSP EQ BOUNDARY: the O+ launch tpInit (prod_<storm>_cusp_b*.tpInit, col5=lat_deg at
R=1.625 RE = 4000 km, col6=T0p launch time) IS the 1e8-gated cusp emission region; per launch-
time bin we take the equatorward-most edge (5th percentile of Northern launch latitude).
CUSPML: drive xgb_abs_eq_mlat with the storm bcwind SW (r054 pipeline), dipole-map 840->4000 km.
INPUT: OpOF sp13_rin15/jun0113_rin15 {bcwind.h5, prod_*_cusp_b*.tpInit} ; cuspML zenodo model.
OUTPUT: cuspML/figures/r055_cusp_eqedge.png + bundles/r055_cusp_eqedge.json
RUN: conda py3.10 ; python r055_cusp_eqedge_fig.py
"""
import json, sys, glob, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/glade/work/yizhu/cuspML/src/scripts/mvp")
import numpy as np, h5py
from datetime import datetime, timedelta
import xgboost as xgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from r044_fixes import BASE
CM="/glade/work/yizhu/cuspML"; OP="/glade/derecho/scratch/yizhu/OpOF"
FEATS=json.load(open(f"{CM}/output/zenodo_models/metadata.json"))["feature_names"]
RE=6371.0; ALT_FAC=np.sqrt((RE+4000)/(RE+840))
STORMS={"sp13":dict(dir="sp13_rin15",date="2013-03-17",tp="prod_sp13_cusp_b*.tpInit"),
        "jun0113":dict(dir="jun0113_rin15",date="2013-06-01",tp="prod_jun0113_cusp_b*.tpInit")}
def alt_map(l): return np.degrees(np.arccos(np.clip(ALT_FAC*np.cos(np.radians(l)),-1,1)))
def feat_order():
    import json as J
    recs=J.load(open(sorted(glob.glob(f"{CM}/output/omni_full_hist_90120/cusp_crossings_*.json"))[0]))
    import pandas as pd; df=pd.DataFrame(recs)
    suff=['mean15','std15','delta15','mean30','std30','delta30','mean60','std60','delta60','int60']
    return BASE+sorted([c for c in df.columns if any(s in c for s in suff)])
def cuspml_eq4000(bcf, cols, date):
    import pandas as pd
    d=h5py.File(bcf,'r'); g=lambda k:np.array(d[k],float)
    mjd=g('MJD'); tilt=g('tilt')*180/np.pi; v=np.sqrt(g('Vx')**2+g('Vy')**2+g('Vz')**2)/1000
    n=g('D'); bx,by,bz=g('Bx'),g('By'),g('Bz'); pdyn=1.6726e-6*n*v**2
    doy=float(pd.Timestamp(date).dayofyear)
    df=pd.DataFrame(dict(imf_bx=bx,imf_by=by,imf_bz=bz,sw_v=v,sw_n=n,sw_pdyn=pdyn,
        dipole_tilt=tilt,hemi_code=1.0,doy=doy))
    df['B_T']=np.sqrt(by**2+bz**2); df['clock_angle']=np.arctan2(by,bz); df['sin_clock_half']=np.sin(df.clock_angle/2)
    df['newell_cf']=v**(4/3)*df.B_T**(2/3)*np.abs(df.sin_clock_half)**(8/3)
    df['kan_lee_ef']=v*df.B_T*df.sin_clock_half**2; df['vBs']=v*np.where(bz<0,-bz,0); df['by_hemi']=by
    for w in (15,30,60):
        for b in ['imf_bx','imf_by','imf_bz','sw_v','sw_n','sw_pdyn']:
            s=df[b]; df[f'{b}_mean{w}']=s.rolling(w,min_periods=1).mean()
            df[f'{b}_std{w}']=s.rolling(w,min_periods=1).std().fillna(0)
            df[f'{b}_delta{w}']=(s-s.shift(w)).fillna(0)
    df['newell_cf_mean60']=df.newell_cf.rolling(60,min_periods=1).mean(); df['newell_cf_int60']=df.newell_cf.rolling(60,min_periods=1).sum()
    df['vBs_mean60']=df.vBs.rolling(60,min_periods=1).mean(); df['vBs_int60']=df.vBs.rolling(60,min_periods=1).sum()
    X=df[cols].values
    meq=xgb.XGBRegressor(); meq.load_model(f"{CM}/output/zenodo_models/xgb_abs_eq_mlat.json")
    mpo=xgb.XGBRegressor(); mpo.load_model(f"{CM}/output/zenodo_models/xgb_abs_pole_mlat.json")
    eq=meq.predict(X); po=mpo.predict(X)                   # cuspML eq + pole boundary (MLT 0830-1530)
    return (mjd-mjd[0])*24.0, alt_map(eq), alt_map(0.5*(eq+po))
def sim_boundaries(tpglob, binhr=0.5, eqpct=5, popct=95, mlt_lo=8.5, mlt_hi=15.5):
    """Sim cusp emission boundaries per T0p bin, on the SAME MLT sector cuspML is trained on
    (0830-1530 = MLT 8.5-15.5). MLT convention: noon at phi=0, MLT=(phi/15+12)%24 (empirically
    puts 80% of cusp launches in the dayside sector). Per 1-hr MLT bin: eq edge = eqpct-th pct,
    pole edge = popct-th pct of Northern launch latitude; centre = (eq+pole)/2. Average each over
    the MLT bins -> sector-mean eq boundary and sector-mean centre. Returns (tc, sim_eq, sim_ctr)."""
    fs=sorted(glob.glob(tpglob))
    if not fs: return None,None,None
    A=np.vstack([np.loadtxt(f) for f in fs]); lat=A[:,5]; phi=A[:,3]; T0=A[:,6]/3600.0
    N=lat>0; lat,phi,T0=lat[N],phi[N],T0[N]; mlt=(phi/15.0+12.0)%24
    sec=(mlt>=mlt_lo)&(mlt<=mlt_hi); lat,T0,mlt=lat[sec],T0[sec],mlt[sec]
    mbins=np.arange(np.floor(mlt_lo),np.ceil(mlt_hi),1.0)
    edges=np.arange(0,np.ceil(T0.max())+binhr,binhr); tc=[];eq=[];ctr=[]
    for i in range(len(edges)-1):
        m=(T0>=edges[i])&(T0<edges[i+1])
        if m.sum()<30: continue
        eqs=[];cts=[]
        for mb in mbins:
            mm=m&(mlt>=mb)&(mlt<mb+1)
            if mm.sum()<10: continue
            e=np.percentile(lat[mm],eqpct); p=np.percentile(lat[mm],popct)
            eqs.append(e); cts.append(0.5*(e+p))
        if not eqs: continue
        tc.append((edges[i]+edges[i+1])/2); eq.append(float(np.mean(eqs))); ctr.append(float(np.mean(cts)))
    return np.array(tc),np.array(eq),np.array(ctr)
def main():
    cols=feat_order(); assert len(cols)==74
    R={}; fig,ax=plt.subplots(2,2,figsize=(13,9))
    for i,(sk,sv) in enumerate(STORMS.items()):
        bcf=f"{OP}/{sv['dir']}/bcwind.h5"
        th,mleq,mlctr=cuspml_eq4000(bcf,cols,sv['date'])      # cuspML eq + centre @4000km
        tc,simeq,simctr=sim_boundaries(f"{OP}/{sv['dir']}/{sv['tp']}")
        R[sk]=dict(cuspml_eq_med=float(np.median(mleq)),cuspml_ctr_med=float(np.median(mlctr)))
        # ---- row 0: equatorward boundary ----
        a=ax[0,i]; a.plot(th,mleq,'-',color='#d62728',lw=1.6,label='cuspML eq-MLAT @4000km')
        if tc is not None:
            a.plot(tc,simeq,'o-',color='#1f77b4',lw=1.4,ms=4,label='sim cusp eq-MLAT (MLT 8.5-15.5 mean)')
            R[sk]['sim_eq_med']=float(np.median(simeq)); R[sk]['off_eq']=float(np.median(mleq)-np.median(simeq))
        a.set_title(f"{sk} ({sv['date']}) — equatorward boundary"); a.set_ylabel("eq-boundary MLAT (deg)")
        # ---- row 1: centre = (eq+pole)/2 ----
        b=ax[1,i]; b.plot(th,mlctr,'-',color='#d62728',lw=1.6,label='cuspML centre (eq+pole)/2 @4000km')
        if tc is not None:
            b.plot(tc,simctr,'o-',color='#1f77b4',lw=1.4,ms=4,label='sim cusp centre (MLT 8.5-15.5 mean)')
            R[sk]['sim_ctr_med']=float(np.median(simctr)); R[sk]['off_ctr']=float(np.median(mlctr)-np.median(simctr))
        b.set_title(f"{sk} ({sv['date']}) — centre (eq+pole avg)"); b.set_ylabel("centre MLAT (deg)")
        for a in (ax[0,i],ax[1,i]):
            a.set_xlabel("hours from run start"); a.set_ylim(60,82); a.legend(fontsize=8); a.grid(alpha=.3)
        print(f"{sk}: EQ cuspML {R[sk]['cuspml_eq_med']:.1f} sim {R[sk].get('sim_eq_med',float('nan')):.1f} off {R[sk].get('off_eq',float('nan')):+.1f}"
              f" | CTR cuspML {R[sk]['cuspml_ctr_med']:.1f} sim {R[sk].get('sim_ctr_med',float('nan')):.1f} off {R[sk].get('off_ctr',float('nan')):+.1f}",flush=True)
    fig.tight_layout(); fig.savefig(f"{CM}/figures/r055_cusp_eqedge.png",dpi=150)
    json.dump(R,open(f"{CM}/src/kernels/cuspmap_mvp/bundles/r055_cusp_eqedge.json","w"),indent=1); print("saved r055")
if __name__=="__main__": main()
