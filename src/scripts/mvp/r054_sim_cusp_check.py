#!/usr/bin/env python
"""R054: cross-check the OpOF simulated cusp equatorward MLAT (sp13 17-Mar-2013, jun0113
1-Jun-2013; regularized cusp mask, hard gate F_en>1e8) against the data-driven cuspML model.
Feeds each storm's real solar-wind driver (from the GAMERA bcwind.h5) into the trained cuspML
eq-MLAT XGBoost and compares the predicted equatorward boundary to the simulation's cusp finder.

PITFALL AVOIDED (jun0113 MJD): bcwind T=0 differs per storm; jun0113 = MJD 56443.5 (31 May 12:00),
NOT 56444.0 midnight (the 2026-07-11 12h bug). We read real time straight from bcwind['MJD'] and
dipole tilt straight from bcwind['tilt'] (RADIANS -> convert to degrees for cuspML).
UNITS: bcwind Vx/Vy/Vz m/s; D cm^-3; Bx/By/Bz nT; tilt RAD; MJD days. cuspML tilt in DEG.
INPUT : bcwind.h5 (sp13_rin15, jun0113_rin15); cuspML output/zenodo_models/xgb_abs_eq_mlat.json;
        sp13 sim cusp finder OpOF/src/kernels/sp13/cusp_center_compare.npz (myfinder center mlatN)
OUTPUT: src/kernels/cuspmap_mvp/bundles/r054_sim_cusp_check.json + figures/r054_sim_cusp_check.png
RUN   : conda py3.10 ; OMP_NUM_THREADS=4 python r054_sim_cusp_check.py
"""
import json, sys, glob, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/glade/work/yizhu/cuspML/src/scripts/mvp")
import numpy as np, pandas as pd, h5py
from datetime import datetime, timedelta
import xgboost as xgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from r044_fixes import BASE
IN="/glade/work/yizhu/cuspML/output/omni_full_hist_90120"
MODEL="/glade/work/yizhu/cuspML/output/zenodo_models/xgb_abs_eq_mlat.json"
OUT="/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r054_sim_cusp_check.json"
FIG="/glade/work/yizhu/cuspML/figures/r054_sim_cusp_check.png"
STORMS={"sp13":"/glade/derecho/scratch/yizhu/OpOF/sp13_rin15/bcwind.h5",
        "jun0113":"/glade/derecho/scratch/yizhu/OpOF/jun0113_rin15/bcwind.h5"}
RE=6371.0; ALT_FAC=np.sqrt((RE+4000)/(RE+840))   # 1.1993 : dipole same-L map 840km -> 4000km
def alt_map(lat_deg):
    """Map cuspML DMSP-840km MLAT down a dipole field line to the sim's ~4000km cusp altitude.
    Same L: cos^2(lat)/r=1/L => lat(r2)=acos(sqrt(r2/r1)*cos(lat1)). Higher altitude -> lower MLAT."""
    return np.degrees(np.arccos(np.clip(ALT_FAC*np.cos(np.radians(lat_deg)),-1,1)))
def feat_order():
    # exact 74-col order = feats(df,60); load ONE json (columns present) to avoid full-data OOM
    import json as J
    recs=J.load(open(sorted(glob.glob(f"{IN}/cusp_crossings_*.json"))[0]))
    df=pd.DataFrame(recs)
    suff=['mean15','std15','delta15','mean30','std30','delta30','mean60','std60','delta60','int60']
    hist=sorted([c for c in df.columns if any(s in c for s in suff)])
    return BASE+hist
def storm_features(bcf, cols, hemi=1):
    d=h5py.File(bcf,'r'); g=lambda k: np.array(d[k],float)
    mjd=g('MJD'); tilt_deg=g('tilt')*180.0/np.pi
    v=np.sqrt(g('Vx')**2+g('Vy')**2+g('Vz')**2)/1000.0     # km/s
    n=g('D'); bx=g('Bx'); by=g('By'); bz=g('Bz')
    pdyn=1.6726e-6*n*v**2
    dt0=datetime(1858,11,17); doy=np.array([(dt0+timedelta(days=float(m))).timetuple().tm_yday for m in mjd],float)
    df=pd.DataFrame(dict(imf_bx=bx,imf_by=by,imf_bz=bz,sw_v=v,sw_n=n,sw_pdyn=pdyn,
        dipole_tilt=tilt_deg,hemi_code=float(hemi),doy=doy))
    df['B_T']=np.sqrt(by**2+bz**2); df['clock_angle']=np.arctan2(by,bz)
    df['sin_clock_half']=np.sin(df.clock_angle/2)
    df['newell_cf']=(v**(4/3))*(df.B_T**(2/3))*(np.abs(df.sin_clock_half)**(8/3))
    df['kan_lee_ef']=v*df.B_T*(df.sin_clock_half**2); df['vBs']=v*np.where(bz<0,-bz,0)
    df['by_hemi']=by*(1 if hemi==1 else -1)
    # history: rolling over the 1-min series (bcwind is 1-min). windows in minutes = samples.
    base6=['imf_bx','imf_by','imf_bz','sw_v','sw_n','sw_pdyn']
    for w in (15,30,60):
        for b in base6:
            s=df[b]
            df[f'{b}_mean{w}']=s.rolling(w,min_periods=1).mean()
            df[f'{b}_std{w}']=s.rolling(w,min_periods=1).std().fillna(0)
            df[f'{b}_delta{w}']=s-s.shift(w); df[f'{b}_delta{w}']=df[f'{b}_delta{w}'].fillna(0)
    df['newell_cf_mean60']=df.newell_cf.rolling(60,min_periods=1).mean()
    df['newell_cf_int60']=df.newell_cf.rolling(60,min_periods=1).sum()
    df['vBs_mean60']=df.vBs.rolling(60,min_periods=1).mean()
    df['vBs_int60']=df.vBs.rolling(60,min_periods=1).sum()
    miss=[c for c in cols if c not in df.columns]
    assert not miss, f"missing feature cols: {miss}"
    return mjd, df[cols].values
def main():
    cols=feat_order(); assert len(cols)==74, f"expected 74 got {len(cols)}"
    m=xgb.XGBRegressor(); m.load_model(MODEL)
    R={}; fig,ax=plt.subplots(1,2,figsize=(13,4.8))
    for i,(name,bcf) in enumerate(STORMS.items()):
        mjd,X=storm_features(bcf,cols)
        pred=m.predict(X)                                    # cuspML eq-MLAT (deg) @840km DMSP, N hemi
        pred4000=alt_map(pred)                               # dipole-mapped to ~4000km sim altitude
        hrs=(mjd-mjd[0])*24.0
        R[name]=dict(cuspml_eqmlat_med=float(np.nanmedian(pred)),cuspml_eqmlat_min=float(np.nanmin(pred)),
                     cuspml_eqmlat_max=float(np.nanmax(pred)),
                     cuspml_eqmlat4000_med=float(np.nanmedian(pred4000)),
                     cuspml_eqmlat4000_range=[float(np.nanmin(pred4000)),float(np.nanmax(pred4000))],n=len(pred),
                     mjd0=float(mjd[0]),bz_med=float(np.nanmedian(np.array(h5py.File(bcf)['Bz'])) ))
        ax[i].plot(hrs,pred,'-',color='#1f77b4',lw=1.2,label='cuspML eq-MLAT @840km (DMSP)')
        ax[i].plot(hrs,pred4000,'--',color='#d62728',lw=1.4,label='cuspML eq-MLAT @4000km (alt-mapped)')
        # Zhang center for reference: invlat->lat approx; center MLAT = 77.3 + 0.77*Bz (invlat)
        bz=np.array(h5py.File(bcf)['Bz'],float); zc=77.3+0.77*bz
        ax[i].plot(hrs,zc,'--',color='#7f7f7f',lw=1,label='Zhang center (invlat)')
        if name=="sp13":
            z=np.load("/glade/work/yizhu/OpOF/src/kernels/sp13/cusp_center_compare.npz")
            th=(z['t']-z['t'][0])/3600.0 + (21600/3600.0)   # cusp t starts 6hr into run
            ax[i].plot(th,z['mlatN'],'o',color='k',ms=3,label='sim myfinder center (N)')
            simc=float(np.nanmedian(z['mlatN'])); R[name]['sim_finder_center_med']=simc
            # verdict: sim cusp centre vs cuspML eq-boundary mapped to 4000km (eq is a few deg equatorward of centre)
            R[name]['offset_840_vs_sim']=float(R[name]['cuspml_eqmlat_med']-simc)
            R[name]['offset_4000_vs_sim']=float(R[name]['cuspml_eqmlat4000_med']-simc)
        ax[i].set_title(f"{name}  (Bz_med {R[name]['bz_med']:.1f} nT)"); ax[i].set_xlabel("hours from run start")
        ax[i].set_ylabel("MLAT (deg)"); ax[i].set_ylim(60,85); ax[i].legend(fontsize=8); ax[i].grid(alpha=.3)
        print(f"{name}: cuspML eq-MLAT med {R[name]['cuspml_eqmlat_med']:.1f} ({R[name]['cuspml_eqmlat_min']:.1f}-{R[name]['cuspml_eqmlat_max']:.1f}) deg | Bz_med {R[name]['bz_med']:.1f}",flush=True)
    fig.tight_layout(); fig.savefig(FIG,dpi=150)
    json.dump(R,open(OUT,"w"),indent=1); print("saved",OUT,FIG)
if __name__=="__main__": main()
