#!/usr/bin/env python
"""R034: parallel IMF-binned cusp occurrence over all OMNI-available DMSP (F10+F11, 1990-1997).
Newell flux standard (peak_flux>=1e8), no MLT hard cut. Accumulates 4-quadrant
numerator/denominator 2D histograms across files via multiprocessing.Pool.
OMNI loaded once in parent -> fork COW shared. Outputs npz + 4-dial figure.
Date: 2026-06-03  Inputs: ncei_ssj_cache, omni_raw  Outputs: bundles/r034_*.npz, figures/r034_*.png
"""
import sys, os, glob, datetime, argparse
import numpy as np, pandas as pd
from multiprocessing import Pool, set_start_method
sys.path.insert(0,'/glade/work/yizhu/cuspML/src'); sys.path.insert(0,'/glade/work/yizhu/cuspML/src/lib')
from parse_ncei_ssj import read_ssj_file, _is_ssj5, CHANNEL_ENERGIES
from identify_cusp import newell_cusp_mask, sliding_window_cusp
from omni_1min import load_omni_1min

CACHE='/glade/derecho/scratch/yizhu/tmp/ncei_ssj_cache'
OMNI_TMPL='/glade/work/yizhu/cuspML/output/omni_raw/omni_min{year}.asc'
OUT_NPZ='/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r034_imf_occurrence_F10F11{suf}.npz'
OUT_FIG='/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/figures/r034_cusp_occurrence_4IMF_F10F11{suf}.png'
N_CH=len(np.asarray(CHANNEL_ENERGIES,float))
FLUX_STD=1e8  # overridden in main() from --flux-std (fork inherits)
MLT_E=np.linspace(0,2*np.pi,49); R_E=np.linspace(0,35,36)  # MLAT 90..55
OMNI_YEARS=list(range(1990,1998))
QUAD=[('q1_Bz-_By-',-1,'<',-3,'<'),('q2_Bz-_By+',-1,'<',3,'>'),
      ('q3_Bz+_By-',1,'>',-3,'<'),('q4_Bz+_By+',1,'>',3,'>')]
_OMNI={}  # year -> (int64 times, bz, by); loaded in parent, COW-shared

def _decode(fn):
    b=os.path.basename(fn)            # jXf NN YY DDD .gz
    nn=int(b[3:5]); yy=int(b[5:7]); doy=int(b[7:10])
    year=1900+yy if yy>=50 else 2000+yy
    return f"F{nn}", year

def process_file(path):
    sat,year=_decode(path)
    if year not in _OMNI: return None
    try: rec=read_ssj_file(path,satellite=sat)
    except Exception: return None
    if len(rec)<100: return None
    t=np.array([r['datetime'] for r in rec])
    ion_avg=np.array([r['ion_avg_energy'] for r in rec],float)
    ele_avg=np.array([r['ele_avg_energy'] for r in rec],float)
    ion_flux=np.array([r['ion_diff_energy_flux'] for r in rec],float)[:,:N_CH]
    lat=np.array([r['cgm_lat'] for r in rec],float); lt=np.array([r['mlt'] for r in rec],float)
    base=sliding_window_cusp(newell_cusp_mask(ion_avg,ele_avg,ion_flux,CHANNEL_ENERGIES),4,3)
    k=(np.abs(lat)>=50)&(lt>=5)&(lt<=19)
    if not k.any(): return None
    flux=np.where(np.abs(ion_flux)>1e10,np.nan,ion_flux)
    pf=np.nanmax(flux,axis=1)
    st=t[k].astype('datetime64[s]').astype(np.int64)
    mlt=lt[k]; mlat=np.abs(lat[k]); pf=pf[k]; cb=base[k]
    ot,obz,oby=_OMNI[year]
    idx=np.clip(np.searchsorted(ot,st),0,len(ot)-1); idxl=np.clip(idx-1,0,len(ot)-1)
    pick=np.where(np.abs(ot[idx]-st)<=np.abs(ot[idxl]-st),idx,idxl)
    ok=np.abs(ot[pick]-st)<=300
    bz=np.where(ok,obz[pick],np.nan); by=np.where(ok,oby[pick],np.nan)
    good=~np.isnan(bz)&~np.isnan(by)
    mlt,mlat,pf,cb,bz,by=mlt[good],mlat[good],pf[good],cb[good],bz[good],by[good]
    th=mlt*np.pi/12; rr=90-mlat
    den=np.zeros((4,len(MLT_E)-1,len(R_E)-1))   # denominator = all dayside per quadrant
    for qi,(_,bzt,bzo,byt,byo) in enumerate(QUAD):
        m=(bz<bzt)&(by<byt) if bzo=='<' and byo=='<' else \
          (bz<bzt)&(by>byt) if bzo=='<' else \
          (bz>bzt)&(by<byt) if byo=='<' else (bz>bzt)&(by>byt)
        if m.any(): den[qi]+=np.histogram2d(th[m],rr[m],bins=[MLT_E,R_E])[0]
    # cb-candidate rows (cusp-type soft-ion = cusp+LLBL+mantle blend) for offline classification
    e=ele_avg[k][good]; iav=ion_avg[k][good]
    c=cb.astype(bool)                                   # keep only cb-passing spectra
    rows=pd.DataFrame(dict(mlt=mlt[c],abs_mlat=mlat[c],ele_avg=e[c],ion_avg=iav[c],
                           peak_flux=pf[c],imf_bz=bz[c],imf_by=by[c]))
    return rows,den

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=120)
    ap.add_argument('--flux-std',type=float,default=1e8); ap.add_argument('--suffix',default=''); a=ap.parse_args()
    global FLUX_STD, OUT_NPZ, OUT_FIG; FLUX_STD=a.flux_std
    OUT_NPZ=OUT_NPZ.format(suf=a.suffix); OUT_FIG=OUT_FIG.format(suf=a.suffix)
    print(f'[R034] FLUX_STD={FLUX_STD:.0e}  out suffix={a.suffix!r}',flush=True)
    print(f"[R034] loading OMNI {OMNI_YEARS[0]}-{OMNI_YEARS[-1]} (parent, COW-shared)...",flush=True)
    for y in OMNI_YEARS:
        p=OMNI_TMPL.format(year=y)
        if not os.path.exists(p): print(f"  miss OMNI {y}"); continue
        om=load_omni_1min(p).sort_values('datetime').reset_index(drop=True)
        _OMNI[y]=(om['datetime'].values.astype('datetime64[s]').astype(np.int64),
                  om['imf_bz'].values.astype(float), om['imf_by'].values.astype(float))
    print(f"  OMNI years loaded: {sorted(_OMNI)}",flush=True)
    files=[f for f in glob.glob(f"{CACHE}/j4f1[01]*.gz") if _decode(f)[1] in _OMNI]
    print(f"[R034] {len(files)} files (F10+F11, OMNI years), {a.workers} workers",flush=True)
    DEN=np.zeros((4,len(MLT_E)-1,len(R_E)-1)); done=0; allrows=[]
    set_start_method('fork',force=True)
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(process_file,files,chunksize=4):
            done+=1
            if r is not None: allrows.append(r[0]); DEN+=r[1]
            if done%500==0: print(f"  {done}/{len(files)} files, candidates so far={sum(len(x) for x in allrows)}",flush=True)
    CAND=pd.concat(allrows,ignore_index=True)
    ROWS_OUT=OUT_NPZ.replace('.npz','_candidates.parquet')
    CAND.to_parquet(ROWS_OUT)
    np.savez(OUT_NPZ,DEN=DEN,MLT_E=MLT_E,R_E=R_E)
    print(f"[R034] candidates={len(CAND)} den={int(DEN.sum())}  saved {ROWS_OUT} + {OUT_NPZ}",flush=True)
    return
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(11,10))
    for qi,(name,*_ ) in enumerate(QUAD):
        occ=np.where(DEN[qi]>=15,NUM[qi]/np.maximum(DEN[qi],1),np.nan)
        ax=fig.add_subplot(2,2,qi+1,projection='polar'); T,R=np.meshgrid(MLT_E,R_E)
        pc=ax.pcolormesh(T,R,occ.T,cmap='viridis',shading='auto',vmin=0)
        ax.set_theta_zero_location('S'); ax.set_theta_direction(1)
        ax.set_rticks([5,15,25,35]); ax.set_yticklabels(['85','75','65','55'],fontsize=7)
        ax.set_xticks(np.linspace(0,2*np.pi,8,endpoint=False)); ax.set_xticklabels(['00','03','06','09','12','15','18','21'],fontsize=7)
        ax.set_title(f"{name}  n_cusp={int(NUM[qi].sum())}",fontsize=9)
        fig.colorbar(pc,ax=ax,shrink=0.6,label='P(cusp)')
    fig.suptitle('R034 cusp OCCURRENCE 4 IMF quadrants (F10+F11 1990-97, flux>=1e8) vs Newell 2004 Fig1',fontsize=11)
    fig.tight_layout(); fig.savefig(OUT_FIG,dpi=140,bbox_inches='tight')
    print(f"[R034] saved {OUT_FIG}",flush=True)

if __name__=='__main__': main()
