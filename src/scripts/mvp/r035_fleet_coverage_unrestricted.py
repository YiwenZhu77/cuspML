#!/usr/bin/env python
"""R035: UNRESTRICTED fleet coverage, ALL cached DMSP (F10,F11,F16,F17,F18).
No MLT cut. Accumulates 2D MLAT-MLT histograms: all-spectra sampling envelope +
cusp_base detections (newell_cusp_mask, no flux/MLT cut). |MLAT|>=50, full 24h MLT.
Date: 2026-06-04  Inputs: ncei_ssj_cache (all sats)  Outputs: bundles/r035_*.npz, figures/r035_*.png
"""
import sys, os, glob, argparse
import numpy as np, pandas as pd
from multiprocessing import Pool, set_start_method
sys.path.insert(0,'/glade/work/yizhu/cuspML/src')
from parse_ncei_ssj import read_ssj_file, _is_ssj5, CHANNEL_ENERGIES
from identify_cusp import newell_cusp_mask, sliding_window_cusp
CACHE='/glade/derecho/scratch/yizhu/tmp/ncei_ssj_cache'
OUT_NPZ='/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r035_fleet_coverage_unrestricted.npz'
OUT_FIG='/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/figures/r035_fleet_coverage_unrestricted.png'
N_CH=len(np.asarray(CHANNEL_ENERGIES,float))
MLT_E=np.linspace(0,2*np.pi,97)      # 0.25h bins, full 24h
R_E=np.linspace(0,40,81)             # MLAT 90..50, 0.5deg

def _decode(fn):
    b=os.path.basename(fn); nn=int(b[3:5]); return f"F{nn}"

def process_file(path):
    sat=_decode(path)
    try: rec=read_ssj_file(path,satellite=sat)
    except Exception: return None
    if len(rec)<100: return None
    ion_avg=np.array([r['ion_avg_energy'] for r in rec],float)
    ele_avg=np.array([r['ele_avg_energy'] for r in rec],float)
    ion_flux=np.array([r['ion_diff_energy_flux'] for r in rec],float)[:,:N_CH]
    lat=np.array([r['cgm_lat'] for r in rec],float); lt=np.array([r['mlt'] for r in rec],float)
    base=sliding_window_cusp(newell_cusp_mask(ion_avg,ele_avg,ion_flux,CHANNEL_ENERGIES),4,3)
    hl=(np.abs(lat)>=50)&np.isfinite(lt)
    if not hl.any(): return None
    th=lt[hl]*np.pi/12; rr=90-np.abs(lat[hl]); cb=base[hl]
    samp=np.histogram2d(th,rr,bins=[MLT_E,R_E])[0]
    cusp=np.histogram2d(th[cb],rr[cb],bins=[MLT_E,R_E])[0] if cb.any() else np.zeros_like(samp)
    return sat,samp,cusp,int(hl.sum()),int(cb.sum())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=48); a=ap.parse_args()
    files=sorted(glob.glob(f"{CACHE}/j[45]f*.gz"))
    print(f"[R035] {len(files)} files (all cached sats), {a.workers} workers",flush=True)
    SAMP=np.zeros((len(MLT_E)-1,len(R_E)-1)); CUSP=np.zeros_like(SAMP)
    persat={}; done=0
    set_start_method('fork',force=True)
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(process_file,files,chunksize=8):
            done+=1
            if r is not None:
                sat,s,c,nhl,ncb=r; SAMP+=s; CUSP+=c
                d=persat.setdefault(sat,[0,0]); d[0]+=nhl; d[1]+=ncb
            if done%1000==0: print(f"  {done}/{len(files)} files, samp={int(SAMP.sum())} cusp={int(CUSP.sum())}",flush=True)
    np.savez(OUT_NPZ,SAMP=SAMP,CUSP=CUSP,MLT_E=MLT_E,R_E=R_E,persat=str(persat))
    print(f"[R035] sampling={int(SAMP.sum())} cusp={int(CUSP.sum())} persat={persat}  saved {OUT_NPZ}",flush=True)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(13,6))
    for k,(H,title,cmap) in enumerate([(SAMP,'All-spectra sampling envelope (|MLAT|>=50, no MLT cut)','viridis'),
                                       (CUSP,'Cusp detections (newell criteria, NO 8.5-15.5 cut)','inferno')]):
        ax=fig.add_subplot(1,2,k+1,projection='polar'); T,R=np.meshgrid(MLT_E,R_E)
        pc=ax.pcolormesh(T,R,np.log10(H.T+1),cmap=cmap,shading='auto')
        ax.set_theta_zero_location('S'); ax.set_theta_direction(1)
        ax.set_rticks([10,20,30,40]); ax.set_yticklabels(['80','70','60','50'],fontsize=7)
        ax.set_xticks(np.linspace(0,2*np.pi,8,endpoint=False)); ax.set_xticklabels(['00','03','06','09','12','15','18','21'],fontsize=8)
        for x in (8.5,15.5): ax.plot([x*np.pi/12]*2,[0,40],'c--',lw=1)
        ax.set_title(title,fontsize=10); fig.colorbar(pc,ax=ax,shrink=0.65,label='log10(count+1)')
    sats=','.join(sorted(persat)); fig.suptitle(f'DMSP unrestricted fleet coverage: {sats}  (dashed = Anderson 8.5-15.5 MLT cut)',fontsize=12)
    fig.tight_layout(); fig.savefig(OUT_FIG,dpi=140,bbox_inches='tight')
    print(f"[R035] saved {OUT_FIG}",flush=True)

if __name__=='__main__': main()
