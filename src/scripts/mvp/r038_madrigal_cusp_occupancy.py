#!/usr/bin/env python
"""R038: process Madrigal calibrated SSJ HDF5 -> cusp occupancy maps + negatives.
Newell cusp criterion on calibrated ion_d_ener. Accumulates AACGM MLAT-MLT histograms:
all-dayside (denominator/negatives) + cusp (positives). Parallel over files (PBS compute).
Date: 2026-06-07  Inputs: scratch/cuspML_madrigal/ssj/<SAT>/<YEAR>/*.hdf5
Outputs: bundles/r038_madrigal_occupancy.npz + per-file cusp table parquet
"""
import sys, os, glob, argparse
import numpy as np, pandas as pd
from multiprocessing import Pool, set_start_method
sys.path.insert(0,'/glade/work/yizhu/cuspML/src')
from identify_cusp import sliding_window_cusp
ROOT='/glade/derecho/scratch/yizhu/cuspML_madrigal/ssj'
OUT='/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r038_madrigal_occupancy.npz'
MLT_E=np.linspace(0,2*np.pi,97); R_E=np.linspace(0,40,81)  # full 24h, MLAT 90..50

def process(path):
    import h5py
    try:
        h=h5py.File(path,'r'); tl=h['Data']['Table Layout'][:]; h.close()
    except Exception: return None
    ut=tl['ut1_unix'].astype(np.int64); che=np.round(tl['ch_energy'],0)
    dde=tl['ion_d_ener']; imen=tl['ion_m_ener']; emen=tl['el_m_ener']
    mlat=tl['mlat']; mlt=tl['mlt']
    order=np.argsort(ut,kind='stable')
    uts=ut[order]; ddes=dde[order]; ches=che[order]; im=imen[order]; em=emen[order]; ml=mlat[order]; lt=mlt[order]
    ut_u=np.unique(uts); bnd=np.searchsorted(uts,ut_u); bnd2=np.append(bnd[1:],len(uts))
    n=len(ut_u); peak=np.zeros(n); peakE=np.zeros(n); iav=np.zeros(n); eav=np.zeros(n); MLA=np.zeros(n); LT=np.zeros(n)
    for i in range(n):
        sl=slice(bnd[i],bnd2[i]); fl=np.where(ddes[sl]>0,ddes[sl],np.nan); en=ches[sl]
        if np.all(np.isnan(fl)): peak[i]=-1
        else:
            j=np.nanargmax(fl); peak[i]=fl[j]; peakE[i]=en[j]
        iav[i]=im[sl][0]; eav[i]=em[sl][0]; MLA[i]=ml[sl][0]; LT[i]=lt[sl][0]
    mask=(iav<=3000)&(eav<=220)&(peak>=2e7)&(peakE>=100)&(peakE<=7000)
    cusp=sliding_window_cusp(mask.astype(bool),4,3)
    hl=(np.abs(MLA)>=50)&np.isfinite(LT)
    if not hl.any(): return None
    th=LT[hl]*np.pi/12; rr=90-np.abs(MLA[hl])
    den=np.histogram2d(th,rr,bins=[MLT_E,R_E])[0]
    cu=cusp[hl]; num=np.histogram2d(th[cu],rr[cu],bins=[MLT_E,R_E])[0] if cu.any() else np.zeros_like(den)
    return den,num,int(hl.sum()),int(cu.sum())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=48); a=ap.parse_args()
    files=sorted(glob.glob(f'{ROOT}/*/*/*.hdf5'))
    print(f'[R038] {len(files)} Madrigal HDF5 files, {a.workers} workers',flush=True)
    DEN=np.zeros((len(MLT_E)-1,len(R_E)-1)); NUM=np.zeros_like(DEN); done=0; tn=0; tc=0
    set_start_method('fork',force=True)
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(process,files,chunksize=4):
            done+=1
            if r is not None: DEN+=r[0]; NUM+=r[1]; tn+=r[2]; tc+=r[3]
            if done%200==0: print(f'  {done}/{len(files)} files, dayside={tn} cusp={tc}',flush=True)
    np.savez(OUT,DEN=DEN,NUM=NUM,MLT_E=MLT_E,R_E=R_E)
    print(f'[R038] dayside={tn} cusp={tc}  saved {OUT}',flush=True)

if __name__=='__main__': main()
