#!/usr/bin/env python
"""R039: validate CALIBRATED Madrigal cusp vs Newell 2004 — IMF-quadrant centroid.
Process Madrigal HDF5 -> cusp spectra (time,mlat,mlt) -> match OMNI Bz/By ->
4 IMF quadrant occupancy histograms. Centroid rule downstream compares to Newell 2004.
Date: 2026-06-08  Inputs: scratch madrigal ssj + omni_raw  Outputs: bundles/r039_*.npz
"""
import sys, os, glob, argparse, datetime
import numpy as np, pandas as pd
from multiprocessing import Pool, set_start_method
sys.path.insert(0,'/glade/work/yizhu/cuspML/src'); sys.path.insert(0,'/glade/work/yizhu/cuspML/src/lib')
from identify_cusp import sliding_window_cusp
from omni_1min import load_omni_1min
ROOT='/glade/derecho/scratch/yizhu/cuspML_madrigal/ssj'
OMNI_TMPL='/glade/work/yizhu/cuspML/output/omni_raw/omni_min{year}.asc'
OUT='/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r039_madrigal_imf_centroid.npz'
MLT_E=np.linspace(0,2*np.pi,49); R_E=np.linspace(0,35,36)
QUAD=[('q1',-1,'<',-3,'<'),('q2',-1,'<',3,'>'),('q3',1,'>',-3,'<'),('q4',1,'>',3,'>')]
_OMNI={}

def cusp_spectra(path):
    import h5py
    try: h=h5py.File(path,'r'); tl=h['Data']['Table Layout'][:]; h.close()
    except Exception: return None
    ut=tl['ut1_unix'].astype(np.int64); che=np.round(tl['ch_energy'],0)
    dde=tl['ion_d_ener']; im=tl['ion_m_ener']; em=tl['el_m_ener']; ml=tl['mlat']; lt=tl['mlt']
    o=np.argsort(ut,kind='stable'); ut,che,dde,im,em,ml,lt=ut[o],che[o],dde[o],im[o],em[o],ml[o],lt[o]
    uu=np.unique(ut); b=np.searchsorted(ut,uu); b2=np.append(b[1:],len(ut))
    n=len(uu); peak=np.zeros(n); pe=np.zeros(n); iav=np.zeros(n); eav=np.zeros(n); MLA=np.zeros(n); LT=np.zeros(n)
    for i in range(n):
        s=slice(b[i],b2[i]); fl=np.where(dde[s]>0,dde[s],np.nan)
        if np.all(np.isnan(fl)): peak[i]=-1
        else: j=np.nanargmax(fl); peak[i]=fl[j]; pe[i]=che[s][j]
        iav[i]=im[s][0]; eav[i]=em[s][0]; MLA[i]=ml[s][0]; LT[i]=lt[s][0]
    mask=(iav<=3000)&(eav<=220)&(peak>=2e7)&(pe>=100)&(pe<=7000)
    cusp=sliding_window_cusp(mask.astype(bool),4,3)
    k=cusp&(np.abs(MLA)>=50)&np.isfinite(LT)
    if not k.any(): return None
    return uu[k], np.abs(MLA[k]), LT[k]

def process(path):
    r=cusp_spectra(path)
    if r is None: return None
    st,mlat,mlt=r; yr=datetime.datetime.utcfromtimestamp(int(st[0])).year
    if yr not in _OMNI: return None
    ot,obz,oby=_OMNI[yr]
    idx=np.clip(np.searchsorted(ot,st),0,len(ot)-1); idxl=np.clip(idx-1,0,len(ot)-1)
    pick=np.where(np.abs(ot[idx]-st)<=np.abs(ot[idxl]-st),idx,idxl)
    ok=np.abs(ot[pick]-st)<=300; bz=np.where(ok,obz[pick],np.nan); by=np.where(ok,oby[pick],np.nan)
    g=~np.isnan(bz)&~np.isnan(by); mlat,mlt,bz,by=mlat[g],mlt[g],bz[g],by[g]
    th=mlt*np.pi/12; rr=90-mlat
    num=np.zeros((4,len(MLT_E)-1,len(R_E)-1))
    for qi,(_,bzt,bzo,byt,byo) in enumerate(QUAD):
        m=(bz<bzt)&(by<byt) if bzo=='<' and byo=='<' else (bz<bzt)&(by>byt) if bzo=='<' else (bz>bzt)&(by<byt) if byo=='<' else (bz>bzt)&(by>byt)
        if m.any(): num[qi]+=np.histogram2d(th[m],rr[m],bins=[MLT_E,R_E])[0]
    return num

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=48); a=ap.parse_args()
    files=sorted(glob.glob(f'{ROOT}/*/*/*.hdf5'))
    years=sorted({int(f.split('/')[-2]) for f in files})
    for y in years:
        p=OMNI_TMPL.format(year=y)
        if os.path.exists(p):
            om=load_omni_1min(p).sort_values('datetime').reset_index(drop=True)
            _OMNI[y]=(om['datetime'].values.astype('datetime64[s]').astype(np.int64),
                      om['imf_bz'].values.astype(float), om['imf_by'].values.astype(float))
    print(f'[R039] {len(files)} files, OMNI years {sorted(_OMNI)}, {a.workers} workers',flush=True)
    NUM=np.zeros((4,len(MLT_E)-1,len(R_E)-1)); done=0
    set_start_method('fork',force=True)
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(process,files,chunksize=4):
            done+=1
            if r is not None: NUM+=r
            if done%200==0: print(f'  {done}/{len(files)}, cusp={int(NUM.sum())}',flush=True)
    np.savez(OUT,NUM=NUM,MLT_E=MLT_E,R_E=R_E)
    print(f'[R039] cusp by quad={[int(NUM[i].sum()) for i in range(4)]} total={int(NUM.sum())} saved {OUT}',flush=True)

if __name__=='__main__': main()
