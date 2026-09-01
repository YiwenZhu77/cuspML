#!/usr/bin/env python
"""R036: sample per-channel raw->calibrated correction across sats/years.
Validates the SSJ/5 (and checks SSJ/4) calibration vector for consistency + physical plausibility.
Downloads small samples NCEI raw + CDAWeb for each (sat,year), derives median per-channel cal/raw.
"""
import urllib.request, cdflib, numpy as np, sys, os
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'/glade/work/yizhu/cuspML/src')
from parse_ncei_ssj import read_ssj_file, CHANNEL_ENERGIES
import re
ce=np.asarray(CHANNEL_ENERGIES,float); KEEP=[i for i in range(20) if i!=10]; ce19=ce[KEEP]
TMP='/glade/derecho/scratch/yizhu/tmp/cal_sample'; os.makedirs(TMP,exist_ok=True)

def cdaweb_files(sat,year,n=5):
    nn=sat[1:]
    u=f'https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmsp{sat.lower()}/ssj/precipitating-electrons-ions/{year}/'
    h=urllib.request.urlopen(u,timeout=30).read().decode('latin1')
    fs=re.findall(r'href=\"([^\"]+\.cdf)\"',h)
    pick=fs[::max(1,len(fs)//n)][:n]
    return u,pick

def date_of(cdf_name):
    m=re.search(r'_(\d{8})_',cdf_name); return m.group(1)  # YYYYMMDD

def proc(sat,year,cdf_url,cdf_name):
    nn=sat[1:]; ymd=date_of(cdf_name); yyyy=ymd[:4]; import datetime
    d=datetime.date(int(ymd[:4]),int(ymd[4:6]),int(ymd[6:8])); doy=d.timetuple().tm_yday; yy=d.year%100
    ssj5 = sat in ('F16','F17','F18'); pre='j5' if ssj5 else 'j4'
    ncei=f'https://www.ncei.noaa.gov/data/dmsp-space-weather-sensors/access/f{nn}/ssj/{yyyy}/{int(ymd[4:6]):02d}/{pre}f{nn}{yy:02d}{doy:03d}.gz'
    lp_c=f'{TMP}/{sat}_{ymd}.cdf'; lp_n=f'{TMP}/{sat}_{ymd}.gz'
    try:
        if not os.path.exists(lp_c): urllib.request.urlretrieve(cdf_url+cdf_name,lp_c)
        if not os.path.exists(lp_n): urllib.request.urlretrieve(ncei,lp_n)
    except Exception as e: return None
    try:
        rec=read_ssj_file(lp_n,satellite=sat)
        nt=np.array([r['datetime'].replace(tzinfo=None) for r in rec],dtype='datetime64[s]').astype(np.int64)
        ndf=np.array([r['ion_diff_energy_flux'] for r in rec],float)[:,KEEP]
        c=cdflib.CDF(lp_c); cdf=np.array(c.varget('ION_DIFF_ENERGY_FLUX'),float)
        ct=np.array(cdflib.cdfepoch.to_datetime(c.varget('Epoch')),dtype='datetime64[s]').astype(np.int64)
        un,ni=np.unique(nt,return_index=True); uc,ci=np.unique(ct,return_index=True)
        comm,ia,ib=np.intersect1d(un,uc,return_indices=True)
        R=np.where(np.abs(ndf[ni[ia]])>1e10,np.nan,ndf[ni[ia]]); C=np.where(np.abs(cdf[ci[ib]])>1e30,np.nan,cdf[ci[ib]])
        return R,C
    except Exception as e: return None

SAMPLES=[('F16',2008),('F16',2014),('F17',2012),('F18',2012),('F13',2005)]
results={}
for sat,year in SAMPLES:
    try: base,fs=cdaweb_files(sat,year,5)
    except Exception as e: print(f'{sat} {year}: list fail {e}'); continue
    Rs=[];Cs=[]
    with ThreadPoolExecutor(8) as ex:
        for out in ex.map(lambda f:proc(sat,year,base,f),fs):
            if out is not None: Rs.append(out[0]);Cs.append(out[1])
    if not Rs: print(f'{sat} {year}: no data'); continue
    R=np.vstack(Rs);C=np.vstack(Cs)
    corr=np.full(19,np.nan)
    for k in range(19):
        m=(R[:,k]>0)&(C[:,k]>0)&np.isfinite(R[:,k])&np.isfinite(C[:,k])
        if m.sum()>=100: corr[k]=np.median(C[m,k]/R[m,k])
    results[f'{sat}_{year}']=corr
    print(f'{sat} {year} ({"SSJ5" if sat in ("F16","F17","F18") else "SSJ4"}): n_spec={len(R)}  corr[0,9,18]={corr[0]:.2f},{corr[9]:.2f},{corr[18]:.2f}')
np.savez('/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r036_cal_vectors.npz',ce19=ce19,**results)
print('saved vectors:',list(results.keys()))
