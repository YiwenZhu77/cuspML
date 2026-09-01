#!/usr/bin/env python
"""R037: pull calibrated DMSP SSJ flux/energy HDF5 from CEDAR Madrigal (inst 8100).
Parallel download to scratch. SSJ 'e' files (kindat 10200+NN). Runs on LOGIN node
(compute nodes lack internet). Skips existing files.
Date: 2026-06-07
"""
import sys, os, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from madrigalWeb.madrigalWeb import MadrigalData
OUT='/glade/derecho/scratch/yizhu/cuspML_madrigal/ssj'
U=('Yiwen Zhu','yz167@rice.edu','Rice University')
MD='http://cedar.openmadrigal.org'

# (sat, kindat) -- 'e' = flux/energy values; kindat = 10200 + sat number
SATS={'F10':10210,'F11':10211,'F12':10212,'F13':10213,'F14':10214,'F15':10215,
      'F16':10216,'F17':10217,'F18':10218}

def pull_year(sat, kindat, year, day_step, workers):
    md=MadrigalData(MD)
    outdir=f'{OUT}/{sat}/{year}'; os.makedirs(outdir,exist_ok=True)
    try:
        exps=md.getExperiments(8100, year,1,1,0,0,0, year,12,31,23,59,59)
    except Exception as e:
        print(f'  {sat} {year}: getExperiments fail {e}',flush=True); return 0
    files=[]
    for e in exps:
        try:
            for f in md.getExperimentFiles(e.id):
                if f.kindat==kindat: files.append(f)
        except Exception: pass
    files=sorted(files,key=lambda f:f.name)[::day_step]
    todo=[f for f in files if not os.path.exists(f'{outdir}/{os.path.basename(f.name)}')]
    print(f'  {sat} {year}: {len(files)} files ({len(todo)} to fetch)',flush=True)
    def dl(f):
        lp=f'{outdir}/{os.path.basename(f.name)}'
        try:
            mdx=MadrigalData(MD); mdx.downloadFile(f.name,lp,*U,'hdf5'); return 1
        except Exception: 
            if os.path.exists(lp): os.remove(lp)
            return 0
    n=0
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(dl,todo): n+=r
    print(f'  {sat} {year}: fetched {n}/{len(todo)}',flush=True)
    return n

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--day-step',type=int,default=2)
    ap.add_argument('--workers',type=int,default=12)
    ap.add_argument('--sample',action='store_true',help='representative sample set')
    a=ap.parse_args()
    if a.sample:
        JOBS=[('F10',1993),('F10',1996),('F11',1992),('F11',1995),
              ('F13',2002),('F13',2005),('F15',2003),('F15',2008),
              ('F16',2008),('F16',2014),('F17',2012),('F18',2014)]
    else:
        JOBS=[(s,y) for s in SATS for y in range(1987,2025)]  # full (huge)
    print(f'[R037] {len(JOBS)} sat-years, day_step={a.day_step}, workers={a.workers}',flush=True)
    tot=0
    for sat,year in JOBS:
        tot+=pull_year(sat,SATS[sat],year,a.day_step,a.workers)
    print(f'[R037] total fetched {tot}. out={OUT}',flush=True)

if __name__=='__main__': main()
