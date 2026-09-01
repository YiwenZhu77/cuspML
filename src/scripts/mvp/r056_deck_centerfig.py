#!/usr/bin/env python
"""R056: deck figure — simulation cusp centre vs the data-driven cuspML centre, JGR style.

Single row (centre only; the equatorward-boundary row of r055 is dropped for the slide). Both
curves are the cusp CENTRE magnetic latitude = midpoint of the equatorward and poleward
boundaries, mapped to the same 4000 km altitude, over the SAME magnetic-local-time sector
(08:30-15:30) that the data-driven model is trained on. Follows ~/.claude/assets/pub-figure-style.md:
black box, inward major+minor ticks all four sides, no grid, restrained palette, spelled-out
English labels, canonical event names, clock-time x-axis, no on-figure provenance stamp.

CLAIM: simulated cusp centre tracks the data-driven model through both storms; median centre
offset March 17 2013 = +4.2 deg, June 1 2013 = +0.1 deg.
INPUT : reuses r055 (bcwind.h5 solar wind + prod_*_cusp_b*.tpInit O+ launch region; cuspML zenodo eq+pole models).
OUTPUT: cuspML/figures/r056_sim_cusp_centre_deck.png
RUN   : conda py3.10 ; OMP_NUM_THREADS=2 python r056_deck_centerfig.py
"""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/glade/work/yizhu/cuspML/src/scripts/mvp")
sys.path.insert(0,"/glade/u/home/yizhu/.claude/assets")
import numpy as np
from datetime import datetime, timedelta
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from jgr_style import set_jgr, finish, PALETTE, date_xaxis
import r055_cusp_eqedge_fig as R  # reuse cuspml_eq4000 / sim_boundaries / feat_order / STORMS / OP / CM
import h5py

def mjd_to_dt(m): return datetime(1858,11,17)+timedelta(days=float(m))

def main():
    cols=R.feat_order(); assert len(cols)==74
    set_jgr(13)
    fig,ax=plt.subplots(1,2,figsize=(12,4.4))
    names={"sp13":"March 17, 2013","jun0113":"June 1, 2013"}
    out={}
    for i,(sk,sv) in enumerate(R.STORMS.items()):
        bcf=f"{R.OP}/{sv['dir']}/bcwind.h5"
        th,_,mlctr=R.cuspml_eq4000(bcf,cols,sv['date'])        # cuspML centre @4000 km
        tc,_,simctr=R.sim_boundaries(f"{R.OP}/{sv['dir']}/{sv['tp']}")
        mjd0=float(np.array(h5py.File(bcf,'r')['MJD'])[0]); base=mjd_to_dt(mjd0)
        a=ax[i]
        a.plot(th,mlctr,'-',color=PALETTE["red"],lw=1.8,
               label="Data-driven model (solar-wind driven)")
        a.plot(tc,simctr,'o-',color=PALETTE["blue"],lw=1.4,ms=4.5,mfc=PALETTE["blue"],mec="black",mew=0.4,
               label="Magnetohydrodynamic simulation")
        a.set_title(names[sk],fontsize=13)
        a.set_ylabel("cusp centre magnetic latitude  [degrees]")
        a.set_xlim(th.min(),th.max()); a.set_ylim(64,80)
        date_xaxis(a,base,hour_step=6,fontsize=9)
        a.legend(fontsize=9,loc="lower right")
        finish(a)
        out[sk]=dict(offset_centre_deg=float(np.median(mlctr)-np.median(simctr)))
        print(f"{sk} ({names[sk]}): centre offset {out[sk]['offset_centre_deg']:+.1f} deg",flush=True)
    # finding + shared element decode
    fig.suptitle("Simulated cusp centre tracks the data-driven model through both storms",
                 fontsize=13.5,y=1.02)
    fig.text(0.5,-0.10,"Cusp centre = midpoint of the equatorward and poleward boundaries, "
             "both mapped to 4000 km altitude and averaged over the 08:30-15:30 magnetic-local-time sector.",
             ha="center",va="top",fontsize=9)
    fig.tight_layout()
    P=f"{R.CM}/figures/r056_sim_cusp_centre_deck.png"
    fig.savefig(P,dpi=300,bbox_inches="tight"); print("saved",P)

if __name__=="__main__": main()
