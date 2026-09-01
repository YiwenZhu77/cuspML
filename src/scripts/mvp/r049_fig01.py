#!/usr/bin/env python
"""R049: regenerate Fig 1 (data coverage) standalone, fixing the half-disk dial bug.

The earlier fig01 rendered the N/S MLT dials as a right-half D-shape that clipped real
post-noon (MLT>12) crossings. This version forces a FULL circle (view from above the
pole): noon (12 MLT) top, dusk (18) left, midnight (00) bottom, dawn (06) right, radius
= 90-|MLAT| toward the center. Cusp data cluster near noon (0830-1530 MLT), which is
physical, not a plotting artifact. Login-safe (no model training).

Input : output/omni_full_hist/cusp_crossings_*.json
Output: paper/figures/fig01_data_coverage.{png,pdf} (+ slides/figures copy)
RUN   : python src/scripts/mvp/r049_fig01.py
"""
import json, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'savefig.dpi': 300,
                     'savefig.bbox': 'tight'})
FIGDIR = "/glade/work/yizhu/cuspML/paper/figures"

def load():
    recs=[]
    for f in sorted(glob.glob("/glade/work/yizhu/cuspML/output/omni_full_hist/cusp_crossings_*.json")):
        recs.extend(json.load(open(f)))
    df=pd.DataFrame(recs).dropna(subset=["imf_bz","sw_v","sw_n","sw_pdyn","eq_mlat","eq_mlt"])
    df["hemi_code"]=(df["hemisphere"]=="N").astype(int)
    df["year"]=pd.to_datetime(df["time_start"]).dt.year
    return df

df=load()
fig=plt.figure(figsize=(15,5))
gs=fig.add_gridspec(1,3,width_ratios=[1.5,1,1])

# (a) stacked bar
ax=fig.add_subplot(gs[0,0])
sats=sorted(df['satellite'].unique()); cmap=plt.cm.tab20
cols={s:cmap(i/max(len(sats)-1,1)) for i,s in enumerate(sats)}
years=np.arange(df['year'].min(),df['year'].max()+1); bottom=np.zeros(len(years))
for s in sats:
    c=df[df['satellite']==s].groupby('year').size().reindex(years,fill_value=0).values
    ax.bar(years,c,bottom=bottom,label=s,color=cols[s],width=0.8,edgecolor='none'); bottom+=c
ax.set_xlabel('Year'); ax.set_ylabel('Number of Crossings')
ax.set_title('(a) Cusp Crossings by Year and Satellite')
ax.legend(ncol=3,fontsize=8,loc='upper left',framealpha=0.9); ax.set_xlim(1986.5,2014.5)

# (b)/(c) full-circle pole-view dials
def dial(ax_p, mask, title, color):
    theta=((df['eq_mlt'].values[mask]-12)/24.0)*2*np.pi
    r=90-df['eq_mlat'].abs().values[mask]
    ax_p.scatter(theta,r,s=2,alpha=0.35,c=color,rasterized=True)
    ax_p.set_theta_zero_location('N'); ax_p.set_theta_direction(1)
    ax_p.set_thetamin(-180); ax_p.set_thetamax(180)          # force FULL circle
    ax_p.set_rlim(0,30); ax_p.set_rticks([5,10,15,20,25])
    ax_p.set_yticklabels(['85°','80°','75°','70°','65°'],fontsize=9)
    ax_p.set_xticks(((np.array([0,6,12,18])-12)/24.0)*2*np.pi)
    ax_p.set_xticklabels(['00','06','12','18'],fontsize=11)
    ax_p.set_title(f'{title}\n($n={int(mask.sum()):,}$)',pad=16)

dial(fig.add_subplot(gs[0,1],projection='polar'),(df['hemi_code'].values==1),'(b) Northern Hemisphere','steelblue')
dial(fig.add_subplot(gs[0,2],projection='polar'),(df['hemi_code'].values==0),'(c) Southern Hemisphere','indianred')

plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig01_data_coverage.png'); fig.savefig(f'{FIGDIR}/fig01_data_coverage.pdf')
print("saved", FIGDIR+"/fig01_data_coverage.png")
