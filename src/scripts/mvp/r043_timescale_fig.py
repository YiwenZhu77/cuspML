#!/usr/bin/env python
"""R043: full history-timescale ablation (instant -> +120 min) + regenerated Fig 6.

Internally consistent version of the paper's timescale ablation, extended to 90 and
120 min for reviewer R2-M1. All feature sets are evaluated on the SAME crossings (rows
with valid 120-min windows) and the same random 80/20 split, so the bars are directly
comparable. Shows the equatorward-MLAT MAE improvement flattening by ~60-90 min.

Input : output/omni_full_hist_90120/cusp_crossings_*.json
Output: paper/figures/fig06_time_window_comparison.{png,pdf}
        src/kernels/cuspmap_mvp/bundles/r043_timescale.json
Date: 2026-07-04
"""
import json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})

IN = "/glade/work/yizhu/cuspML/output/omni_full_hist_90120"
FIG = "/glade/work/yizhu/cuspML/paper/figures/fig06_time_window_comparison"
JS = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r043_timescale.json"
XGB = dict(n_estimators=1000, max_depth=8, learning_rate=0.02, subsample=0.8,
           colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
           random_state=42, n_jobs=32, verbosity=0)
BASE = ['dipole_tilt','hemi_code','doy','imf_bx','imf_by','imf_bz','sw_v','sw_n','sw_pdyn',
        'B_T','clock_angle','sin_clock_half','newell_cf','kan_lee_ef','vBs','by_hemi']
TARGETS = ['abs_eq_mlat','abs_pole_mlat','eq_mlt','mean_mlt']
WIN_SUFF = {15:['mean15','std15','delta15'], 30:['mean30','std30','delta30'],
            60:['mean60','std60','delta60','int60'],
            90:['mean90','std90','delta90'], 120:['mean120','std120','delta120']}


def load():
    recs = []
    for f in sorted(glob.glob(f"{IN}/cusp_crossings_*.json")):
        recs.extend(json.load(open(f)))
    df = pd.DataFrame(recs).dropna(subset=["eq_mlat","pole_mlat","imf_bz","sw_v","sw_n","sw_pdyn"])
    df["abs_eq_mlat"]=df["eq_mlat"].abs(); df["abs_pole_mlat"]=df["pole_mlat"].abs()
    df["hemi_code"]=(df["hemisphere"]=="N").astype(float)
    df["doy"]=pd.to_datetime(df["time_start"]).dt.dayofyear
    df["B_T"]=np.sqrt(df["imf_by"]**2+df["imf_bz"]**2)
    df["clock_angle"]=np.arctan2(df["imf_by"],df["imf_bz"]); df["sin_clock_half"]=np.sin(df["clock_angle"]/2)
    df["newell_cf"]=(df["sw_v"]**(4/3))*(df["B_T"]**(2/3))*(np.abs(df["sin_clock_half"])**(8/3))
    df["kan_lee_ef"]=df["sw_v"]*df["B_T"]*(df["sin_clock_half"]**2)
    df["vBs"]=df["sw_v"]*np.where(df["imf_bz"]<0,-df["imf_bz"],0)
    df["by_hemi"]=df["imf_by"]*np.where(df["hemisphere"]=="N",1,-1)
    return df


def feats(df, maxwin):
    suff = sum((WIN_SUFF[w] for w in [15,30,60,90,120] if w <= maxwin), [])
    hist = sorted([c for c in df.columns if suff and any(s in c for s in suff) and c not in BASE])
    return BASE + hist


def main():
    df = load()
    keep = list(dict.fromkeys(feats(df,120)+TARGETS))
    dfc = df[[c for c in keep if c in df.columns]].dropna().reset_index(drop=True)
    y = dfc[TARGETS].values.astype(np.float32)
    itr, ite = train_test_split(np.arange(len(dfc)), test_size=0.2, random_state=42)
    print(f"rows {len(dfc)}")

    labels = ['Instant\n(no history)','+15 min','+30 min','+60 min','+90 min','+120 min']
    windows = [0,15,30,60,90,120]
    maes = []
    for mw in windows:
        F = [c for c in feats(df,mw) if c in dfc.columns]
        X = dfc[F].values.astype(np.float32)
        m = MultiOutputRegressor(XGBRegressor(**XGB)); m.fit(X[itr], y[itr])
        mae = float(mean_absolute_error(y[ite,0], m.predict(X[ite])[:,0]))
        maes.append(mae); print(mw, len(F), round(mae,4))

    json.dump({'windows':windows,'n_features':[len(feats(df,w)) for w in windows],
               'random_MAE_eqmlat':maes,'n_rows':int(len(dfc))}, open(JS,'w'), indent=1)

    fig, ax = plt.subplots(figsize=(7,4))
    colors = ['#999999','#7fb3d5','#5499c7','#2e86c1','#21618c','#1b3a5c']
    bars = ax.bar(range(len(windows)), maes, color=colors, edgecolor='k', linewidth=0.6, width=0.7)
    for b,mv in zip(bars,maes):
        ax.text(b.get_x()+b.get_width()/2, mv+0.004, f"{mv:.3f}", ha='center', va='bottom', fontsize=10)
    ax.set_xticks(range(len(windows))); ax.set_xticklabels(labels)
    ax.set_ylabel("Equatorward MLAT MAE (°)")
    ax.set_ylim(0, max(maes)*1.15)
    ax.axhline(maes[3], color='#2e86c1', ls='--', lw=0.8, alpha=0.6)
    ax.set_title("History window vs prediction error (random split)")
    fig.tight_layout(); fig.savefig(FIG+".png"); fig.savefig(FIG+".pdf")
    print("saved", FIG)


if __name__ == "__main__":
    main()
