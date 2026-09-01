#!/usr/bin/env python
"""R042: history-window sweep 60 vs 90 vs 120 min (reviewer R2-M1).

Trains XGBoost with maximum history window in {60, 90, 120} min on the IDENTICAL
crossings and split, and reports equatorward-MLAT MAE for each. Shows skill saturates
at 60 min (adding 90/120 does not reduce MAE), consistent with the ~1-hour
magnetospheric reconfiguration timescale.

Input : output/omni_full_hist_90120/cusp_crossings_*.json  (r041 output: base + 15/30/
        60/90/120 window features)
Output: src/kernels/cuspmap_mvp/bundles/r042_window_sweep.json
Date: 2026-07-04
"""
import json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

IN = "/glade/work/yizhu/cuspML/output/omni_full_hist_90120"
OUT = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles/r042_window_sweep.json"
XGB = dict(n_estimators=1000, max_depth=8, learning_rate=0.02, subsample=0.8,
           colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
           random_state=42, n_jobs=32, verbosity=0)


def load():
    recs = []
    for f in sorted(glob.glob(f"{IN}/cusp_crossings_*.json")):
        recs.extend(json.load(open(f)))
    df = pd.DataFrame(recs).dropna(subset=["eq_mlat", "pole_mlat", "imf_bz", "sw_v", "sw_n", "sw_pdyn"])
    df["abs_eq_mlat"] = df["eq_mlat"].abs(); df["abs_pole_mlat"] = df["pole_mlat"].abs()
    df["hemi_code"] = (df["hemisphere"] == "N").astype(float)
    df["doy"] = pd.to_datetime(df["time_start"]).dt.dayofyear
    df["year"] = pd.to_datetime(df["time_start"]).dt.year
    df["B_T"] = np.sqrt(df["imf_by"]**2 + df["imf_bz"]**2)
    df["clock_angle"] = np.arctan2(df["imf_by"], df["imf_bz"])
    df["sin_clock_half"] = np.sin(df["clock_angle"] / 2)
    df["newell_cf"] = (df["sw_v"]**(4/3)) * (df["B_T"]**(2/3)) * (np.abs(df["sin_clock_half"])**(8/3))
    df["kan_lee_ef"] = df["sw_v"] * df["B_T"] * (df["sin_clock_half"]**2)
    df["vBs"] = df["sw_v"] * np.where(df["imf_bz"] < 0, -df["imf_bz"], 0)
    df["by_hemi"] = df["imf_by"] * np.where(df["hemisphere"] == "N", 1, -1)
    return df


BASE = ['dipole_tilt','hemi_code','doy','imf_bx','imf_by','imf_bz','sw_v','sw_n','sw_pdyn',
        'B_T','clock_angle','sin_clock_half','newell_cf','kan_lee_ef','vBs','by_hemi']
TARGETS = ['abs_eq_mlat','abs_pole_mlat','eq_mlt','mean_mlt']
# window-suffix groups
WIN_SUFF = {15:['mean15','std15','delta15'], 30:['mean30','std30','delta30'],
            60:['mean60','std60','delta60','int60'],
            90:['mean90','std90','delta90'], 120:['mean120','std120','delta120']}


def feats_for_maxwin(df, maxwin):
    wins = [w for w in [15,30,60,90,120] if w <= maxwin]
    suff = sum((WIN_SUFF[w] for w in wins), [])
    hist = sorted([c for c in df.columns if any(s in c for s in suff) and c not in BASE])
    return BASE + hist


def main():
    df = load()
    # common clean rows across the LARGEST feature set so all models use identical rows
    feat_all = feats_for_maxwin(df, 120)
    keep = list(dict.fromkeys(feat_all + TARGETS + ['year']))
    dfc = df[[c for c in keep if c in df.columns]].dropna().reset_index(drop=True)
    y = dfc[TARGETS].values.astype(np.float32)
    yr = dfc['year'].values
    print(f"common clean rows: {len(dfc)}")

    idx = np.arange(len(dfc))
    itr, ite = train_test_split(idx, test_size=0.2, random_state=42)
    tr_t = yr < 2008; te_t = yr >= 2008

    R = {'n_rows': int(len(dfc))}
    for mw in [60, 90, 120]:
        feats = [c for c in feats_for_maxwin(df, mw) if c in dfc.columns]
        X = dfc[feats].values.astype(np.float32)
        # random split
        m = MultiOutputRegressor(XGBRegressor(**XGB)); m.fit(X[itr], y[itr])
        pr = m.predict(X[ite])
        # temporal holdout
        mt = MultiOutputRegressor(XGBRegressor(**XGB)); mt.fit(X[tr_t], y[tr_t])
        pt = mt.predict(X[te_t])
        R[f'max{mw}min'] = dict(
            n_features=len(feats),
            random_MAE=float(mean_absolute_error(y[ite,0], pr[:,0])),
            random_r=float(np.corrcoef(y[ite,0], pr[:,0])[0,1]),
            temporal_MAE=float(mean_absolute_error(y[te_t,0], pt[:,0])),
        )
        print(mw, R[f'max{mw}min'])
    json.dump(R, open(OUT, 'w'), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
