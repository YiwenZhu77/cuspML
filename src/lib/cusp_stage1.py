"""Stage 1: P(cusp observed | SW) for the two-stage cusp probability product.

Reads NASA OMNI2 hourly flat file (omni2_all_years.dat). For each hour in the
crossings era, labels positive if any DMSP cusp crossing in our 48k table fell
in that hour, negative otherwise. Trains a binary XGBoost classifier on 16
base SW features.

Combined product at inference: P(cusp at (MLAT, MLT) | SW)
                              = stage1(SW) * stage2_R002(MLAT, MLT, SW)

OMNI2 hourly column layout (1-indexed):
  1  YEAR
  2  DOY (1-366)
  3  HOUR (0-23)
  ...
  13 Bx GSE/GSM (nT)
  16 By GSM (nT)
  17 Bz GSM (nT)
  24 N proton density (n/cc)
  25 V flow speed (km/s)
  29 Pressure (nPa)
  43 AE (nT)

Fill values: 999.9 for B/V/N, 9999 for T/AE, etc.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd


OMNI2_COLS_NEEDED = {
    "year": 0,
    "doy": 1,
    "hour": 2,
    "imf_bx": 12,
    "imf_by": 15,
    "imf_bz": 16,
    "sw_n": 23,
    "sw_v": 24,
    "sw_pdyn": 28,
    "ae_index": 42,
}

FILL_LOOKUP = {
    "imf_bx": 999.9, "imf_by": 999.9, "imf_bz": 999.9,
    "sw_n": 999.9, "sw_v": 9999.0, "sw_pdyn": 99.99,
    "ae_index": 9999,
}


def load_omni2_hourly(path: str, year_min: int = 1987, year_max: int = 2014) -> pd.DataFrame:
    """Parse omni2_all_years.dat into a DataFrame restricted to [year_min, year_max]."""
    cols = list(OMNI2_COLS_NEEDED.values())
    names = list(OMNI2_COLS_NEEDED.keys())
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=cols, names=names,
                     engine="c", low_memory=False)
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)].reset_index(drop=True)

    # build datetime from year + doy + hour
    df["datetime"] = (
        pd.to_datetime(df["year"].astype(str) + df["doy"].astype(str).str.zfill(3),
                       format="%Y%j")
        + pd.to_timedelta(df["hour"], unit="h")
    )

    # mark fill values as NaN
    for k, fill in FILL_LOOKUP.items():
        df.loc[df[k] >= fill * 0.99, k] = np.nan  # tolerate float drift

    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the same derived features paper 1 uses (B_T, clock_angle, newell_cf, etc).
    Excludes history features (mean15/30/60 etc) — those require sub-hour cadence.
    """
    df = df.copy()
    df["doy_feat"] = df["datetime"].dt.dayofyear  # for consistency with stage 2 'doy'
    df["B_T"] = np.sqrt(df["imf_by"] ** 2 + df["imf_bz"] ** 2)
    df["clock_angle"] = np.arctan2(df["imf_by"], df["imf_bz"])
    df["sin_clock_half"] = np.sin(df["clock_angle"] / 2)
    df["newell_cf"] = (
        df["sw_v"] ** (4 / 3)
        * df["B_T"] ** (2 / 3)
        * np.abs(df["sin_clock_half"]) ** (8 / 3)
    )
    df["kan_lee_ef"] = df["sw_v"] * df["B_T"] * df["sin_clock_half"] ** 2
    df["vBs"] = df["sw_v"] * np.where(df["imf_bz"] < 0, -df["imf_bz"], 0)
    return df


STAGE1_BASE_FEATURES = [
    "doy_feat",
    "imf_bx", "imf_by", "imf_bz",
    "sw_v", "sw_n", "sw_pdyn",
    "B_T", "clock_angle", "sin_clock_half",
    "newell_cf", "kan_lee_ef", "vBs",
]
# AE intentionally excluded: the 48k positives were pre-filtered by add_omni.py to
# AE < 100 nT (Anderson 2024 criterion). Including AE would teach stage 1 the
# filter, not the physics, and would systematically underpredict during storms.


def label_hours(omni: pd.DataFrame, crossings: pd.DataFrame) -> np.ndarray:
    """For each hour row in omni, label 1 if any crossing in 48k table fell in that hour.

    Returns boolean array len(omni).
    """
    cross_time = pd.to_datetime(crossings["time_start"])
    cross_hour = cross_time.dt.floor("h")
    pos_hour_set = set(cross_hour.astype(np.int64).tolist())
    omni_hour = omni["datetime"].dt.floor("h").astype(np.int64).values
    return np.isin(omni_hour, list(pos_hour_set))


def fit_stage1(X_train, y_train, X_val, y_val):
    from xgboost import XGBClassifier
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=10,
        scale_pos_weight=1.0,  # default; let isotonic absorb whatever imbalance remains
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model
