"""Load 1-min OMNI flat file and compute history statistics matching paper-1.

NASA high-res OMNI 1-min column layout (1-indexed, see
https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/hroformat.txt):
  1  Year
  2  DOY
  3  Hour
  4  Minute
  ...
  15 Bx GSE/GSM (nT)
  18 By GSM (nT)
  19 Bz GSM (nT)
  22 Flow speed (km/s)
  26 Proton density (n/cc)
  28 Flow pressure (nPa)

Fill values: 9999.99 (B), 99999.9 (V), 999.99 (n, Pdyn).
"""
import numpy as np
import pandas as pd


COLS_NEEDED = {
    "year": 0, "doy": 1, "hour": 2, "minute": 3,
    "imf_bx": 14, "imf_by": 17, "imf_bz": 18,
    "sw_v": 21, "sw_n": 25, "sw_pdyn": 27,
}
FILL_TOL = {
    "imf_bx": 9000.0, "imf_by": 9000.0, "imf_bz": 9000.0,
    "sw_v": 90000.0, "sw_n": 900.0, "sw_pdyn": 90.0,
}


def load_omni_1min(path: str) -> pd.DataFrame:
    cols = list(COLS_NEEDED.values())
    names = list(COLS_NEEDED.keys())
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=cols, names=names,
                     engine="c", low_memory=False)
    df["datetime"] = (
        pd.to_datetime(df["year"].astype(str) + df["doy"].astype(str).str.zfill(3),
                       format="%Y%j")
        + pd.to_timedelta(df["hour"], unit="h")
        + pd.to_timedelta(df["minute"], unit="m")
    )
    for k, tol in FILL_TOL.items():
        df.loc[df[k].abs() >= tol, k] = np.nan
    return df


def compute_history(df: pd.DataFrame, base_cols=("imf_bx", "imf_by", "imf_bz",
                                                  "sw_v", "sw_n", "sw_pdyn")) -> pd.DataFrame:
    """For each minute row, compute mean/std/delta over 15/30/60-min windows.

    delta_w(t) = value(t) - value(t-w)  (matching paper-1 add_omni.py convention)
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    for c in base_cols:
        for w in (15, 30, 60):
            r = df[c].rolling(window=w, min_periods=max(3, w // 3))
            df[f"{c}_mean{w}"] = r.mean()
            df[f"{c}_std{w}"] = r.std()
            df[f"{c}_delta{w}"] = df[c] - df[c].shift(w)
    return df


def derive_paper1_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["B_T"] = np.sqrt(df["imf_by"] ** 2 + df["imf_bz"] ** 2)
    df["clock_angle"] = np.arctan2(df["imf_by"], df["imf_bz"])
    df["sin_clock_half"] = np.sin(df["clock_angle"] / 2)
    df["newell_cf"] = (df["sw_v"] ** (4/3) * df["B_T"] ** (2/3)
                       * np.abs(df["sin_clock_half"]) ** (8/3))
    df["kan_lee_ef"] = df["sw_v"] * df["B_T"] * df["sin_clock_half"] ** 2
    df["vBs"] = df["sw_v"] * np.where(df["imf_bz"] < 0, -df["imf_bz"], 0)
    # 60-min averages of derived
    df["newell_cf_mean60"] = df["newell_cf"].rolling(60, min_periods=20).mean()
    df["newell_cf_int60"] = df["newell_cf"].rolling(60, min_periods=20).sum()
    df["vBs_mean60"] = df["vBs"].rolling(60, min_periods=20).mean()
    df["vBs_int60"] = df["vBs"].rolling(60, min_periods=20).sum()
    df["doy"] = df["datetime"].dt.dayofyear
    df["hemi_code"] = 1.0
    df["by_hemi"] = df["imf_by"]
    df["dipole_tilt"] = 0.0
    return df
