"""Cusp probability map MVP — library functions.

Builds calibrated 2D P(cusp | MLAT, MLT, SW) from the existing 48k DMSP
crossings table by mining stratified synthetic negatives (5 near + 5 far per
positive), encoding spatial coordinates as polar Cartesian (x, y), and training
an XGBoost binary classifier with optional isotonic calibration.

Reuses paper-1's feature engineering from src/nn_dse.py:46-67.
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- data loading

def load_crossings(omni_dir: str = "output/omni_full_hist") -> pd.DataFrame:
    """Load all OMNI-matched cusp crossings, derive engineered features.

    Returns one row per crossing with columns:
      - identifiers: time_start, satellite, hemisphere
      - boundaries: eq_mlat, pole_mlat, eq_mlt, pole_mlt, mean_mlat, mean_mlt
      - 74 input features (SW + IMF + running stats + derived)
    """
    files = sorted(glob.glob(f"{omni_dir}/cusp_crossings_*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files in {omni_dir}")

    rows = []
    for f in files:
        with open(f) as fh:
            rows.extend(json.load(fh))
    df = pd.DataFrame(rows)

    # paper-1 feature engineering (src/nn_dse.py:42-52)
    df["abs_eq_mlat"] = df["eq_mlat"].abs()
    df["abs_pole_mlat"] = df["pole_mlat"].abs()
    df["abs_mean_mlat"] = df["mean_mlat"].abs()
    df["hemi_code"] = (df["hemisphere"] == "N").astype(float)
    df["doy"] = pd.to_datetime(df["time_start"]).dt.dayofyear
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
    df["by_hemi"] = df["imf_by"] * np.where(df["hemisphere"] == "N", 1, -1)

    # crossing_id = stable row index in dropna-cleaned order
    df = df.reset_index(drop=True)
    return df


SW_BASE_FEATURES = [
    "dipole_tilt", "hemi_code", "doy",
    "imf_bx", "imf_by", "imf_bz",
    "sw_v", "sw_n", "sw_pdyn",
    "B_T", "clock_angle", "sin_clock_half",
    "newell_cf", "kan_lee_ef", "vBs",
    "by_hemi",
]


def sw_feature_names(df: pd.DataFrame) -> list[str]:
    """Return the same ~74 SW features paper 1 used (base + history)."""
    hist = [
        c for c in df.columns
        if any(s in c for s in
               ["mean15", "mean30", "mean60",
                "std15", "std30", "std60",
                "delta15", "delta30", "delta60",
                "int60", "_mean60"])
    ]
    hist = [c for c in hist if c not in SW_BASE_FEATURES]
    return SW_BASE_FEATURES + sorted(hist)


# ----------------------------------------------------------- spatial encoding

def polar_xy(abs_mlat: np.ndarray, mlt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(|MLAT|, MLT) -> (x, y) on polar dial. Origin = magnetic pole."""
    r = 90.0 - np.asarray(abs_mlat)
    theta = 2 * np.pi * np.asarray(mlt) / 24.0
    return r * np.cos(theta), r * np.sin(theta)


# ----------------------------------------------------------- negative sampling

def _sample_far_negatives(n: int, exclude_box: tuple[float, float, float, float],
                          dial_box=(50.0, 90.0, 0.0, 24.0),
                          rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sample n random (|MLAT|, MLT) points on the dial, rejecting any inside exclude_box."""
    rng = rng or np.random.default_rng()
    eq_lo, pole_hi, mlt_lo, mlt_hi = exclude_box
    lat_lo, lat_hi, m_lo, m_hi = dial_box
    out_lat, out_mlt = [], []
    while len(out_lat) < n:
        cand_lat = rng.uniform(lat_lo, lat_hi, n * 3)
        cand_mlt = rng.uniform(m_lo, m_hi, n * 3)
        # exclude_box on (MLAT, MLT) is (eq-buf, pole+buf, mlt_lo-buf, mlt_hi+buf)
        in_box = (
            (cand_lat >= eq_lo) & (cand_lat <= pole_hi)
            & (cand_mlt >= mlt_lo) & (cand_mlt <= mlt_hi)
        )
        keep_lat = cand_lat[~in_box]
        keep_mlt = cand_mlt[~in_box]
        need = n - len(out_lat)
        out_lat.extend(keep_lat[:need].tolist())
        out_mlt.extend(keep_mlt[:need].tolist())
    return np.array(out_lat), np.array(out_mlt)


def expand_crossing(row: pd.Series, n_pos: int = 5, k_neg: int = 10,
                    buffer_lat: float = 2.0, buffer_mlt: float = 1.0,
                    rng: np.random.Generator | None = None) -> pd.DataFrame:
    """One crossing -> n_pos positives + n_pos * k_neg negatives.

    Negatives are split evenly: half near (boundary offset 1-5 deg / 0-1 h)
    and half far (random on dial with rejection of crossing region +/- buffer).
    """
    rng = rng or np.random.default_rng()
    eq_lat = abs(row["eq_mlat"])
    pole_lat = abs(row["pole_mlat"])
    eq_mlt = row["eq_mlt"]
    pole_mlt = row["pole_mlt"]
    # ensure ordered (eq_lat < pole_lat in absolute terms; some crossings reverse)
    if eq_lat > pole_lat:
        eq_lat, pole_lat = pole_lat, eq_lat
    if eq_mlt > pole_mlt:
        eq_mlt, pole_mlt = pole_mlt, eq_mlt

    n_near_half = k_neg // 2  # 5
    n_far_half = k_neg - n_near_half  # 5
    rows = []

    for _ in range(n_pos):
        # positive
        pos_lat = rng.uniform(eq_lat, pole_lat)
        pos_mlt = rng.uniform(eq_mlt, pole_mlt)
        rows.append({"abs_mlat": pos_lat, "mlt": pos_mlt, "label": 1})

        # near negatives: equator side + polar side
        eq_side_lat_lo = max(50.0, eq_lat - 5.0)
        eq_side_lat_hi = max(eq_side_lat_lo + 0.1, eq_lat - 1.0)
        pole_side_lat_lo = min(90.0, pole_lat + 1.0)
        pole_side_lat_hi = min(90.0, pole_lat + 5.0)
        if pole_side_lat_hi <= pole_side_lat_lo:
            pole_side_lat_hi = pole_side_lat_lo + 0.1
        n_eq = n_near_half // 2
        n_pole = n_near_half - n_eq
        near_lats = np.concatenate([
            rng.uniform(eq_side_lat_lo, eq_side_lat_hi, n_eq),
            rng.uniform(pole_side_lat_lo, pole_side_lat_hi, n_pole),
        ])
        near_mlts = rng.uniform(max(0.0, eq_mlt - 1.0), min(24.0, pole_mlt + 1.0), n_near_half)
        for lat, mlt in zip(near_lats, near_mlts):
            rows.append({"abs_mlat": lat, "mlt": mlt, "label": 0})

        # far negatives
        far_lats, far_mlts = _sample_far_negatives(
            n_far_half,
            exclude_box=(max(50.0, eq_lat - buffer_lat),
                         min(90.0, pole_lat + buffer_lat),
                         max(0.0, eq_mlt - buffer_mlt),
                         min(24.0, pole_mlt + buffer_mlt)),
            rng=rng,
        )
        for lat, mlt in zip(far_lats, far_mlts):
            rows.append({"abs_mlat": lat, "mlt": mlt, "label": 0})

    return pd.DataFrame(rows)


def expand_dataset(df: pd.DataFrame, n_pos: int = 5, k_neg: int = 10,
                   buffer_lat: float = 2.0, buffer_mlt: float = 1.0,
                   seed: int = 42, verbose: bool = True) -> pd.DataFrame:
    """Expand every crossing into n_pos + n_pos*k_neg rows.

    Returns expanded df with crossing_id, abs_mlat, mlt, x, y, label, plus
    all SW + identifier columns from the parent crossing.
    """
    rng = np.random.default_rng(seed)
    sw_cols = sw_feature_names(df)
    keep_cols = list(set(sw_cols + ["satellite", "hemisphere", "time_start"]))
    keep_cols = [c for c in keep_cols if c in df.columns]

    chunks = []
    t0 = time.time()
    for cid, row in df.iterrows():
        sub = expand_crossing(row, n_pos=n_pos, k_neg=k_neg,
                              buffer_lat=buffer_lat, buffer_mlt=buffer_mlt, rng=rng)
        sub["crossing_id"] = cid
        for c in keep_cols:
            sub[c] = row[c]
        chunks.append(sub)
        if verbose and (cid + 1) % 5000 == 0:
            print(f"  expanded {cid + 1} crossings  ({time.time() - t0:.1f}s)")
    out = pd.concat(chunks, ignore_index=True)
    x, y = polar_xy(out["abs_mlat"].values, out["mlt"].values)
    out["x_polar"] = x
    out["y_polar"] = y
    if verbose:
        print(f"  expand_dataset: {len(df)} crossings -> {len(out)} rows  ({time.time() - t0:.1f}s)")
    return out


# ---------------------------------------------------------------- model layer

@dataclass
class TrainedModel:
    model: object  # XGBClassifier
    isotonic: Optional[object]  # IsotonicRegression or None
    feature_names: list[str]
    used_calibration: bool


def build_feature_matrix(df: pd.DataFrame, sw_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Stack SW features + (x_polar, y_polar). Drop hemi_code duplication (it's in sw_cols already)."""
    feats = list(sw_cols) + ["x_polar", "y_polar"]
    # dedup preserving order
    seen = set()
    feats = [f for f in feats if not (f in seen or seen.add(f))]
    X = df[feats].values.astype(np.float32)
    return X, feats


def fit_xgb(X_train, y_train, X_val, y_val, hp_overrides: dict | None = None):
    from xgboost import XGBClassifier
    hp = dict(
        objective="binary:logistic",
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        scale_pos_weight=10.0,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="logloss",
    )
    if hp_overrides:
        hp.update(hp_overrides)
    model = XGBClassifier(**hp)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def maybe_calibrate(model, X_cal, y_cal, deviation_threshold: float = 0.05) -> tuple[Optional[object], dict]:
    """Fit isotonic if raw reliability deviates from diagonal beyond threshold in [0.1, 0.9]."""
    from sklearn.calibration import calibration_curve
    from sklearn.isotonic import IsotonicRegression

    raw_p = model.predict_proba(X_cal)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_cal, raw_p, n_bins=15)
    mask = (mean_pred >= 0.1) & (mean_pred <= 0.9)
    if mask.any():
        dev = np.max(np.abs(frac_pos[mask] - mean_pred[mask]))
    else:
        dev = np.max(np.abs(frac_pos - mean_pred))
    info = {"raw_reliability_deviation": float(dev),
            "frac_pos": frac_pos.tolist(),
            "mean_pred": mean_pred.tolist()}
    if dev < deviation_threshold:
        info["calibration"] = "skipped"
        return None, info
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_p, y_cal)
    info["calibration"] = "isotonic"
    return iso, info


def predict_proba(trained: TrainedModel, X: np.ndarray) -> np.ndarray:
    raw = trained.model.predict_proba(X)[:, 1]
    if trained.isotonic is not None:
        return trained.isotonic.transform(raw)
    return raw


# ------------------------------------------------------------------ inference

def infer_grid(trained: TrainedModel, sw_state: dict, hemisphere: str = "N",
               mlat_range=(50, 90, 1.0), mlt_range=(0, 24, 0.5)) -> dict:
    """Predict P over a (|MLAT|, MLT) grid for a given SW state.

    sw_state: dict with values for each of the 74 SW features (e.g. imf_bz=-5, sw_v=500, ...).
    Returns {'mlat': arr40, 'mlt': arr48, 'P': arr40x48}.
    """
    lat_lo, lat_hi, dlat = mlat_range
    mlt_lo, mlt_hi, dmlt = mlt_range
    mlat_axis = np.arange(lat_lo, lat_hi + 1e-9, dlat)
    mlt_axis = np.arange(mlt_lo, mlt_hi + 1e-9, dmlt)
    MM, LL = np.meshgrid(mlt_axis, mlat_axis)
    n = MM.size
    x, y = polar_xy(LL.ravel(), MM.ravel())

    # broadcast SW state to n rows; add x, y; pick out features in trained.feature_names order
    rec = dict(sw_state)
    rec["hemi_code"] = 1.0 if hemisphere == "N" else 0.0
    grid = {k: np.full(n, v, dtype=np.float32) for k, v in rec.items()}
    grid["x_polar"] = x.astype(np.float32)
    grid["y_polar"] = y.astype(np.float32)
    df = pd.DataFrame(grid)
    missing = [f for f in trained.feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"sw_state is missing features: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    X = df[trained.feature_names].values.astype(np.float32)
    P = predict_proba(trained, X).reshape(LL.shape)
    return {"mlat": mlat_axis, "mlt": mlt_axis, "P": P}


# -------------------------------------------------------------------- splits

def crossing_random_split(df_expanded: pd.DataFrame, frac_test: float = 0.2,
                          frac_val: float = 0.1, frac_cal: float = 0.1,
                          seed: int = 42) -> dict:
    """Group-by-crossing_id split: train / val / cal / test.

    Train fraction = 1 - frac_test - frac_val - frac_cal of crossings (60% by default).
    """
    rng = np.random.default_rng(seed)
    cids = df_expanded["crossing_id"].unique()
    rng.shuffle(cids)
    n = len(cids)
    n_test = int(n * frac_test)
    n_val = int(n * frac_val)
    n_cal = int(n * frac_cal)
    test_ids = set(cids[:n_test])
    val_ids = set(cids[n_test:n_test + n_val])
    cal_ids = set(cids[n_test + n_val:n_test + n_val + n_cal])
    train_ids = set(cids[n_test + n_val + n_cal:])

    return {
        "train": df_expanded[df_expanded["crossing_id"].isin(train_ids)].reset_index(drop=True),
        "val": df_expanded[df_expanded["crossing_id"].isin(val_ids)].reset_index(drop=True),
        "cal": df_expanded[df_expanded["crossing_id"].isin(cal_ids)].reset_index(drop=True),
        "test": df_expanded[df_expanded["crossing_id"].isin(test_ids)].reset_index(drop=True),
    }


def crossing_temporal_split(df_expanded: pd.DataFrame,
                            cutoff_year: int = 2008,
                            frac_val: float = 0.125,
                            frac_cal: float = 0.125,
                            seed: int = 42) -> dict:
    """Train < cutoff_year, test >= cutoff_year; val + cal carved from train."""
    rng = np.random.default_rng(seed)
    yr = pd.to_datetime(df_expanded["time_start"]).dt.year.values
    test_mask = yr >= cutoff_year
    train_pool_mask = ~test_mask

    pool_cids = df_expanded.loc[train_pool_mask, "crossing_id"].unique()
    rng.shuffle(pool_cids)
    n_pool = len(pool_cids)
    n_val = int(n_pool * frac_val)
    n_cal = int(n_pool * frac_cal)
    val_ids = set(pool_cids[:n_val])
    cal_ids = set(pool_cids[n_val:n_val + n_cal])
    train_ids = set(pool_cids[n_val + n_cal:])

    df = df_expanded
    return {
        "train": df[train_pool_mask & df["crossing_id"].isin(train_ids)].reset_index(drop=True),
        "val": df[train_pool_mask & df["crossing_id"].isin(val_ids)].reset_index(drop=True),
        "cal": df[train_pool_mask & df["crossing_id"].isin(cal_ids)].reset_index(drop=True),
        "test": df[test_mask].reset_index(drop=True),
    }


# ----------------------------------------------------------------- evaluation

def evaluate(trained: TrainedModel, test_df: pd.DataFrame, sw_cols: list[str]) -> dict:
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss)
    from sklearn.calibration import calibration_curve

    X, _ = build_feature_matrix(test_df, sw_cols)
    y = test_df["label"].values.astype(int)
    p = predict_proba(trained, X)

    metrics = {
        "auc_roc": float(roc_auc_score(y, p)),
        "auc_pr": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "n_test": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
    }
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=15)
    mask = (mean_pred >= 0.1) & (mean_pred <= 0.9)
    if mask.any():
        metrics["reliability_deviation"] = float(np.max(np.abs(frac_pos[mask] - mean_pred[mask])))
    else:
        metrics["reliability_deviation"] = float(np.max(np.abs(frac_pos - mean_pred)))
    metrics["reliability_frac_pos"] = frac_pos.tolist()
    metrics["reliability_mean_pred"] = mean_pred.tolist()

    # per-MLT-bin AUC
    mlt_bins = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
    per_mlt = {}
    for lo, hi in mlt_bins:
        m = (test_df["mlt"].values >= lo) & (test_df["mlt"].values < hi)
        if m.sum() < 50 or y[m].sum() < 5 or (y[m] == 0).sum() < 5:
            per_mlt[f"{lo}-{hi}"] = None
        else:
            per_mlt[f"{lo}-{hi}"] = float(roc_auc_score(y[m], p[m]))
    metrics["per_mlt_auc"] = per_mlt

    # hemisphere stratified
    metrics["hemi"] = {}
    for hcode, hname in [(1.0, "N"), (0.0, "S")]:
        m = (test_df["hemi_code"].values == hcode)
        if m.sum() < 50 or y[m].sum() < 5 or (y[m] == 0).sum() < 5:
            metrics["hemi"][hname] = None
            continue
        metrics["hemi"][hname] = {
            "auc_roc": float(roc_auc_score(y[m], p[m])),
            "brier": float(brier_score_loss(y[m], p[m])),
            "n_test": int(m.sum()),
        }
    return metrics
