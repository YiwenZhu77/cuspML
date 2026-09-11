#!/usr/bin/env python3
"""Train the revised manuscript models from the released, ordered crossing table.

Inputs: rows.parquet, manifest.json, calibration.parquet (baselines),
        dst.parquet (storm controls). No private paths or prediction-cache reads.
The supplied precipitation labels are inputs; this is not a raw DMSP decoder.
Latitude targets are absolute AACGM latitude [degrees]; MLT targets are [hours].
Example: python train.py --data-dir data/current --output-dir results/current \
             --tasks primary baselines other_targets windows balanced --jobs 6
Long runs including controls should use a compute node. No test-set tuning occurs.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

TARGETS = ("abs_eq_mlat", "abs_pole_mlat", "eq_mlt", "mean_mlt")
TASKS = ("primary", "other_targets", "baselines", "windows", "balanced", "controls", "independent")
RAW_KEYS = ("imf_bx", "imf_by", "imf_bz", "sw_v", "sw_n", "sw_pdyn")


def sha256(path):
    """Hash a released input without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_input(data_dir, name, manifest=None):
    """Refuse altered release inputs before using them for scientific computation."""
    if manifest is None:
        manifest = json.loads((data_dir / "manifest.json").read_text())
    entry = manifest.get("files", {}).get(name)
    if not isinstance(entry, dict) or not entry.get("sha256"):
        raise ValueError(f"Missing manifest SHA-256 for {name}")
    actual = sha256(data_dir / name)
    if actual != entry["sha256"]:
        raise ValueError(f"Input SHA-256 mismatch: {name}")
    return actual


def metrics(observed, predicted):
    """Use float64 residual arithmetic, matching the manuscript analysis."""
    y, p = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError("Metrics require equal, nonempty one-dimensional arrays")
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("Nonfinite observations or predictions")
    error = p - y
    variance = np.sum((y - y.mean()) ** 2)
    return dict(n=len(y), MAE=float(abs(error).mean()),
                RMSE=float(np.sqrt(np.mean(error ** 2))),
                r=float(np.corrcoef(y, p)[0, 1]) if np.std(y) and np.std(p) else None,
                R2=float(1 - np.sum(error ** 2) / variance) if variance else None,
                mean=float(error.mean()), median=float(np.median(error)),
                within1=float(np.mean(abs(error) <= 1)),
                within2=float(np.mean(abs(error) <= 2)),
                above3=float(np.mean(abs(error) > 3)))


def load_inputs(data_dir):
    """Enforce released feature order, finite primary inputs, and chronological split."""
    manifest = json.loads((data_dir / "manifest.json").read_text())
    features = manifest.get("features", manifest.get("feature_names"))
    parameters = manifest.get("parameters", manifest.get("xgboost_parameters"))
    if not isinstance(features, list) or len(features) != 74 or len(set(features)) != 74:
        raise ValueError("manifest must contain 74 unique ordered features")
    if not isinstance(parameters, dict):
        raise ValueError("manifest must contain XGBoost parameters")
    verify_input(data_dir, "rows.parquet", manifest)
    frame = pd.read_parquet(data_dir / "rows.parquet")
    missing = set(features + list(TARGETS) + ["year", "time_start", "hemisphere"]) - set(frame)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    x = frame[features].to_numpy(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(frame[list(TARGETS)].to_numpy(float)).all():
        raise ValueError("Primary features and targets must be complete; rows are never filtered")
    times = pd.to_datetime(frame.time_start, utc=True)
    years = frame.year.to_numpy()
    if not np.array_equal(years, times.dt.year.to_numpy()):
        raise ValueError("year and time_start disagree")
    train, test = np.flatnonzero(years < 2008), np.flatnonzero(years >= 2008)
    if (len(frame), len(train), len(test)) != (39668, 29935, 9733):
        raise ValueError("Expected the released 39,668 rows, with 29,935 train and 9,733 test")
    return frame, features, parameters.copy(), x, train, test


def newell_predictor(frame):
    """Newell 2006 E_WAV^(2/3), using the table's retained derived inputs.

    E_WAV = v * B_T * sin(clock_angle / 2)^4. This is not dPhi/dt
    raised again to 2/3. The separated powers match calibration arithmetic.
    """
    return (frame.sw_v.to_numpy(dtype=float) ** (2 / 3)
            * frame.B_T.to_numpy(dtype=float) ** (2 / 3)
            * np.abs(frame.sin_clock_half.to_numpy(dtype=float)) ** (8 / 3))


def fit_newell(data_dir, frame, test):
    """Refit the frozen training calibration, then predict the current test table.

    Baseline calibration precedes the final feature refresh. Its training rows
    are released explicitly rather than silently refitting a different baseline.
    """
    path = data_dir / "calibration.parquet"
    verify_input(data_dir, path.name)
    calibration = pd.read_parquet(path)
    if len(calibration) != 29935 or not (calibration.year.to_numpy() < 2008).all():
        raise ValueError("Newell calibration must contain exactly the pre-2008 training rows")
    z = newell_predictor(calibration)
    y = calibration.abs_eq_mlat.to_numpy(np.float32).astype(float)
    if not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite calibration input")
    coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(z)), z]), y, rcond=None)[0]
    prediction = coefficients[0] + coefficients[1] * newell_predictor(frame.iloc[test])
    return prediction, dict(coefficients=coefficients.tolist(),
                            calibration_file=path.name, calibration_sha256=sha256(path),
                            n_calibration=len(calibration),
                            calibration_scope="frozen pre-refresh pre-2008 training calibration",
                            equation="intercept + slope * (v * B_T * sin(clock/2)^4)^(2/3)")


def history_features(features, window):
    """Cumulative histories: retain all shorter windows and the 60-min coupling block."""
    selected = features[:16] + [name for name in features[16:]
                               if int(re.search(r"(\d+)$", name)[1]) <= window]
    if window >= 90:
        selected += sorted(f"{key}_{stat}{width}" for key in RAW_KEYS
                           for stat in ("mean", "std", "delta")
                           for width in (90, 120) if width <= window)
    return selected


def storm_groups(frame, data_dir):
    """Reproduce the published event grouping from the released hourly Dst table."""
    verify_input(data_dir, "dst.parquet")
    dst = pd.read_parquet(data_dir / "dst.parquet")
    t = pd.to_datetime(dst.time, utc=True).dt.tz_localize(None).to_numpy()
    v = dst.dst.to_numpy()
    if not np.isfinite(v).all() or np.any(t[1:] < t[:-1]):
        raise ValueError("Dst must be finite and time-ordered")
    events, i = [], 0
    while i < len(v):
        if v[i] > -30:
            i += 1
            continue
        j = i
        while j < len(v) and v[j] <= -30:
            j += 1
        a = i
        while a > 0 and v[a - 1] <= -15:
            a -= 1
        b = j
        while b < len(v):
            if v[b] > -20 and np.all(v[b:min(b + 6, len(v))] > -20):
                break
            b += 1
        events.append((t[a], t[min(b, len(v) - 1)]))
        i = max(j, b)
    merged = []
    for start, end in events:
        if merged and start - merged[-1][1] <= np.timedelta64(24, "h"):
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    times = pd.to_datetime(frame.time_start, utc=True).dt.tz_localize(None)
    tv = times.to_numpy()
    groups = times.dt.strftime("Q%Y-%m").to_numpy(dtype=object)
    is_storm = np.zeros(len(frame), dtype=bool)
    for index, (start, end) in enumerate(merged):
        mask = (tv >= start) & (tv <= end)
        groups[mask] = f"S{index}"
        is_storm |= mask
    return groups, is_storm


def control_folds(frame, x, y, data_dir):
    """Every returned fold is evaluated only on that fold's excluded observations."""
    times = pd.to_datetime(frame.time_start, utc=True)
    days = times.dt.strftime("%Y-%m-%d")
    day = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(x, y, days))
    groups, is_storm = storm_groups(frame, data_dir)
    folds = {"day": [day], "storm": list(GroupKFold(n_splits=5).split(x, y, groups))}
    order = np.argsort(times.to_numpy(), kind="stable")
    folds["block"] = [(np.setdiff1d(np.arange(len(y)), b), b) for b in np.array_split(order, 5)]
    years = frame.year.to_numpy()
    folds["year"] = [(np.flatnonzero(years != yr), np.flatnonzero(years == yr))
                     for yr in sorted(np.unique(years)) if (years == yr).sum() >= 100]
    return folds, is_storm


def run(args):
    """Train requested tasks from scratch and write predictions, models and metrics."""
    data_dir, output = args.data_dir.resolve(), args.output_dir.resolve()
    if data_dir == output:
        raise ValueError("Output directory must differ from input directory")
    output.mkdir(parents=True, exist_ok=True)
    predictions_dir, models_dir = output / "predictions", output / "models"
    predictions_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    frame, features, parameters, x, train, test = load_inputs(data_dir)
    parameters["n_jobs"] = args.jobs
    y = frame.abs_eq_mlat.to_numpy(np.float32)
    selected = set(TASKS if "all" in args.tasks else args.tasks)
    report = dict(n_train=len(train), n_test=len(test), features=features, parameters=parameters,
                  models={}, tasks=sorted(selected), split="train <2008; test >=2008",
                  input_sha256={name: sha256(data_dir / name) for name in ("rows.parquet", "manifest.json")},
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                pandas=pd.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__),
                  elapsed_seconds={})
    primary_prediction = None

    def save_report():
        (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def save_prediction(name, prediction, target, indices):
        np.savez_compressed(predictions_dir / f"{name}.npz", predicted=prediction,
                            observed=target[indices], test_indices=indices)

    def fit_xgb(name, matrix, target, training, testing, names):
        if np.intersect1d(training, testing).size:
            raise ValueError("Training and test rows overlap")
        started = time.monotonic()
        model = XGBRegressor(**parameters).fit(matrix[training], target[training])
        prediction = model.predict(matrix[testing])
        save_prediction(name, prediction, target, testing)
        model.save_model(models_dir / f"{name}.ubj")
        (models_dir / f"{name}.features.json").write_text(json.dumps(names, indent=2) + "\n")
        report["elapsed_seconds"][name] = time.monotonic() - started
        print(f"{name}: MAE={metrics(target[testing], prediction)['MAE']:.10f}; "
              f"{report['elapsed_seconds'][name]:.1f} s", flush=True)
        return prediction

    if selected & {"primary", "windows", "baselines", "independent"}:
        primary_prediction = fit_xgb("primary", x, y, train, test, features)
        report["primary"] = metrics(y[test], primary_prediction)
        report["models"]["XGBoost74"] = report["primary"]
        save_report()
    if "other_targets" in selected:
        report["other_targets"] = {}
        for target in TARGETS[1:]:
            target_y = frame[target].to_numpy(np.float32)
            prediction = fit_xgb(target, x, target_y, train, test, features)
            report["other_targets"][target] = metrics(target_y[test], prediction)
            save_report()
    if "baselines" in selected:
        baseline, calibration = fit_newell(data_dir, frame, test)
        save_prediction("Newell2006", baseline, y, test)
        report["baseline"] = metrics(y[test], baseline)
        report["newell_calibration"] = calibration
        report["models"]["Newell (2006) form"] = report["baseline"]
        report["reduction_pct"] = 100 * (1 - report["primary"]["MAE"] / report["baseline"]["MAE"])
        baseline_models = {
            "Ridge74": make_pipeline(StandardScaler(), Ridge(alpha=1.)),
            "GBR300": GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42),
            "MLP74": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 128),
                activation="relu", alpha=1e-3, max_iter=500, early_stopping=True,
                n_iter_no_change=15, random_state=42)),
        }
        import joblib
        for name, estimator in baseline_models.items():
            started = time.monotonic()
            # One BLAS thread reproduces the frozen Ridge float32 predictions.
            # MLP retains its original six-thread setting.
            with threadpool_limits(limits=1 if name == "Ridge74" else args.jobs):
                estimator.fit(x[train], y[train])
                prediction = estimator.predict(x[test])
            save_prediction(name, prediction, y, test)
            joblib.dump(estimator, models_dir / f"{name}.joblib")
            report["models"][name] = metrics(y[test], prediction)
            report["elapsed_seconds"][name] = time.monotonic() - started
            print(f"{name}: MAE={report['models'][name]['MAE']:.10f}; "
                  f"{report['elapsed_seconds'][name]:.1f} s", flush=True)
            save_report()
    if "windows" in selected:
        report["windows"] = {}
        for width in (0, 15, 30, 60, 90, 120):
            names = history_features(features, width)
            missing = set(names) - set(frame)
            if missing:
                raise ValueError(f"Missing extended window columns: {sorted(missing)}")
            matrix = frame[names].to_numpy(np.float32)
            # Extended inputs may contain NaN; XGBoost handles these without dropping rows.
            if np.isinf(matrix).any():
                raise ValueError("Infinite extended input")
            prediction = primary_prediction if width == 60 else fit_xgb(
                f"window{width}", matrix, y, train, test, names)
            report["windows"][str(width)] = dict(n_features=len(names), **metrics(y[test], prediction))
            save_report()
    if "balanced" in selected:
        rng = np.random.default_rng(42)
        h = frame.hemisphere.to_numpy()
        north, south = train[h[train] == "N"], train[h[train] == "S"]
        count = min(len(north), len(south))
        balanced = np.concatenate([rng.choice(north, count, replace=False),
                                   rng.choice(south, count, replace=False)])
        prediction = fit_xgb("balanced", x, y, balanced, test, features)
        report["balanced"] = {"n_per_hemisphere": count, **{
            hemi: metrics(y[test][h[test] == hemi], prediction[h[test] == hemi]) for hemi in ("N", "S")}}
        save_report()
    if "independent" in selected:
        path = data_dir / "independent" / "history_features.npz"
        verify_input(data_dir, "independent/history_features.npz")
        history = np.load(path)["features"]
        if history.shape != (3, len(frame), 22):
            raise ValueError("Independent history input must have shape (3, 39668, 22)")
        block = [f"{key}_{stat}60" for key in RAW_KEYS for stat in ("mean", "std", "delta")]
        block += ["newell_cf_mean60", "newell_cf_int60", "vBs_mean60", "vBs_int60"]
        np.testing.assert_array_equal(history[0].astype(np.float32), frame[block].to_numpy(np.float32))
        report["input_sha256"]["independent/history_features.npz"] = sha256(path)
        report["independent"] = {"models": {"anchor60": report["primary"]}}
        predictions = {"anchor60": primary_prediction}
        for group, replaced in (("coupling4", block[-4:]), ("full_block22", block)):
            for width in (90, 120):
                matrix, names = x.copy(), features.copy()
                for column in replaced:
                    index = features.index(column)
                    matrix[:, index] = history[(60, 90, 120).index(width), :, block.index(column)].astype(np.float32)
                    names[index] = column[:-2] + str(width)
                label = f"{group}_{width}"
                prediction = fit_xgb(label, matrix, y, train, test, names)
                predictions[label] = prediction
                report["independent"]["models"][label] = dict(
                    **metrics(y[test], prediction), feature_names=names,
                    replaced_original_columns=replaced,
                    matrix_sha256=hashlib.sha256(matrix.tobytes()).hexdigest())
                save_report()
        months = pd.to_datetime(frame.iloc[test].time_start, utc=True).dt.strftime("%Y-%m").to_numpy()
        unique, inverse = np.unique(months, return_inverse=True)
        sizes = np.bincount(inverse)
        draws = np.random.default_rng(20260909).integers(len(unique), size=(5000, len(unique)))
        denominators = sizes[draws].sum(axis=1)
        base_error = abs(primary_prediction.astype(float) - y[test])
        bootstrap = dict(months=len(unique), replicates=5000, seed=20260909, comparisons={})
        for label, prediction in predictions.items():
            if label == "anchor60":
                continue
            difference = abs(prediction.astype(float) - y[test]) - base_error
            distribution = np.bincount(inverse, weights=difference)[draws].sum(axis=1) / denominators
            bootstrap["comparisons"][label] = dict(MAE_difference=float(difference.mean()),
                                                   CI95=np.quantile(distribution, [.025, .975]).tolist())
        report["independent"]["paired_month_bootstrap"] = bootstrap
        save_report()
    if "controls" in selected:
        folds, is_storm = control_folds(frame, x, y, data_dir)
        report["storm_fraction"] = float(is_storm.mean())
        report["input_sha256"]["dst.parquet"] = sha256(data_dir / "dst.parquet")
        report["controls"] = {}
        for kind, splits in folds.items():
            scores = []
            for index, (training, testing) in enumerate(splits):
                prediction = fit_xgb(f"control_{kind}_{index}", x, y, training, testing, features)
                scores.append(metrics(y[testing], prediction))
            maes = [score["MAE"] for score in scores]
            report["controls"][kind] = dict(MAE_mean=float(np.mean(maes)),
                                             MAE_std=float(np.std(maes)), folds=scores)
            save_report()
    save_report()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=(*TASKS, "all"), default=["primary"])
    parser.add_argument("--jobs", type=int, choices=range(1, 7), default=6)
    args = parser.parse_args()
    if set(args.tasks) & {"all", "baselines"} and args.jobs != 6:
        parser.error("Exact MLP reproduction requires --jobs 6 (the original BLAS thread count)")
    with threadpool_limits(limits=args.jobs):
        run(args)


if __name__ == "__main__":
    main()
