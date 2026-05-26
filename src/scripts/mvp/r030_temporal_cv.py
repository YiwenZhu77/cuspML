"""R030 (thesis chapter, deliverable 1) — 5-fold temporal cross-validation.

Splits 27 years into 5 contiguous folds:
  fold 0: train all-except-{1990-1994}, test 1990-1994
  fold 1: train all-except-{1995-1999}, test 1995-1999
  fold 2: train all-except-{2000-2004}, test 2000-2004
  fold 3: train all-except-{2005-2009}, test 2005-2009
  fold 4: train all-except-{2010-2014}, test 2010-2014

Retrains stage 2 ONLY (stage 1 left as R015 — opportunity-restricted is
quasi-temporal-invariant). Per fold, measures end-to-end median peak distance
and mean true-cell logp on held-out crossings of that fold's test years.
"""
import json, os, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, "/glade/work/yizhu/cuspML/src/lib")
sys.path.insert(0, "/glade/work/yizhu/cuspML/src/scripts/mvp")
from cusp_map import (load_crossings, sw_feature_names, expand_dataset,
                       build_feature_matrix, fit_xgb, maybe_calibrate, TrainedModel,
                       predict_proba, polar_xy)
from cusp_stage1 import STAGE1_BASE_FEATURES
from r014_endtoend_eval import (LAT_AXIS, MLT_AXIS, cell_of, stage2_dial,
                                 normalize_pmf, haversine_deg)
from r016_combined_v2 import load_stage1_v2, stage1_scalar_v2

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles"

FOLDS = [
    ("1990-1994", 1990, 1994),
    ("1995-1999", 1995, 1999),
    ("2000-2004", 2000, 2004),
    ("2005-2009", 2005, 2009),
    ("2010-2014", 2010, 2014),
]


def main():
    t0 = time.time()
    # load stage 1 (shared across folds)
    s1_model, s1_iso, s1_feats = load_stage1_v2()

    # load expanded dataset (cached)
    parquet = f"{OUT_DIR}/expanded_full.parquet"
    if not os.path.exists(parquet):
        print("regenerating expanded ...")
        df = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
        sw = sw_feature_names(df)
        keep = sw + ["eq_mlat", "pole_mlat", "eq_mlt", "pole_mlt",
                     "satellite", "hemisphere", "time_start"]
        df_clean = df[keep].dropna().reset_index(drop=True)
        expanded = expand_dataset(df_clean, n_pos=5, k_neg=10, seed=42, verbose=False)
        expanded.to_parquet(parquet)
    else:
        expanded = pd.read_parquet(parquet)
    expanded["year"] = pd.to_datetime(expanded["time_start"]).dt.year
    print(f"expanded {len(expanded)} rows, years {expanded['year'].min()}-{expanded['year'].max()}")
    sw_cols = sw_feature_names(expanded)

    # also load full crossings (for end-to-end eval on actual crossings, not expanded rows)
    df_x = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    sw_x = sw_feature_names(df_x)
    keep_x = sw_x + ["eq_mlat", "pole_mlat", "eq_mlt", "pole_mlt",
                     "mean_mlat", "mean_mlt",
                     "satellite", "hemisphere", "time_start"]
    df_x = df_x[keep_x].dropna().reset_index(drop=True)
    df_x["year"] = pd.to_datetime(df_x["time_start"]).dt.year

    fold_results = []
    for label, y0, y1 in FOLDS:
        print(f"\n=== FOLD {label} ({y0}-{y1}) ===")
        t_fold = time.time()
        # train: all years OUTSIDE the fold window
        train_mask = (expanded["year"] < y0) | (expanded["year"] > y1)
        test_xings = df_x[(df_x["year"] >= y0) & (df_x["year"] <= y1)].reset_index(drop=True)
        if len(test_xings) == 0:
            print(f"  no test crossings, skip")
            continue
        # split train into train/val/cal
        train_idx_arr = np.where(train_mask)[0]
        rng = np.random.default_rng(42)
        rng.shuffle(train_idx_arr)
        n = len(train_idx_arr)
        n_val = int(n * 0.1); n_cal = int(n * 0.1)
        val_idx = train_idx_arr[:n_val]
        cal_idx = train_idx_arr[n_val:n_val + n_cal]
        train_idx = train_idx_arr[n_val + n_cal:]
        print(f"  train {len(train_idx)} rows, val {len(val_idx)}, cal {len(cal_idx)}, test {len(test_xings)} crossings")

        X, feat_names = build_feature_matrix(expanded, sw_cols)
        y = expanded["label"].values.astype(int)

        t_train = time.time()
        model = fit_xgb(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
        print(f"  trained {time.time()-t_train:.1f}s")
        iso, cal_info = maybe_calibrate(model, X[cal_idx], y[cal_idx], deviation_threshold=0.05)
        trained = TrainedModel(model=model, isotonic=iso, feature_names=feat_names,
                               used_calibration=(iso is not None))

        # eval on test crossings (end-to-end)
        sample_n = min(500, len(test_xings))
        sample = test_xings.sample(n=sample_n, random_state=99).reset_index(drop=True)
        peak_dists = []
        logps = []
        eps = 1e-12
        for _, row in sample.iterrows():
            true_lat = abs(row["mean_mlat"])
            true_mlt = row["mean_mlt"]
            hemi = "N" if row["hemisphere"] == "N" else "S"
            sw = {c: row[c] for c in sw_cols if c in row}
            sw_s1 = {c: row[c] for c in STAGE1_BASE_FEATURES if c in row}
            sw_s1["doy_feat"] = sw.get("doy", pd.to_datetime(row["time_start"]).dayofyear)
            if "B_T" not in sw_s1:
                sw_s1["B_T"] = float(np.sqrt(row["imf_by"] ** 2 + row["imf_bz"] ** 2))
            s1_p = stage1_scalar_v2(s1_model, s1_iso, s1_feats, sw_s1)
            s2_p = stage2_dial(trained, sw, hemisphere=hemi)
            pmf = normalize_pmf(s2_p, area_weighted=True)
            comb = s1_p * pmf
            ci, cj = cell_of(true_lat, true_mlt)
            peak = np.unravel_index(np.argmax(comb), comb.shape)
            peak_dists.append(haversine_deg(true_lat, true_mlt,
                                             LAT_AXIS[peak[0]], MLT_AXIS[peak[1]]))
            logps.append(np.log(comb[ci, cj] + eps))

        fold_results.append({
            "fold": label,
            "n_test_crossings": int(len(test_xings)),
            "n_sample": sample_n,
            "median_peak_dist": float(np.median(peak_dists)),
            "p90_peak_dist": float(np.percentile(peak_dists, 90)),
            "mean_true_logp": float(np.mean(logps)),
            "best_iter": int(model.best_iteration),
            "elapsed_fold_sec": float(time.time() - t_fold),
        })
        r = fold_results[-1]
        print(f"  FOLD {label}: median {r['median_peak_dist']:.2f} deg, p90 {r['p90_peak_dist']:.2f} deg, logp {r['mean_true_logp']:.3f}")
        print(f"  fold elapsed {r['elapsed_fold_sec']:.1f}s, cumulative {time.time()-t0:.1f}s")

    # summary
    print("\n[R030] 5-fold temporal CV results:")
    print(f"{'fold':>12s}  {'n_test':>8s}  {'med_dist':>10s}  {'p90_dist':>10s}  {'logp':>8s}")
    meds = []; p90s = []; logps = []
    for r in fold_results:
        print(f"  {r['fold']:>10s}  {r['n_test_crossings']:>8d}  {r['median_peak_dist']:>10.3f}  "
              f"{r['p90_peak_dist']:>10.3f}  {r['mean_true_logp']:>8.3f}")
        meds.append(r["median_peak_dist"])
        p90s.append(r["p90_peak_dist"])
        logps.append(r["mean_true_logp"])
    print(f"\n  median across folds: {np.mean(meds):.3f} +/- {np.std(meds):.3f}")
    print(f"  p90    across folds: {np.mean(p90s):.3f} +/- {np.std(p90s):.3f}")
    print(f"  logp   across folds: {np.mean(logps):.3f} +/- {np.std(logps):.3f}")

    with open(f"{OUT_DIR}/r030_temporal_cv.json", "w") as f:
        json.dump({"folds": fold_results,
                    "summary": {"median_dist_mean": float(np.mean(meds)),
                                 "median_dist_std": float(np.std(meds)),
                                 "p90_dist_mean": float(np.mean(p90s)),
                                 "p90_dist_std": float(np.std(p90s)),
                                 "logp_mean": float(np.mean(logps)),
                                 "logp_std": float(np.std(logps))},
                    "elapsed_total_sec": float(time.time() - t0)}, f, indent=2)
    print(f"\ntotal elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
