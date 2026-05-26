"""R022 (pilot) — apply both R002 (synthetic-trained) and R021 (real-trained)
to the SAME F10 1993-94 held-out crossings; compare end-to-end accuracy.

Held-out crossings = the F10 1993-94 entries from the 48k crossings table
filtered to dates not used in R021 training (we use the same hour-bin split
to avoid leakage — but R002 saw none of these crossings as its 80/20 split
is at crossing-level seed=42, and ~20% of F10 1993-94 crossings happen to
be in R002's test split).

For each held-out crossing, run R002 combined and R021 combined, compute
2D peak distance, log-prob of true cell.
"""
import json
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, "/glade/work/yizhu/cuspML/src/lib")
sys.path.insert(0, "/glade/work/yizhu/cuspML/src/scripts/mvp")
from cusp_map import load_crossings, sw_feature_names, polar_xy, predict_proba, TrainedModel
from cusp_stage1 import STAGE1_BASE_FEATURES
from r014_endtoend_eval import (LAT_AXIS, MLT_AXIS, MM, LL,
                                 cell_of, stage2_dial, normalize_pmf, haversine_deg)
from r016_combined_v2 import load_stage1_v2, stage1_scalar_v2
from r012_case_studies_2stage import load_stage2

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles"


def load_r021(tag):
    from xgboost import XGBClassifier
    m = XGBClassifier()
    m.load_model(f"{OUT_DIR}/r021_real_negs_{tag}_model.ubj")
    iso = None
    iso_path = f"{OUT_DIR}/r021_real_negs_{tag}_isotonic.pkl"
    if os.path.exists(iso_path):
        with open(iso_path, "rb") as f:
            iso = pickle.load(f)
    with open(f"{OUT_DIR}/r021_real_negs_{tag}_features.json") as f:
        feats = json.load(f)
    return TrainedModel(model=m, isotonic=iso, feature_names=feats,
                        used_calibration=(iso is not None))


def stage2_dial_custom(trained, sw_state, hemisphere="N"):
    """Like stage2_dial but uses trained.feature_names not the global s2."""
    x, y = polar_xy(LL.ravel(), MM.ravel())
    rec = dict(sw_state)
    rec["hemi_code"] = 1.0 if hemisphere == "N" else 0.0
    n = LL.size
    grid = {k: np.full(n, v, dtype=np.float32) for k, v in rec.items()}
    grid["x_polar"] = x.astype(np.float32)
    grid["y_polar"] = y.astype(np.float32)
    df = pd.DataFrame(grid)
    for m_ in trained.feature_names:
        if m_ not in df.columns:
            df[m_] = 0.0
    X = df[trained.feature_names].values.astype(np.float32)
    P = predict_proba(trained, X).reshape(LL.shape)
    return P


def main():
    s1_model, s1_iso, s1_feats = load_stage1_v2()
    s2_synth = load_stage2()  # R002
    s2_real = load_r021("K10_F10_1993_1994")
    print(f"  loaded R002 ({len(s2_synth.feature_names)} feats) and R021 ({len(s2_real.feature_names)} feats)")

    df = load_crossings("/glade/work/yizhu/cuspML/output/omni_full_hist")
    sw_cols = sw_feature_names(df)
    required = ["eq_mlat", "pole_mlat", "eq_mlt", "pole_mlt"]
    keep = sw_cols + required + ["satellite", "hemisphere", "time_start"]
    df_clean = df[keep].dropna().reset_index(drop=True)

    # F10 1993-94 subset
    f10 = df_clean[df_clean["satellite"] == "F10"].copy()
    f10["year"] = pd.to_datetime(f10["time_start"]).dt.year
    f10_pilot = f10[f10["year"].isin([1993, 1994])].reset_index(drop=True)
    print(f"  F10 1993-94 crossings: {len(f10_pilot)}")

    # Take 200 random for eval
    sample = f10_pilot.sample(n=min(200, len(f10_pilot)), random_state=99).reset_index(drop=True)
    print(f"  sampled {len(sample)} for eval")

    n_cells = LAT_AXIS.size * MLT_AXIS.size
    eps = 1e-12
    uniform_logp = np.log(1.0 / n_cells)
    rec = {"synth_dist": [], "real_dist": [],
           "synth_logp_true": [], "real_logp_true": [],
           "synth_top10": [], "real_top10": [],
           "synth_top1pct": [], "real_top1pct": []}

    for _, row in sample.iterrows():
        true_lat = abs(row["mean_mlat"]) if "mean_mlat" in row else (abs(row["eq_mlat"]) + abs(row["pole_mlat"])) / 2
        true_mlt = row["mean_mlt"] if "mean_mlt" in row else (row["eq_mlt"] + row["pole_mlt"]) / 2
        hemi = "N" if row["hemisphere"] == "N" else "S"
        sw_for_s2 = {c: row[c] for c in sw_cols if c in row}
        sw_for_s1 = {c: row[c] for c in STAGE1_BASE_FEATURES if c in row}
        sw_for_s1["doy_feat"] = sw_for_s2.get("doy", pd.to_datetime(row["time_start"]).dayofyear)
        if "B_T" not in sw_for_s1:
            sw_for_s1["B_T"] = float(np.sqrt(row["imf_by"]**2 + row["imf_bz"]**2))
        s1_p = stage1_scalar_v2(s1_model, s1_iso, s1_feats, sw_for_s1)

        for tag, trained, rec_dist, rec_lp, rec_t10, rec_t1p in [
            ("synth", s2_synth, "synth_dist", "synth_logp_true", "synth_top10", "synth_top1pct"),
            ("real", s2_real, "real_dist", "real_logp_true", "real_top10", "real_top1pct"),
        ]:
            s2_p = stage2_dial_custom(trained, sw_for_s2, hemisphere=hemi)
            s2_pmf = normalize_pmf(s2_p, area_weighted=True)
            combined = s1_p * s2_pmf
            i, j = cell_of(true_lat, true_mlt)
            rec[rec_lp].append(np.log(combined[i, j] + eps))
            peak_idx = np.unravel_index(np.argmax(combined), combined.shape)
            rec[rec_dist].append(haversine_deg(true_lat, true_mlt,
                                               LAT_AXIS[peak_idx[0]], MLT_AXIS[peak_idx[1]]))
            flat = combined.flatten()
            rank = (flat > combined[i, j]).sum() + 1
            rec[rec_t10].append(rank <= 10)
            rec[rec_t1p].append(rank <= n_cells // 100)

    summary = {
        "n_eval": len(sample),
        "synth": {
            "median_peak_dist_deg": float(np.median(rec["synth_dist"])),
            "p90_peak_dist_deg": float(np.percentile(rec["synth_dist"], 90)),
            "frac_top10": float(np.mean(rec["synth_top10"])),
            "frac_top1pct": float(np.mean(rec["synth_top1pct"])),
            "mean_logp_true": float(np.mean(rec["synth_logp_true"])),
            "improvement_over_uniform": float(np.mean(rec["synth_logp_true"]) - uniform_logp),
        },
        "real": {
            "median_peak_dist_deg": float(np.median(rec["real_dist"])),
            "p90_peak_dist_deg": float(np.percentile(rec["real_dist"], 90)),
            "frac_top10": float(np.mean(rec["real_top10"])),
            "frac_top1pct": float(np.mean(rec["real_top1pct"])),
            "mean_logp_true": float(np.mean(rec["real_logp_true"])),
            "improvement_over_uniform": float(np.mean(rec["real_logp_true"]) - uniform_logp),
        },
    }
    print(f"\n[R022] COMPARISON: synthetic R002 vs real R021 on {len(sample)} F10 1993-94 crossings")
    print(f"{'metric':>34s}  {'synth':>10s}  {'real':>10s}  delta")
    for k in ["median_peak_dist_deg", "p90_peak_dist_deg", "frac_top10",
              "frac_top1pct", "mean_logp_true", "improvement_over_uniform"]:
        s = summary["synth"][k]; r = summary["real"][k]
        d = r - s
        print(f"  {k:>34s}: {s:>10.4f}  {r:>10.4f}  {d:>+.4f}")

    # gate: real should not be much worse than synth (within 0.5 deg dist, within 0.5 nat logp)
    pass_dist = summary["real"]["median_peak_dist_deg"] <= summary["synth"]["median_peak_dist_deg"] + 0.5
    pass_logp = summary["real"]["mean_logp_true"] >= summary["synth"]["mean_logp_true"] - 0.5
    pass_overall = pass_dist and pass_logp
    pass_strong = (summary["real"]["median_peak_dist_deg"] < summary["synth"]["median_peak_dist_deg"]) or \
                  (summary["real"]["mean_logp_true"] > summary["synth"]["mean_logp_true"])
    print(f"\n[R022] GATE:")
    print(f"  PASS (real not >0.5 deg worse): {pass_dist}")
    print(f"  PASS (real logp not >0.5 nat worse): {pass_logp}")
    print(f"  OVERALL pass: {pass_overall}")
    print(f"  STRONG (real strictly better on at least one): {pass_strong}")

    out = {"summary": summary, "gate_pass": pass_overall, "gate_strong": pass_strong}
    with open(f"{OUT_DIR}/r022_compare.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
