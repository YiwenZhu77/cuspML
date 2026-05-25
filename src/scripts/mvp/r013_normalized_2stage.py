"""R013 (B-fix-1, v2) — stage 2 renormalized as conditional spatial shape over dial.

v2 (post codex round 1):
- drop MLT=24 endpoint duplicate (sampled at 0 already; same point on dial)
- use bin-edge convention (cell centers), not inclusive endpoints
- pole (MLAT=90) handled separately as a single point, not 48 redundant MLT samples
- area-weight cells by cos((90-MLAT)*pi/180) so high-lat tiny cells don't over-weight
- renormalized stage 2 explicitly NOT a calibrated probability; it is a spatial
  shape function. Calibrated quantity is stage1(SW) * shape; sums to stage1(SW)
  in (cell-area-weighted) integral over the dial.

Stage 2 outputs P((MLAT, MLT) is the cusp | observed, SW) which is a calibrated
binary classifier, not a density. Renormalize over the polar dial so each
SW-conditional map sums to 1, then multiply by stage 1 scalar to get a joint
per-cell occupancy probability that sums to stage1(SW) across the dial.

Combined formula:
    combined(MLAT, MLT, SW) = stage1(SW) * stage2(MLAT, MLT, SW)
                              ---------------------------------
                                  sum_{cell} stage2(cell, SW)

Combined map sums to stage1(SW) across the dial. Peak per-cell probability is
now interpretable as "probability of cusp footprint covering this cell during
this hour, given current SW".
"""
import json
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/glade/work/yizhu/cuspML/src/lib")
sys.path.insert(0, "/glade/work/yizhu/cuspML/src/scripts/mvp")
from cusp_map import polar_xy, predict_proba, TrainedModel
from cusp_stage1 import STAGE1_BASE_FEATURES
from r002_case_studies import CASES, build_sw_state
from r012_case_studies_2stage import (
    load_stage1, load_stage2, stage1_scalar, stage2_grid, plot_dial,
)

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp"


def combined_renorm(stage1_p, stage2_grid_p, mlat_axis, mlt_axis):
    """Renormalize stage 2 over the dial with cell-area weighting, then multiply by stage 1 scalar.

    Cell area weight = cos(|MLAT|) since polar dial cells get smaller toward the pole.
    Sum of returned map (area-weighted) equals stage1_p.
    """
    MM, LL = np.meshgrid(mlt_axis, mlat_axis)
    area_w = np.cos(np.deg2rad(LL))
    area_w = area_w / area_w.sum()
    weighted = stage2_grid_p * area_w
    s2_sum = weighted.sum()
    if s2_sum < 1e-12:
        return np.zeros_like(stage2_grid_p)
    return stage1_p * weighted / s2_sum


def main():
    s1_model, s1_iso, s1_feats = load_stage1()
    s2 = load_stage2()
    figs_dir = f"{OUT_DIR}/figures"
    os.makedirs(figs_dir, exist_ok=True)

    all_maps = {}
    for c in CASES:
        sw = build_sw_state(c, hemisphere="N")
        s1_p = stage1_scalar(s1_model, s1_iso, s1_feats, sw)
        mlat_axis, mlt_axis, s2_p = stage2_grid(s2, sw, hemisphere="N")
        combined = combined_renorm(s1_p, s2_p, mlat_axis, mlt_axis)
        all_maps[c["name"]] = {"s1": s1_p, "s2": s2_p, "combined": combined,
                                "mlat": mlat_axis, "mlt": mlt_axis,
                                "title": c["title"]}
        peak_p = combined.max()
        peak_idx = np.unravel_index(np.argmax(combined), combined.shape)
        print(f"  {c['name']:>30s}  s1={s1_p:.4f}  s2_peak={s2_p.max():.3f}  "
              f"combined_peak={peak_p:.5f} at lat={mlat_axis[peak_idx[0]]:.0f}, "
              f"MLT={mlt_axis[peak_idx[1]]:.1f}  sum={combined.sum():.4f}")

    vmax = max(d["combined"].max() for d in all_maps.values())
    print(f"\n  shared vmax for renormalized plots: {vmax:.5f}")

    for name, d in all_maps.items():
        plot_dial(d["mlat"], d["mlt"], d["combined"],
                  f"{d['title']}  (renorm, stage1={d['s1']:.3f})",
                  f"{figs_dir}/2stage_renorm_{name}.png", vmax=vmax)

    # physics sanity
    s = {name: {"s1": d["s1"],
                 "combined_peak": float(d["combined"].max()),
                 "combined_sum": float(d["combined"].sum()),
                 "peak_lat": float(d["mlat"][np.unravel_index(np.argmax(d["combined"]),
                                                              d["combined"].shape)[0]]),
                 "peak_mlt": float(d["mlt"][np.unravel_index(np.argmax(d["combined"]),
                                                             d["combined"].shape)[1]]),
                 "midnight_mean": float(d["combined"][:,
                     (d["mlt"] < 4) | (d["mlt"] > 20)].mean())}
         for name, d in all_maps.items()}
    checks = {
        "storm_peak_higher_than_quiet": s["case6_storm"]["combined_peak"] > s["case5_quiet"]["combined_peak"],
        "south_Bz_peak_higher_than_north_Bz": s["case1_strong_south_Bz"]["combined_peak"] > s["case2_strong_north_Bz"]["combined_peak"],
        "south_Bz_lat_lower_than_north_Bz": s["case1_strong_south_Bz"]["peak_lat"] < s["case2_strong_north_Bz"]["peak_lat"],
        "quiet_below_active_in_stage1": s["case5_quiet"]["s1"] < s["case1_strong_south_Bz"]["s1"],
        "strong_driving_peak_above_quiet": s["case1_strong_south_Bz"]["combined_peak"] > 5 * s["case5_quiet"]["combined_peak"]
            if s["case5_quiet"]["combined_peak"] > 0 else s["case1_strong_south_Bz"]["combined_peak"] > 0,
        "midnight_low": all(s[k]["midnight_mean"] < 0.05 * max(s[kk]["combined_peak"] for kk in s) for k in s),
    }
    print(f"\n[physics sanity 2-stage-renorm]")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    out = {"summary": s, "physics_checks": checks,
           "n_pass": sum(checks.values()), "n_total": len(checks)}
    with open(f"{OUT_DIR}/bundles/r013_normalized_2stage.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  figures -> {figs_dir}/2stage_renorm_*.png")
    print(f"  sanity: {out['n_pass']}/{out['n_total']}")


if __name__ == "__main__":
    main()
