"""
Compare reproduced cusp dataset with Anderson & Bukowski (2024) statistics.

Anderson 2024 key numbers:
- 41,000+ cusp crossings from 14 DMSP satellites, 40 years
- Filtered: AE < 100 nT, IMF available
- MLAT range: ~70-82 degrees
- MLT range: ~8.5-15.5 hours
- Cusp center: ~77-78 MLAT, ~12 MLT
- Tilt dependence: 0.043-0.051 deg MLAT per deg tilt
- By effect: ~1 hour MLT shift between By>3 and By<-3
"""

import json
import glob
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_all_crossings(output_dir):
    """Load all crossing JSON files from output directory."""
    files = sorted(glob.glob(f"{output_dir}/cusp_crossings_*.json"))
    all_crossings = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            all_crossings.extend(data)
    return all_crossings


def compare(crossings):
    """Print comparison statistics."""
    n = len(crossings)
    if n == 0:
        print("No crossings found!")
        return

    # Extract arrays
    eq_mlat = np.array([c["eq_mlat"] for c in crossings])
    pole_mlat = np.array([c["pole_mlat"] for c in crossings])
    mean_mlat = np.array([c["mean_mlat"] for c in crossings])
    mean_mlt = np.array([c["mean_mlt"] for c in crossings])
    hemispheres = np.array([c["hemisphere"] for c in crossings])
    sats = np.array([c.get("satellite", "?") for c in crossings])
    tilts = np.array([c.get("dipole_tilt", np.nan) for c in crossings], dtype=float)

    # IMF data (if matched)
    has_imf = any(c.get("imf_by") is not None for c in crossings)
    if has_imf:
        imf_by = np.array([c.get("imf_by", np.nan) for c in crossings], dtype=float)
        imf_bz = np.array([c.get("imf_bz", np.nan) for c in crossings], dtype=float)
        ae = np.array([c.get("ae_index", np.nan) for c in crossings], dtype=float)

    print("=" * 60)
    print("COMPARISON WITH ANDERSON & BUKOWSKI (2024)")
    print("=" * 60)

    print(f"\n--- Dataset Size ---")
    print(f"Our crossings:       {n:,}")
    print(f"Anderson 2024:       41,000+")
    print(f"Ratio:               {n/41000:.1%}")

    print(f"\n--- Satellites ---")
    unique_sats = np.unique(sats)
    print(f"Our satellites:      {', '.join(unique_sats)} ({len(unique_sats)} total)")
    print(f"Anderson 2024:       14 DMSP satellites")

    print(f"\n--- Hemisphere ---")
    n_north = (hemispheres == "N").sum()
    n_south = (hemispheres == "S").sum()
    print(f"Northern:            {n_north:,} ({n_north/n:.1%})")
    print(f"Southern:            {n_south:,} ({n_south/n:.1%})")

    print(f"\n--- Equatorward boundary MLAT ---")
    abs_eq = np.abs(eq_mlat)
    print(f"Our mean |MLAT|:     {np.nanmean(abs_eq):.1f}° ± {np.nanstd(abs_eq):.1f}°")
    print(f"Our range:           [{np.nanmin(abs_eq):.1f}°, {np.nanmax(abs_eq):.1f}°]")
    print(f"Anderson expected:   ~75-80° (quiet, depends on coupling)")

    print(f"\n--- Poleward boundary MLAT ---")
    abs_pole = np.abs(pole_mlat)
    print(f"Our mean |MLAT|:     {np.nanmean(abs_pole):.1f}° ± {np.nanstd(abs_pole):.1f}°")

    print(f"\n--- Mean MLT ---")
    print(f"Our mean MLT:        {np.nanmean(mean_mlt):.1f} ± {np.nanstd(mean_mlt):.1f} hr")
    print(f"Anderson expected:   ~11-13 hr (centered near noon)")

    print(f"\n--- Dipole Tilt ---")
    valid_tilt = ~np.isnan(tilts)
    if valid_tilt.sum() > 0:
        print(f"Tilt range:          [{np.nanmin(tilts):.1f}°, {np.nanmax(tilts):.1f}°]")

        # Global fit (for reference)
        valid = valid_tilt & ~np.isnan(abs_eq)
        if valid.sum() > 50:
            coeffs = np.polyfit(tilts[valid], abs_eq[valid], 1)
            print(f"d(MLAT)/d(tilt) global: {coeffs[0]:.3f} deg/deg")

        # Anderson method: fit per 1-hour MLT running bin
        # Anderson multiplies tilt by -1 for S hemisphere
        print(f"\n--- Tilt slope per MLT bin (Anderson method) ---")
        # Combine hemispheres: flip tilt sign for S
        combined_tilt = tilts.copy()
        combined_tilt[hemispheres == "S"] *= -1

        mlt_centers = np.arange(9.0, 15.0, 0.5)  # running 1-hr bins
        print(f"{'MLT bin':>12s}  {'slope':>8s}  {'r':>6s}  {'N':>6s}")
        for mlt_c in mlt_centers:
            mlt_lo, mlt_hi = mlt_c - 0.5, mlt_c + 0.5
            sel = (mean_mlt >= mlt_lo) & (mean_mlt < mlt_hi) & valid_tilt & ~np.isnan(abs_eq)
            if sel.sum() > 30:
                c = np.polyfit(combined_tilt[sel], abs_eq[sel], 1)
                r = np.corrcoef(combined_tilt[sel], abs_eq[sel])[0, 1]
                print(f"  [{mlt_lo:.1f}-{mlt_hi:.1f}]  {c[0]:8.4f}  {r:6.3f}  {sel.sum():6d}")
        print(f"Anderson 2024:       0.043-0.051 deg/deg (at cusp center MLT)")

    if has_imf:
        print(f"\n--- IMF Effects ---")
        valid_by = ~np.isnan(imf_by)
        by_pos = (imf_by > 3) & valid_by
        by_neg = (imf_by < -3) & valid_by
        if by_pos.sum() > 10 and by_neg.sum() > 10:
            mlt_bypos = np.nanmean(mean_mlt[by_pos])
            mlt_byneg = np.nanmean(mean_mlt[by_neg])
            print(f"MLT for By > 3 nT:   {mlt_bypos:.1f} hr (n={by_pos.sum()})")
            print(f"MLT for By < -3 nT:  {mlt_byneg:.1f} hr (n={by_neg.sum()})")
            print(f"MLT shift:           {mlt_bypos - mlt_byneg:.1f} hr")
            print(f"Anderson expected:    ~1 hr shift")

        print(f"\n--- AE Index ---")
        valid_ae = ~np.isnan(ae)
        print(f"AE < 100 crossings:  {(ae[valid_ae] < 100).sum():,}")
        print(f"AE >= 100:           {(ae[valid_ae] >= 100).sum():,}")

    print("\n" + "=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    crossings = load_all_crossings(args.output_dir)
    compare(crossings)


if __name__ == "__main__":
    main()
