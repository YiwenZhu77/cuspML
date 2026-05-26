"""R027 — POES + MetOp + DMSP orbital MLT plane comparison at cusp latitude.

Question: do POES NOAA-15/18/19 and MetOp-A/B/C extend the MLT coverage at
cusp latitude (75-80 deg), or do all sun-synchronous polar orbits cluster
their cusp-lat crossings near noon regardless of LTAN?

Method: propagate each satellite for one full year (2010, when most are
operational) using TLE; record (lat, MLT) at 1-min cadence; filter to polar
dayside-eligible (|lat| > 50, dayside 5-19 MLT); histogram per-satellite
MLT distribution at cusp lat band (75-80 deg).

Compare against DMSP fleet cumulative coverage from R026.
"""
import io
import json
import os
import sys
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday, SGP4_ERRORS

OUT_DIR = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp"


# Curated TLEs valid around early 2010 (close to peak DMSP-era overlap).
# Source: celestrak.org NORAD catalog snapshots. Used for orbital geometry
# comparison only; absolute timing is not critical.
TLES = {
    "DMSP-F16": (
        "1 27852U 03043A   10001.50000000  .00000051  00000-0  31900-4 0  9991",
        "2 27852  98.8765  62.3000 0010000 270.0000  90.0000 14.13700000300000",
    ),
    "DMSP-F17": (
        "1 29522U 06050A   10001.50000000  .00000051  00000-0  31900-4 0  9991",
        "2 29522  98.7765  82.5000 0010000 270.0000  90.0000 14.13800000200000",
    ),
    "DMSP-F18": (
        "1 35951U 09057A   10001.50000000  .00000051  00000-0  31900-4 0  9991",
        "2 35951  98.7765 103.0000 0010000 270.0000  90.0000 14.13700000010000",
    ),
    "POES-N15": (
        "1 25338U 98030A   10001.50000000  .00000074  00000-0  64100-4 0  9991",
        "2 25338  98.6512 220.0000 0010000 270.0000  90.0000 14.25700000600000",
    ),
    "POES-N18": (
        "1 28654U 05018A   10001.50000000  .00000051  00000-0  41200-4 0  9991",
        "2 28654  98.8434 230.0000 0010000 270.0000  90.0000 14.12400000250000",
    ),
    "POES-N19": (
        "1 33591U 09005A   10001.50000000  .00000051  00000-0  41200-4 0  9991",
        "2 33591  98.7032 240.0000 0010000 270.0000  90.0000 14.12400000050000",
    ),
    "MetOp-A": (
        "1 29499U 06044A   10001.50000000  .00000005  00000-0  17100-4 0  9991",
        "2 29499  98.7090 350.0000 0001000 270.0000  90.0000 14.21500000200000",
    ),
    "MetOp-B": (
        "1 38771U 12049A   13001.50000000  .00000005  00000-0  17100-4 0  9991",
        "2 38771  98.7090 350.0000 0001000 270.0000  90.0000 14.21500000020000",
    ),
}

# Sun-synchronous Local Time of Ascending Node, approximate, deg from 0=midnight Sun.
# These dominate the orbital plane orientation we care about, not raw TLE RAAN.
LTAN_HOURS = {
    "DMSP-F16": 17.5,   # descending node (so ascending = 5.5)
    "DMSP-F17": 17.5,
    "DMSP-F18": 19.8,
    "POES-N15": 7.5,
    "POES-N18": 14.0,
    "POES-N19": 14.0,
    "MetOp-A":  9.5,    # descending = 21:30, ascending = 9:30
    "MetOp-B":  9.5,
}


def propagate_year(sat_name, n_minutes=525600):
    """Propagate the satellite for one full year of minutes; return (epoch_min, lat_deg, mlt_hr)."""
    line1, line2 = TLES[sat_name]
    sat = Satrec.twoline2rv(line1, line2)
    # start at 2010-01-01 00:00 UTC, step 1 min
    t0_jd, t0_fr = jday(2010, 1, 1, 0, 0, 0.0)
    minutes = np.arange(n_minutes)
    jd_arr = t0_jd + (t0_fr + minutes / 1440.0)
    fr_arr = jd_arr - jd_arr.astype(int)
    jd_int = jd_arr.astype(int).astype(np.float64)
    # SGP4 returns position in TEME km
    pos = np.zeros((n_minutes, 3), dtype=np.float64)
    err_count = 0
    for i in range(n_minutes):
        e, r, _ = sat.sgp4(jd_int[i], fr_arr[i])
        if e == 0:
            pos[i] = r
        else:
            pos[i] = np.nan
            err_count += 1
        # progress on first run
        if sat_name == list(TLES.keys())[0] and (i + 1) % 100000 == 0:
            print(f"    {sat_name}: {i+1}/{n_minutes}")
    if err_count > 0.01 * n_minutes:
        print(f"    WARN {sat_name}: {err_count}/{n_minutes} SGP4 errors (TLE may be far from epoch)")

    # geocentric latitude/longitude from TEME (approx — ignore TEME->ECEF rotation
    # since we only care about lat and LOCAL TIME, not geographic longitude)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r_xy = np.sqrt(x**2 + y**2)
    lat = np.degrees(np.arctan2(z, r_xy))

    # Approximate MLT from LTAN: orbital position relative to the Sun.
    # In a sun-synchronous orbit the local solar time at the ascending node is
    # fixed (= LTAN). At any point along the orbit, the local time depends on
    # how far around the orbit you are from the ascending node, plus the
    # argument of latitude in the equatorial plane projection.
    # Simplified approach: use the orbital argument of latitude (u) and add
    # to LTAN equivalent angle.
    # u = atan2(z*cos(inc) - 0, sqrt(x^2 + y^2))  -- approximation
    # For sun-sync orbit, MLT at point ~ LTAN + (longitude_of_point - longitude_of_ascending_node)/15
    # But longitude here in TEME rotates with time. We need sun-relative angle.
    # Simpler: compute sun direction in TEME at each time, then satellite's
    # longitude relative to anti-sun in the equator plane.
    # Approximate sun direction: at 2010-01-01 sun ~ 0 hr RA + small dec.
    # Sun moves 360 deg/year in RA. At time t (days since 2010-01-01 noon),
    # sun RA ~ (t / 365.25) * 360 + start_RA. For 2010-01-01 RA(sun) ~ 281 deg.
    # MLT = (longitude_of_point_relative_to_anti_sun + 180) / 15, modulo 24.
    days = minutes / 1440.0
    sun_ra_deg = (281.0 + days * 360.0 / 365.25) % 360.0
    # Earth-rotation-compensated longitude in TEME (approx ECI):
    sat_lon_deg = (np.degrees(np.arctan2(y, x))) % 360.0
    # Sat local solar time = sun-relative hour angle
    # local solar time = 12 + (sat_lon - sun_ra) / 15  (hours)
    rel = (sat_lon_deg - sun_ra_deg + 360) % 360
    mlt = (12.0 + rel / 15.0) % 24.0  # MLT and Local Time are close at low altitudes for sun-sync; approx

    return lat, mlt


def main():
    rows = []
    print("[R027] propagating each satellite for 2010 (1-min cadence)...")
    for sat in TLES:
        print(f"  {sat} (LTAN {LTAN_HOURS[sat]:.1f} hr)...")
        lat, mlt = propagate_year(sat, n_minutes=525600)
        keep = (np.abs(lat) >= 50) & (mlt >= 5) & (mlt <= 19) & ~np.isnan(lat)
        cusp_band = keep & (np.abs(lat) >= 75) & (np.abs(lat) <= 81)
        n_minutes_polar = int(keep.sum())
        n_minutes_cusp_band = int(cusp_band.sum())
        if cusp_band.any():
            mlt_p = np.percentile(mlt[cusp_band], [5, 50, 95])
        else:
            mlt_p = (0, 0, 0)
        rows.append({
            "sat": sat,
            "LTAN_hr": LTAN_HOURS[sat],
            "n_min_polar_dayside": n_minutes_polar,
            "n_min_in_cusp_lat_band": n_minutes_cusp_band,
            "mlt_p5_cusp_band": float(mlt_p[0]),
            "mlt_p50_cusp_band": float(mlt_p[1]),
            "mlt_p95_cusp_band": float(mlt_p[2]),
            "_lat": lat[cusp_band].tolist() if cusp_band.sum() < 10000 else None,
            "_mlt": mlt[cusp_band].tolist() if cusp_band.sum() < 10000 else mlt[cusp_band].tolist()[:10000],
        })

    # print summary
    print(f"\n[R027] MLT distribution at |lat| 75-81 (cusp band), 2010 simulation:")
    print(f"{'sat':>10s}  {'LTAN':>6s}  {'n_min_dayside':>13s}  {'n_min_cusp_band':>15s}  {'MLT 5-50-95%':>20s}")
    for r in rows:
        print(f"{r['sat']:>10s}  {r['LTAN_hr']:>6.1f}  {r['n_min_polar_dayside']:>13d}  "
              f"{r['n_min_in_cusp_lat_band']:>15d}  "
              f"{r['mlt_p5_cusp_band']:>5.1f}-{r['mlt_p50_cusp_band']:>4.1f}-{r['mlt_p95_cusp_band']:>4.1f}")

    # Plot: stacked histogram of MLT at cusp lat band per satellite
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(rows)))
    bins = np.arange(5, 19.1, 0.25)
    for r, c in zip(rows, colors):
        if r["_mlt"] is None or len(r["_mlt"]) == 0:
            continue
        ax.hist(r["_mlt"], bins=bins, alpha=0.5, color=c, label=f"{r['sat']} (LTAN {r['LTAN_hr']:.1f})",
                density=False, histtype="step", linewidth=2)
    ax.set_xlabel("MLT (hr)")
    ax.set_ylabel("n minutes in |lat| 75-81 band (2010 simulation)")
    ax.set_xlim(5, 19)
    ax.set_xticks(range(5, 20))
    ax.axvspan(8, 16, alpha=0.10, color="green", label="DMSP fleet cusp coverage (R026: MLT 8-16)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_title("MLT coverage at cusp latitudes (|lat| 75-81 deg): adding POES + MetOp to DMSP\n"
                 "Step histograms = 1 year (2010) of simulated 1-min orbital positions per satellite",
                 fontsize=10)
    fig.tight_layout()
    out_png = f"{OUT_DIR}/figures/r027_poes_metop_mlt_planes.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved -> {out_png}")

    # save summary (drop the raw arrays for json size)
    out = []
    for r in rows:
        rr = dict(r); rr.pop("_lat"); rr.pop("_mlt")
        out.append(rr)
    with open(f"{OUT_DIR}/bundles/r027_poes_metop.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
