"""R020 (pilot) — emit all polar 1Hz spectra for given (sat, year) with cusp_mask label.

Reuses parse_ncei_ssj.read_ssj_file and identify_cusp.newell_cusp_mask logic.
Instead of returning the small list of cusp crossings, emits every 1Hz spectrum
in the polar dayside-eligible region (|MLAT|>50, MLT 5-19) along with the
Anderson cusp mask label (0 / 1).

Output: <out_dir>/pilot_spectra_<sat>_<year>.parquet with columns
  [time, satellite, hemisphere, abs_mlat, mlt, cusp_mask]

Per sat-year ~hundreds of thousands of rows. Time: ~30 min on Casper htc serial.
"""
import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/glade/work/yizhu/cuspML/src")
from parse_ncei_ssj import read_ssj_file, _is_ssj5, CHANNEL_ENERGIES
from identify_cusp import newell_cusp_mask, sliding_window_cusp

OUT_DIR = "/glade/work/yizhu/cuspML/output/pilot_spectra"
CACHE = "/glade/derecho/scratch/yizhu/tmp/ncei_ssj_cache"

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger(__name__)


def process_one_day(satellite, date):
    """Return DataFrame of polar 1Hz spectra for one (sat, day)."""
    yy = date.year % 100
    doy = date.timetuple().tm_yday
    prefix = "j5" if _is_ssj5(satellite) else "j4"
    sat_num = satellite.lower().replace("f", "")  # F10 -> 10
    fname = f"{prefix}f{sat_num}{yy:02d}{doy:03d}.gz"
    gz_path = Path(CACHE) / fname
    if not gz_path.exists():
        return None

    try:
        records = read_ssj_file(gz_path, satellite=satellite)
    except Exception as e:
        log.warning(f"parse error {gz_path}: {e}")
        return None
    if len(records) < 100:
        return None

    times = np.array([r["datetime"] for r in records])
    ion_avg = np.array([r["ion_avg_energy"] for r in records], dtype=np.float64)
    ele_avg = np.array([r["ele_avg_energy"] for r in records], dtype=np.float64)
    ion_flux = np.array([r["ion_diff_energy_flux"] for r in records], dtype=np.float64)
    aacgm_lat = np.array([r["cgm_lat"] for r in records], dtype=np.float64)
    aacgm_lt = np.array([r["mlt"] for r in records], dtype=np.float64)

    cusp_spec = newell_cusp_mask(ion_avg, ele_avg, ion_flux, CHANNEL_ENERGIES)
    cusp_win = sliding_window_cusp(cusp_spec, window=4, threshold=3)

    # polar dayside-eligible region: |MLAT|>50, MLT 5-19 (slightly wider than crossing filter)
    polar = (np.abs(aacgm_lat) >= 50.0) & (aacgm_lt >= 5.0) & (aacgm_lt <= 19.0)
    keep = polar
    if not keep.any():
        return None
    df = pd.DataFrame({
        "time": pd.to_datetime(times[keep]),
        "satellite": satellite,
        "hemisphere": np.where(aacgm_lat[keep] > 0, "N", "S"),
        "abs_mlat": np.abs(aacgm_lat[keep]),
        "mlt": aacgm_lt[keep],
        "cusp_mask": cusp_win[keep].astype(np.int8),
    })
    return df


def process_year(satellite, year):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = Path(OUT_DIR) / f"pilot_spectra_{satellite}_{year}.parquet"
    if out_path.exists():
        print(f"  exists, skipping: {out_path}")
        return out_path

    rows_all = []
    n_pos = 0
    n_days = 0
    d0 = datetime.date(year, 1, 1)
    for i in range(366):
        d = d0 + datetime.timedelta(days=i)
        if d.year != year:
            break
        df = process_one_day(satellite, d)
        if df is None or len(df) == 0:
            continue
        rows_all.append(df)
        n_pos += int(df["cusp_mask"].sum())
        n_days += 1
        if (i + 1) % 30 == 0:
            print(f"    {satellite} {d}: cumulative {sum(len(x) for x in rows_all)} rows, {n_pos} cusp-positive, {n_days} days processed")
    if not rows_all:
        print(f"  no data for {satellite} {year}")
        return None
    out = pd.concat(rows_all, ignore_index=True)
    out.to_parquet(out_path, index=False)
    print(f"  saved {len(out)} rows ({int(out['cusp_mask'].sum())} cusp+) -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sat", required=True)
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    print(f"[R020] processing {args.sat} {args.year} ...")
    import time
    t0 = time.time()
    process_year(args.sat, args.year)
    print(f"  elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
