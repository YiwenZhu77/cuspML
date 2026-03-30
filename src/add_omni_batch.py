"""
Batch OMNI matching: download all years in parallel, then match all crossings.
Much faster than per-file sequential processing.
"""

import datetime
import json
import glob
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def download_omni_year(year):
    """Download 1-min OMNI data for one year from CDAWeb."""
    from cdasws import CdasWs
    cdas = CdasWs()

    ds = "OMNI_HRO_1MIN"
    t0 = datetime.datetime(year, 1, 1)
    t1 = datetime.datetime(year + 1, 1, 1)

    varnames = [
        "BX_GSE", "BY_GSM", "BZ_GSM",
        "flow_speed", "proton_density", "Pressure",
        "AE_INDEX", "SYM_H",
    ]

    log.info(f"Downloading OMNI {year}...")
    _, data = cdas.get_data(ds, varnames, t0, t1)
    log.info(f"OMNI {year} done: {len(data.get('Epoch', [])) if data else 0} records")
    return year, data


def prep_omni(omni_data):
    """Pre-process OMNI data for fast matching."""
    epoch = np.array(omni_data["Epoch"])
    arrays = {}
    for key, cdaw_key in [
        ("imf_bx", "BX_GSE"), ("imf_by", "BY_GSM"), ("imf_bz", "BZ_GSM"),
        ("sw_v", "flow_speed"), ("sw_n", "proton_density"), ("sw_pdyn", "Pressure"),
        ("ae_index", "AE_INDEX"),
    ]:
        arr = np.array(omni_data.get(cdaw_key, []), dtype=float)
        arr[np.abs(arr) > 9999] = np.nan
        arrays[key] = arr

    epoch_sec = epoch.astype("datetime64[s]").astype(np.int64)
    return epoch_sec, arrays


def match_crossings(crossings, epoch_sec, arrays):
    """Vectorized matching using searchsorted.

    Adds instantaneous OMNI values (with 10-min bow shock→cusp delay)
    plus rolling window statistics (mean, std, delta) for 15/30/60 min windows.
    """
    BS_TO_CUSP_DELAY = 10 * 60  # 10 min in seconds
    WINDOWS_MIN = [15, 30, 60]  # rolling windows in minutes
    SW_KEYS = ["imf_bx", "imf_by", "imf_bz", "sw_v", "sw_n", "sw_pdyn"]

    crossing_times = []
    valid_indices = []
    for i, c in enumerate(crossings):
        try:
            t = np.datetime64(c["time_start"][:19]).astype("datetime64[s]").astype(np.int64)
            crossing_times.append(t)
            valid_indices.append(i)
        except Exception:
            continue

    if not crossing_times:
        return crossings

    csec = np.array(crossing_times)
    # Apply 10-min delay: look at OMNI data 10 min before crossing
    csec_delayed = csec - BS_TO_CUSP_DELAY

    idx = np.searchsorted(epoch_sec, csec_delayed)
    idx = np.clip(idx, 1, len(epoch_sec) - 1)
    left = np.abs(csec_delayed - epoch_sec[idx - 1])
    right = np.abs(csec_delayed - epoch_sec[idx])
    nearest = np.where(left <= right, idx - 1, idx)

    # dt between OMNI records (should be 60s for 1-min data)
    dt = 60  # seconds

    for j, ci in enumerate(valid_indices):
        c = crossings[ci]
        ni = nearest[j]

        # 1. Instantaneous values (at t - 10 min)
        for key, arr in arrays.items():
            val = arr[ni]
            c[key] = float(val) if not np.isnan(val) else None

        # 2. Rolling window stats for solar wind keys
        for win_min in WINDOWS_MIN:
            win_samples = win_min  # 1-min data → N samples = N minutes
            i_start = max(0, ni - win_samples)
            i_end = ni + 1

            for key in SW_KEYS:
                arr = arrays[key]
                window = arr[i_start:i_end]
                valid = window[~np.isnan(window)]

                if len(valid) >= win_samples // 3:  # require ≥ 1/3 valid
                    c[f"{key}_mean{win_min}"] = float(np.mean(valid))
                    c[f"{key}_std{win_min}"] = float(np.std(valid))
                    # Delta: last - first (change over window)
                    c[f"{key}_delta{win_min}"] = float(valid[-1] - valid[0]) if len(valid) >= 2 else None
                else:
                    c[f"{key}_mean{win_min}"] = None
                    c[f"{key}_std{win_min}"] = None
                    c[f"{key}_delta{win_min}"] = None

        # 3. Integrated coupling functions over 60 min (Newell 2006)
        i60 = max(0, ni - 60)
        bz_win = arrays["imf_bz"][i60:ni+1]
        by_win = arrays["imf_by"][i60:ni+1]
        v_win = arrays["sw_v"][i60:ni+1]

        valid_mask = ~(np.isnan(bz_win) | np.isnan(by_win) | np.isnan(v_win))
        if valid_mask.sum() >= 20:
            bz_v = bz_win[valid_mask]
            by_v = by_win[valid_mask]
            v_v = v_win[valid_mask]
            B_T = np.sqrt(by_v**2 + bz_v**2)
            clock = np.arctan2(by_v, bz_v)
            sin_half = np.abs(np.sin(clock / 2))
            # Newell coupling function integrated over window
            ncf = (v_v ** (4/3)) * (B_T ** (2/3)) * (sin_half ** (8/3))
            c["newell_cf_int60"] = float(np.sum(ncf) * dt)
            c["newell_cf_mean60"] = float(np.mean(ncf))
            # vBs integrated
            vBs = v_v * np.where(bz_v < 0, -bz_v, 0)
            c["vBs_int60"] = float(np.sum(vBs) * dt)
            c["vBs_mean60"] = float(np.mean(vBs))
        else:
            c["newell_cf_int60"] = None
            c["newell_cf_mean60"] = None
            c["vBs_int60"] = None
            c["vBs_mean60"] = None

    return crossings


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ae-filter", type=float, default=100)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Find all input files and determine needed years
    files = sorted(glob.glob(f"{args.input_dir}/cusp_crossings_*.json"))
    log.info(f"Found {len(files)} input files")

    file_years = {}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if data:
            year = int(data[0]["date"][:4])
            file_years[f] = (year, data)

    needed_years = sorted(set(y for y, _ in file_years.values()))
    log.info(f"Need OMNI for {len(needed_years)} years: {needed_years}")

    # 2. Download all years in parallel
    omni_cache = {}
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {pool.submit(download_omni_year, y): y for y in needed_years}
        for fut in as_completed(futures):
            y = futures[fut]
            try:
                year, data = fut.result()
                if data and "Epoch" in data:
                    omni_cache[year] = prep_omni(data)
                else:
                    log.warning(f"No OMNI data for {year}")
            except Exception as e:
                log.error(f"Failed to download OMNI {y}: {e}")
                # Retry once sequentially
                try:
                    log.info(f"Retrying OMNI {y}...")
                    year, data = download_omni_year(y)
                    if data and "Epoch" in data:
                        omni_cache[year] = prep_omni(data)
                except Exception as e2:
                    log.error(f"Retry failed for {y}: {e2}")

    log.info(f"Downloaded OMNI for {len(omni_cache)}/{len(needed_years)} years")

    # 3. Match all files
    total_before = 0
    total_after = 0
    total_imf = 0

    for f, (year, crossings) in file_years.items():
        basename = os.path.basename(f)
        outfile = os.path.join(args.output_dir, basename)

        if year not in omni_cache:
            log.warning(f"Skipping {basename}: no OMNI for {year}")
            with open(outfile, "w") as fh:
                json.dump([], fh)
            continue

        epoch_sec, arrays = omni_cache[year]
        crossings = match_crossings(crossings, epoch_sec, arrays)

        n_before = len(crossings)
        total_before += n_before

        # AE filter
        filtered = [c for c in crossings
                     if c.get("ae_index") is not None
                     and c["ae_index"] < args.ae_filter]

        # IMF available filter
        imf_valid = [c for c in filtered
                      if c.get("imf_by") is not None
                      and c.get("imf_bz") is not None]

        total_after += len(filtered)
        total_imf += len(imf_valid)

        with open(outfile, "w") as fh:
            json.dump(imf_valid, fh, indent=2, default=str)

        log.info(f"{basename}: {n_before} -> AE<{args.ae_filter}: {len(filtered)} -> IMF valid: {len(imf_valid)}")

    log.info(f"=== TOTAL: {total_before} -> AE filter: {total_after} -> IMF valid: {total_imf} ===")

    # 4. Quick comparison stats
    all_c = []
    for f in sorted(glob.glob(f"{args.output_dir}/cusp_crossings_*.json")):
        with open(f) as fh:
            all_c.extend(json.load(fh))

    if all_c:
        eq = np.abs(np.array([c["eq_mlat"] for c in all_c]))
        mlt = np.array([c["mean_mlt"] for c in all_c])
        tilt = np.array([c.get("dipole_tilt", np.nan) for c in all_c], dtype=float)
        hemi = np.array([c["hemisphere"] for c in all_c])

        log.info(f"Final dataset: {len(all_c)} crossings")
        log.info(f"Mean |MLAT|: {np.nanmean(eq):.2f} +/- {np.nanstd(eq):.2f}")
        log.info(f"Mean MLT: {np.nanmean(mlt):.2f} +/- {np.nanstd(mlt):.2f}")
        log.info(f"N: {(hemi=='N').sum()}, S: {(hemi=='S').sum()}")

        north = hemi == "N"
        valid = north & ~np.isnan(tilt) & ~np.isnan(eq)
        if valid.sum() > 50:
            coeffs = np.polyfit(tilt[valid], eq[valid], 1)
            r = np.corrcoef(tilt[valid], eq[valid])[0, 1]
            log.info(f"North tilt slope: {coeffs[0]:.4f} deg/deg (Anderson: 0.043-0.051)")
            log.info(f"North tilt r: {r:.3f}")


if __name__ == "__main__":
    main()
