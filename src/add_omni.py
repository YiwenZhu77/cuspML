"""
Match cusp crossings with OMNI solar wind data.
Add IMF (Bx, By, Bz), solar wind (V, n, Pdyn), and AE index.
Filter by AE < 100 nT (Anderson 2024 criterion).
"""

import argparse
import datetime
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def download_omni_year(year):
    """Download 1-min OMNI data for one year from CDAWeb."""
    from cdasws import CdasWs
    cdas = CdasWs()

    ds = "OMNI_HRO_1MIN"
    t0 = datetime.datetime(year, 1, 1)
    t1 = datetime.datetime(year + 1, 1, 1)

    varnames = [
        "BX_GSE", "BY_GSM", "BZ_GSM",  # IMF components
        "flow_speed", "proton_density", "Pressure",  # Solar wind
        "AE_INDEX", "SYM_H",  # Geomagnetic indices
    ]

    _, data = cdas.get_data(ds, varnames, t0, t1)
    return data


def match_crossings_with_omni(crossings, omni_data):
    """
    For each cusp crossing, find the closest OMNI data point
    and add solar wind / IMF / index values.
    Also apply AE < 100 nT filter.
    """
    if omni_data is None:
        return crossings

    omni_epoch = np.array(omni_data["Epoch"])
    by = np.array(omni_data.get("BY_GSM", []), dtype=float)
    bz = np.array(omni_data.get("BZ_GSM", []), dtype=float)
    bx = np.array(omni_data.get("BX_GSE", []), dtype=float)
    v = np.array(omni_data.get("flow_speed", []), dtype=float)
    n = np.array(omni_data.get("proton_density", []), dtype=float)
    pdyn = np.array(omni_data.get("Pressure", []), dtype=float)
    ae = np.array(omni_data.get("AE_INDEX", []), dtype=float)

    # Replace fill values
    for arr in [by, bz, bx, v, n, pdyn, ae]:
        arr[np.abs(arr) > 9999] = np.nan

    # Vectorized matching using searchsorted (O(N log M) instead of O(N*M))
    omni_sec = omni_epoch.astype("datetime64[s]").astype(np.int64)

    # Parse all crossing times at once
    crossing_times = []
    valid_indices = []
    for i, c in enumerate(crossings):
        t_str = c.get("time_start", "")
        try:
            t = np.datetime64(t_str[:19]).astype("datetime64[s]").astype(np.int64)
            crossing_times.append(t)
            valid_indices.append(i)
        except Exception:
            continue

    if not crossing_times:
        return crossings

    crossing_sec = np.array(crossing_times)

    # Use searchsorted to find nearest OMNI index for each crossing
    insert_idx = np.searchsorted(omni_sec, crossing_sec)
    insert_idx = np.clip(insert_idx, 1, len(omni_sec) - 1)
    # Check which neighbor is closer
    left = np.abs(crossing_sec - omni_sec[insert_idx - 1])
    right = np.abs(crossing_sec - omni_sec[insert_idx])
    nearest_idx = np.where(left <= right, insert_idx - 1, insert_idx)

    matched = []
    for j, ci in enumerate(valid_indices):
        c = crossings[ci]
        idx = nearest_idx[j]

        c["imf_bx"] = float(bx[idx]) if not np.isnan(bx[idx]) else None
        c["imf_by"] = float(by[idx]) if not np.isnan(by[idx]) else None
        c["imf_bz"] = float(bz[idx]) if not np.isnan(bz[idx]) else None
        c["sw_v"] = float(v[idx]) if not np.isnan(v[idx]) else None
        c["sw_n"] = float(n[idx]) if not np.isnan(n[idx]) else None
        c["sw_pdyn"] = float(pdyn[idx]) if not np.isnan(pdyn[idx]) else None
        c["ae_index"] = float(ae[idx]) if not np.isnan(ae[idx]) else None

        matched.append(c)

    return matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON with crossings")
    parser.add_argument("--output", help="Output JSON (default: overwrite input)")
    parser.add_argument("--ae-filter", type=float, default=None,
                        help="Filter by AE < threshold (Anderson uses 100)")
    args = parser.parse_args()

    with open(args.input) as f:
        crossings = json.load(f)

    if not crossings:
        log.info("No crossings to process")
        return

    # Determine year from first crossing
    year = int(crossings[0]["date"][:4])
    log.info(f"Downloading OMNI data for {year}")
    omni = download_omni_year(year)

    log.info(f"Matching {len(crossings)} crossings with OMNI")
    crossings = match_crossings_with_omni(crossings, omni)

    if args.ae_filter is not None:
        before = len(crossings)
        crossings = [c for c in crossings
                     if c.get("ae_index") is not None
                     and c["ae_index"] < args.ae_filter]
        log.info(f"AE < {args.ae_filter}: {before} -> {len(crossings)} crossings")

    outfile = args.output or args.input
    with open(outfile, "w") as f:
        json.dump(crossings, f, indent=2, default=str)
    log.info(f"Saved {len(crossings)} crossings to {outfile}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
