"""
Identify cusp crossings from DMSP SSJ data using Anderson & Bukowski (2024) criteria.

Reproduces the methodology of Anderson & Bukowski (2024, JGR, 10.1029/2023JA031886):
- Uses DMSP SSJ precipitating particle data from CDAWeb
- Applies MODIFIED Newell criteria (Anderson 2024 version)
- Extracts equatorward/poleward boundaries per cusp crossing
- Computes dipole tilt angle

Anderson 2024 criteria (modified from Newell+2006):
  1) Ion average energy <= 8000 eV  (Newell used 3000 eV)
  2) Electron average energy <= 220 eV  (same as Newell)
  3) Total integrated ion flux >= 0.1 erg/cm^2/s  (Newell used spectral peak >= 2e7)
  4) Ion spectral flux peak channel between 100-7000 eV  (same as Newell)

Sliding window: 4 consecutive spectra, 3/4 must satisfy => cusp entry/exit.
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------- Newell 2006 original criteria ----------

def newell_cusp_mask(ion_avg, ele_avg, ion_flux, channel_energies):
    """
    Apply Newell+2006 cusp criteria to each 1-second spectrum.
    Works with both 19-channel (CDAWeb) and 20-channel (NCEI) data.
    """
    ion_avg = np.where(np.abs(ion_avg) > 1e10, np.nan, ion_avg)
    ele_avg = np.where(np.abs(ele_avg) > 1e10, np.nan, ele_avg)
    ion_flux = np.where(np.abs(ion_flux) > 1e10, np.nan, ion_flux)

    flux_safe = np.nan_to_num(ion_flux, nan=-1.0)
    peak_ch = np.argmax(flux_safe, axis=1)
    peak_flux = np.nanmax(ion_flux, axis=1)
    peak_energy = channel_energies[peak_ch]

    c1 = ion_avg <= 3000.0
    c2 = ele_avg <= 220.0
    c3 = peak_flux >= 2e7
    c4 = (peak_energy >= 100.0) & (peak_energy <= 7000.0)
    valid = ~np.isnan(ion_avg) & ~np.isnan(ele_avg) & ~np.isnan(peak_flux)

    return c1 & c2 & c3 & c4 & valid


# ---------- Anderson 2024 criteria ----------

def anderson_cusp_mask(ion_avg, ele_avg, ion_flux, channel_energies,
                       ion_total_eflux=None):
    """
    Apply Anderson & Bukowski (2024) modified cusp criteria to each 1-second spectrum.

    Changes from Newell+2006:
    - Ion avg energy threshold raised from 3000 to 8000 eV
    - Uses total integrated ion flux (>= 0.1 erg/cm^2/s) instead of spectral peak

    Parameters
    ----------
    ion_avg : (N,) array, ion average energy in eV
    ele_avg : (N,) array, electron average energy in eV
    ion_flux : (N, 19) array, differential energy flux per channel
    channel_energies : (19,) array, channel center energies in eV
    ion_total_eflux : (N,) array or None, total ion energy flux from CDAWeb
                      Units: eV/(cm^2 s sr). Convert to erg/cm^2/s for threshold.

    Returns
    -------
    cusp_mask : (N,) bool array, True where all 4 criteria are met
    """
    # Replace fill values
    ion_avg = np.where(np.abs(ion_avg) > 1e10, np.nan, ion_avg)
    ele_avg = np.where(np.abs(ele_avg) > 1e10, np.nan, ele_avg)
    ion_flux = np.where(np.abs(ion_flux) > 1e10, np.nan, ion_flux)

    # Peak energy channel (for criterion 4)
    flux_safe = np.nan_to_num(ion_flux, nan=-1.0)
    peak_ch = np.argmax(flux_safe, axis=1)
    peak_energy = channel_energies[peak_ch]

    # Total integrated ion flux in erg/cm^2/s
    if ion_total_eflux is not None:
        # CDAWeb provides ION_TOTAL_ENERGY_FLUX in eV/(cm^2 s sr)
        # Convert: eV -> erg (1 eV = 1.602e-12 erg), integrate over 2*pi sr
        ion_total_eflux = np.where(np.abs(ion_total_eflux) > 1e10, np.nan, ion_total_eflux)
        total_erg = ion_total_eflux * 2 * np.pi * 1.602e-12
    else:
        # Fallback: integrate differential flux manually
        n_ch = len(channel_energies)
        dE = np.zeros(n_ch)
        for i in range(n_ch):
            if i == 0:
                dE[i] = channel_energies[1] - channel_energies[0]
            elif i == n_ch - 1:
                dE[i] = channel_energies[-1] - channel_energies[-2]
            else:
                dE[i] = (channel_energies[i+1] - channel_energies[i-1]) / 2.0
        total_erg = np.nansum(ion_flux * dE[np.newaxis, :], axis=1) * 2 * np.pi * 1.602e-12

    c1 = ion_avg <= 8000.0          # Anderson: 8000 eV (not 3000)
    c2 = ele_avg <= 220.0           # Same as Newell
    c3 = total_erg >= 0.1           # Anderson: total integrated >= 0.1 erg/cm^2/s
    c4 = (peak_energy >= 100.0) & (peak_energy <= 7000.0)
    valid = ~np.isnan(ion_avg) & ~np.isnan(ele_avg) & ~np.isnan(total_erg)

    return c1 & c2 & c3 & c4 & valid


def sliding_window_cusp(cusp_spec, window=4, threshold=3):
    """
    Apply Newell+2006 sliding window rule:
    cusp entry = first point where >= threshold of window spectra satisfy criteria
    cusp exit  = first point where >= threshold of window spectra do NOT satisfy

    Returns (N,) bool array marking cusp region.
    """
    n = len(cusp_spec)
    in_cusp = np.zeros(n, dtype=bool)

    # Compute rolling sum of cusp_spec
    cusp_int = cusp_spec.astype(int)
    # Use cumsum for efficient rolling sum
    cs = np.cumsum(cusp_int)
    cs = np.insert(cs, 0, 0)
    roll_sum = np.zeros(n, dtype=int)
    for i in range(n):
        j = min(i + window, n)
        roll_sum[i] = cs[j] - cs[i]

    inside = False
    for i in range(n):
        if not inside:
            if roll_sum[i] >= threshold:
                inside = True
                in_cusp[i] = True
        else:
            # Check if we should exit: >= threshold spectra do NOT satisfy
            non_cusp_count = window - roll_sum[i]
            if i + window <= n and non_cusp_count >= threshold:
                inside = False
            else:
                in_cusp[i] = True

    return in_cusp


# ---------- Orbit segmentation ----------

def segment_orbits(aacgm_lat, min_gap_sec=300):
    """
    Segment data into polar passes by detecting latitude reversals.
    A new segment starts when the latitude changes direction or
    there's a gap > min_gap_sec.

    Returns list of (start_idx, end_idx) tuples for dayside high-lat passes.
    """
    n = len(aacgm_lat)
    # Fill NaN lat with 0 (equator) for segmentation
    lat = np.nan_to_num(aacgm_lat, nan=0.0)

    segments = []
    seg_start = 0

    for i in range(1, n):
        # New segment on sign change of latitude derivative or large gap
        if i > 1:
            dlat_prev = lat[i - 1] - lat[i - 2]
            dlat_curr = lat[i] - lat[i - 1]
            if (dlat_prev * dlat_curr < 0 and abs(lat[i - 1]) > 60):
                if i - seg_start > 10:
                    segments.append((seg_start, i))
                seg_start = i

    if n - seg_start > 10:
        segments.append((seg_start, n))

    return segments


# ---------- Extract cusp crossings ----------

def extract_crossings(epoch, aacgm_lat, aacgm_lt, cusp_mask_windowed):
    """
    From the windowed cusp mask, extract individual cusp crossings.
    Each crossing is a contiguous block of cusp=True points.

    Returns list of dicts with boundary info.
    """
    crossings = []
    in_crossing = False
    start_idx = 0

    for i in range(len(cusp_mask_windowed)):
        if cusp_mask_windowed[i] and not in_crossing:
            in_crossing = True
            start_idx = i
        elif not cusp_mask_windowed[i] and in_crossing:
            in_crossing = False
            crossings.append(_crossing_info(
                epoch, aacgm_lat, aacgm_lt, start_idx, i))

    if in_crossing:
        crossings.append(_crossing_info(
            epoch, aacgm_lat, aacgm_lt, start_idx, len(cusp_mask_windowed)))

    return crossings


def _crossing_info(epoch, lat, lt, i0, i1):
    """Compute boundary info for a single cusp crossing."""
    lats = lat[i0:i1]
    lts = lt[i0:i1]
    times = epoch[i0:i1]

    valid = ~np.isnan(lats)
    if valid.sum() == 0:
        return None

    lats_v = lats[valid]
    lts_v = lts[valid]

    # Determine hemisphere from mean latitude
    mean_lat = np.nanmean(lats_v)
    abs_lats = np.abs(lats_v)

    # Equatorward = minimum |MLAT|, poleward = maximum |MLAT|
    eq_idx = np.argmin(abs_lats)
    pole_idx = np.argmax(abs_lats)

    return {
        "time_start": str(times[0]),
        "time_end": str(times[-1]),
        "duration_sec": i1 - i0,
        "hemisphere": "N" if mean_lat > 0 else "S",
        "eq_mlat": float(lats_v[eq_idx]),
        "pole_mlat": float(lats_v[pole_idx]),
        "eq_mlt": float(lts_v[eq_idx]),
        "pole_mlt": float(lts_v[pole_idx]),
        "mean_mlat": float(np.nanmean(lats_v)),
        "mean_mlt": float(np.nanmean(lts_v)),
        "n_spectra": int(valid.sum()),
    }


# ---------- Dipole tilt ----------

def dipole_tilt_angle(dt):
    """
    Compute Earth's dipole tilt angle in degrees.
    Positive = northern dipole tilted toward Sun.
    Uses simple formula from Hapgood (1992).
    """
    # Day of year
    doy = dt.timetuple().tm_yday
    # Fraction of year
    T = (dt.year - 2000) + (doy - 1) / 365.25
    # Solar declination (approximate)
    dec_sun = 23.44 * np.sin(np.radians(360 / 365.25 * (doy - 80)))
    # Subsolar point geographic latitude ≈ solar declination
    # Dipole tilt ≈ angle between dipole axis and GSM Z
    # Simple approximation: tilt ≈ -(geographic colatitude of dipole pole - 90) + dec_sun
    # More precisely, we need the UT-dependent rotation
    # Use the standard formula: tilt = arcsin(sin(dec)*sin(pole_lat) +
    #   cos(dec)*cos(pole_lat)*cos(pole_lon - subsolar_lon))
    # Dipole pole: 80.1°N, 287.8°E (epoch 2005)
    pole_lat = np.radians(80.1)
    pole_lon = np.radians(287.8)

    # Subsolar longitude depends on UT
    ut_hours = dt.hour + dt.minute / 60 + dt.second / 3600
    subsolar_lon = np.radians((12.0 - ut_hours) * 15.0)  # rough

    dec_rad = np.radians(dec_sun)
    tilt = np.degrees(np.arcsin(
        np.sin(dec_rad) * np.sin(pole_lat) +
        np.cos(dec_rad) * np.cos(pole_lat) * np.cos(pole_lon - subsolar_lon)
    ))
    return tilt


# ---------- Main processing ----------

def process_one_day(satellite, date, output_dir):
    """
    Process one day of DMSP SSJ data:
    1. Download from CDAWeb
    2. Apply Newell cusp criteria
    3. Extract cusp crossings
    4. Save results

    Returns list of crossing dicts.
    """
    from cdasws import CdasWs
    cdas = CdasWs()

    ds = f"DMSP-{satellite}_SSJ_PRECIPITATING-ELECTRONS-IONS"
    t0 = datetime.datetime(date.year, date.month, date.day)
    t1 = t0 + datetime.timedelta(days=1)

    varnames = [
        "ION_AVG_ENERGY", "ELE_AVG_ENERGY", "ION_DIFF_ENERGY_FLUX",
        "ION_TOTAL_ENERGY_FLUX",
        "SC_AACGM_LAT", "SC_AACGM_LTIME", "SC_GEOCENTRIC_LAT",
    ]

    try:
        _, data = cdas.get_data(ds, varnames, t0, t1)
    except Exception as e:
        log.warning(f"Failed to download {ds} {date}: {e}")
        return []

    if data is None or "Epoch" not in data:
        log.warning(f"No data for {ds} {date}")
        return []

    epoch = np.array(data["Epoch"])
    ion_avg = np.array(data["ION_AVG_ENERGY"])
    ele_avg = np.array(data["ELE_AVG_ENERGY"])
    ion_flux = np.array(data["ION_DIFF_ENERGY_FLUX"])
    ion_total_eflux = np.array(data.get("ION_TOTAL_ENERGY_FLUX", []))
    aacgm_lat = np.array(data["SC_AACGM_LAT"])
    aacgm_lt = np.array(data["SC_AACGM_LTIME"])
    energies = np.array(data["CHANNEL_ENERGIES"])

    if len(epoch) < 100:
        return []

    # Step 1: Anderson 2024 per-spectrum criteria
    cusp_spec = anderson_cusp_mask(
        ion_avg, ele_avg, ion_flux, energies,
        ion_total_eflux=ion_total_eflux if len(ion_total_eflux) == len(epoch) else None
    )

    # Dayside filter (8.5-15.5 MLT)
    dayside = (aacgm_lt >= 8.5) & (aacgm_lt <= 15.5)
    # High latitude filter (|MLAT| > 60)
    highlat = np.abs(aacgm_lat) > 60.0

    cusp_spec = cusp_spec & dayside & highlat

    if cusp_spec.sum() == 0:
        return []

    # Step 2: Sliding window (3/4 rule)
    cusp_windowed = sliding_window_cusp(cusp_spec, window=4, threshold=3)

    if cusp_windowed.sum() == 0:
        return []

    # Step 3: Extract crossings
    crossings = extract_crossings(epoch, aacgm_lat, aacgm_lt, cusp_windowed)
    crossings = [c for c in crossings if c is not None]

    # Step 4: Add metadata
    for c in crossings:
        c["satellite"] = satellite
        c["date"] = str(date)
        # Compute dipole tilt at crossing midpoint time
        try:
            t_start_str = c.get("time_start", "")
            t_end_str = c.get("time_end", "")
            t_start = datetime.datetime.fromisoformat(str(t_start_str)[:19])
            t_end = datetime.datetime.fromisoformat(str(t_end_str)[:19])
            t_mid = t_start + (t_end - t_start) / 2
            c["dipole_tilt"] = round(dipole_tilt_angle(t_mid), 2)
        except Exception:
            try:
                c["dipole_tilt"] = round(dipole_tilt_angle(
                    datetime.datetime(date.year, date.month, date.day, 12)), 2)
            except Exception:
                c["dipole_tilt"] = None

    log.info(f"{satellite} {date}: {cusp_spec.sum()} cusp spectra, "
             f"{cusp_windowed.sum()} windowed, {len(crossings)} crossings")

    return crossings


def process_satellite_daterange(satellite, start_date, end_date, output_dir):
    """Process a date range for one satellite."""
    all_crossings = []
    date = start_date
    while date <= end_date:
        try:
            crossings = process_one_day(satellite, date, output_dir)
            all_crossings.extend(crossings)
        except Exception as e:
            log.error(f"Error processing {satellite} {date}: {e}")
        date += datetime.timedelta(days=1)

    return all_crossings


def main():
    parser = argparse.ArgumentParser(description="Identify cusp crossings from DMSP SSJ")
    parser.add_argument("--satellite", required=True, help="DMSP satellite (e.g., F16)")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    args = parser.parse_args()

    start = datetime.date.fromisoformat(args.start)
    end = datetime.date.fromisoformat(args.end)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info(f"Processing {args.satellite} from {start} to {end}")
    crossings = process_satellite_daterange(args.satellite, start, end, outdir)

    # Save results
    outfile = outdir / f"cusp_crossings_{args.satellite}_{start}_{end}.json"
    with open(outfile, "w") as f:
        json.dump(crossings, f, indent=2, default=str)

    log.info(f"Saved {len(crossings)} crossings to {outfile}")


if __name__ == "__main__":
    main()
