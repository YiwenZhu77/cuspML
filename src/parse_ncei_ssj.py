"""
Parse NCEI old-format DMSP SSJ/4 and SSJ/5 binary data files.

Supported satellites:
  - SSJ/4: F10, F11
  - SSJ/5: F16, F17, F18

Binary format specification from AFRL-RV-PS-TR-2014-0174:
  - Gzip-compressed files named j4fNNYYddd.gz (SSJ/4) or j5fNNYYddd.gz (SSJ/5)
  - Big-endian uint16 words
  - Each minute block = 2640 words (5280 bytes)
  - Block layout: 15-word header + 60 x 43-word per-second records + 45 padding words

SSJ/5 differences from SSJ/4:
  - Per-second record word[2] encodes milliseconds (sec = word[2] // 1000)
    instead of plain seconds
  - Integration time is 1.0 s (vs 0.098 s for SSJ/4)
  - Same channel energies, widths, and channel shuffle order

Supports downloading from NCEI and applying Anderson & Bukowski (2024) cusp criteria.
"""

import argparse
import datetime
import gzip
import json
import logging
import struct
import sys
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORDS_PER_MINUTE = 2640
BYTES_PER_MINUTE = WORDS_PER_MINUTE * 2       # 5280
HEADER_WORDS = 15
SECONDS_PER_MINUTE = 60
WORDS_PER_SECOND = 43
PADDING_WORDS = 45                             # 2640 - 15 - 60*43

N_CHANNELS = 20
INTEGRATION_TIME_SSJ4 = 0.098                  # seconds, SSJ/4
INTEGRATION_TIME_SSJ5 = 1.0                    # seconds, SSJ/5

# Keep legacy alias for backward compatibility
INTEGRATION_TIME = INTEGRATION_TIME_SSJ4

# Satellite -> instrument mapping
SSJ4_SATELLITES = {"F10", "F11"}
SSJ5_SATELLITES = {"F16", "F17", "F18"}
ALL_SATELLITES = SSJ4_SATELLITES | SSJ5_SATELLITES

# Channel center energies, high-to-low (eV), channels 1-20
# Same for SSJ/4 and SSJ/5
CHANNEL_ENERGIES = np.array([
    30000, 20400, 13900, 9450, 6460,
    4400,  3000,  2040,  1392, 949,
    949,   646,   440,   300,  204,
    139,   95,    65,    44,   30,
], dtype=np.float64)

# Channel widths (dE) in eV, channels 1-20
CHANNEL_WIDTHS = np.array([
    9800, 6700, 4600, 3100, 2120,
    1440, 985,  670,  457,  311,
    311,  212,  144,  99,   67,
    46,   31,   21,   15,   10,
], dtype=np.float64)

# Approximate geometric factor for all channels (cm^2 sr)
# True values vary per channel and satellite; this is a first-order estimate.
G_APPROX = 4.5e-4  # cm^2 sr

# Mapping from file word order to channel order.
# Words 4-23 in a per-second record are grouped as:
#   [ch4, ch3, ch2, ch1], [ch8, ch7, ch6, ch5], ...
# We need to unscramble to channel order 1..20.
_CHANNEL_ORDER = []
for group_start in range(0, 20, 4):
    # Within each group of 4, the order is reversed: ch(g+4), ch(g+3), ch(g+2), ch(g+1)
    for offset in [3, 2, 1, 0]:
        _CHANNEL_ORDER.append(group_start + offset)
# _CHANNEL_ORDER[file_position] = channel_index  (0-based)
# We need the inverse: for each channel 0..19, which file position holds it?
CHANNEL_UNSHUFFLE = [0] * 20
for file_pos, ch_idx in enumerate(_CHANNEL_ORDER):
    CHANNEL_UNSHUFFLE[ch_idx] = file_pos
CHANNEL_UNSHUFFLE = np.array(CHANNEL_UNSHUFFLE)

# NCEI URL templates
NCEI_URL_TEMPLATE_SSJ4 = (
    "https://www.ncei.noaa.gov/data/dmsp-space-weather-sensors/"
    "access/f{nn}/ssj/{yyyy}/{mm}/j4f{NN}{yy}{ddd}.gz"
)
NCEI_URL_TEMPLATE_SSJ5 = (
    "https://www.ncei.noaa.gov/data/dmsp-space-weather-sensors/"
    "access/f{nn}/ssj/{yyyy}/{mm}/j5f{NN}{yy}{ddd}.gz"
)

# Keep legacy alias
NCEI_URL_TEMPLATE = NCEI_URL_TEMPLATE_SSJ4

# Default cache directory
DEFAULT_CACHE_DIR = Path("/glade/derecho/scratch/yizhu/tmp/ncei_ssj_cache")


def _is_ssj5(satellite):
    """Return True if the satellite uses SSJ/5 instrument."""
    return satellite.upper() in SSJ5_SATELLITES


def _get_integration_time(satellite):
    """Return integration time in seconds for the given satellite."""
    return INTEGRATION_TIME_SSJ5 if _is_ssj5(satellite) else INTEGRATION_TIME_SSJ4


# ---------------------------------------------------------------------------
# Log decompression
# ---------------------------------------------------------------------------

def decompress_log_counts(raw_value):
    """
    Decompress a log-compressed SSJ count value.

    Parameters
    ----------
    raw_value : int (uint16)
        Raw compressed value from file.

    Returns
    -------
    float
        Decompressed counts, or np.nan for missing data (raw_value == 0).
    """
    if raw_value == 0:
        return np.nan  # missing data
    if raw_value == 1:
        return 0.0     # zero counts
    X = raw_value % 32         # 5-bit mantissa
    Y = (raw_value - X) // 32  # 4-bit exponent (equivalently raw_value >> 5)
    return float((X + 32) * (2 ** Y) - 33)


# Vectorized version for arrays
def _decompress_log_counts_array(raw):
    """Vectorized log decompression for a numpy array of uint16 values."""
    raw = np.asarray(raw, dtype=np.float64)
    result = np.empty_like(raw, dtype=np.float64)

    missing = (raw == 0)
    zero = (raw == 1)
    valid = ~missing & ~zero

    result[missing] = np.nan
    result[zero] = 0.0

    if valid.any():
        r = raw[valid].astype(np.int64)
        X = r % 32
        Y = (r - X) // 32
        result[valid] = (X + 32) * (2.0 ** Y) - 33.0

    return result


# ---------------------------------------------------------------------------
# Header / coordinate conversions
# ---------------------------------------------------------------------------

def _convert_lat(raw):
    """Convert raw uint16 latitude word per AFRL spec."""
    if raw <= 1800:
        return (raw - 900) / 10.0
    else:
        return (raw - 4995) / 10.0


def _convert_lon(raw):
    """Convert raw uint16 longitude word to degrees (0-360)."""
    return raw / 10.0


def _parse_header(words):
    """
    Parse a 15-word minute-block header.

    Parameters
    ----------
    words : array-like of 15 uint16 values

    Returns
    -------
    dict with header fields
    """
    doy = int(words[0])
    hour = int(words[1])
    minute = int(words[2])
    second = int(words[3])
    year = int(words[4]) + 1950

    return {
        "year": year,
        "doy": doy,
        "hour": hour,
        "minute": minute,
        "second": second,
        "glat": _convert_lat(int(words[5])),
        "glon": _convert_lon(int(words[6])),
        "alt_km": int(words[7]) * 1.852,  # nautical miles -> km
        "fp_lat": _convert_lat(int(words[8])),
        "fp_lon": _convert_lon(int(words[9])),
        "cgm_lat": _convert_lat(int(words[10])),
        "cgm_lon": _convert_lon(int(words[11])),
        "mlt_hour": int(words[12]),
        "mlt_minute": int(words[13]),
        "mlt_second": int(words[14]),
    }


# ---------------------------------------------------------------------------
# Physics: flux computation
# ---------------------------------------------------------------------------

def compute_fluxes(counts_electrons, counts_ions, integration_time=None):
    """
    Compute differential/total fluxes and average energies from raw counts.

    Parameters
    ----------
    counts_electrons : (20,) array of decompressed counts (may contain NaN)
    counts_ions : (20,) array of decompressed counts (may contain NaN)
    integration_time : float, optional
        Integration time in seconds. Defaults to INTEGRATION_TIME_SSJ4 (0.098 s).

    Returns
    -------
    dict with:
        ion_avg_energy, ele_avg_energy (eV)
        ion_diff_energy_flux (20,) array in eV/(cm^2 s sr eV)
        ion_total_energy_flux (eV/(cm^2 s sr))
        ele_total_energy_flux (eV/(cm^2 s sr))
    """
    E = CHANNEL_ENERGIES
    dE = CHANNEL_WIDTHS
    G = G_APPROX
    dt = integration_time if integration_time is not None else INTEGRATION_TIME_SSJ4

    result = {}
    for prefix, counts in [("ion", counts_ions), ("ele", counts_electrons)]:
        # Differential number flux: J_i = counts_i / (G * dt * dE_i)
        # Units: #/(cm^2 s sr eV)
        J = counts / (G * dt * dE)

        # Differential energy flux: JE_i = J_i * E_i
        # Units: eV/(cm^2 s sr eV)
        JE = J * E

        # Total number flux: integral of J_i * dE_i
        total_number_flux = np.nansum(J * dE)   # #/(cm^2 s sr)

        # Total energy flux: integral of JE_i * dE_i = sum(counts_i * E_i / (G * dt))
        total_energy_flux = np.nansum(JE * dE)  # eV/(cm^2 s sr)

        # Average energy
        if total_number_flux > 0:
            avg_energy = total_energy_flux / total_number_flux
        else:
            avg_energy = np.nan

        result[f"{prefix}_avg_energy"] = avg_energy
        result[f"{prefix}_total_energy_flux"] = total_energy_flux
        if prefix == "ion":
            result["ion_diff_energy_flux"] = JE
        if prefix == "ele":
            result["ele_diff_energy_flux"] = JE

    return result


# ---------------------------------------------------------------------------
# File reader
# ---------------------------------------------------------------------------

def read_ssj_file(filepath, satellite=None):
    """
    Read one NCEI old-format DMSP SSJ/4 or SSJ/5 .gz binary file.

    Parameters
    ----------
    filepath : str or Path
        Path to the .gz compressed binary file.
    satellite : str, optional
        Satellite identifier (e.g. "F16"). Used to determine SSJ/4 vs SSJ/5
        time decoding and integration time. If None, auto-detected from filename.

    Returns
    -------
    list of dict
        Per-second records with fields:
            datetime, glat, glon, alt_km, cgm_lat, cgm_lon, mlt,
            ion_avg_energy, ele_avg_energy (eV),
            ion_diff_energy_flux (20-element list),
            ion_total_energy_flux, ele_total_energy_flux (eV/(cm^2 s sr))
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Auto-detect SSJ version from filename if satellite not given
    if satellite is not None:
        ssj5 = _is_ssj5(satellite)
    else:
        ssj5 = filepath.name.startswith("j5")

    int_time = INTEGRATION_TIME_SSJ5 if ssj5 else INTEGRATION_TIME_SSJ4

    # Read and decompress
    with gzip.open(filepath, "rb") as f:
        raw_bytes = f.read()

    n_bytes = len(raw_bytes)
    if n_bytes < BYTES_PER_MINUTE:
        log.warning(f"File too short ({n_bytes} bytes): {filepath}")
        return []

    n_blocks = n_bytes // BYTES_PER_MINUTE
    if n_bytes % BYTES_PER_MINUTE != 0:
        log.warning(
            f"File size {n_bytes} not a multiple of {BYTES_PER_MINUTE}; "
            f"processing {n_blocks} complete blocks"
        )

    # Unpack all words as big-endian uint16
    n_words = n_blocks * WORDS_PER_MINUTE
    words = struct.unpack(f">{n_words}H", raw_bytes[:n_words * 2])

    records = []

    for block_idx in range(n_blocks):
        block_start = block_idx * WORDS_PER_MINUTE
        block_words = words[block_start:block_start + WORDS_PER_MINUTE]

        # Parse header
        try:
            header = _parse_header(block_words[:HEADER_WORDS])
        except Exception as e:
            log.debug(f"Bad header in block {block_idx}: {e}")
            continue

        # Validate header fields
        if not (1950 <= header["year"] <= 2030):
            log.debug(f"Invalid year {header['year']} in block {block_idx}, skipping")
            continue
        if not (1 <= header["doy"] <= 366):
            continue

        # Compute CGM MLT as decimal hours
        mlt = header["mlt_hour"] + header["mlt_minute"] / 60.0 + header["mlt_second"] / 3600.0

        # Process 60 per-second records
        data_start = HEADER_WORDS
        for sec_idx in range(SECONDS_PER_MINUTE):
            sec_offset = data_start + sec_idx * WORDS_PER_SECOND
            sec_words = block_words[sec_offset:sec_offset + WORDS_PER_SECOND]

            if len(sec_words) < WORDS_PER_SECOND:
                continue

            ut_hour = int(sec_words[0])
            ut_min = int(sec_words[1])
            raw_time_word = int(sec_words[2])

            if ssj5:
                # SSJ/5: word[2] encodes milliseconds (e.g. 30015 = 30s + 15ms)
                ut_sec = raw_time_word // 1000
            else:
                # SSJ/4: word[2] is plain seconds
                ut_sec = raw_time_word

            # Validate time
            if ut_hour > 23 or ut_min > 59 or ut_sec > 59:
                continue

            # Construct datetime
            try:
                dt = datetime.datetime(
                    header["year"], 1, 1,
                    ut_hour, ut_min, ut_sec
                ) + datetime.timedelta(days=header["doy"] - 1)
            except (ValueError, OverflowError):
                continue

            # Extract raw channel values (words 4-23 = electrons, 24-43 = ions)
            raw_ele = np.array(sec_words[3:23], dtype=np.uint16)   # 20 values
            raw_ion = np.array(sec_words[23:43], dtype=np.uint16)  # 20 values

            # Unshuffle channel order: file order -> channel 1..20
            raw_ele = raw_ele[CHANNEL_UNSHUFFLE]
            raw_ion = raw_ion[CHANNEL_UNSHUFFLE]

            # Decompress log-compressed counts
            counts_ele = _decompress_log_counts_array(raw_ele)
            counts_ion = _decompress_log_counts_array(raw_ion)

            # Check if all channels are missing
            if np.all(np.isnan(counts_ele)) and np.all(np.isnan(counts_ion)):
                continue

            # Compute fluxes
            fluxes = compute_fluxes(counts_ele, counts_ion, integration_time=int_time)

            records.append({
                "datetime": dt,
                "glat": header["glat"],
                "glon": header["glon"],
                "alt_km": header["alt_km"],
                "cgm_lat": header["cgm_lat"],
                "cgm_lon": header["cgm_lon"],
                "mlt": mlt,
                "ion_avg_energy": fluxes["ion_avg_energy"],
                "ele_avg_energy": fluxes["ele_avg_energy"],
                "ion_diff_energy_flux": fluxes["ion_diff_energy_flux"].tolist(),
                "ion_total_energy_flux": fluxes["ion_total_energy_flux"],
                "ele_total_energy_flux": fluxes["ele_total_energy_flux"],
            })

    instr = "SSJ/5" if ssj5 else "SSJ/4"
    log.info(f"Read {len(records)} {instr} records from {filepath.name} ({n_blocks} minute blocks)")
    return records


def read_ssj4_file(filepath):
    """Legacy wrapper: read an SSJ/4 file. See read_ssj_file() for details."""
    return read_ssj_file(filepath, satellite=None)


# ---------------------------------------------------------------------------
# Download from NCEI
# ---------------------------------------------------------------------------

def download_ncei_day(satellite, date, cache_dir=None):
    """
    Download one day's SSJ file from NCEI.

    Parameters
    ----------
    satellite : str
        Satellite identifier, e.g. "F10", "F11", "F16", "F17", "F18".
    date : datetime.date
        Date to download.
    cache_dir : str or Path, optional
        Directory to save files. Defaults to DEFAULT_CACHE_DIR.

    Returns
    -------
    Path or None
        Path to the downloaded .gz file, or None if not available.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sat_num = satellite.upper().replace("F", "")
    nn = sat_num          # e.g. "10" or "16"
    NN = sat_num          # same for filename
    yy = f"{date.year % 100:02d}"
    yyyy = f"{date.year:04d}"
    mm = f"{date.month:02d}"
    ddd = f"{date.timetuple().tm_yday:03d}"

    ssj5 = _is_ssj5(satellite)
    prefix = "j5" if ssj5 else "j4"
    url_template = NCEI_URL_TEMPLATE_SSJ5 if ssj5 else NCEI_URL_TEMPLATE_SSJ4

    filename = f"{prefix}f{NN}{yy}{ddd}.gz"
    local_path = cache_dir / filename

    if local_path.exists():
        log.debug(f"Using cached file: {local_path}")
        return local_path

    url = url_template.format(
        nn=nn, yyyy=yyyy, mm=mm, NN=NN, yy=yy, ddd=ddd,
    )

    log.info(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, local_path)
        log.info(f"Saved to {local_path}")
        return local_path
    except urllib.error.HTTPError as e:
        if e.code == 404:
            log.warning(f"File not found on NCEI (404): {filename}")
        else:
            log.warning(f"HTTP error {e.code} downloading {url}: {e}")
        return None
    except Exception as e:
        log.warning(f"Failed to download {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Process one day with Anderson cusp criteria
# ---------------------------------------------------------------------------

def process_ncei_day(satellite, date, cache_dir=None):
    """
    Download, parse, and apply Anderson (2024) cusp criteria to one day of
    NCEI old-format SSJ data.

    Produces output compatible with identify_cusp.py's process_one_day().

    Parameters
    ----------
    satellite : str
        "F10", "F11", "F16", "F17", or "F18"
    date : datetime.date
        Date to process.
    cache_dir : str or Path, optional
        Cache directory for downloads.

    Returns
    -------
    list of dict
        Cusp crossing dicts in the same format as identify_cusp.process_one_day().
    """
    # Import cusp identification machinery from identify_cusp.py
    # This file lives in the same directory.
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from identify_cusp import (
        newell_cusp_mask,
        sliding_window_cusp,
        extract_crossings,
        dipole_tilt_angle,
    )

    # Step 1: Download
    gz_path = download_ncei_day(satellite, date, cache_dir=cache_dir)
    if gz_path is None:
        return []

    # Step 2: Parse binary file
    try:
        records = read_ssj_file(gz_path, satellite=satellite)
    except Exception as e:
        log.error(f"Error parsing {gz_path}: {e}")
        return []

    if len(records) < 100:
        log.info(f"Too few records ({len(records)}) for {satellite} {date}")
        return []

    # Step 3: Build arrays compatible with identify_cusp
    n = len(records)
    epoch = np.array([r["datetime"] for r in records])
    ion_avg = np.array([r["ion_avg_energy"] for r in records], dtype=np.float64)
    ele_avg = np.array([r["ele_avg_energy"] for r in records], dtype=np.float64)

    # ion_diff_energy_flux: (N, 20) -> use all 20 channels for cusp identification
    # identify_cusp expects (N, 19) for CDAWeb data, but anderson_cusp_mask can
    # handle any channel count as long as channel_energies matches.
    ion_flux = np.array([r["ion_diff_energy_flux"] for r in records], dtype=np.float64)

    # Total ion energy flux in eV/(cm^2 s sr) -- already computed
    ion_total_eflux = np.array([r["ion_total_energy_flux"] for r in records], dtype=np.float64)

    aacgm_lat = np.array([r["cgm_lat"] for r in records], dtype=np.float64)
    aacgm_lt = np.array([r["mlt"] for r in records], dtype=np.float64)

    # Step 4: Apply Newell 2006 cusp criteria
    cusp_spec = newell_cusp_mask(
        ion_avg, ele_avg, ion_flux, CHANNEL_ENERGIES,
    )

    # Dayside filter (8.5-15.5 MLT)
    dayside = (aacgm_lt >= 8.5) & (aacgm_lt <= 15.5)
    # High latitude filter (|MLAT| > 60)
    highlat = np.abs(aacgm_lat) > 60.0

    cusp_spec = cusp_spec & dayside & highlat

    if cusp_spec.sum() == 0:
        log.info(f"No cusp spectra for {satellite} {date}")
        return []

    # Step 5: Sliding window (3/4 rule)
    cusp_windowed = sliding_window_cusp(cusp_spec, window=4, threshold=3)

    if cusp_windowed.sum() == 0:
        return []

    # Step 6: Extract crossings
    crossings = extract_crossings(epoch, aacgm_lat, aacgm_lt, cusp_windowed)
    crossings = [c for c in crossings if c is not None]

    # Step 7: Add metadata
    instr = "SSJ/5" if _is_ssj5(satellite) else "SSJ/4"
    for c in crossings:
        c["satellite"] = satellite
        c["date"] = str(date)
        c["data_source"] = f"NCEI_binary_{instr.replace('/', '')}"
        try:
            c["dipole_tilt"] = round(dipole_tilt_angle(
                datetime.datetime(date.year, date.month, date.day, 12)
            ), 2)
        except Exception:
            c["dipole_tilt"] = None

    log.info(
        f"{satellite} {date}: {cusp_spec.sum()} cusp spectra, "
        f"{cusp_windowed.sum()} windowed, {len(crossings)} crossings"
    )
    return crossings


# ---------------------------------------------------------------------------
# Main: batch processing over a date range
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse NCEI old-format DMSP SSJ/4 and SSJ/5 binary files "
                    "and identify cusp crossings."
    )
    parser.add_argument(
        "--satellite", required=True,
        choices=sorted(ALL_SATELLITES),
        help="DMSP satellite (F10, F11 for SSJ/4; F16, F17, F18 for SSJ/5)",
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory for output JSON files",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help=f"Cache directory for downloaded files (default: {DEFAULT_CACHE_DIR})",
    )
    args = parser.parse_args()

    start = datetime.date.fromisoformat(args.start)
    end = datetime.date.fromisoformat(args.end)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info(f"Processing {args.satellite} from {start} to {end}")

    all_crossings = []
    date = start
    while date <= end:
        try:
            crossings = process_ncei_day(
                args.satellite, date, cache_dir=args.cache_dir
            )
            all_crossings.extend(crossings)
        except Exception as e:
            log.error(f"Error processing {args.satellite} {date}: {e}")
        date += datetime.timedelta(days=1)

    # Save results
    outfile = outdir / f"cusp_crossings_{args.satellite}_{start}_{end}_ncei.json"
    with open(outfile, "w") as f:
        json.dump(all_crossings, f, indent=2, default=str)

    log.info(f"Saved {len(all_crossings)} crossings to {outfile}")
    return all_crossings


if __name__ == "__main__":
    main()
