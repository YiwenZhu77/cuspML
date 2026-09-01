"""
R041 (robust): add 90- and 120-minute OMNI window features to the model's actual
input crossings, for the reviewer-requested timescale sweep (R2-M1).

Rewrite of the first attempt, which cached ALL ~28 OMNI years in RAM at once and was
OOM-killed on the login node. This version processes ONE year at a time: download ->
match that year's file -> write -> free. A failed/corrupt year is skipped, not fatal.

Input : output/omni_full_hist/cusp_crossings_*.json  (the 48,056-crossing DB the
        1D model is trained on; each crossing already has time_start + 15/30/60 feats)
Output: output/omni_full_hist_90120/<same names>  (same records + *_mean90/std90/
        delta90 and *_mean120/std120/delta120 for the six SW keys)
Claim : lets us train XGBoost with max-window in {60, 90, 120} on identical crossings
        and show MAE does not improve past 60 min.
"""
import datetime, json, glob, logging, os
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

IN_DIR  = "/glade/work/yizhu/cuspML/output/omni_full_hist"
OUT_DIR = "/glade/work/yizhu/cuspML/output/omni_full_hist_90120"
BS_TO_CUSP_DELAY = 10 * 60          # 10 min, same as the base pipeline
NEW_WINDOWS = [90, 120]            # minutes
SW_KEYS = ["imf_bx", "imf_by", "imf_bz", "sw_v", "sw_n", "sw_pdyn"]


def download_omni_year(year, tries=3):
    """1-min OMNI for one year from CDAWeb, with retries on transient CDF errors."""
    from cdasws import CdasWs
    cdas = CdasWs()
    ds = "OMNI_HRO_1MIN"
    varnames = ["BX_GSE", "BY_GSM", "BZ_GSM", "flow_speed", "proton_density", "Pressure"]
    t0 = datetime.datetime(year, 1, 1)
    t1 = datetime.datetime(year + 1, 1, 1)
    last = None
    for k in range(tries):
        try:
            _, data = cdas.get_data(ds, varnames, t0, t1)
            if data and "Epoch" in data:
                return data
            last = "empty payload"
        except Exception as e:
            last = repr(e)
            log.warning(f"OMNI {year} attempt {k+1}/{tries} failed: {last}")
    log.error(f"OMNI {year} giving up: {last}")
    return None


def prep_omni(data):
    epoch = np.array(data["Epoch"])
    arrays = {}
    for key, cdaw in [("imf_bx", "BX_GSE"), ("imf_by", "BY_GSM"), ("imf_bz", "BZ_GSM"),
                      ("sw_v", "flow_speed"), ("sw_n", "proton_density"), ("sw_pdyn", "Pressure")]:
        arr = np.array(data.get(cdaw, []), dtype=float)
        arr[np.abs(arr) > 9999] = np.nan
        arrays[key] = arr
    epoch_sec = epoch.astype("datetime64[s]").astype(np.int64)
    return epoch_sec, arrays


def add_windows(crossings, epoch_sec, arrays):
    """Add *_mean/std/delta for NEW_WINDOWS to each crossing (in place)."""
    times, vidx = [], []
    for i, c in enumerate(crossings):
        try:
            t = np.datetime64(c["time_start"][:19]).astype("datetime64[s]").astype(np.int64)
            times.append(t); vidx.append(i)
        except Exception:
            continue
    if not times:
        return crossings
    csec = np.array(times) - BS_TO_CUSP_DELAY
    idx = np.clip(np.searchsorted(epoch_sec, csec), 1, len(epoch_sec) - 1)
    left = np.abs(csec - epoch_sec[idx - 1]); right = np.abs(csec - epoch_sec[idx])
    nearest = np.where(left <= right, idx - 1, idx)

    for j, ci in enumerate(vidx):
        c = crossings[ci]; ni = nearest[j]
        for win in NEW_WINDOWS:
            i0 = max(0, ni - win); i1 = ni + 1
            for key in SW_KEYS:
                w = arrays[key][i0:i1]
                v = w[~np.isnan(w)]
                if len(v) >= win // 3:
                    c[f"{key}_mean{win}"]  = float(np.mean(v))
                    c[f"{key}_std{win}"]   = float(np.std(v))
                    c[f"{key}_delta{win}"] = float(v[-1] - v[0]) if len(v) >= 2 else None
                else:
                    c[f"{key}_mean{win}"] = c[f"{key}_std{win}"] = c[f"{key}_delta{win}"] = None
    return crossings


def year_of(crossings):
    for c in crossings:
        ts = c.get("time_start") or c.get("date")
        if ts:
            return int(str(ts)[:4])
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(f"{IN_DIR}/cusp_crossings_*.json"))
    log.info(f"{len(files)} input files")

    # group files by year so each OMNI year is downloaded once
    by_year = {}
    for f in files:
        crossings = json.load(open(f))
        y = year_of(crossings)
        by_year.setdefault(y, []).append((f, crossings))

    years = sorted(k for k in by_year if k is not None)
    log.info(f"{len(years)} years: {years}")

    done_files = 0
    for y in years:
        out_paths = [os.path.join(OUT_DIR, os.path.basename(f)) for f, _ in by_year[y]]
        if all(os.path.exists(p) for p in out_paths):
            log.info(f"{y}: all outputs exist, skip")
            done_files += len(out_paths)
            continue
        data = download_omni_year(y)
        if data is None:
            log.error(f"{y}: no OMNI, writing inputs unchanged (90/120 absent)")
            for f, crossings in by_year[y]:
                json.dump(crossings, open(os.path.join(OUT_DIR, os.path.basename(f)), "w"),
                          default=str)
            continue
        epoch_sec, arrays = prep_omni(data)
        for f, crossings in by_year[y]:
            add_windows(crossings, epoch_sec, arrays)
            json.dump(crossings, open(os.path.join(OUT_DIR, os.path.basename(f)), "w"),
                      default=str)
            done_files += 1
        del data, epoch_sec, arrays
        log.info(f"{y}: {len(by_year[y])} file(s) done  ({done_files}/{len(files)})")

    log.info(f"=== DONE: {done_files}/{len(files)} files written to {OUT_DIR} ===")


if __name__ == "__main__":
    main()
