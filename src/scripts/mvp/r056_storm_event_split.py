#!/usr/bin/env python
"""R056: storm-event-based train/test split (Reviewer 1 Major 1).

WHAT:  Group every cusp crossing by the geomagnetic STORM EVENT it belongs to (from hourly Dst),
       so that all crossings of one storm land entirely in train or entirely in test, then report
       XGBoost equatorward-MLAT MAE under (a) 5-fold GroupKFold over events and (b) a
       leave-one-storm-out test restricted to storm-time crossings. This is the split the reviewer
       asked for ("all crossings associated with a given geomagnetic event assigned exclusively to
       either training or testing"), stronger than the day-grouped / contiguous-block controls.
CLAIM: prints and saves the event-grouped MAE next to the existing split ladder
       (random 0.956, temporal 1.107, contiguous-block LOBO 1.272). A value near the temporal/LOBO
       band (~1.1-1.3 deg) confirms the random split was optimistic but the skill is genuine under
       true event independence.
PHYSICS: a storm event = a contiguous interval where hourly Dst <= DST_THR (moderate storm),
       expanded to onset (last hour Dst > -15 before) and recovery (first 6-h stretch Dst > -20
       after), with events merged if their windows lie within MERGE_H hours. Crossings outside any
       storm are grouped by calendar month so quiet-time crossings are also leak-free groups.
UNITS: Dst [nT]; MAE [deg magnetic latitude]. Crossing time is UT (time_start).
INPUTS:  output/omni_full_hist_90120/cusp_crossings_*.json (via r046.load, the 39,668 frame);
         hourly Dst pulled once from CDAWeb HAPI (OMNI2_H0_MRG1HR / DST1800) and cached to
         src/kernels/cuspmap_mvp/bundles/omni_dst_hourly_1987_2014.parquet.
OUTPUTS: src/kernels/cuspmap_mvp/bundles/r052_storm_event.json
RUN:     conda activate py3.10 && python src/scripts/mvp/r056_storm_event_split.py
DEPS:    py3.10 (xgboost, sklearn, pandas, hapiclient). Imports load/feats/XGB/mae from
         r046_matched_splits (documented sibling with a __main__ guard) to guarantee an IDENTICAL
         modeling frame; no other sibling logic is reused.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from r046_matched_splits import load, feats, XGB, mae               # identical frame + model config

BUND = "/glade/work/yizhu/cuspML/src/kernels/cuspmap_mvp/bundles"
DST_CACHE = f"{BUND}/omni_dst_hourly_1987_2014.parquet"
OUT = f"{BUND}/r052_storm_event.json"

DST_THR = -30.0      # nT: moderate-storm threshold defining event cores
ONSET_LVL = -15.0    # walk back to where Dst was last quieter than this
RECOV_LVL = -20.0    # recovery when Dst climbs above this and stays for RECOV_H hours
RECOV_H = 6
MERGE_H = 24         # merge two events whose windows are within this many hours
MIN_TEST = 40        # min storm crossings for a storm to be a leave-one-out test fold


def _naive(d):
    # crossing times are tz-naive UT; force the Dst series to match so comparisons never silently offset
    if getattr(d['time'].dt, 'tz', None) is not None:
        d = d.copy()
        d['time'] = d['time'].dt.tz_localize(None)
    return d


def get_dst():
    if os.path.exists(DST_CACHE):
        d = _naive(pd.read_parquet(DST_CACHE))
        print(f"Dst cache: {len(d)} hourly rows {d.time.min()}..{d.time.max()}")
        return d
    from hapiclient import hapi
    srv = "https://cdaweb.gsfc.nasa.gov/hapi"
    frames = []
    for y0 in range(1987, 2015, 4):                                 # pull in 4-year chunks
        y1 = min(y0 + 4, 2015)
        data, meta = hapi(srv, "OMNI2_H0_MRG1HR", "DST1800",
                          f"{y0}-01-01T00:00:00", f"{y1}-01-01T00:00:00")
        t = pd.to_datetime([x.decode() if isinstance(x, bytes) else x for x in data["Time"]])
        frames.append(pd.DataFrame({"time": t, "dst": np.asarray(data["DST1800"], float)}))
        print(f"  pulled Dst {y0}-{y1}: {len(frames[-1])} rows")
    d = pd.concat(frames, ignore_index=True)
    d = d[d.dst < 99990].sort_values("time").reset_index(drop=True)  # drop fill values
    d = _naive(d)
    d.to_parquet(DST_CACHE)
    print(f"Dst pulled + cached: {len(d)} rows")
    return d


def build_events(dst):
    """Return a list of (start, end) storm-window timestamps from the hourly Dst series."""
    t = dst.time.values
    v = dst.dst.values
    below = v <= DST_THR
    events = []
    i = 0
    n = len(v)
    while i < n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j < n and below[j]:                                   # core: contiguous Dst<=thr
            j += 1
        # onset: walk back to last hour quieter than ONSET_LVL
        a = i
        while a > 0 and v[a - 1] <= ONSET_LVL:
            a -= 1
        # recovery: from j, first hour after which Dst stays > RECOV_LVL for RECOV_H hours
        b = j
        while b < n:
            if v[b] > RECOV_LVL and np.all(v[b:min(b + RECOV_H, n)] > RECOV_LVL):
                break
            b += 1
        events.append((t[a], t[min(b, n - 1)]))
        i = max(j, b)
    # merge events within MERGE_H hours
    merged = []
    for s, e in events:
        if merged and (s - merged[-1][1]) <= np.timedelta64(MERGE_H, "h"):
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def assign_groups(times, events):
    """Each crossing -> 'S<k>' if inside storm event k, else 'Q<YYYY-MM>' (leak-free quiet group)."""
    tv = times.values.astype("datetime64[ns]")
    grp = np.array([None] * len(tv), dtype=object)
    is_storm = np.zeros(len(tv), bool)
    for k, (s, e) in enumerate(events):
        m = (tv >= np.datetime64(s)) & (tv <= np.datetime64(e))
        grp[m] = f"S{k}"
        is_storm |= m
    q = ~is_storm
    ym = pd.Series(times).dt.strftime("Q%Y-%m").values
    grp[q] = ym[q]
    assert all(g is not None for g in grp), "every crossing must get a group"
    return grp, is_storm


def main():
    df = load()
    f74 = feats(df, 60)
    need = f74 + ['abs_eq_mlat', 'ae_index', 'hemisphere', 'date', 'year', 'newell_cf']
    df = df[df[[c for c in need if c in df.columns]].notna().all(axis=1)].sort_values('t').reset_index(drop=True)
    N = len(df)
    print("rows", N)
    assert 39000 < N < 40500, f"frame row count {N} does not match the ~39,668 modeling set"
    X = df[f74].values.astype(np.float32)
    y = df['abs_eq_mlat'].values.astype(np.float32)

    dst = get_dst()
    events = build_events(dst)
    grp, is_storm = assign_groups(df['t'], events)

    # --- review sanity: event catalogue + a known storm must be captured ---
    n_storm_cross = int(is_storm.sum())
    # the Nov-2003 superstorm (Dst min -422 at 2003-11-20 20:30 UT) must fall inside some event window
    hall = [(s, e) for (s, e) in events
            if pd.Timestamp(s) <= pd.Timestamp("2003-11-20 20:30") <= pd.Timestamp(e)]
    print(f"events: {len(events)} storms | storm-time crossings: {n_storm_cross}"
          f" ({100*n_storm_cross/N:.1f}%) | groups total: {len(set(grp))}")
    print(f"  Nov-2003 superstorm captured as an event: {'YES' if hall else 'NO'}")
    dmin = float(dst.dst.min())
    print(f"  Dst series min (should be ~-422): {dmin}")
    assert hall, "sanity check failed: Nov-2003 superstorm (Dst -422) not captured as an event"

    R = {'n_rows': int(N), 'n_events': len(events), 'n_storm_crossings': n_storm_cross,
         'params': dict(DST_THR=DST_THR, ONSET_LVL=ONSET_LVL, RECOV_LVL=RECOV_LVL,
                        RECOV_H=RECOV_H, MERGE_H=MERGE_H)}

    # --- (a) event-grouped 5-fold GroupKFold: no storm spans train/test ---
    gkf = GroupKFold(n_splits=5)
    fold_mae = []
    for tr, te in gkf.split(X, y, groups=grp):
        assert not (set(grp[tr]) & set(grp[te])), "group leak between train and test"
        xg = XGBRegressor(**XGB).fit(X[tr], y[tr])
        fold_mae.append(mae(y[te], xg.predict(X[te])))
    R['event_groupkfold'] = dict(MAE_mean=round(float(np.mean(fold_mae)), 4),
                                 MAE_std=round(float(np.std(fold_mae)), 4),
                                 per_fold=[round(x, 4) for x in fold_mae])
    print("event GroupKFold XGB:", R['event_groupkfold'])

    # --- (b) single 80/20 event-grouped split (mirrors the day-grouped report) ---
    gtr, gte = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups=grp))
    xg = XGBRegressor(**XGB).fit(X[gtr], y[gtr])
    R['event_groupshuffle_80_20'] = round(mae(y[gte], xg.predict(X[gte])), 4)
    print("event GroupShuffleSplit 80/20 XGB:", R['event_groupshuffle_80_20'])

    # --- (c) leave-one-storm-out, tested ONLY on that storm's crossings (the reviewer's worry) ---
    storm_ids = [g for g in set(grp[is_storm])]
    loso = []
    for sid in storm_ids:
        te = grp == sid
        if te.sum() < MIN_TEST:
            continue
        tr = ~te
        xg = XGBRegressor(**XGB).fit(X[tr], y[tr])
        loso.append(mae(y[te], xg.predict(X[te])))
    if loso:
        R['leave_one_storm_out'] = dict(MAE_mean=round(float(np.mean(loso)), 4),
                                        MAE_std=round(float(np.std(loso)), 4),
                                        n_storms=len(loso), min_test=MIN_TEST)
        print("leave-one-storm-out XGB (storm-time test only):", R['leave_one_storm_out'])
    else:
        R['leave_one_storm_out'] = f"no storm had >= {MIN_TEST} crossings"
        print(R['leave_one_storm_out'])

    json.dump(R, open(OUT, 'w'), indent=1)
    print("saved", OUT)
    try:
        from runlog import log_run
        log_run(inputs=[DST_CACHE], outputs=[OUT], note="storm-event split (R1-M1)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
