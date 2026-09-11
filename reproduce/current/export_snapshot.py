"""Export the manuscript's processed inputs and reference outputs for distribution.

PHYSICS: No new identification, filtering, fitting or imputation is performed.
UNITS: MLAT degrees; MLT hours; solar-wind speed km/s, IMF nT, pressure nPa.
INPUTS: --source-root author's project, with the September 2026 audit artifacts.
OUTPUTS: --output-dir portable parquet/NPZ/UBJ files and SHA256 manifest.
RUN: python reproduce/current/export_snapshot.py --source-root .
DEPS: numpy, pandas, pyarrow. This export step is for maintainers only;
      consumers use the bundled snapshot directly, without author files.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'data')
    args = ap.parse_args()
    root, out = args.source_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = root / 'output/audits/20260907_temporal_manuscript'
    baseline = root / 'output/audits/20260906_temporal_baselines'
    independent = root / 'output/audits/20260909_independent_history_windows'
    analysis = json.loads((source / 'analysis.json').read_text())
    rows = pd.read_pickle(source / 'extended_rows.pkl').reset_index(drop=True)
    reference = pd.read_pickle(baseline / 'audit_rows_all_refreshed.pkl')
    features = analysis['features']
    assert len(rows) == 39668 and len(features) == len(set(features)) == 74
    np.testing.assert_array_equal(rows[features].to_numpy(np.float32), reference[features].to_numpy(np.float32))
    rows['row_index'] = np.arange(len(rows))
    rows.to_parquet(out / 'rows.parquet', index=False, compression='zstd')
    records, sources = [], {}
    for path in sorted((root / 'output/omni_full_hist').glob('cusp_crossings_*.json')):
        records.extend(json.loads(path.read_text()))
        sources[str(path.relative_to(root))] = digest(path)
    assert len(records) == 48056
    coverage = pd.DataFrame(records).dropna(subset=['imf_bz','sw_v','sw_n','sw_pdyn','eq_mlat','eq_mlt'])
    coverage = coverage[['time_start','satellite','hemisphere','eq_mlat','eq_mlt']].copy()
    coverage['year'] = pd.to_datetime(coverage.time_start).dt.year
    assert len(coverage) == 40813
    coverage.to_parquet(out / 'coverage.parquet', index=False, compression='zstd')
    calibration = pd.read_pickle(baseline / 'audit_rows.pkl')
    calibration['row_index'] = np.arange(len(calibration))
    calibration = calibration.loc[calibration.year < 2008, ['row_index','year','abs_eq_mlat','sw_v','B_T','sin_clock_half']]
    assert len(calibration) == 29935
    calibration.to_parquet(out / 'calibration.parquet', index=False, compression='zstd')
    shutil.copy2(root / 'src/kernels/cuspmap_mvp/bundles/omni_dst_hourly_1987_2014.parquet', out / 'dst.parquet')
    (out / 'predictions').mkdir(exist_ok=True)
    (out / 'models').mkdir(exist_ok=True)
    for path in sorted(source.glob('*.npz')):
        shutil.copy2(path, out / 'predictions' / path.name)
    shutil.copy2(source / 'primary.ubj', out / 'models/primary.ubj')
    ref = np.load(baseline / 'fully_corrected_temporal_predictions.npz')
    test = np.flatnonzero(rows.year.to_numpy() >= 2008)
    observed = rows.abs_eq_mlat.to_numpy(np.float32)[test]
    for name in ['Ridge74', 'GBR300']:
        np.savez_compressed(out / 'predictions' / (name + '.npz'), predicted=ref[name], observed=observed, test_indices=test)
    b = np.load(root / 'paper/audit/20260906_instant_formula_predictions.npz')
    np.savez_compressed(out / 'predictions/Newell2006.npz', predicted=b['existing_train_fitted_coefficients_instant'], observed=observed, test_indices=test)
    (out / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    (out / 'independent').mkdir(exist_ok=True)
    for name in ['history_features.npz', 'anchor60.npz', 'coupling4_90.npz', 'coupling4_120.npz', 'full_block22_90.npz', 'full_block22_120.npz']:
        shutil.copy2(independent / name, out / 'independent' / name)
    # The portable manifest records source hashes, never machine-specific paths.
    for path in [source / 'extended_rows.pkl', baseline / 'audit_rows_all_refreshed.pkl', baseline / 'audit_rows.pkl', source / 'analysis.json', source / 'primary.ubj', independent / 'history_features.npz']:
        sources[str(path.relative_to(root))] = digest(path)
    window_cols = {str(w): features[:16] + [c for c in features[16:] if int(''.join(filter(str.isdigit,c))) <= w] for w in [0,15,30,60,90,120]}
    for w in [90,120]:
        window_cols[str(w)] += sorted(f'{k}_{s}{win}' for k in ['imf_bx','imf_by','imf_bz','sw_v','sw_n','sw_pdyn'] for s in ['mean','std','delta'] for win in [90,120] if win <= w)
    manifest = dict(schema_version=1, snapshot='September 2026 temporal manuscript',
        scope='Processed crossing-table reproduction; not raw DMSP label re-identification.',
        features=features, parameters=analysis['parameters'], targets=['abs_eq_mlat','abs_pole_mlat','eq_mlt','mean_mlt'],
        n_rows=len(rows), split=dict(method='temporal',train='year < 2008',test='year >= 2008',n_train=29935,n_test=9733),
        interpretation=dict(n=2000,seed=42,sampling='numpy.random.default_rng, without replacement from temporal test indices'),
        baseline=dict(formula='intercept + slope * sw_v**(2/3) * B_T**(2/3) * abs(sin_clock_half)**(8/3)',
            calibration='calibration.parquet contains the pre-2008 inputs used for the frozen manuscript fit, before the full OMNI refresh; test inference uses rows.parquet.',
            coefficients=[78.40937950077127,-0.02247589222708998]),
        cumulative_window_features=window_cols, author_source_sha256=sources)
    manifest['files'] = {str(p.relative_to(out)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {len(manifest["files"])} files; {sum(v["bytes"] for v in manifest["files"].values()) / 1e6:.1f} MB to {out}')


if __name__ == '__main__':
    main()
