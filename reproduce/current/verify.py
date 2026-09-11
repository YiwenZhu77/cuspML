"""Verify the distributed processed-data reproduction against manuscript outputs.

PHYSICS: Newell 2006 E_WAV^(2/3) fit, temporal MLAT prediction, Tree SHAP and PDP.
UNITS: latitude/residuals degrees, MLT hours; see README for solar-wind units.
INPUTS: --data-dir snapshot, optional freshly trained --run-dir.
OUTPUTS: --output-dir/verification.json and a numerical analysis.json for figures.
RUN: python reproduce/current/verify.py --explain
DEPS: numpy, pandas, pyarrow, xgboost. No network access or model training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, DMatrix


def metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError('Expected equal nonempty one-dimensional prediction/label arrays')
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError('Nonfinite prediction or label')
    e = p-y
    return dict(n=len(y), MAE=float(abs(e).mean()), RMSE=float(np.sqrt(np.mean(e**2))),
                r=float(np.corrcoef(y,p)[0,1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=Path(__file__).parent/'data')
    ap.add_argument('--run-dir', type=Path, help='Fresh train.py output to compare with the snapshot')
    ap.add_argument('--output-dir', type=Path, default=Path('reproduction-check'))
    ap.add_argument('--explain', action='store_true', help='Recompute Tree SHAP and all six PDP curves')
    args = ap.parse_args()
    data, out = args.data_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not leave an earlier PASS as the apparent current result.
    (out/'verification.json').write_text(json.dumps({'status':'INCOMPLETE'})+'\n')
    manifest = json.loads((data/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = data/name
        assert path.is_file(), f'Missing snapshot file: {name}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], f'Checksum mismatch: {name}'
    rows = pd.read_parquet(data/'rows.parquet')
    cols = manifest['features']
    assert len(cols) == len(set(cols)) == 74 and len(rows) == 39668
    np.testing.assert_array_equal(rows.row_index, np.arange(len(rows)))
    x = rows[cols].to_numpy(np.float32)
    y = rows.abs_eq_mlat.to_numpy(np.float32)
    assert np.isfinite(x).all()
    tr, te = np.flatnonzero(rows.year.to_numpy()<2008), np.flatnonzero(rows.year.to_numpy()>=2008)
    assert (len(tr),len(te)) == (29935,9733)
    assert set(tr).isdisjoint(te)
    reference = json.loads((data/'analysis.json').read_text())
    analysis = json.loads(json.dumps(reference))
    run_dir = args.run_dir.resolve() if args.run_dir else None
    predictions = (run_dir or data)/'predictions'
    modeldir = (run_dir or data)/'models'
    checks = {}
    if run_dir:
        run = json.loads((run_dir/'analysis.json').read_text())
        required_tasks = {'primary','other_targets','baselines','windows','balanced','controls','independent'}
        if set(run['tasks']) != required_tasks:
            raise ValueError('--run-dir requires a completed train.py --tasks all run')
        np.testing.assert_array_equal(run['features'],cols)
        assert (run['n_train'],run['n_test']) == (len(tr),len(te))
        for name in ('rows.parquet','manifest.json','dst.parquet','independent/history_features.npz'):
            assert run['input_sha256'][name] == hashlib.sha256((data/name).read_bytes()).hexdigest(), name
        for section in ('primary','other_targets','baseline','models','windows','balanced','controls','independent'):
            if section not in run:
                raise ValueError(f'Incomplete training report: missing {section}')
        if 'paired_month_bootstrap' not in run['independent']:
            raise ValueError('Incomplete independent-window analysis')

    def compare_metrics(actual, expected, name):
        assert actual['n'] == expected['n'], name
        for key in ('MAE','RMSE','r'):
            np.testing.assert_allclose(actual[key],expected[key],rtol=0,atol=2e-5,err_msg=f'{name}: {key}')

    def check_prediction(name, target, expected=None, test_indices=te, golden_group='predictions',
                         run_name=None):
        golden_path = data/golden_group/f'{name}.npz'
        path = predictions/f'{run_name or name}.npz' if run_dir else golden_path
        if not path.is_file():
            raise FileNotFoundError(f'Required prediction is missing: {path}')
        with np.load(golden_path,allow_pickle=False) as z:
            golden = z['predicted'].copy()
            golden_observed = z['observed'].copy()
            # The historical MLP archive predates explicit stored row indices.
            golden_indices = z['test_indices'].copy() if 'test_indices' in z else test_indices
        np.testing.assert_array_equal(golden_indices,test_indices)
        with np.load(path,allow_pickle=False) as z:
            if run_dir and 'test_indices' not in z:
                raise ValueError(f'Fresh prediction lacks test_indices: {path}')
            ids = z['test_indices'].copy() if 'test_indices' in z else test_indices
            observed = z['observed'].copy()
            predicted = z['predicted'].copy()
        np.testing.assert_array_equal(ids,test_indices)
        obs = rows[target].to_numpy(np.float32)[ids]
        np.testing.assert_array_equal(observed,obs)
        np.testing.assert_array_equal(golden_observed,obs)
        measured = metrics(obs,predicted)
        compare_metrics(measured,metrics(obs,golden),name+' vs archived prediction')
        if expected is not None:
            compare_metrics(measured,expected,name+' vs manuscript')
        np.testing.assert_allclose(predicted,golden,rtol=0,atol=2e-4,err_msg=name)
        checks[name] = measured
        if run_dir and name not in ('Newell2006','anchor60'):
            ext = '.joblib' if name in ('Ridge74','GBR300','MLP74') else '.ubj'
            if not (modeldir/f'{name}{ext}').is_file():
                raise FileNotFoundError(f'Required trained model is missing: {name}{ext}')
            if ext == '.ubj' and not (modeldir/f'{name}.features.json').is_file():
                raise FileNotFoundError(f'Required ordered feature manifest is missing: {name}')
        return predicted

    p = check_prediction('primary','abs_eq_mlat',reference['primary'])
    m = XGBRegressor(n_jobs=2)
    m.load_model(modeldir/'primary.ubj')
    m.set_params(n_jobs=2)
    np.testing.assert_allclose(m.predict(x[te]),p,rtol=0,atol=1e-6)
    analysis['primary'].update(checks['primary'])
    analysis['models']['XGBoost74'].update(checks['primary'])
    if run_dir:
        compare_metrics(checks['primary'],run['primary'],'training report primary')
    for target in manifest['targets'][1:]:
        check_prediction(target,target,reference['other_targets'][target])
        analysis['other_targets'][target].update(checks[target])
        if run_dir:
            compare_metrics(checks[target],run['other_targets'][target],target+' training report')
    for name in ['Ridge74','GBR300','MLP74']:
        check_prediction(name,'abs_eq_mlat',reference['models'][name])
        analysis['models'][name].update(checks[name])
        if run_dir:
            compare_metrics(checks[name],run['models'][name],name+' training report')
    calibration = pd.read_parquet(data/'calibration.parquet')
    assert (calibration.year<2008).all()
    np.testing.assert_array_equal(calibration.row_index,tr)
    def coupling(df):
        return df.sw_v.to_numpy()**(2/3)*df.B_T.to_numpy()**(2/3)*abs(df.sin_clock_half.to_numpy())**(8/3)
    a = np.column_stack([np.ones(len(calibration)), coupling(calibration)])
    coef = np.linalg.lstsq(a,calibration.abs_eq_mlat.to_numpy(np.float32).astype(float),rcond=None)[0]
    np.testing.assert_allclose(coef,manifest['baseline']['coefficients'],rtol=0,atol=1e-10)
    baseline = coef[0]+coef[1]*coupling(rows.iloc[te])
    bp = check_prediction('Newell2006','abs_eq_mlat',reference['baseline'])
    np.testing.assert_allclose(bp,baseline,rtol=0,atol=1e-10)
    if run_dir:
        compare_metrics(checks['Newell2006'],run['baseline'],'Newell training report')
        np.testing.assert_allclose(coef,run['newell_calibration']['coefficients'],rtol=0,atol=1e-10)
        assert run['newell_calibration']['calibration_sha256'] == hashlib.sha256((data/'calibration.parquet').read_bytes()).hexdigest()
    analysis['baseline'].update(checks['Newell2006'])
    analysis['models']['Newell (2006) form'].update(checks['Newell2006'])
    analysis['reduction_pct'] = 100*(1-checks['primary']['MAE']/checks['Newell2006']['MAE'])
    analysis['windows']['60'].update(checks['primary'])
    for w in [0,15,30,90,120]:
        name = f'window{w}'
        check_prediction(name,'abs_eq_mlat',reference['windows'][str(w)])
        analysis['windows'][str(w)].update(checks[name])
        if run_dir:
            compare_metrics(checks[name],run['windows'][str(w)],name+' training report')
    if run_dir:
        compare_metrics(checks['primary'],run['windows']['60'],'window60 training report')
    for kind, summary in reference['controls'].items():
        actual_folds = []
        if run_dir:
            assert len(run['controls'][kind]['folds']) == len(summary['folds']), kind
        for i, expected in enumerate(summary['folds']):
            name = f'control_{kind}_{i}'
            with np.load(data/'predictions'/f'{name}.npz',allow_pickle=False) as z:
                test_ids = z['test_indices'].copy()
            check_prediction(name,'abs_eq_mlat',expected,test_ids)
            actual_folds.append(checks[name])
            if run_dir:
                compare_metrics(checks[name],run['controls'][kind]['folds'][i],name+' training report')
        mae = [v['MAE'] for v in actual_folds]
        for key,value in [('MAE_mean',float(np.mean(mae))),('MAE_std',float(np.std(mae)))]:
            np.testing.assert_allclose(value,summary[key],rtol=0,atol=2e-5)
            analysis['controls'][kind][key] = value
            if run_dir:
                np.testing.assert_allclose(value,run['controls'][kind][key],rtol=0,atol=2e-5)
        analysis['controls'][kind]['folds'] = actual_folds
    balanced = check_prediction('balanced','abs_eq_mlat')
    hemisphere = rows.iloc[te].hemisphere.to_numpy()
    for h in ('N','S'):
        mask = hemisphere == h
        observed = y[te][mask]
        measured = metrics(observed,balanced[mask])
        compare_metrics(measured,reference['balanced'][h],'balanced '+h)
        analysis['balanced'][h].update(measured)
        if run_dir:
            compare_metrics(measured,run['balanced'][h],'balanced training report '+h)
        primary_hemi = metrics(observed,p[mask])
        compare_metrics(primary_hemi,reference['hemisphere'][h],'primary '+h)
        analysis['hemisphere'][h].update(primary_hemi)
    training_counts = rows.iloc[tr].hemisphere.value_counts()
    assert reference['balanced']['n_per_hemisphere'] == min(training_counts['N'],training_counts['S'])
    if run_dir:
        assert run['balanced']['n_per_hemisphere'] == reference['balanced']['n_per_hemisphere']
    ae = rows.iloc[te].ae_index.to_numpy()
    for record in analysis['activity']:
        mask = (ae>=record['lower']) & (ae<(record['upper'] if record['upper'] is not None else np.inf))
        measured = metrics(y[te][mask],p[mask])
        compare_metrics(measured,record,'AE activity bin')
        record.update(measured)
    anchor = check_prediction('anchor60','abs_eq_mlat',reference['primary'],
                              golden_group='independent',run_name='primary')
    np.testing.assert_allclose(anchor,p,rtol=0,atol=1e-6)
    independent = {'anchor60':anchor}
    for group in ('coupling4','full_block22'):
        for width in (90,120):
            name = f'{group}_{width}'
            independent[name] = check_prediction(name,'abs_eq_mlat',golden_group='independent')
            if run_dir:
                compare_metrics(checks[name],run['independent']['models'][name],name+' training report')
    if run_dir:
        compare_metrics(checks['anchor60'],run['independent']['models']['anchor60'],'independent anchor report')
        months = pd.to_datetime(rows.iloc[te].time_start,utc=True).dt.strftime('%Y-%m').to_numpy()
        unique,inverse = np.unique(months,return_inverse=True)
        sizes = np.bincount(inverse)
        draws = np.random.default_rng(20260909).integers(len(unique),size=(5000,len(unique)))
        denominators = sizes[draws].sum(axis=1)
        bootstrap = run['independent']['paired_month_bootstrap']
        assert (bootstrap['months'],bootstrap['replicates'],bootstrap['seed']) == (len(unique),5000,20260909)
        for name,pred in independent.items():
            if name == 'anchor60':
                continue
            difference = abs(pred.astype(float)-y[te])-abs(anchor.astype(float)-y[te])
            distribution = np.bincount(inverse,weights=difference)[draws].sum(axis=1)/denominators
            expected = bootstrap['comparisons'][name]
            np.testing.assert_allclose(difference.mean(),expected['MAE_difference'],rtol=0,atol=2e-5)
            np.testing.assert_allclose(np.quantile(distribution,[.025,.975]),expected['CI95'],rtol=0,atol=2e-5)
    coverage = pd.read_parquet(data/'coverage.parquet')
    assert coverage.hemisphere.value_counts().to_dict() == {'N':36805,'S':4008}
    assert rows.iloc[te].hemisphere.value_counts().to_dict() == {'N':9582,'S':151}
    ids = np.random.default_rng(42).choice(te,2000,replace=False)
    assert rows.iloc[ids].hemisphere.value_counts().to_dict() == {'N':1959,'S':41}
    explanation = 'not recomputed (use --explain)'
    if args.explain:
        print('Recomputing Tree SHAP on 2,000 temporal test crossings',flush=True)
        contrib = m.get_booster().predict(DMatrix(x[ids]),pred_contribs=True)
        np.testing.assert_allclose(contrib.sum(axis=1),m.predict(x[ids]),rtol=0,atol=3e-4)
        shap = np.mean(abs(contrib[:,:-1]),axis=0)
        gain = m.feature_importances_
        for i,c in enumerate(cols):
            np.testing.assert_allclose(shap[i],reference['shap'][c],rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(gain[i],reference['gain'][c],rtol=1e-5,atol=1e-7)
        analysis['shap'] = {cols[i]:float(shap[i]) for i in np.argsort(shap)[::-1]}
        analysis['gain'] = {cols[i]:float(gain[i]) for i in np.argsort(gain)[::-1]}
        for c, expected in reference['pdp'].items():
            j = cols.index(c)
            grid = np.linspace(*np.quantile(x[tr,j],[.02,.98]),25)
            z = x[ids].copy()
            values = []
            for value in grid:
                z[:,j] = value
                values.append(float(m.predict(z).mean()))
            np.testing.assert_allclose(grid,expected['grid'],rtol=0,atol=1e-8)
            np.testing.assert_allclose(values,expected['prediction'],rtol=0,atol=2e-5)
            analysis['pdp'][c] = {'grid':grid.tolist(),'prediction':values}
        explanation = 'Tree SHAP, gain and all 150 PDP grid predictions recomputed and matched'
    report = dict(status='PASS',scope='Processed data, saved predictions and model inference; retraining only if run_dir is set.',
                  required_tasks_verified=['primary','other_targets','baselines','windows','balanced','controls','independent'],
                  prediction_sets_verified=len(checks),
                  compared_run=bool(args.run_dir),snapshot_files_verified=len(manifest['files']),
                  n_features=74,n_train=len(tr),n_test=len(te),metrics=checks,
                  reduction_pct=analysis['reduction_pct'],explanation=explanation,
                  snapshot_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='metrics'},indent=2))


if __name__ == '__main__':
    main()
