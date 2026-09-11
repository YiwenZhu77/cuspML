"""Current manuscript entry point: verify temporal results and render all ten figures.

Run from any directory. --train additionally refits every reported model/control.
Historical experiments remain in repository history; this entry point uses the current temporal snapshot.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=here/'current/data')
    ap.add_argument('--output-dir', type=Path, default=here/'results')
    ap.add_argument('--train', action='store_true', help='Train all models and controls from scratch before verification')
    ap.add_argument('--jobs', type=int, default=6, choices=range(1,7))
    args = ap.parse_args()
    if args.train and args.jobs != 6:
        ap.error('Exact full reproduction requires --jobs 6 for the original MLP BLAS setting')
    data, out = args.data_dir.resolve(), args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose a fresh --output-dir: {out}')
    out.mkdir(parents=True,exist_ok=True)
    current = here/'current'
    def run(script,*extra):
        subprocess.run([sys.executable,str(current/script),*map(str,extra)],check=True)
    if args.train:
        run('train.py','--data-dir',data,'--output-dir',out/'trained','--tasks','all','--jobs',args.jobs)
    extra = ['--run-dir',out/'trained'] if args.train else []
    run('verify.py','--data-dir',data,'--output-dir',out/'checked','--explain',*extra)
    render_data = out/'render-inputs'
    render_data.mkdir()
    for name in ['rows.parquet','coverage.parquet']:
        shutil.copy2(data/name,render_data/name)
    shutil.copy2(out/'checked/analysis.json',render_data/'analysis.json')
    shutil.copytree((out/'trained' if args.train else data)/'predictions',render_data/'predictions')
    run('figures.py','--data-dir',render_data,'--output-dir',out/'figures')
    print(f'Completed: {out / "checked/verification.json"}; all ten figures: {out / "figures"}')


if __name__ == '__main__':
    main()
