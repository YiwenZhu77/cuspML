#!/bin/bash
#PBS -N add_omni
#PBS -A P28100045
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=04:00:00
#PBS -o /glade/work/yizhu/cuspML/output/add_omni.log
#PBS -e /glade/work/yizhu/cuspML/output/add_omni.err
#PBS -j oe

source /glade/u/home/yizhu/work/miniconda3/etc/profile.d/conda.sh
conda activate py3.10

cd /glade/work/yizhu/cuspML

# Process each JSON file: match with OMNI, save to omni/ subdir
mkdir -p output/omni

for f in output/old_newell/cusp_crossings_*.json; do
    base=$(basename "$f")
    outfile="output/omni/${base}"

    # Skip if already processed
    if [ -f "$outfile" ]; then
        echo "SKIP: $outfile already exists"
        continue
    fi

    echo "Processing: $f -> $outfile"
    python3 src/add_omni.py --input "$f" --output "$outfile" --ae-filter 100
done

echo "=== DONE ==="

# Merge all filtered results and run comparison
python3 -c "
import json, glob, os
import numpy as np

files = sorted(glob.glob('output/omni/cusp_crossings_*.json'))
all_c = []
for f in files:
    with open(f) as fh:
        all_c.extend(json.load(fh))

print(f'Total crossings after AE<100 filter: {len(all_c):,}')

# Save merged
with open('output/omni/cusp_crossings_all_filtered.json', 'w') as fh:
    json.dump(all_c, fh, indent=2, default=str)

# IMF available filter (Anderson also requires IMF data)
imf_valid = [c for c in all_c if c.get('imf_by') is not None and c.get('imf_bz') is not None]
print(f'After IMF available filter: {len(imf_valid):,}')

with open('output/omni/cusp_crossings_all_anderson_filter.json', 'w') as fh:
    json.dump(imf_valid, fh, indent=2, default=str)

# Quick stats
if imf_valid:
    eq = np.array([c['eq_mlat'] for c in imf_valid])
    mlt = np.array([c['mean_mlt'] for c in imf_valid])
    tilt = np.array([c.get('dipole_tilt', np.nan) for c in imf_valid], dtype=float)
    hemi = np.array([c['hemisphere'] for c in imf_valid])

    abs_eq = np.abs(eq)
    print(f'Mean |MLAT|: {np.nanmean(abs_eq):.2f} +/- {np.nanstd(abs_eq):.2f}')
    print(f'Mean MLT: {np.nanmean(mlt):.2f} +/- {np.nanstd(mlt):.2f}')
    print(f'N hemisphere: {(hemi==\"N\").sum()}, S hemisphere: {(hemi==\"S\").sum()}')

    # Tilt slope - North only
    north = hemi == 'N'
    valid = north & ~np.isnan(tilt) & ~np.isnan(abs_eq)
    if valid.sum() > 50:
        coeffs = np.polyfit(tilt[valid], abs_eq[valid], 1)
        r = np.corrcoef(tilt[valid], abs_eq[valid])[0,1]
        print(f'North tilt slope: {coeffs[0]:.4f} deg/deg (Anderson: 0.043-0.051)')
        print(f'North tilt r: {r:.3f}')

    # By satellite
    from collections import Counter
    sat_count = Counter(c['satellite'] for c in imf_valid)
    print('Per satellite:')
    for s, n in sorted(sat_count.items()):
        print(f'  {s}: {n}')
"
