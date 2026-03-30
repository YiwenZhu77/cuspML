#!/bin/bash
#PBS -N omni_full
#PBS -A P28100045
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o /glade/work/yizhu/cuspML/output/omni_full_hist.log

source /glade/u/home/yizhu/work/miniconda3/etc/profile.d/conda.sh
conda activate py3.10
cd /glade/work/yizhu/cuspML

# No AE filter (ae-filter=99999 effectively disables it)
python3 src/add_omni_batch.py \
    --input-dir output/old_newell \
    --output-dir output/omni_full_hist \
    --ae-filter 99999 \
    --threads 4
