#!/bin/bash
#PBS -N omni_batch
#PBS -A P28100045
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=01:00:00
#PBS -o /glade/work/yizhu/cuspML/output/omni_batch.log
#PBS -e /glade/work/yizhu/cuspML/output/omni_batch.log
#PBS -j oe

source /glade/u/home/yizhu/work/miniconda3/etc/profile.d/conda.sh
conda activate py3.10

cd /glade/work/yizhu/cuspML

python3 src/add_omni_batch.py \
    --input-dir output/old_newell \
    --output-dir output/omni_all \
    --ae-filter 99999 \
    --threads 4
