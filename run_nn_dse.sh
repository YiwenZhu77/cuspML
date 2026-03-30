#!/bin/bash
#PBS -N nn_dse
#PBS -A P28100045
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=32GB:ngpus=1:gpu_type=v100
#PBS -l walltime=02:00:00
#PBS -o output/nn_dse_quiet.log
#PBS -j oe

source /glade/u/home/yizhu/work/miniconda3/etc/profile.d/conda.sh
conda activate py3.10

cd /glade/work/yizhu/cuspML
python3 src/nn_dse.py
