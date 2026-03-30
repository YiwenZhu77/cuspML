#!/bin/bash
#PBS -N nn_dse_hist
#PBS -A P28100045
#PBS -q casper
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=v100
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /glade/work/yizhu/cuspML/output/nn_dse_hist.log

source /glade/u/home/yizhu/work/miniconda3/etc/profile.d/conda.sh
conda activate py3.10
cd /glade/work/yizhu/cuspML
export OMNI_DIR=output/omni_hist
python3 src/nn_dse.py
