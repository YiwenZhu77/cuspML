#!/bin/bash
#PBS -N tree_dse
#PBS -A P28100045
#PBS -q casper
#PBS -l select=1:ncpus=16:mem=64gb:ngpus=1:gpu_type=v100
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o /glade/work/yizhu/cuspML/output/tree_dse.log

source /glade/u/home/yizhu/work/miniconda3/etc/profile.d/conda.sh
conda activate py3.10
cd /glade/work/yizhu/cuspML
export OMNI_DIR=output/omni_hist
export USE_GPU=1
export N_JOBS=16
python3 src/tree_dse.py
