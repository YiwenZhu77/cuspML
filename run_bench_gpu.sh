#!/bin/bash
#PBS -N bench_gpu
#PBS -A P28100045
#PBS -q casper
#PBS -l select=1:ncpus=8:ngpus=1:mem=32GB:gpu_type=v100
#PBS -l walltime=00:30:00
#PBS -o /glade/work/yizhu/cuspML/output/bench_gpu.log
#PBS -j oe
source /glade/u/home/yizhu/work/miniconda3/etc/profile.d/conda.sh
conda activate py3.10
cd /glade/work/yizhu/cuspML
python3 bench_gpu.py
