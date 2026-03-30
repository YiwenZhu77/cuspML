#!/bin/bash
#PBS -N cusp_ncei_new
#PBS -A P28100045
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=16GB
#PBS -l walltime=04:00:00
#PBS -o /glade/work/yizhu/cuspML/output/cusp_ncei_new.log
#PBS -j oe
#PBS -J 0-29

source /glade/u/home/yizhu/work/miniconda3/etc/profile.d/conda.sh
conda activate py3.10
cd /glade/work/yizhu/cuspML

JOBS=(F16,2015 F16,2016 F16,2017 F16,2018 F16,2019 F16,2020 F16,2021 F16,2022 F16,2023 F16,2024 F17,2015 F17,2016 F17,2017 F17,2018 F17,2019 F17,2020 F17,2021 F17,2022 F17,2023 F17,2024 F18,2015 F18,2016 F18,2017 F18,2018 F18,2019 F18,2020 F18,2021 F18,2022 F18,2023 F18,2024)
IFS=',' read SAT YEAR <<< "${JOBS[$PBS_ARRAY_INDEX]}"

python3 src/parse_ncei_ssj.py --satellite $SAT --start $YEAR-01-01 --end $YEAR-12-31 \
    --output-dir output/old_newell
