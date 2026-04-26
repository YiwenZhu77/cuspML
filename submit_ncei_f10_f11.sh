#!/bin/bash
# Submit PBS jobs for F10 (1990-1997) and F11 (1991-2000, skip 1996)
cd /glade/work/yizhu/cuspML

mkdir -p output/old_newell

# F10: 1990-1997
for YEAR in 1990 1991 1992 1993 1994 1995 1996 1997; do
    qsub -v SAT=F10,YEAR=$YEAR << 'PBSEOF'
#!/bin/bash
#PBS -N cusp_ncei
#PBS -A P28100045
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=04:00:00
#PBS -o /glade/work/yizhu/cuspML/output/cusp_ncei_${SAT}_${YEAR}.log
#PBS -j oe

source /glade/u/home/yizhu/work/miniconda3/etc/profile.d/conda.sh
conda activate py3.10
cd /glade/work/yizhu/cuspML

python3 src/parse_ncei_ssj.py \
    --satellite $SAT \
    --start ${YEAR}-01-01 \
    --end ${YEAR}-12-31 \
    --output-dir output/old_newell
PBSEOF
    echo "Submitted F10 $YEAR"
done

# F11: 1991-2000 (skip 1996 - no data per NCEI)
for YEAR in 1991 1992 1993 1994 1995 1997 1998 1999 2000; do
    qsub -v SAT=F11,YEAR=$YEAR << 'PBSEOF'
#!/bin/bash
#PBS -N cusp_ncei
#PBS -A P28100045
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=04:00:00
#PBS -o /glade/work/yizhu/cuspML/output/cusp_ncei_${SAT}_${YEAR}.log
#PBS -j oe

source /glade/u/home/yizhu/work/miniconda3/etc/profile.d/conda.sh
conda activate py3.10
cd /glade/work/yizhu/cuspML

python3 src/parse_ncei_ssj.py \
    --satellite $SAT \
    --start ${YEAR}-01-01 \
    --end ${YEAR}-12-31 \
    --output-dir output/old_newell
PBSEOF
    echo "Submitted F11 $YEAR"
done

echo "=== All submitted ==="
