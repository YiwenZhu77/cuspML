#!/bin/bash
# Submit PBS jobs to re-run cusp identification with Anderson 2024 criteria
# One job per satellite-year combination

WORKDIR=/glade/work/yizhu/cuspML
LOGDIR=${WORKDIR}/output/anderson/logs
mkdir -p ${LOGDIR}

COUNT=0

submit_job() {
    local SAT=$1
    local YEAR=$2

    cat <<PBSEOF | qsub
#!/bin/bash
#PBS -N cusp_${SAT}_${YEAR}
#PBS -A P28100045
#PBS -q casper
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o ${LOGDIR}/cusp_${SAT}_${YEAR}.log

source /glade/u/home/yizhu/work/miniconda3/etc/profile.d/conda.sh
conda activate py3.10
cd ${WORKDIR}
python3 src/identify_cusp.py --satellite ${SAT} --start ${YEAR}-01-01 --end ${YEAR}-12-31 --output-dir output/anderson
PBSEOF

    COUNT=$((COUNT + 1))
    echo "Submitted: ${SAT} ${YEAR} (job #${COUNT})"
}

# F06: 1987
submit_job F06 1987

# F07: 1987
submit_job F07 1987

# F08: 1987
submit_job F08 1987

# F09: 1988
submit_job F09 1988

# F12: 2000-2002 (2000-12-28 to 2002-07-27)
for YEAR in 2000 2001 2002; do
    submit_job F12 $YEAR
done

# F13: 2000-2007
for YEAR in $(seq 2000 2007); do
    submit_job F13 $YEAR
done

# F14: 2000-2005 (2000-12-28 to 2005-09-29)
for YEAR in $(seq 2000 2005); do
    submit_job F14 $YEAR
done

# F15: 2000-2009 (2000-12-28 to 2009-03-24)
for YEAR in $(seq 2000 2009); do
    submit_job F15 $YEAR
done

# F16: 2003-2014 (2003-10-29 to 2014-12-31)
for YEAR in $(seq 2003 2014); do
    submit_job F16 $YEAR
done

# F17: 2009-2014
for YEAR in $(seq 2009 2014); do
    submit_job F17 $YEAR
done

# F18: 2009-2014 (2009-10-24 to 2014-12-31)
for YEAR in $(seq 2009 2014); do
    submit_job F18 $YEAR
done

echo ""
echo "Total jobs submitted: ${COUNT}"
