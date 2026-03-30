#!/bin/bash
# Submit PBS jobs for all DMSP satellites and years available on CDAWeb
# Anderson 2024 uses F06-F19, but CDAWeb has limited coverage

declare -A SAT_YEARS
SAT_YEARS[F06]="1987"
SAT_YEARS[F07]="1987"
SAT_YEARS[F08]="1987"
SAT_YEARS[F09]="1988"
SAT_YEARS[F12]="2000 2001 2002"
SAT_YEARS[F13]="2000 2001 2002 2003 2004 2005 2006 2007"
SAT_YEARS[F14]="2000 2001 2002 2003 2004 2005"
SAT_YEARS[F15]="2000 2001 2002 2003 2004 2005 2006 2007 2008 2009"
SAT_YEARS[F16]="2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014"
SAT_YEARS[F17]="2009 2010 2011 2012 2013 2014"
SAT_YEARS[F18]="2009 2010 2011 2012 2013 2014"

cd /glade/work/yizhu/cuspML

for SAT in "${!SAT_YEARS[@]}"; do
    for YEAR in ${SAT_YEARS[$SAT]}; do
        echo "Submitting $SAT $YEAR"
        qsub -v SAT=$SAT,YEAR=$YEAR scripts/run_cusp_id.pbs
        sleep 1
    done
done

echo "All jobs submitted."
