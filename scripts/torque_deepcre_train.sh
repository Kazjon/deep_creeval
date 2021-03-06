#!/bin/sh
#
# ======= PBS OPTIONS ======= (user input required)
#
### Specify queue to run
#PBS -q python
### Set the job name
#PBS -N deep_creeval
### Specify the # of cpus for your job.
#PBS -l nodes=1:ppn=4:gpus=1
### Adjust walltime below (default walltime = 7 days, or 168 hours)
### if you require > 7 days, INCREASE to estimated # hours needed
### if you DON'T require 7 days DECREASE to estimated # hours needed
### (hint: jobs with smaller walltime value tend to run sooner)
#PBS -l walltime=168:00:00
### pass the full environment
#PBS -V
# send PBS output to /dev/null  (we redirect it below)
#PBS -o /dev/null
#PBS -e /dev/null
#
# ===== END PBS OPTIONS =====

# ======= APP OPTIONS ======= (user input required)
#
### specify additional options
OPTS=""
# ===== END APP OPTIONS =====

### Get the short $PBS_JOBID
SHORT_JOBID=`echo $PBS_JOBID |cut -d. -f1`

### Use this to redirect STDOUT and STDERR to working dir
exec 1>$PBS_O_WORKDIR/$PBS_JOBNAME-$SHORT_JOBID.out  2>$PBS_O_WORKDIR/$PBS_JOBNAME-$SHORT_JOBID.err

### run job
./startmongo.sh
cd deep_creeval
python deep_creeval.py ebird_top10_2008_2012 -m train_exp --epochs 5 --sample_limit 0 --exp_name exp_5yr_monthly_5epochs  --pretrain_start 2008 --pretrain_stop 2009 --train_stop 2013 --time_slice 0.0833
kill -9 `pidof mongod`
