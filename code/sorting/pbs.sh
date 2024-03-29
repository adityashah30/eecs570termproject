#!/bin/bash
#PBS -S /bin/bash
#PBS -N DBCuda
#PBS -M aditysha@umich.edu
#PBS -l qos=flux
#PBS -q fluxg
#PBS -A eecs570w16_fluxg
#PBS -l nodes=1:gpus=1,mem=2gb,walltime=01:00:00


# GPGPU job submission script for EECS 570.

# Running batch job under PBS

HOST=`cat $PBS_NODEFILE`
USERDIR=/home/aditysha/570termproject/code/sorting
OUTFILE=$USERDIR/output/$PBS_JOBID.stdout
echo "I'm running on: $HOST" > $OUTFILE
# Launching job to GPU
$USERDIR/sortingtest.cu.out >> $OUTFILE
