#!/bin/bash
#PBS -S /bin/bash
#PBS -N DBCuda
#PBS -M xfkong@umich.edu
#PBS -l qos=flux
#PBS -q flux
#PBS -l nodes=1:gpus=1,mem=4gb,walltime=01:00:00


# GPGPU job submission script for EECS 570.

# Running batch job under PBS


  HOST=`cat $PBS_NODEFILE`
  USERDIR=/home/$USER/eecs570termproject/code/selection
  OUTFILE=$USERDIR/$PBS_JOBID.stdout
  echo "I'm running on: $HOST" > $OUTFILE
  # Launching job to GPU
  ./selectiontest.cu.out >> $OUTFILE



