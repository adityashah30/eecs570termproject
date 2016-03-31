#!/bin/bash
#PBS -S /bin/bash
#PBS -N beamform
#PBS -l nodes=Typhon2:ppn=1:gpus=1
#PBS -l walltime=10:00:00
#PBS -q test

if [[ $PBS_JOBID == "" ]] ; then
  # Not running under PBS
  USE_LOCAL_MIC_NUMBER=4
  echo "Running interactive job. Use 'qsub micTest.sh' to submit a batch job."
  micnativeloadex scalingtests.mic.out
else
  # Running batch job under PBS
  HOST=`cat $PBS_NODEFILE`
  MICNUM=$(cat $PBS_GPUFILE | cut -c12-)
  USERDIR=/n/typhon/home/$USER/termproject/code/tests
  OUTFILE=$USERDIR/output/$PBS_JOBID.stdout
  echo "I'm running on: $HOST mic$MICNUM" > $OUTFILE
  # Launching job to MIC
  # timeout set to 5 minutes
  micnativeloadex $USERDIR/scalingtests.mic.out >> $OUTFILE
fi

