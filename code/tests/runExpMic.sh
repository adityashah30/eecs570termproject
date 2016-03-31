make scalingtests.out
./scalingtests.out
gnuplot plotFile.txt
make scalingtests.mic.out
qsub micTest.sh
gnuplot plotFileMic.txt
