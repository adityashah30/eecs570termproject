make scalingtests.out
./scalingtests.out
gnuplot plotFile.txt
make scalingtests.mic.out
./micTest.sh
./copyData.sh < pass.txt
gnuplot plotFileMic.txt
