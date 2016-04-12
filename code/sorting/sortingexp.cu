#include "sorting.cuh"
#include "../timer/timer.h"
#include <string>
#include <cassert>
#include <fstream>

using namespace std;

int main()
{
    Dataset originalInput;
    Dataset input;
    Dataset output;

    string resultFile = "sortingScalingGPUResults.txt";

    int fieldIdx = 2;
    int expCount = 1;

    cout << "Loading data" << endl;
    loadData(originalInput);
    cout << "Data loaded" << endl;

    cout << "Converting to nearest power of 2" << endl;
    nearestPowerOf2DS(input, originalInput);
    cout << "Conversion complete" << endl;

    ofstream out(resultFile);
    out << "#NumThreads Time" << endl;

    for(int numThreads = 32; numThreads <= 1024; numThreads <<= 1)
    {
        Timer timer;
        long long expTime = 0;
        for(int c = 0; c < expCount; c++)
        {
            timer.startTimer();
            sortData(output, input, fieldIdx, numThreads);
            timer.stopTimer();
            expTime += timer.getElapsedTime();
        }
        expTime /= expCount;
        cout << "Time to sort data on " << numThreads << " threads : " 
             << expTime << endl;
        out << numThreads << " " << expTime << endl;
    }

    out.close();

    return 0;
}
