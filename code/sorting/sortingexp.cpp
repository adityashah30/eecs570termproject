#include "sorting.h"
#include "../timer/timer.h"
#include <string>
#include <cassert>
#include <fstream>

using namespace std;

int main()
{
    Dataset input;
    Dataset output;

    string filename = "../../data/ratings.csv";
    string resultFile = "sortingScalingResults.txt";

    int fieldIdx = 2;
    int expCount = 100;

    cout << "Loading data" << endl;
    loadData(input, filename);
    cout << "Data loaded" << endl;

    ofstream out(resultFile);
    out << "#NumThreads Time" << endl;

    for(int numThreads = 2; numThreads <= 16; numThreads <<= 1)
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
