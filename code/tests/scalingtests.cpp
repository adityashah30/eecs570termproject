#include "../sorting/sorting.h"
#include "../timer/timer.h"
#include <string>
#include <cassert>
#include <limits>
#include <fstream>

using namespace std;

void testSorting(Dataset& output, Dataset& input, int expCount);

int main()
{
    Dataset input;
    Dataset output;

    string filename = "../../data/ratings.csv";

    cout << "Loading data" << endl;
    loadData(input, filename);
    cout << "Data loaded" << endl;

    int expCount = 1;

    testSorting(output, input, expCount);

    return 0;
}

void testSorting(Dataset& output, Dataset& input, int expCount)
{
    string threadResultFile = "sortingThreadScalingResults.txt";
    string sizeResultFile = "sortingSizeScalingResults.txt";

    int fieldIdx = 2;

    int optimalNumThreads = 2;
    long long optimalTime = numeric_limits<long long>::max();

    ofstream out(threadResultFile);
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

        if(optimalTime > expTime)
        {
            optimalTime = expTime;
            optimalNumThreads = numThreads;
        }

        cout << "Time to sort data on " << numThreads << " threads : " 
             << expTime << endl;
        out << numThreads << " " << expTime << endl;
    }

    out.close();

    double fractions[] = {0.25, 0.5, 0.75};
    int numFractions = sizeof(fractions)/sizeof(double);

    Dataset smallDS;

    out.open(sizeResultFile);
    out << "#Fraction Time (Optimal numThreads: " << optimalNumThreads << ")" << endl;

    for(int i=0; i<numFractions; i++)
    {
        extractSmallDS(smallDS, input, fractions[i]);

        Timer timer;
        long long expTime = 0;
        for(int c = 0; c < expCount; c++)
        {
            timer.startTimer();
            sortData(output, smallDS, fieldIdx, optimalNumThreads);
            timer.stopTimer();
            expTime += timer.getElapsedTime();
        }
        expTime /= expCount;

        cout << "Time to sort data fraction: " << fractions[i] 
             << " using optimal numThread: " << optimalNumThreads
             << " is : " << expTime << endl;
        out << fractions[i] << " " << expTime << endl;
    }
    cout << "Time to sort data fraction: 1.0" << 
             << " using optimal numThread: " << optimalNumThreads
             << " is : " << optimalTime << endl;
    out << "1.0 " << optimalTime << endl;

    out.close();
}
