#include "../sorting/sorting.h"
#include "../selection/selection.h"
#include "../aggregation/group.h"
#include "../timer/timer.h"
#include <string>
#include <cassert>
#include <limits>
#include <fstream>

using namespace std;

void testAggregation(Dataset& output, Dataset& input, int expCount);
void testSelection(Dataset& output, Dataset& input, int expCount);
void testSorting(Dataset& output, Dataset& input, int expCount);

int main()
{
    Dataset input;
    Dataset output;

    string filename = "../../data/ratings.csv";

    cout << "Loading data" << endl;
    loadData(input, filename);
    cout << "Data loaded" << endl;

    int expCount = 100;

    cout << "Conduction tests on Aggregation..." << endl;
    testAggregation(output, input, expCount);
    cout << "Tests on Aggregation complete..." << endl;
    cout << "Conduction tests on Selection..." << endl;
    testSelection(output, input, expCount);
    cout << "Tests on Selection complete..." << endl;
    cout << "Conduction tests on Sorting..." << endl;
    testSorting(output, input, expCount);
    cout << "Tests on Sorting complete..." << endl;

    return 0;
}

void testAggregation(Dataset& output, Dataset& input, int expCount)
{
    string threadResultFile = "aggregationThreadScalingResults.txt";
    string sizeResultFile = "aggregationSizeScalingResults.txt";

    int group_idx = 1;
    int target_idx = 2;

    int optimalNumThreads = 2;
    long long optimalTime = numeric_limits<long long>::max();

    int minNumThreads = 2;
    int maxNumThreads = 16;

    ofstream out(threadResultFile);
    out << "#NumThreads Time" << endl;

    for(int numThreads = minNumThreads; numThreads <= maxNumThreads; numThreads <<= 1)
    {
        Timer timer;
        long long expTime = 0;
        for(int c = 0; c < expCount; c++)
        {
            timer.startTimer();
            group(output, input, group_idx, target_idx, numThreads);
            timer.stopTimer();
            expTime += timer.getElapsedTime();
        }
        expTime /= expCount;

        if(optimalTime > expTime)
        {
            optimalTime = expTime;
            optimalNumThreads = numThreads;
        }

        cout << "Time to aggregate data on " << numThreads << " threads : " 
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
            group(output, input, group_idx, target_idx, optimalNumThreads);
            timer.stopTimer();
            expTime += timer.getElapsedTime();
        }
        expTime /= expCount;

        cout << "Time to aggregate data on fraction: " << fractions[i] 
             << " using optimal numThread: " << optimalNumThreads
             << " is : " << expTime << endl;
        out << fractions[i] << " " << expTime << endl;
    }
    cout << "Time to aggregate data on fraction: 1.0"
         << " using optimal numThread: " << optimalNumThreads
         << " is : " << optimalTime << endl;
    out << "1.0 " << optimalTime << endl;

    out.close();
}

void testSelection(Dataset& output, Dataset& input, int expCount)
{
    string threadResultFile = "selectionThreadScalingResults.txt";
    string sizeResultFile = "selectionSizeScalingResults.txt";

    int index = 2;
    double rating = 3.5;
    Field cons = rating;

    int optimalNumThreads = 2;
    long long optimalTime = numeric_limits<long long>::max();

    int minNumThreads = 2;
    int maxNumThreads = 16;

    ofstream out(threadResultFile);
    out << "#NumThreads Time" << endl;

    for(int numThreads = minNumThreads; numThreads <= maxNumThreads; numThreads <<= 1)
    {
        Timer timer;
        long long expTime = 0;
        for(int c = 0; c < expCount; c++)
        {
            timer.startTimer();
            selData(output, input, index, cons, numThreads);
            timer.stopTimer();
            expTime += timer.getElapsedTime();
        }
        expTime /= expCount;

        if(optimalTime > expTime)
        {
            optimalTime = expTime;
            optimalNumThreads = numThreads;
        }

        cout << "Time to select data on " << numThreads << " threads : " 
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
            selData(output, smallDS, index, cons, optimalNumThreads);
            timer.stopTimer();
            expTime += timer.getElapsedTime();
        }
        expTime /= expCount;

        cout << "Time to select data on fraction: " << fractions[i] 
             << " using optimal numThread: " << optimalNumThreads
             << " is : " << expTime << endl;
        out << fractions[i] << " " << expTime << endl;
    }
    cout << "Time to select data on fraction: 1.0"
         << " using optimal numThread: " << optimalNumThreads
         << " is : " << optimalTime << endl;
    out << "1.0 " << optimalTime << endl;

    out.close();
}

void testSorting(Dataset& output, Dataset& input, int expCount)
{
    string threadResultFile = "sortingThreadScalingResults.txt";
    string sizeResultFile = "sortingSizeScalingResults.txt";

    int fieldIdx = 2;

    int optimalNumThreads = 2;
    long long optimalTime = numeric_limits<long long>::max();

    int minNumThreads = 2;
    int maxNumThreads = 16;

    ofstream out(threadResultFile);
    out << "#NumThreads Time" << endl;

    for(int numThreads = minNumThreads; numThreads <= maxNumThreads; numThreads <<= 1)
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
    cout << "Time to sort data on fraction: 1.0"
         << " using optimal numThread: " << optimalNumThreads
         << " is : " << optimalTime << endl;
    out << "1.0 " << optimalTime << endl;

    out.close();
}
