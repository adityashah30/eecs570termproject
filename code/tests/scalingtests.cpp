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

    cout << "Loading data" << endl;
    loadData(input);
    cout << "Data loaded" << endl;

    int expCount = 20;

    cout << "Conducting tests on Aggregation..." << endl;
    testAggregation(output, input, expCount);
    cout << "Tests on Aggregation complete..." << endl;
    cout << "Conducting tests on Selection..." << endl;
    testSelection(output, input, expCount);
    cout << "Tests on Selection complete..." << endl;
    cout << "Conducting tests on Sorting..." << endl;
    testSorting(output, input, expCount);
    cout << "Tests on Sorting complete..." << endl;

    return 0;
}

void testAggregation(Dataset& output, Dataset& input, int expCount)
{
#ifdef __MIC__
    string threadResultFile = "/home/micuser/aggregation.mic.txt";
#else
    string threadResultFile = "cpuresults/aggregation.txt";
#endif

    int group_idx = 1;
    int target_idx = 2;

    long long singleThreadTime = 0;

    int minNumThreads = 1;
#ifdef __MIC__
    int maxNumThreads = 256;
#else
    int maxNumThreads = 32;
#endif

#ifdef __INTEL_COMPILER
    double fraction = 5;
    Dataset expandedDS;
    duplicateDS(expandedDS, input, fraction);
#endif

    ofstream out(threadResultFile);
    out << "#NumThreads Time" << endl;

    for(int numThreads = minNumThreads; numThreads <= maxNumThreads; numThreads <<= 1)
    {
        Timer timer;
        long long expTime = 0;
        for(int c = 0; c < expCount; c++)
        {
            timer.startTimer();
        #ifdef __INTEL_COMPILER
            group(output, expandedDS, numThreads);
        #else
            group(output, input, numThreads);
        #endif
            timer.stopTimer();
            expTime += timer.getElapsedTime();
        }
        expTime /= expCount;

        if(numThreads == minNumThreads)
        {
            singleThreadTime = expTime;
        }

        double speedup = (double)singleThreadTime/expTime;

        cout << "Time to aggregate data on " << numThreads << " threads : " 
             << expTime << "; Speedup: " << speedup << endl;
        out << numThreads << " " << expTime << " " << speedup << endl; 
    }

    out.close();
}

void testSelection(Dataset& output, Dataset& input, int expCount)
{

#ifdef __MIC__
    string threadResultFile = "/home/micuser/selection.mic.txt";
#else
    string threadResultFile = "cpuresults/selection.txt";
#endif

    double cons = 3.5;

    long long singleThreadTime = 0;

    int minNumThreads = 1;
#ifdef __MIC__
    int maxNumThreads = 256;
#else
    int maxNumThreads = 32;
#endif

#ifdef __INTEL_COMPILER
    double fraction = 5;
    Dataset expandedDS;
    duplicateDS(expandedDS, input, fraction);
#endif

    ofstream out(threadResultFile);
    out << "#NumThreads Time" << endl;

    for(int numThreads = minNumThreads; numThreads <= maxNumThreads; numThreads <<= 1)
    {
        Timer timer;
        long long expTime = 0;
        for(int c = 0; c < expCount; c++)
        {
            timer.startTimer();
        #ifdef __INTEL_COMPILER
            selData(output, expandedDS, cons, numThreads);
        #else
            selData(output, input, cons, numThreads);
        #endif
            timer.stopTimer();
            expTime += timer.getElapsedTime();
        }
        expTime /= expCount;

        if(numThreads == minNumThreads)
        {
            singleThreadTime = expTime;
        }

        double speedup = (double)singleThreadTime/expTime;

        cout << "Time to select data on " << numThreads << " threads : " 
             << expTime << "; Speedup: " << speedup << endl;
        out << numThreads << " " << expTime << " " << speedup << endl; 
    }

    out.close();
}

void testSorting(Dataset& output, Dataset& input, int expCount)
{

#ifdef __MIC__
    string threadResultFile = "/home/micuser/sorting.mic.txt";
#else
    string threadResultFile = "cpuresults/sorting.txt";
#endif

    Dataset powerOf2DS;

    cout << "Coonverting to nearest power of 2" << endl; 
    nearestPowerOf2DS(powerOf2DS, input);
    cout << "Conversion complete" << endl;

    int fieldIdx = 2;

    long long singleThreadTime = 0;

    int minNumThreads = 1;
#ifdef __MIC__
    int maxNumThreads = 256;
#else
    int maxNumThreads = 32;
#endif

    ofstream out(threadResultFile);
    out << "#NumThreads Time" << endl;

    for(int numThreads = minNumThreads; numThreads <= maxNumThreads; numThreads <<= 1)
    {
        Timer timer;
        long long expTime = 0;
        for(int c = 0; c < expCount; c++)
        {
            timer.startTimer();	
            sortData(output, powerOf2DS, fieldIdx, numThreads);
            timer.stopTimer();
            expTime += timer.getElapsedTime();
        }
        expTime /= expCount;

        if(numThreads == minNumThreads)
        {
            singleThreadTime = expTime;
        }

        double speedup = (double)singleThreadTime/expTime;

        cout << "Time to sort data on " << numThreads << " threads : " 
             << expTime << "; Speedup: " << speedup << endl;
        out << numThreads << " " << expTime << " " << speedup << endl;
    }

    out.close();
}
