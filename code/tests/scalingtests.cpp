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

    int expCount = 10;

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
    string threadResultFile = "/home/micuser/aggregationThreadScalingResults.mic.txt";
    string sizeResultFile = "/home/micuser/aggregationSizeScalingResults.mic.txt";
#else
    string threadResultFile = "aggregationThreadScalingResults.txt";
    string sizeResultFile = "aggregationSizeScalingResults.txt";
#endif

    int group_idx = 1;
    int target_idx = 2;

    int optimalNumThreads = 2;
    long long optimalTime = numeric_limits<long long>::max();

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

        if(numThreads == 1)
        {
            singleThreadTime = expTime;
        }

        long long idealTime = singleThreadTime/numThreads;

        if(optimalTime > expTime)
        {
            optimalTime = expTime;
            optimalNumThreads = numThreads;
        }

        cout << "Time to aggregate data on " << numThreads << " threads : " 
             << expTime << "; Ideal Time: " << idealTime << endl;
        out << numThreads << " " << expTime << " " << idealTime << endl; 
    }

    out.close();

    double fractions[] = {0.25, 0.5, 0.75};
    int numFractions = sizeof(fractions)/sizeof(double);

    Dataset smallDS;

    out.open(sizeResultFile);
    out << "#Fraction Time (Optimal numThreads: " << optimalNumThreads << ")" << endl;

    for(int i=0; i<numFractions; i++)
    {
    #ifdef __INTEL_COMPILER
        extractSmallDS(smallDS, expandedDS, fractions[i]);
    #else
        extractSmallDS(smallDS, input, fractions[i]);
    #endif

        Timer timer;
        long long expTime = 0;
        for(int c = 0; c < expCount; c++)
        {
            timer.startTimer();
            group(output, smallDS, optimalNumThreads);
            timer.stopTimer();
            expTime += timer.getElapsedTime();
        }
        expTime /= expCount;

        long long idealTime = fractions[i]*optimalTime;

        cout << "Time to aggregate data on fraction: " << fractions[i] 
             << " using optimal numThread: " << optimalNumThreads
             << " is : " << expTime << "; Ideal Time: " << idealTime << endl;
        out << fractions[i] << " " << expTime << " " << idealTime << endl;
    }
    cout << "Time to aggregate data on fraction: 1.0"
         << " using optimal numThread: " << optimalNumThreads
         << " is : " << optimalTime << "; Ideal Time: " << optimalTime << endl;
    out << "1.0 " << optimalTime << " " << optimalTime << endl;

    out.close();
}

void testSelection(Dataset& output, Dataset& input, int expCount)
{

#ifdef __MIC__
    string threadResultFile = "/home/micuser/selectionThreadScalingResults.mic.txt";
    string sizeResultFile = "/home/micuser/selectionSizeScalingResults.mic.txt";
#else
    string threadResultFile = "selectionThreadScalingResults.txt";
    string sizeResultFile = "selectionSizeScalingResults.txt";
#endif

    double cons = 3.5;

    int optimalNumThreads = 2;
    long long optimalTime = numeric_limits<long long>::max();

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

        if(numThreads == 1)
        {
            singleThreadTime = expTime;
        }

        long long idealTime = singleThreadTime/numThreads;

        if(optimalTime > expTime)
        {
            optimalTime = expTime;
            optimalNumThreads = numThreads;
        }

        cout << "Time to select data on " << numThreads << " threads : " 
             << expTime << "; Ideal Time: " << idealTime << endl;
        out << numThreads << " " << expTime << " " << idealTime << endl;
    }

    out.close();

    double fractions[] = {0.25, 0.5, 0.75};
    int numFractions = sizeof(fractions)/sizeof(double);

    Dataset smallDS;

    out.open(sizeResultFile);
    out << "#Fraction Time (Optimal numThreads: " << optimalNumThreads << ")" << endl;

    for(int i=0; i<numFractions; i++)
    {
    #ifdef __INTEL_COMPILER
        extractSmallDS(smallDS, expandedDS, fractions[i]);
    #else
        extractSmallDS(smallDS, input, fractions[i]);
    #endif

        Timer timer;
        long long expTime = 0;
        for(int c = 0; c < expCount; c++)
        {
            timer.startTimer();
            selData(output, smallDS, cons, optimalNumThreads);
            timer.stopTimer();
            expTime += timer.getElapsedTime();
        }
        expTime /= expCount;

        long long idealTime = fractions[i]*optimalTime;

        cout << "Time to select data on fraction: " << fractions[i] 
             << " using optimal numThread: " << optimalNumThreads
             << " is : " << expTime << "; Ideal Time: " << idealTime << endl;
        out << fractions[i] << " " << expTime << " " << idealTime << endl;
    }
    cout << "Time to select data on fraction: 1.0"
         << " using optimal numThread: " << optimalNumThreads
         << " is : " << optimalTime << "; Ideal Time: " << optimalTime << endl;
    out << "1.0 " << optimalTime << " " << optimalTime << endl;

    out.close();
}

void testSorting(Dataset& output, Dataset& input, int expCount)
{

#ifdef __MIC__
    string threadResultFile = "/home/micuser/sortingThreadScalingResults.mic.txt";
    string sizeResultFile = "/home/micuser/sortingSizeScalingResults.mic.txt";
#else
    string threadResultFile = "sortingThreadScalingResults.txt";
    string sizeResultFile = "sortingSizeScalingResults.txt";
#endif

    Dataset powerOf2DS;

    cout << "Coonverting to nearest power of 2" << endl; 
    nearestPowerOf2DS(powerOf2DS, input);
    cout << "Conversion complete" << endl;

    int fieldIdx = 2;

    int optimalNumThreads = 2;
    long long optimalTime = numeric_limits<long long>::max();

    long long singleThreadTime = 0;

    int minNumThreads = 1;
#ifdef __MIC__
    int maxNumThreads = 256;
#else
    int maxNumThreads = 32;
#endif

    ofstream out(threadResultFile);
    out << "#NumThreads Time" << endl;

    ///////////////////Reduce Dataset Size for Sorting//////
    Dataset reducedDS;
    extractSmallDS(reducedDS, powerOf2DS, 0.5);
    ////////////////////////////////////////////////////////

    for(int numThreads = minNumThreads; numThreads <= maxNumThreads; numThreads <<= 1)
    {
        Timer timer;
        long long expTime = 0;
        for(int c = 0; c < expCount; c++)
        {
            timer.startTimer();	
            // sortData(output, powerOf2DS, fieldIdx, numThreads);
            sortData(output, reducedDS, fieldIdx, numThreads);
            timer.stopTimer();
            expTime += timer.getElapsedTime();
        }
        expTime /= expCount;

        if(numThreads == 1)
        {
            singleThreadTime = expTime;
        }

        long long idealTime = singleThreadTime/numThreads;

        if(optimalTime > expTime)
        {
            optimalTime = expTime;
            optimalNumThreads = numThreads;
        }

        cout << "Time to sort data on " << numThreads << " threads : " 
             << expTime << "; Ideal Time: " << idealTime << endl;
        out << numThreads << " " << expTime << " " << idealTime << endl;
    }

    out.close();

    double fractions[] = {0.25, 0.5, 0.75};
    int numFractions = sizeof(fractions)/sizeof(double);

    Dataset smallDS;

    out.open(sizeResultFile);
    out << "#Fraction Time (Optimal numThreads: " << optimalNumThreads << ")" << endl;

    for(int i=0; i<numFractions; i++)
    { 
        extractSmallDS(smallDS, reducedDS, fractions[i]);
	//tractSmallDS(smallDS, powerOf2DS, fractions[i]);

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

        long long idealTime = fractions[i]*optimalTime;

        cout << "Time to sort data on fraction: " << fractions[i] 
             << " using optimal numThread: " << optimalNumThreads
             << " is : " << expTime << "; Ideal Time: " << idealTime << endl;
        out << fractions[i] << " " << expTime << " " << idealTime << endl;
    }
    cout << "Time to sort data on fraction: 1.0"
         << " using optimal numThread: " << optimalNumThreads
         << " is : " << optimalTime << "; Ideal Time: " << optimalTime << endl;
    out << "1.0 " << optimalTime << " " << optimalTime << endl;

    out.close();
}
