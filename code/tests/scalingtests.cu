#include "../sorting/sorting.cuh"
#include "../selection/selection.cuh"
#include "../aggregation/group.cuh"
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
    string threadResultFile = "/home/aditysha/570termproject/code/tests/gpuresults/aggregation.cu.txt";

    int id_num = 0;

    cout << "Preprocessing Data" << endl;
    group_preprocessing(input, id_num);
    cout << "Preprocessing complete" << endl;

    long long singleThreadTime = 0;

    int minNumThreads = 1;
    int maxNumThreads = 8192;

    double fraction = 5;
    Dataset expandedDS;
    duplicateDS(expandedDS, input, fraction);

    ofstream out(threadResultFile);
    out << "#NumThreads Time" << endl;

    for(int numThreads = minNumThreads; numThreads <= maxNumThreads; numThreads <<= 1)
    {
        Timer timer;
        long long expTime = 0;
        for(int c = 0; c < expCount; c++)
        {
            timer.startTimer();
            group(output, expandedDS, numThreads, id_num);
            timer.stopTimer();
            expTime += timer.getElapsedTime();
        }
        expTime /= expCount;
        double speedup = (double)singleThreadTime/expTime;

        if(numThreads == 1)
        {
            singleThreadTime = expTime;
            numThreads = 16;
        }

        cout << "Time to aggregate data on " << numThreads << " threads : " 
             << expTime << "; Speedup: " << speedup << endl;
        out << numThreads << " " << expTime << " " << speedup << endl; 
    }

    out.close();
}

void testSelection(Dataset& output, Dataset& input, int expCount)
{

    string threadResultFile = "/home/aditysha/570termproject/code/tests/gpuresults/selection.cu.txt";

    double constraint = 3.5;

    long long singleThreadTime = 0;

    int minNumThreads = 1;
    int maxNumThreads = 8192;

    double fraction = 5;
    Dataset expandedDS;
    duplicateDS(expandedDS, input, fraction);

    ofstream out(threadResultFile);
    out << "#NumThreads Time" << endl;

    for(int numThreads = minNumThreads; numThreads <= maxNumThreads; numThreads <<= 1)
    {
        Timer timer;
        long long expTime = 0;
        for(int c = 0; c < expCount; c++)
        {
            timer.startTimer();
            selData(output, expandedDS, constraint, numThreads);
            timer.stopTimer();
            expTime += timer.getElapsedTime();
        }
        expTime /= expCount;
        double speedup = (double)singleThreadTime/expTime;

        if(numThreads == 1)
        {
            singleThreadTime = expTime;
            numThreads = 16;
        }

        cout << "Time to select data on " << numThreads << " threads : " 
             << expTime << "; Speedup: " << speedup << endl;
        out << numThreads << " " << expTime << " " << speedup << endl; 
    }

    out.close();
}

void testSorting(Dataset& output, Dataset& input, int expCount)
{

    string threadResultFile = "/home/aditysha/570termproject/code/tests/gpuresults/sorting.cu.txt";

    int fieldIdx = 2;

    long long singleThreadTime = 0;

    int minNumThreads = 1;
    int maxNumThreads = 8192;

    Dataset powerOf2DS;

    cout << "Coonverting to nearest power of 2" << endl; 
    nearestPowerOf2DS(powerOf2DS, input);
    cout << "Conversion complete" << endl;

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
        double speedup = (double)singleThreadTime/expTime;

        if(numThreads == 1)
        {
            singleThreadTime = expTime;
            numThreads = 16;
        }

        cout << "Time to sort data on " << numThreads << " threads : " 
             << expTime << "; Speedup: " << speedup << endl;
        out << numThreads << " " << expTime << " " << speedup << endl; 
    }

    out.close();
}
