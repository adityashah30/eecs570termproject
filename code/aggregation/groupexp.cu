#include "group.cuh"
#include "../timer/timer.h"
#include <string>
#include <cassert>
#include <algorithm>

using namespace std;

//void populateData(Dataset& input, Dataset& expectedOutput);

//static bool compare(const Record& first, const Record& second){
//	return first.movieId < second.movieId;
//}

int main()
{
//    Dataset originalInput;
    Dataset input;
    Dataset output;
	
	int expCount = 20;
	
    string resultFile = "AggregationScalingGPUResults.txt";	
	
    cout << "Loading data" << endl;
    loadData(input);
    cout << "Data loaded" << endl;

    ofstream out(resultFile);
    out << "#NumThreads Time" << endl;

    for(int numThreads = 32; numThreads <= 1024; numThreads <<= 1)
    {
        Timer timer;
        long long expTime = 0;
        for(int c = 0; c < expCount; c++)
        {
            timer.startTimer();
            gorup(output, input, numThreads);
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
