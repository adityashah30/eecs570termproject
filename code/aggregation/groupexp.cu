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
    Dataset expandDS;	
	int expCount = 1;
	
    string resultFile = "AggregationScalingGPUResults.txt";	
	
    cout << "Loading data" << endl;
    loadData(input);
    cout << "Data loaded" << endl;
    int id_num = 0;
    group_preprocessing(input, id_num); 

    duplicateDS(expandDS, input, 5);
    ofstream out(resultFile);
    out << "#NumThreads Time" << endl;

    for(int numThreads = 32; numThreads <= 4096; numThreads <<= 1)
    {
        Timer timer;
        long long expTime = 0;
        for(int c = 0; c < expCount; c++)
        {
            timer.startTimer();
            group(output, expandDS, numThreads, id_num);
            timer.stopTimer();
            expTime += timer.getElapsedTime();
        }
        expTime /= expCount;
        cout << "Time to group data on " << numThreads << " threads : " 
             << expTime << endl;
        out << numThreads << " " << expTime << endl;
    }

    out.close();

    return 0;
}	
