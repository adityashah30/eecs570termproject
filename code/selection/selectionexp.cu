#include "selection.cuh"
#include "../timer/timer.h"
#include <string>
#include <cassert>

using namespace std;

void populateData(Dataset&, Dataset&);

int main()
{
    	Dataset input;
    	Dataset output;
	
    	string resultFile = "sortingScalingResults.txt";

    	double 	cons = 3.5;
	int expCount = 1;
    	
	cout << "Loading data" << endl;
	loadData(input);
    	cout << "Data loaded" << endl;
	
    	ofstream out(resultFile);
    	out << "#NumThreads Time" << endl;

	for(int numThreads = 1; numThreads <= 1024; numThreads <<= 1) 
	{
		Timer timer;
		long long expTime = 0;
		for(int c = 0; c < expCount; c++)
		{
			timer.startTimer();
    			selData(output, input, cons, numThreads);
    			timer.stopTimer();
            		expTime += timer.getElapsedTime();
        	}
        	expTime /= expCount;
        	cout << "Time to select data on " << numThreads << " threads : " 
             	     << expTime << endl;
      	  	out << numThreads << " " << expTime << endl;
    	}

    	out.close();

    	return 0;
}
