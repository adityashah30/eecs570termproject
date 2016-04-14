#include "selection.cuh"
#include "../timer/timer.h"
#include <string>
#include <cassert>

using namespace std;

void populateData(Dataset&, Dataset&);

int main()
{
	double fraction = 5;
    	Dataset input;
    	Dataset output;
	Dataset expandedDS;	
	
 	string resultFile = "sortingScalingResults.txt";

    	double 	cons = 3.5;
	int expCount = 10;
    	
	cout << "Loading data" << endl;
	loadData(input);
    	cout << "Data loaded" << endl;

 	cout << "Duplicating data" << endl;	
    	duplicateDS(expandedDS, input, fraction);
	cout << "Data duplicated" << endl;
/*	
    	ofstream out(resultFile);
    	out << "#NumThreads Time" << endl;
*/
	for(int numThreads = 32; numThreads <= 8192; numThreads <<= 1) 
	{
		Timer timer;
		long long expTime = 0;
		for(int c = 0; c < expCount; c++)
		{
			timer.startTimer();
    			selData(output, expandedDS, cons, numThreads);
    			timer.stopTimer();
            		expTime += timer.getElapsedTime();
        	}
        	expTime /= expCount;
        	cout << "Time to select data on " << numThreads << " threads : " 
             	     << expTime << endl;
      	  	//out << numThreads << " " << expTime << endl;
    	}

    	//out.close();

    	return 0;
}
