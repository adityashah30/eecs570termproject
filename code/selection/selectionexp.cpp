#include "selection.h"
#include "../timer/timer.h"
#include <string>
#include <cassert>

using namespace std;

void populateData(Dataset&, Dataset&);

int main()
{
    	Dataset input;
    	Dataset output;
	Dataset expectedOutput;

    	int 	numThreads = 4;

    	double 	cons = 3.5;

    	Timer timer;
    	timer.startTimer();
    	loadData(input);
    	timer.stopTimer();
    	std::cout << "Time to load data: " << timer.getElapsedTime() << std::endl;

	for(int numThreads = 1; numThreads <= 16; numThreads <<= 1) 
	{
		timer.startTimer();
    		selData(output, input, cons, numThreads);
    		timer.stopTimer();
    		std::cout << "Time to select data on " << numThreads << " threads : " 
        	   	  << timer.getElapsedTime() << std::endl;
	
        	
		std::cout << "output size" << output.size() << " expected output size" << expectedOutput.size() << endl;
		assert(output == expectedOutput);
        	std::cout << "NumThreads: " << numThreads << "; Test passed!" << std::endl;
	}
  
    	return 0;
}

void populateData(Dataset& input, Dataset& expectedOutput)
{
    input.clear();
    expectedOutput.clear();

    double rating[2] = {2.5, 3.5};

    int numRecords = 1024;
    for(int i=0; i<numRecords; i++)
    {
    	int r = rand() % 2;
        Record inputRecord = {i, 0, rating[r], 0};
        input.push_back(inputRecord);
        if(r == 1) {
		expectedOutput.push_back(inputRecord);
	}
    }
}
