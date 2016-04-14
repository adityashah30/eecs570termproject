#include "group.cuh"
#include "../timer/timer.h"
#include <string>
#include <cassert>
#include <algorithm>

using namespace std;

void populateData(Dataset& input, Dataset& expectedOutput);

static bool compare(const Record& first, const Record& second){
	return first.movieId < second.movieId;
}

int main()
{
    Dataset input;
    Dataset output;
	Dataset expectedOutput;
	Timer timer;
	
    int numThreads = 1024;
    
	cout << "start populate data" << endl;
	populateData(input, expectedOutput);
	cout << "complete populate data" << endl;
	
	for(; numThreads < 1025; numThreads <<= 1){
		cout << "///////start testing on " << numThreads << " threads" << endl;
		timer.startTimer();
		cout << "start group function" << endl;
		group(output, input, numThreads);
		cout << "complete group function" << endl;
		timer.stopTimer();
		
		sort(output.begin(), output.end(), compare);
		
		/*cout << "My output" << endl;
		
		for(auto it : output)
			cout << it.movieId << ' ' << it.rating << endl;
		
		cout << "Expected output" << endl;
		
		for(auto it : expectedOutput)
			cout << it.movieId << ' ' << it.rating << endl;*/
		
		assert(output.size() == expectedOutput.size());
		
		for(int i = 0; i < output.size(); ++i){
			assert(output[i].movieId == expectedOutput[i].movieId);
			assert(output[i].rating == expectedOutput[i].rating);
		}
		
		//assert(output == expectedOutput);
		
		cout << "Time to group data on " << numThreads << " threads : " 
              << timer.getElapsedTime() << std::endl;
		cout << '\n';
    }
	
    return 0;
}

void populateData(Dataset& input, Dataset& expectedOutput){
	input.clear();
	expectedOutput.clear();
	
	int numRecords = 3333 * 111;
	//int offset = 123;
	int numid = 3333;
	
	for(int i = 0; i < numRecords/numid; ++i){
		double total = 0;
		for(int j = 0; j < numid; ++j){
			double rating = rand()%5;
			Record inputRecord = {0, i, rating, 0 };
			total += rating;
			input.push_back(inputRecord);
		}
		double avg = total/numid;
		Record outputRecord = {0, i, avg, 0};
		expectedOutput.push_back(outputRecord);
	}
	
}
