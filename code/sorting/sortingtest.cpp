#include "sorting.h"
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

    int index = 0;
    int numThreads = 8;
    
    populateData(input, expectedOutput);

    Timer timer;
    timer.startTimer();
    sortData(output, input, index, numThreads);
    timer.stopTimer();
    std::cout << "Time to sort data on " << numThreads << " threads : " 
              << timer.getElapsedTime() << std::endl;
    
    assert(output == expectedOutput);
    std::cout << "Test passed!" << std::endl;

    return 0;
}

void populateData(Dataset& input, Dataset& expectedOutput)
{
    int numRecords = 1024;
    int offset = 123;
    for(int i=0; i<numRecords; i++)
    {
        Record inputRecord = {(i+offset)%numRecords, 0, 0.0, 0};
        Record outputRecord = {i, 0, 0.0, 0};
        input.push_back(inputRecord);
        expectedOutput.push_back(outputRecord);
    }
}
