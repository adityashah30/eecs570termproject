#include "sorting.h"
#include "../timer/timer.h"
#include <string>
#include <cassert>

using namespace std;

int main()
{
    string filename;
    Dataset input;
    Dataset output;

    filename = "../../data/links.csv";
    int index = 1;
    int numThreads = 8;
    
    Timer timer;
    timer.startTimer();
    loadData(input, filename);
    timer.stopTimer();
    std::cout << "Time to load data: " << timer.getElapsedTime() << std::endl;

    timer.startTimer();
    sortData(output, input, index, numThreads);
    timer.stopTimer();
    std::cout << "Time to sort data on " << numThreads << " threads : " 
              << timer.getElapsedTime() << std::endl;
    
    return 0;
}
