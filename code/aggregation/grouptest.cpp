#include "group.h"
#include "../timer/timer.h"
#include <string>
#include <cassert>

using namespace std;

int main()
{
    string filename;
    Dataset input;
    Dataset output;

    filename = "../../data/ratings.csv";
    int group_idx = 1;
	int target_idx = 2;
    int numThreads = 3;
    
    Timer timer;
    timer.startTimer();
    loadData(input, filename);
    timer.stopTimer();
    std::cout << "Time to load data: " << timer.getElapsedTime() << std::endl;

    timer.startTimer();
	group(output, input, group_idx, target_idx, numThreads);
    timer.stopTimer();
	
	cout << "Group by\t" << "Value" << endl; 
	for(auto it = output.begin(); it != output.end(); ++it){
		cout << it->at(0) << ("\t\t") << it->at(1) << endl;
	}
	
    std::cout << "Time to group data on " << numThreads << " threads : " 
              << timer.getElapsedTime() << std::endl;
    
    return 0;
}
