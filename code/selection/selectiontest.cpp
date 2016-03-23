#include "selection.h"
#include "../timer/timer.h"
#include <string>
#include <cassert>

using namespace std;

int main()
{
    	Dataset input;
    	Dataset output;

    	int 	numThreads = 4;

    	double 	cons = 3.5;

    	Timer timer;
    	timer.startTimer();
    	loadData(input);
    	timer.stopTimer();
    	std::cout << "Time to load data: " << timer.getElapsedTime() << std::endl;


	timer.startTimer();
    	selData(output, input, cons, numThreads);
    	timer.stopTimer();
    	std::cout << "Time to select data on " << numThreads << " threads : " 
           	  << timer.getElapsedTime() << std::endl;
	
    	cout << " Output  Size: " << output.size() << endl; 
	
    	cout << " Input Size: " << input.size() << endl; 

/*
    for (int i=0; i < input.size(); i++) {
	   	cout << input[i].userId << ' ' << input[i].movieId << ' ' << input[i].rating << ' ' << input[i].timestamp;
	   	cout << endl;
    } 
    
    cout << " Output Size: " << output.size() << endl; 

    for (int i=0; i < output.size(); i++) {
	   	cout << output[i].userId << ' ' << output[i].movieId << ' ' << output[i].rating << ' ' << output[i].timestamp;
		cout <<  endl;
    } 
*/
  
    return 0;
}
