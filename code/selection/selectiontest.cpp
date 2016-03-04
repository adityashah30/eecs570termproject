#include "selection.h"
#include "../timer/timer.h"
#include <string>
#include <cassert>

using namespace std;

int main()
{
    	string filename;
    	Dataset input;
    	Dataset output;

    	filename = "../../data/ratings25.csv";
    	int index = 2;
    	int numThreads = 4;
    	//boost::variant<long long, double, std::string> cons;

    	long long 	a = 862;
    	string    	b = "Mark Waters";
    	double 		c = 3.5;

    	Field cons = c;    

    	Timer timer;
    	timer.startTimer();
    	loadData(input, filename);
    	timer.stopTimer();
    	std::cout << "Time to load data: " << timer.getElapsedTime() << std::endl;


	timer.startTimer();
    	selData(output, input, index, cons, numThreads);
    	timer.stopTimer();
    	std::cout << "Time to select data on " << numThreads << " threads : " 
           	  << timer.getElapsedTime() << std::endl;
	
    	cout << " Output  Size: " << output.size() << endl; 
/*	

    for (int i=0; i < input.size(); i++) {
	  	for (int j=0; j < input[i].size();j++) {
		  cout << input[i][j] << ' ';
		}
		  cout << endl;
    } 
    
    cout << " Output  Size: " << output.size() << endl; 

    for (int i=0; i < output.size(); i++) {
	  	for (int j=0; j < output[i].size();j++) {
		  cout << output[i][j] << ' ';
		}
	  cout <<  endl;
    } 

*/  
    return 0;
}
