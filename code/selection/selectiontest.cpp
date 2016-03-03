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

    filename = "../../data/tags.csv";
    int index = 2;
    int numThreads = 2;
    //boost::variant<long long, double, std::string> cons;

    long long a = 862;
    string    b = "Mark Waters";
    long long c = 3;

    Field cons = b;    

    Timer timer;
    timer.startTimer();
    loadData(input, filename);
    timer.stopTimer();
    std::cout << "Time to load data: " << timer.getElapsedTime() << std::endl;


    for(int i=2; i<=2; i=i+2) 
    {
    	timer.startTimer();
    	selData(output, input, index, cons, i);
    	timer.stopTimer();
    	std::cout << "Time to select data on " << i << " threads : " 
           	  << timer.getElapsedTime() << std::endl;
	
    	cout << " Output  Size: " << output.size() << endl; 

    }
	

    for (int i=0; i < 3; i++) {
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

    
    return 0;
}
