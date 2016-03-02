#include "sorting.h"
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
    
    loadData(input, filename);

    sortData(output, input, index, numThreads);
    
    return 0;
}
