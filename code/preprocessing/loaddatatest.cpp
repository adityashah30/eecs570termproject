#include "loaddata.h"
#include <string>
#include <cassert>
#include <fstream>

using namespace std;

int main()
{
    Dataset data;

    loadData(data);
    assert(data.size() == 20000263);
    Record firstRec3 = {1, 2, 3.5, 1112486027};
    assert(data[0] == firstRec3);
    cout << "loadData pass" << endl;

    Dataset data25;
    double fraction = 0.25;
    extractSmallDS(data25, data, fraction);
    int newSize = data25.size();
    int newActualSize = data.size()*fraction;
    assert(newSize == newActualSize);
    cout << "extractSmallDS pass" << endl;

    Dataset data200;
    fraction = 2.0;
    duplicateDS(data200, data, fraction);
    newSize = data200.size();
    newActualSize = data.size()*fraction;
    assert(newSize == newActualSize);
    cout << "duplicateDS pass" << endl;
    
    return 0;
}
