#include "loaddata.h"
#include <string>
#include <cassert>

using namespace std;

int main()
{
    string filename;
    Dataset data;

    filename = "../../data/links.csv";
    loadData(data, filename);
    assert(data.size() == 27278);
    assert(data[0].size() == 3);
    cout << "links pass" << endl;

    filename = "../../data/movies.csv";
    loadData(data, filename);
    assert(data.size() == 27278);
    assert(data[0].size() == 3);
    cout << "movie pass" << endl;

    filename = "../../data/ratings.csv";
    loadData(data, filename);
    assert(data.size() == 20000263);
    assert(data[0].size() == 4);
    cout << "ratings pass" << endl;

    filename = "../../data/tags.csv";
    loadData(data, filename);
    assert(data.size() == 465564);
    assert(data[0].size() == 4);
    cout << "tags pass" << endl;
    
    return 0;
}
