#include "loaddata.h"
#include <string>

using namespace std;

int main()
{
    string filename;
    Dataset data;

    filename = "../../data/links.csv";
    loadData(data, filename);
    assert(data.size() == 27279);
    assert(data[0].size() == 3);
    cout << "links pass" << endl;

    filename = "../../data/movies.csv";
    loadData(data, filename);
    assert(data.size() == 27279);
    assert(data[0].size() == 3);
    cout << "movie pass" << endl;

    filename = "../../data/ratings.csv";
    loadData(data, filename);
    assert(data.size() == 20000264);
    assert(data[0].size() == 4);
    cout << "ratings pass" << endl;

    filename = "../../data/tags.csv";
    loadData(data, filename);
    assert(data.size() == 465565);
    assert(data[0].size() == 4);
    cout << "tags pass" << endl;
    
    return 0;
}
