#include "loaddata.h"
#include <string>
#include <cassert>
#include <fstream>

using namespace std;

int main()
{
    string filename;
    Dataset data;

    filename = "../../data/links.csv";
    loadData(data, filename);
    assert(data.size() == 27278);
    assert(data[0].size() == 3);
    long long field11 = 1, field12 = 114709, field13 = 862;
    Record firstRec1 = {field11, field12, field13};
    assert(data[0] == firstRec1);
    cout << "links pass" << endl;

    filename = "../../data/movies.csv";
    loadData(data, filename);
    assert(data.size() == 27278);
    assert(data[0].size() == 3);
    long long field21 = 1;
    string field22 = "Toy Story (1995)";
    string field23 = "Adventure|Animation|Children|Comedy|Fantasy";
    Record firstRec2 = {field21, field22, field23};
    assert(data[0] == firstRec2);
    cout << "movie pass" << endl;

    filename = "../../data/ratings.csv";
    loadData(data, filename);
    assert(data.size() == 20000263);
    assert(data[0].size() == 4);
    long long field31 = 1, field32 = 2, field34 = 1112486027;
    double field33 = 3.5;
    Record firstRec3 = {field31, field32, field33, field34};
    assert(data[0] == firstRec3);
    cout << "ratings pass" << endl;

    filename = "../../data/tags.csv";
    loadData(data, filename);
    assert(data.size() == 465564);
    assert(data[0].size() == 4);
    long long field41 = 18, field42 = 4141, field44 = 1240597180;
    string field43 = "Mark Waters";
    Record firstRec4 = {field41, field42, field43, field44};
    assert(data[0] == firstRec4);
    cout << "tags pass" << endl;
    
    return 0;
}
