/**
 * LoadData module that parses the given CSV file and loads the data
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cassert>
#include "boost/variant.hpp"

typedef boost::variant<long long, double, std::string> Field;
typedef std::vector<Field> Record;
typedef std::vector<Record> Dataset;

void loadData(Dataset& data, std::string filename);
