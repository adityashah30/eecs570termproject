#include "loaddata.h"

void loadData(Dataset& data, std::string filename)
{
    data.clear();
    std::ifstream in(filename);
    std::string line;
    int numFields = 0;
    std::getline(in ,line);
    std::stringstream lineStream(line);
    std::string cell;
    while(std::getline(lineStream, cell, ','))
    {
        numFields++;
    }
    while(std::getline(in, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        Record record;
        for(int i=0; i<numFields; i++)
        {
            std::getline(lineStream, cell, ',');
            Field field;
            try
            {
                double val = stod(cell);
                if(static_cast<long long>(val) == val)
                {
                    field = static_cast<long long>(val);
                }
                else
                {
                    field = val;
                }
            }
            catch(const std::invalid_argument&)
            {
                field = cell;   
            }
            record.push_back(field);
        }
        assert(record.size() == numFields);
        data.push_back(record);
    }
    in.close();
}

void extractSmallDS(Dataset& smallDataset, Dataset& originalDataset, double fraction)
{
    smallDataset.clear();
    int newSize = originalDataset.size()*fraction;
    smallDataset.resize(newSize);
    std::copy(originalDataset.begin(), originalDataset.begin()+newSize, smallDataset.begin());
}
