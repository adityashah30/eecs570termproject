#include "loaddata.h"

void loadData(Dataset& data, std::string filename)
{
    data.clear();
    std::ifstream in(filename);
    std::string line;
    while(std::getline(in, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        Record record;
        while(std::getline(lineStream, cell, ','))
        {
            Field field;
            try
            {
                field = stoll(cell);
            }
            catch(const std::invalid_argument&)
            {
                try
                {
                    field = stod(cell);
                }
                catch(const std::invalid_argument&)
                {
                    
                    field = cell;
                }
            }
            record.push_back(field);
        }
        data.push_back(record);
    }
    in.close();
}
