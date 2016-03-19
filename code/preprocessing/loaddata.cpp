#include "loaddata.h"

void loadData(Dataset& data)
{
    data.clear();

    std::string filename = "../../data/ratings.csv";
    std::ifstream in(filename);
    std::string line;

    assert(in.is_open());

    in >> line;

    while(in >> line)
    {
        std::stringstream lineStream(line);
        std::string field;
        Record record;

        std::getline(lineStream, field, ',');
        record.userId = stoi(field);

        std::getline(lineStream, field, ',');
        record.movieId = stoi(field);

        std::getline(lineStream, field, ',');
        record.rating = stod(field);

        std::getline(lineStream, field, ',');
        record.timestamp = stoll(field);

        data.push_back(record);
    }
    in.close();
}

void duplicateDS(Dataset& bigDataset, Dataset& originalDataset, double fraction)
{
    bigDataset.clear();
    size_t chunkSize = originalDataset.size();
    size_t newSize = chunkSize*fraction;
    bigDataset.resize(newSize);
    Dataset::iterator bigIt = bigDataset.begin();
    for(int i=0; i<(int)fraction; i++)
    {
        std::copy(originalDataset.begin(), originalDataset.end(), bigIt);
        bigIt += chunkSize;
    }
    int remaining = chunkSize*(fraction - (int)fraction);
    if (remaining > 0)
    {
        std::copy(originalDataset.begin(), originalDataset.begin()+remaining, bigIt);
    }
}

void extractSmallDS(Dataset& smallDataset, Dataset& originalDataset, double fraction)
{
    smallDataset.clear();
    int newSize = originalDataset.size()*fraction;
    smallDataset.resize(newSize);
    std::copy(originalDataset.begin(), originalDataset.begin()+newSize, smallDataset.begin());
}
