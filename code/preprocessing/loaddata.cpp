#include "loaddata.h"

bool operator==(const Record& a, const Record& b)
{
    return (a.userId == b.userId) &&
           (a.movieId == b.movieId) &&
           (a.rating == b.rating) &&
           (a.timestamp == b.timestamp);
}

void loadData(Dataset& data)
{
    data.clear();

#ifdef __MIC__
    std::string filename = "/home/micuser/ratings.csv";
#else
    std::string filename = "../../data/ratings.csv";
#endif

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
    size_t remaining = chunkSize*(fraction - (int)fraction);
    if (remaining > 0)
    {
        std::copy(originalDataset.begin(), originalDataset.begin()+remaining, bigIt);
    }
}

void extractSmallDS(Dataset& smallDataset, Dataset& originalDataset, double fraction)
{
    smallDataset.clear();
    size_t newSize = originalDataset.size()*fraction;
    smallDataset.resize(newSize);
    std::copy(originalDataset.begin(), originalDataset.begin()+newSize, smallDataset.begin());
}

void nearestPowerOf2DS(Dataset& powerDataset, Dataset& originalDataset)
{
    size_t nearestPowerOf2 = 1;
    size_t size = originalDataset.size();
    while(nearestPowerOf2 < size)
    {
        nearestPowerOf2 <<= 1;
    }
    nearestPowerOf2 >>= 1;

    powerDataset.clear();
    powerDataset.resize(nearestPowerOf2);
    std::copy(originalDataset.begin(), originalDataset.begin()+nearestPowerOf2, powerDataset.begin());
}
