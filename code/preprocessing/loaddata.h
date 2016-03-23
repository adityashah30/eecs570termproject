/**
 * LoadData module that parses the given CSV file and loads the data
 */
#ifndef LOADDATA_H
#define LOADDATA_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cassert>

struct Record
{
    int userId;
    int movieId;
    double rating;
    long long timestamp;

    Record() : userId(0), movieId(0), rating(0.0), timestamp(0)
    {
    }

    Record(int uid, int mid, double r, long long ts)
    : userId(uid), movieId(mid), rating(r), timestamp(ts)
    {
    }

    Record(const Record& other)
    : userId(other.userId), movieId(other.movieId), 
      rating(other.rating), timestamp(other.timestamp)
    {
    }

};

bool operator==(const Record& a, const Record& b);

typedef std::vector<Record> Dataset;

void loadData(Dataset& data);
void duplicateDS(Dataset& bigDataset, Dataset& originalDataset, double fraction);
void extractSmallDS(Dataset& smallDataset, Dataset& originalDataset, double fraction);
void nearestPowerOf2DS(Dataset& powerDataset, Dataset& originalDataset);

#endif
