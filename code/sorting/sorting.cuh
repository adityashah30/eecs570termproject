/**
 * Sorting GPU module
 *
 * Author  : Aditya Shah
 * Version : 0.1
 * Date    : April 07, 2016
 *
 * Description
 * ===========
 *
 * Sorts the given dataset using the provided field index using given
 * number of threads
 */
#ifndef SORTING_CUH_
#define SORTING_CUH_

#include "../preprocessing/loaddata.h"
#include "../preprocessing/cudafoo.h"
#include <algorithm>
#include <cstdlib>

typedef thrust::device_vector<Record> Dataset_d;

/**
 * The main sorting function.
 * @param out        The output dataset
 * @param in         The input dataset
 * @param index      The field index by which data is to be sorted
 * @param numThreads The number of threads spawned
 */
void sortData(Dataset& out, Dataset& in, int index, int numThreads);


#endif
