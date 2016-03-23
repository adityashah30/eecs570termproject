/**
 * Sorting module
 *
 * Author  : Aditya Shah
 * Version : 0.1
 * Date    : March 01, 2016
 *
 * Description
 * ===========
 *
 * Sorts the given dataset using the provided field index using given
 * number of threads
 */
#ifndef SORTING_H
#define SORTING_H

#include "../preprocessing/loaddata.h"
#include <algorithm>
#include <cstdlib>
#include <pthread.h>

/**
 * The main sorting function.
 * @param out        The output dataset
 * @param in         The input dataset
 * @param index      The field index by which data is to be sorted
 * @param numThreads The number of threads spawned
 */
void sortData(Dataset& out, Dataset& in, int index, int numThreads);

#endif
