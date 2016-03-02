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

#include "../preprocessing/loaddata.h"
#include <algorithm>
#include <cstdlib>
#include <pthread.h>

void sortData(Dataset& out, Dataset& in, int index, int numThreads);
