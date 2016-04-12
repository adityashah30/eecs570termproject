/**
 * Selection GPU module
 *
 * Author  : Xiangfei Kong
 * Version : 0.1
 * Date    : April 10, 2016
 *
 * Description
 * ===========
 *
 * Select the given dataset using the provided field index using given
 * number of threads
 */
#ifndef SELECTION_CUH_
#define SELECTION_CUH_

#include "../preprocessing/loaddata.h"
#include "../preprocessing/cudafoo.h"
#include <algorithm>
#include <cstdlib>

typedef thrust::device_vector<Record> Dataset_d;

/**
 * The main sorting function.
 * @param out        	The output dataset
 * @param in         	The input dataset
 * @param constraint    The constraint to select
 * @param numThreads 	The number of threads spawned
 */
void selData(Dataset& out, Dataset& in, double constraint, int numThreads);


#endif
