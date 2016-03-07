#include "../preprocessing/loaddata.h"
#include <algorithm>
#include <cstdlib>
#include <pthread.h>
#include <unordered_map>

/**
 * The main aggregation & group function.
 * Only can perform AVG operation & goup by string type for now
 * @param out        The output dataset
 * @param in         The input dataset
 * @param tar_idx    The field index that need to operate aggregation
 * @param tar_idx    The field index that need to be grouped
 * @param numThreads The number of threads spawned
 */
 
 void group(Dataset& out, Dataset& in, int group_idx, int tar_idx, int numThreads);