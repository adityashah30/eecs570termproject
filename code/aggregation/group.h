#include "../preprocessing/loaddata.h"
#include <algorithm>
#include <cstdlib>
#include <pthread.h>
#include <unordered_map>

/**
 * The main aggregation & group function.
 * Only can perform AVG operation on rating & goup by movie id for now
 * @param out        The output dataset
 * @param in         The input dataset
 * @param numThreads The number of threads spawned
 */
 
 void group(Dataset& out, Dataset& in, int numThreads);