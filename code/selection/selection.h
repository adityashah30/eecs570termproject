/**
 * Selection Module
 *
 * Author  : Xiangfei Kong
 * Version : 0.1
 * Date    : March 02, 2016
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
#include "boost/variant.hpp"

//typedef boost::variant<long long, double, std::string> Constraint;

/**
 * The main sorting function.
 * @param out        The output dataset
 * @param in         The input dataset
 * @param index      The field index by which data is to be selected
 * @param constrait  The field constraint by which data is to be matched
 * @param numThreads The number of threads spawned
 */
void selData(Dataset& out, Dataset& in, int index, Field constraint, int numThreads);
