#ifndef AGGREGATION_CUH_
#define AGGREGATION_CUH_


#include "../preprocessing/loaddata.h"
#include "../preprocessing/cudafoo.h"
#include <algorithm>
#include <cstdlib>

/**
 * The main aggregation & group function.
 * Only can perform AVG operation on rating & goup by movie id for now
 * @param out        The output dataset
 * @param in         The input dataset
 * @param numThreads The number of threads spawned
 */
 
 
 struct Entry {
	int key;
	int cnt;
	double value;
	Entry *next;
	
	Entry() : key(-1), value(0.0), cnt(0), next(NULL){};
};

 typedef thrust::device_vector<Entry> Entries_d;
 typedef thrust::device_vector<Entry*> Entries_ptr_d;

/*struct Table {
	Entries_ptr_d entries;
	Entries_d pool;
	int first_free;
	
	Table() : first_free(0) {};
};*/
 
 typedef thrust::device_vector<Record> Dataset_d;
 //typedef thrust::device_vector<Table> Tables_d;
 typedef thrust::device_vector<int> vecint_d;
 
 void group(Dataset& out, Dataset& in, int numThreads);
 
 #endif