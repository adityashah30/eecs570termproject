#include "group.cuh"
#include <sstream>

#define ELEMENTS 26744
#define HASH_ENTRIES 1024

__device__ __host__ 
int hash(int value, size_t count) {
	return value % count;
}

__global__ 
void aggre_kernel(Entry** entries, Entry* pool, int* first_free, size_t size, Record* in){
    size_t blockId = blockIdx.x
                  + blockIdx.y*gridDim.x
                  + blockIdx.z*gridDim.y*gridDim.x;
    size_t threadId = threadIdx.x
                   + threadIdx.y*blockDim.x
                   + threadIdx.z*blockDim.x*blockDim.y
                   + blockId*blockDim.x*blockDim.y*blockDim.z;
    size_t numThreads = blockDim.x*blockDim.y*blockDim.z
                        *gridDim.x*gridDim.y*gridDim.z;

	if(threadId < numThreads){
		int chunksize = (size + numThreads-1)/numThreads;
	
		Entry** mytable = entries + HASH_ENTRIES * threadId;
		int* my_first_free = first_free;
		Entry* mypool = pool + ELEMENTS * threadId;
	
		assert(mypool[0].key == -1);
		for(int i = 0; i < chunksize; ++i){
			int cur_id = threadId * chunksize + i;
			int key = in[cur_id].movieId;
			int hashval = hash(key, HASH_ENTRIES);
		
			Entry *cur = mytable[hashval];	
			while(cur){
				if(cur->key == key){
					cur->value += in[cur_id].rating;
					++(cur->cnt);
					break;
				}
				cur = cur->next;
			}
			if(!cur){
				Entry *new_entry = &(mypool[*my_first_free]);
				(*my_first_free)++;
				new_entry->key = key;
				new_entry->value = in[cur_id].rating;
				new_entry->cnt = 1;
				new_entry->next = mytable[hashval];
				mytable[hashval] = new_entry;
			}
		}
	}
	__syncthreads();
}


void group(Dataset& out, Dataset& in, int numThreads){
    Dataset_d in_d = in;
	size_t size = in.size();

	int tpb = (numThreads<1024)?numThreads:1024;
    dim3 block(tpb,1,1);
	int numBlocks = (numThreads+tpb-1)/tpb;
	dim3 grid(numBlocks,1,1);
	

	//int totalThreads = numThreads * block_1;
	//Tables_d tables_d(numThreads);
	
	//for(int i = 0; i < numThreads; ++i)
	//	initialize_table(tables_d[i], HASH_ENTRIES, ELEMENTS);
	Entries_ptr_d total_entries_d(HASH_ENTRIES * numThreads);
	Entries_d total_pool_d(ELEMENTS * numThreads);
	vecint_d first_free_d(numThreads);
	
	Record* in_begin = thrust::raw_pointer_cast(in_d.data());
	//Table* table_begin = thrust::raw_pointer_cast(tables_d.data());
	Entry** entries_begin = thrust::raw_pointer_cast(total_entries_d.data());
	Entry*  pool_begin = thrust::raw_pointer_cast(total_pool_d.data());
	int* first_free_begin = thrust::raw_pointer_cast(first_free_d.data());
	
	aggre_kernel<<<grid, block>>>(entries_begin, pool_begin, first_free_begin, size, in_begin);
	
	std::vector<Entry> host_pools(ELEMENTS * numThreads);
	
	thrust::copy(total_pool_d.begin(), total_pool_d.end(), host_pools.begin());

	std::vector<Entry*> final_entries;
	std::vector<Entry> final_pool;
	int final_first_free = 0;
	
	// Build the table at host
	for(int i = 0; i < numThreads; ++i){
		int start = i * ELEMENTS;
		int end = start + ELEMENTS;
		for(int j = start; j < end; ++j){
			if(host_pools[j].key == -1)
				break;
			
			int key = host_pools[j].key;
			int hashval = hash(key, HASH_ENTRIES);
		
			Entry *cur = final_entries[hashval];
			while(cur){
				if(cur->key == key){
					cur->value += host_pools[j].value;
					cur->cnt   += host_pools[j].cnt;
				break;
				}
				cur = cur->next;
			}
			if(!cur){
				Entry *new_entry = &final_pool[final_first_free++];
				new_entry->key = key;
				new_entry->value = host_pools[j].value;
				new_entry->cnt   = host_pools[j].cnt;
				new_entry->next = final_entries[hashval];
				final_entries[hashval] = new_entry;
			}
		}
	}
	
	// Traverse the table, calculate the avg
	assert(final_first_free == ELEMENTS);
	out.resize(size);
	
	int k = 0;
	for(int i = 0; i < HASH_ENTRIES; ++i){
		Entry *cur = final_entries[i];
		while(cur){
			double avg_rating = cur->value / cur->cnt;
			Record record(-1, cur->key, avg_rating, -1);
			out[k++] = record;
		}
	}
	
}