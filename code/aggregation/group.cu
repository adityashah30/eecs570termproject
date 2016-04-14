#include "group.cuh"
#include <sstream>
#include <unordered_set>

//#define ELEMENTS 26744 
#define HASH_ENTRIES 10000 

__device__ __host__ 
int hash(int value, size_t count) {
	return value % count;
}

__global__ 
void aggre_kernel(Entry** entries, Entry* pool, int* first_free, size_t size, Record* in, int ELEMENTS){
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
		int* my_first_free = first_free + threadId;
		Entry* mypool = pool + ELEMENTS * threadId;
	
		//assert(mypool[0].key == -1);
		int idx1 = threadId * chunksize;
     		int idx2 = (threadId + 1) * chunksize;
        	if(idx1 >= size) return;
		if(idx2 >= size){
        	    idx2 = size;
        	}
		for(int cur_id = idx1; cur_id < idx2; ++cur_id){
			//int cur_id = threadId * chunksize + i;
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


void group(Dataset& out, Dataset& in, int numThreads, int id_num){
	Dataset_d in_d = in;
	out.clear();
	size_t size = in.size();
//	std::cout << "Elements count: " << id_num << std::endl; 
	int ELEMENTS = id_num;
//	std::cout << "Dataset size: " << size << std::endl;
	int tpb = (numThreads<1024)?numThreads:1024;
    dim3 block(tpb,1,1);
	int numBlocks = (numThreads+tpb-1)/tpb;
	dim3 grid(numBlocks,1,1);
	
	std::vector<Entry> final_pool(ELEMENTS * numThreads);

	Entries_ptr_d total_entries_d(HASH_ENTRIES * numThreads, NULL);
	Entries_d total_pool_d = final_pool;
	vecint_d first_free_d(numThreads, 0);
	
	Record* in_begin = thrust::raw_pointer_cast(in_d.data());
	Entry** entries_begin = thrust::raw_pointer_cast(total_entries_d.data());
	Entry*  pool_begin = thrust::raw_pointer_cast(total_pool_d.data());
	int* first_free_begin = thrust::raw_pointer_cast(first_free_d.data());

	//std::cout << "start aggregation kernel" << std::endl;	
	aggre_kernel<<<grid, block>>>(entries_begin, pool_begin, first_free_begin, size, in_begin, ELEMENTS);
	checkCudaErrorKernel("aggregationKernel");
	//std::cout << "end aggregation kernel" << std::endl;
	
	std::vector<Entry> host_pools(ELEMENTS * numThreads);
	
	thrust::copy(total_pool_d.begin(), total_pool_d.end(), host_pools.begin());

	std::vector<Entry*> final_entries(HASH_ENTRIES, NULL);
	int final_first_free = 0;
	
	// Build the table at host
	//std::cout << "start build table at host" << std::endl;
	for(int i = 0; i < numThreads; ++i){
	//	std::cout << "start building table on thread id: " << i << std::endl;
		int start = i * ELEMENTS;
		int end = start + ELEMENTS;
		for(int j = start; j < end; ++j){
	//		std::cout << "key value: " << host_pools[j].key << std::endl;
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
	//std::cout << "final first free: " << final_first_free << std::endl;
	//assert(final_first_free == ELEMENTS);
	//std::cout << "finish build table at host" << std::endl;
	out.resize(ELEMENTS);
	
	int k = 0;
	for(int i = 0; i < HASH_ENTRIES; ++i){
		Entry *cur = final_entries[i];
		while(cur){
			double avg_rating = cur->value / cur->cnt;
			Record record(-1, cur->key, avg_rating, -1);
			out[k++] = record;
			cur = cur->next;
		}
	}
	//std::cout << "complete populate to out" << std::endl;
}

 void group_preprocessing(Dataset& in, int& id_num){
	 std::unordered_set<int> id_set;
	 int size = in.size();
	 for(int i = 0; i < size; ++i)
		 id_set.insert(in[i].movieId);
	 
	 id_num = id_set.size();
 }
