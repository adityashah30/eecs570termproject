#include"group.h"
#include<string>

using namespace std;

pthread_mutex_t mutex;
// keep sum of all the ratings & count of rating 
//unordered_map<long long, pair<double, int>> look_up;


struct ThreadArg{
public:
    Dataset::iterator beginIt;
    Dataset::iterator endIt;
	Dataset *outptr;
	// map from name to pair of rating & count in local dataset
	unordered_map<int, pair<double, int>> local;
public:
    void setArgs(Dataset::iterator bIt, Dataset::iterator eIt, Dataset *out){
        beginIt = bIt;
        endIt = eIt;
		outptr = out;
    }
};

struct MergeArg{
public:
	unordered_map<int, pair<double, int>>* first;
	unordered_map<int, pair<double, int>>* second;
	
public:
	void setArgs(unordered_map<int, pair<double, int>>* first_in, unordered_map<int, pair<double, int>>* second_in){
		first = first_in;
		second = second_in;
	}
};

static void* groupThread(void* args){
    ThreadArg* arg = static_cast<ThreadArg*>(args);

	auto &local = arg->local;
	
	// perform group by and aggregation in local dataset, keep the result in a hash table
	for(auto it = arg->beginIt; it != arg->endIt; ++it){
		int &cur_id = it->movieId;
		double &cur_rating = it->rating;
		
		if(local.find(cur_id) == local.end()){
			local.insert({cur_id, {cur_rating, 1}});
		}else{
			auto& cur = local[cur_id];
			cur.first += cur_rating;
			cur.second++;
		}
	}
}

static void* mergeThread(void* args){
    MergeArg* arg = static_cast<MergeArg*>(args);
	unordered_map<int, pair<double, int>> &first = *(arg->first);
	unordered_map<int, pair<double, int>> &second = *(arg->second);
	
	// Merge data into the first hash table;
	for(auto it = second.begin(); it != second.end(); it++){
		double& cur_rating = ((it->second).first);
		int& cur_cnt = (it->second).second;		
		
		if(first.find(it->first) == first.end()){
			first[it->first] = {cur_rating, cur_cnt};
		}else{
			double& prev_rating = first[it->first].first;
			int& prev_cnt = first[it->first].second;
			
			prev_rating += cur_rating;
			prev_cnt += cur_cnt;
		}
	}
}

void group(Dataset& out, Dataset& in, int numThreads){
	out.clear();
    pthread_t* threads = new pthread_t[numThreads];
    ThreadArg* args = new ThreadArg[numThreads];
	MergeArg* margs = new MergeArg[numThreads];
    int* beginIndex = new int[numThreads];
    int* endIndex = new int[numThreads];

    int chunkSize = in.size()/numThreads;

    for(int i=0; i < numThreads - 1; i++)
    {
	    beginIndex[i] = chunkSize * i;
	    endIndex[i] = chunkSize * (i+1);
	}
	beginIndex[numThreads-1] = chunkSize*(numThreads-1);
	endIndex[numThreads-1] = in.size();
	
	for(int i = 0; i < numThreads; ++i){
        args[i].setArgs(in.begin() + beginIndex[i], in.begin() + endIndex[i], &out);
		
		pthread_create(&threads[i], NULL, groupThread, (void*)&args[i]);
	}
	
	for(int i = 0; i < numThreads; ++i){
		pthread_join(threads[i], NULL);
	}
	
	
	// Merge
	int numChunks = 1;
	while(numChunks < numThreads)
		numChunks <<= 1;
	numChunks >>= 1;
	
	for(int stride = 1; stride < numThreads; stride <<= 1, numChunks >>= 1){
		for(int i = 0; i < numChunks; ++i){
			int idx1 = 2*i*stride;
			int idx2 = idx1 + stride;
			if(idx2 >= numThreads)
				break;
			
			margs[i].setArgs(&args[idx1].local, &args[idx2].local);
			
			pthread_create(&threads[i], NULL, mergeThread, (void*)&margs[i]);
		}
		for(int i = 0; i < numChunks; ++i)	{
			pthread_join(threads[i], NULL);
		}
	}
	
	// Calculate average
	for(auto it = args[0].local.begin(); it != args[0].local.end(); ++it){
		double avg_rating = it->second.first / it->second.second;
		Record record(-1, it->first, avg_rating, -1);
		
		out.push_back(record);
	}
	
	delete[] threads;
    delete[] args;
 }
