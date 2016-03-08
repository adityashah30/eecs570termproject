#include"group.h"
#include<string>

using namespace std;

pthread_mutex_t mutex;
// keep sum of all the ratings & count of rating 
unordered_map<long long, pair<double, int>> look_up;


struct ThreadArg{
public:
    Dataset::iterator beginIt;
    Dataset::iterator endIt;
	Dataset *outptr;
    int g_idx;
	int t_idx;
public:
    void setArgs(Dataset::iterator bIt, Dataset::iterator eIt, Dataset *out, int group_idx, int tar_idx){
        beginIt = bIt;
        endIt = eIt;
		outptr = out;
        g_idx = group_idx;
		t_idx = tar_idx;
    }
};

static void* groupThread(void* args){
    ThreadArg* arg = static_cast<ThreadArg*>(args);
	// map from name to pair of rating & count in local dataset
	unordered_map<long long, pair<double, int>> local;
	int &g_idx = arg->g_idx;
	int &t_idx = arg->t_idx;
	
	// perform group by and aggregation in local dataset, keep the result in a hash table
	for(auto it = arg->beginIt; it != arg->endIt; ++it){
		long long cur_id;
		/*try{
			cur_name = boost::get<string>(it->at(g_idx));
		}catch(const boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::bad_get> >&){
			try {
				cur_name = to_string(boost::get<double>(it->at(g_idx)));
			}
			catch(const boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::bad_get> >&){
				cur_name = to_string(boost::get<long long>(it->at(g_idx)));
			}
		}*/
		cur_id = boost::get<long long>(it->at(g_idx));
		double cur_rating;
		
		try{
			cur_rating = boost::get<double>(it->at(t_idx));
		}catch(const boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::bad_get> >&){
			cur_rating = boost::get<long long>(it->at(t_idx));
		}
		
		if(local.find(cur_id) == local.end()){
			local.insert({cur_id, {cur_rating, 1}});
		}else{
			auto& cur = local[cur_id];
			cur.first += cur_rating;
			cur.second++;
		}
	}
	
	// populate into hash table
	pthread_mutex_lock(&mutex);
	for(auto it = local.begin(); it != local.end(); ++it){
		double& cur_rating = ((it->second).first);
		int& cur_cnt = (it->second).second;
		
		// if it's a new data, push it into hash table
		if(look_up.find(it->first) == look_up.end()){
			look_up[it->first] = {cur_rating, cur_cnt};
		}
		// if it's already exist in hash table, update rating and count
		else{
			double& prev_rating = look_up[it->first].first;
			int& prev_cnt = look_up[it->first].second;
			
			prev_rating += cur_rating;
			prev_cnt += cur_cnt;
		}
	}
	pthread_mutex_unlock(&mutex);
}

 void group(Dataset& out, Dataset& in, int group_idx, int tar_idx, int numThreads){
	 
    pthread_t* threads = new pthread_t[numThreads];
    ThreadArg* args = new ThreadArg[numThreads];

    int chunkSize = in.size()/numThreads;
	
	for(int i = 0; i < numThreads - 1; ++i){
        args[i].setArgs(in.begin() + chunkSize*i, in.begin() + chunkSize*(i + 1), &out, group_idx, tar_idx);
		
		pthread_create(&threads[i], NULL, groupThread, (void*)&args[i]);
	}

    args[numThreads-1].setArgs(in.begin() + chunkSize*(numThreads-1), in.end(), &out, group_idx, tar_idx);
	pthread_create(&threads[numThreads-1], NULL, groupThread, (void*)&args[numThreads-1]);
	
	for(int i = 0; i < numThreads; ++i){
		pthread_join(threads[i], NULL);
	}
	
	// Calculate average
	for(auto it = look_up.begin(); it != look_up.end(); ++it){
		Record record;
		
		record.push_back(it->first);
		record.push_back(it->second.first / it->second.second);
		
		out.push_back(record);
	}
	
	delete[] threads;
    delete[] args;
 }
