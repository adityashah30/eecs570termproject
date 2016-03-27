#include "selection.h"
#include <iostream>

static void* selDataThread(void* args);
static void* mergeDataThread(void* args);

struct SelThreadArg
{
public:
    	// Input
	Dataset::iterator 	beginIt;
    	Dataset::iterator 	endIt;
    	double			constraint;
	Dataset			*tmp_result;		

public:
    void setArgs( Dataset::iterator bIt, 
                  Dataset::iterator eIt, 
		  double cst,
		  Dataset *tmp_results)
    {
        beginIt		= bIt;
        endIt 		= eIt;
	constraint	= cst;
	tmp_result 	= tmp_results;
    }
};

struct MergeThreadArg
{
public:
	// Input
	Dataset			*merge_result;
	Dataset::iterator 	beginIt;
    	Dataset::iterator 	endIt;
public:
    void setArgs( Dataset *tmp_result,
                  Dataset::iterator bIt, 
                  Dataset::iterator eIt 
		)
    {
	    merge_result = tmp_result;
            beginIt = bIt;
            endIt = eIt;
    }
};

void selData(Dataset& out, Dataset& in, double constraint, int numThreads)
{
    	out.clear();
    	pthread_t* threads = new pthread_t[numThreads];
    	SelThreadArg* sArgs = new SelThreadArg[numThreads];
	MergeThreadArg* mArgs = new MergeThreadArg[numThreads];
    	int* beginIndex = new int[numThreads];
    	int* endIndex = new int[numThreads];

    	int chunkSize = in.size()/numThreads;
	int rc = 0;

	Dataset  tmp_results[numThreads];

    	for(int i=0; i < numThreads - 1; i++)
    	{
	    beginIndex[i] = chunkSize * i;
	    endIndex[i] = chunkSize * (i+1);
	}
	beginIndex[numThreads-1] = chunkSize*(numThreads-1);
	endIndex[numThreads-1] = in.size();


    	// Selection Phase
    	for(int i = 0; i < numThreads; i++)
    	{
        	sArgs[i].setArgs( in.begin() + beginIndex[i], 
                                  in.begin() + endIndex[i], 
                                  constraint, &tmp_results[i]
				);

		rc = pthread_create( &threads[i], NULL, 
                             	     selDataThread, (void*)&sArgs[i]);

		/*
		if(rc)
        	{
            	std::cerr << "Error: Return code from pthread_create on threadId: " 
                 	  << i << " is " << rc << std::endl;
            	exit(EXIT_FAILURE);
        	}
		*/
    	}

    	for(int i = 0; i < numThreads; i++)
    	{
        	rc = pthread_join(threads[i], NULL);
        	/*
		if(rc)
        	{
            	std::cerr << "Error: Return code from pthread_create on threadId: " 
            	          << i << " is " << rc << std::endl;
            	exit(EXIT_FAILURE);
        	}
		*/
    	}

	// Merge Phase
	int numChunks = 1;
	while(numChunks < numThreads) 
	{
		numChunks <<= 1;
	}
	numChunks >>= 1;

	for(int stride = 1; stride < numThreads; stride <<= 1, numChunks >>= 1)
	{	
		for(int i=0; i < numChunks; i++)
		{
			int idx1 = stride*(2*i);
			int idx2 = stride*(2*i+1);
		
			if(idx2 >= numThreads)
			{
				break;
			}

			mArgs[i].setArgs( &tmp_results[idx1], 
				 	  tmp_results[idx2].begin(),
					  tmp_results[idx2].end()
					 );

            		rc = pthread_create(&threads[i], NULL, 
                                            mergeDataThread, (void*)&mArgs[i]);
            		/*
			if(rc)
            		{
                		std::cerr << "Error: Return code from pthread_create on threadId: " 
                        	          << i << " is " << rc << std::endl;
                		exit(EXIT_FAILURE);
        		}
			*/
		}
	
		for(int i=0; i < numChunks; i++)
        	{
        		rc = pthread_join(threads[i], NULL);
          		/*
			if(rc)
            		{	
                		std::cerr << "Error: Return code from pthread_create on threadId: " 
                	 		  << i << " is " << rc << std::endl;
                		exit(EXIT_FAILURE);
            		}
			*/
        	}
	}

	out = tmp_results[0];
	
/*
    	for(int i = 0; i < numThreads; i++)
    	{
	    out.insert(out.end(), (sArgs[i].tmp_result).begin(), (sArgs[i].tmp_result).end()); 
    	}
*/
	delete[] beginIndex;
	delete[] endIndex;
    	delete[] threads;
    	delete[] sArgs;
	delete[] mArgs;
}

static void* selDataThread(void* args)
{
	SelThreadArg* arg = static_cast<SelThreadArg*>(args);
	
	for (auto it=arg->beginIt; it!=arg->endIt; it++) 
	{
		if (it->rating == arg->constraint)	
		{	
			(arg->tmp_result)->push_back(*it);
		}
	}

    	pthread_exit(NULL);    
}

static void* mergeDataThread(void* args)
{
	MergeThreadArg* arg = static_cast<MergeThreadArg*>(args);
        (arg->merge_result)->insert( (arg->merge_result)->end(), arg->beginIt, arg->endIt);
	
	pthread_exit(NULL);
}

