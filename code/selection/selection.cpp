#include "selection.h"
#include <iostream>

static void* selDataThread(void* args);

struct ThreadArg
{
public:
    	// Input
	Dataset::iterator 	beginIt;
    	Dataset::iterator 	endIt;
    	int 			index;
    	Field			constraint;
    	// Result
	Dataset			tmp_result;		

public:
    void setArgs(Dataset::iterator bIt, 
                 Dataset::iterator eIt, 
		 int idx, Field cst)
    {
        beginIt		= bIt;
        endIt 		= eIt;
        index 		= idx;
	constraint	= cst;
    }
};

void selData(Dataset& out, Dataset& in, int index, Field constraint, int numThreads)
{
    out.clear();
    pthread_t* threads = new pthread_t[numThreads];
    ThreadArg* args = new ThreadArg[numThreads];

    int chunkSize = in.size()/numThreads;
    int rc = 0;

    // Thread Creation
    for(int i = 0; i < numThreads - 1; i++)
    {
        args[i].setArgs( in.begin() + chunkSize * i, 
                         in.begin() + chunkSize * (i + 1), 
                         index, constraint);

	rc = pthread_create( &threads[i], NULL, 
                             selDataThread, (void*)&args[i]);

	if(rc)
        {
            std::cerr << "Error: Return code from pthread_create on threadId: " 
                      << i << " is " << rc << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Thread Creation for Last One
    args[numThreads-1].setArgs( in.begin() + chunkSize * (numThreads - 1),
                               	in.end(), index, constraint);

    rc = pthread_create( &threads[numThreads-1], NULL, 
                         selDataThread, (void*)&args[numThreads-1]);

    if(rc)
    {
        std::cerr << "Error: Return code from pthread_create on threadId: " 
                  << numThreads-1 << " is " << rc << std::endl;
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < numThreads; i++)
    {
        rc = pthread_join(threads[i], NULL);
        if(rc)
        {
            std::cerr << "Error: Return code from pthread_create on threadId: " 
                      << i << " is " << rc << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    for(int i = 0; i < numThreads; i++)
    {
	/*	    if((args[i].tmp_result).size()!=0)
	    {
		// For Debug
		// std::cout << "find a match" << std::endl;
		for(auto it=(args[i].tmp_result).begin(); it!= (args[i].tmp_result).end(); it++)  
    		{
			out.push_back(*it);
		}
	    	
	    }
	*/
	    out.insert(out.end(), (args[i].tmp_result).begin(), (args[i].tmp_result).end()); 
    }

    delete[] threads;
    delete[] args;
}


static void* selDataThread(void* args)
{
	ThreadArg* arg = static_cast<ThreadArg*>(args);
	
	for (auto it=arg->beginIt; it!=arg->endIt; it++) 
	{
		if ((*it)[arg->index] == arg->constraint)	
		{
			
			// For Debug
			// std::cout << "thread find a match" << std::endl;
			(arg->tmp_result).push_back(*it);
		}
	}

    	pthread_exit(NULL);    
}
