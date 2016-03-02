#include "sorting.h"

static void* sortDataThread(void* args);

class SortComparator
{
public:
    SortComparator(int idx)
    {
        index = idx;
    }
    bool operator() (Record& rec1, Record& rec2)
    {
        return rec1.at(index) < rec2.at(index);
    }
private:
    int index;
};

struct ThreadArg
{
public:
    Dataset::iterator beginIt;
    Dataset::iterator endIt;
    int index;
public:
    void setArgs(Dataset::iterator bIt, 
                 Dataset::iterator eIt, int idx)
    {
        beginIt = bIt;
        endIt = eIt;
        index = idx;
    }
};

void sortData(Dataset& out, Dataset& in, int index, int numThreads)
{
    out = in;

    pthread_t* threads = new pthread_t[numThreads];
    ThreadArg* args = new ThreadArg[numThreads];

    int chunkSize = out.size()/numThreads;
    int rc = 0;

    for(int i=0; i<numThreads-1; i++)
    {
        args[i].setArgs(out.begin() + chunkSize*i, 
                        out.begin() + (chunkSize+1)*i, 
                        index);
        rc = pthread_create(&threads[i], NULL, 
                            sortDataThread, (void*)&args[i]);
        if(rc)
        {
            std::cerr << "Error: Return code from pthread_create on threadId: " 
                      << i << " is " << rc << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    args[numThreads-1].setArgs(out.begin() + chunkSize*(numThreads-1),
                               out.end(), index);
    rc = pthread_create(&threads[numThreads-1], NULL, 
                        sortDataThread, (void*)&args[numThreads-1]);
    if(rc)
    {
        std::cerr << "Error: Return code from pthread_create on threadId: " 
                  << numThreads-1 << " is " << rc << std::endl;
        exit(EXIT_FAILURE);
    }

    for(int i=0; i<numThreads; i++)
    {
        rc = pthread_join(threads[i], NULL);
        if(rc)
        {
            std::cerr << "Error: Return code from pthread_create on threadId: " 
                      << i << " is " << rc << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    delete[] threads;
    delete[] args;
}

static void* sortDataThread(void* args)
{
    ThreadArg* arg = static_cast<ThreadArg*>(args);

    // int index = arg->index;
    // Dataset::iterator beginIt = arg->beginIt;
    // Dataset::iterator endIt = arg->endIt;

    SortComparator compObj(arg->index);
    std::sort(arg->beginIt, arg->endIt, compObj);

    pthread_exit(NULL);    
}
