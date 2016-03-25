#include "sorting.h"

static void* sortDataThread(void* args);
static void* mergeDataThread(void* args);

static pthread_barrier_t barr;

class SortComparator
{
public:
    SortComparator(int idx)
    {
        index = idx;
    }
    bool operator() (const Record& rec1, const Record& rec2)
    {
        switch(index)
        {
            case 0:
                return rec1.userId < rec2.userId;
            case 1:
                return rec1.movieId < rec2.movieId;
            case 2:
                return rec1.rating < rec2.rating;
            case 3:
                return rec1.timestamp < rec2.timestamp;
            default:
                return rec1.rating < rec2.rating;
        }

    }
private:
    int index;
};

struct SortThreadArg
{
public:
    Dataset::iterator out;
    size_t size;
    int numThreads;
    int threadId;
    int fieldIdx;
public:
    void setArgs(Dataset::iterator o, size_t s,
                 int nThreads, int tId, int fIdx)
    {
        out = o;
        size = s;
        numThreads = nThreads;
        threadId = tId;
        fieldIdx = fIdx;
    }
};

void sortData(Dataset& out, Dataset& in, int index, int numThreads)
{
    out = in;

    pthread_t* threads = new pthread_t[numThreads];
    SortThreadArg* sArgs = new SortThreadArg[numThreads];

    int rc = 0;

    if(pthread_barrier_init(&barr, NULL, numThreads))
    {
        std::cerr << "Could not create a barrier" << std::endl;
        exit(EXIT_FAILURE);
    }

    for(int i=0; i<numThreads; i++)
    {
        sArgs[i].setArgs(out.begin(), out.size(),
                         numThreads, i, index);
        rc = pthread_create(&threads[i], NULL, 
                            sortDataThread, (void*)&sArgs[i]);
        if(rc)
        {
            std::cerr << "Error: Return code from pthread_create on threadId: " 
                      << i << " is " << rc << std::endl;
            exit(EXIT_FAILURE);
        }
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

    delete[] sArgs;
    delete[] threads;
}

static void* sortDataThread(void* args)
{
    SortThreadArg* arg = static_cast<SortThreadArg*>(args);

    Dataset::iterator out = arg->out;
    size_t size = arg->size;
    int numThreads = arg->numThreads;
    int threadId = arg->threadId;
    int fieldIdx = arg->fieldIdx;

    SortComparator compObj(fieldIdx);

    int numComp = size/(2*numThreads);

    for(int ostep = 2; ostep <= size; ostep <<= 1)
    {
        int halfStep = ostep >> 1;
        for(int istep = ostep; istep > 1; istep >>= 1)
        {
            int stride = istep >> 1;
            for(int i=0; i<numComp; i++)
            {
                int compId = threadId*numComp + i;
                int idx1 = (compId/stride)*istep + (compId%stride);
                int idx2 = idx1 + stride;
                Dataset::iterator it1 = out + idx1;
                Dataset::iterator it2 = out + idx2;
                bool dir = (compId%ostep) < halfStep;
                if(dir)
                {
                    if(compObj(*it2, *it1))
                    {
                        std::iter_swap(it1, it2);
                    }
                }
                else
                {
                    if(compObj(*it1, *it2))
                    {
                        std::iter_swap(it1, it2);
                    }
                }
            }
            
            int rc = pthread_barrier_wait(&barr);
            if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
            {
                std::cerr << "Could not wait on barrier" << std::endl;
                exit(EXIT_FAILURE);
            }

        }
    }

    pthread_exit(NULL);
}
