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
    int beginIdx;
    int endIdx;
    int index;
    Dataset::iterator out;
    size_t size;
public:
    void setArgs(Dataset::iterator o, size_t s,
                 int bIdx, int eIdx, int idx)
    {
        out = o;
        size = s;
        beginIdx = bIdx;
        endIdx = eIdx;
        index = idx;
    }
};

void sortData(Dataset& out, Dataset& in, int index, int numThreads)
{
    out = in;

    pthread_t* threads = new pthread_t[numThreads];
    SortThreadArg* sArgs = new SortThreadArg[numThreads];
    int* beginIndex = new int[numThreads];
    int* endIndex = new int[numThreads];

    int chunkSize = out.size()/numThreads;
    int rc = 0;

    for(int i=0; i<numThreads-1; i++)
    {
        beginIndex[i] = chunkSize*i;
        endIndex[i] = chunkSize*(i+1);
    }
    beginIndex[numThreads-1] = chunkSize*(numThreads-1);
    endIndex[numThreads-1] = out.size();

    if(pthread_barrier_init(&barr, NULL, numThreads))
    {
        std::cerr << "Could not create a barrier" << std::endl;
        exit(EXIT_FAILURE);
    }

    for(int i=0; i<numThreads; i++)
    {
        sArgs[i].setArgs(out.begin(), out.size(),
                         beginIndex[i], endIndex[i], index);
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

    delete[] beginIndex;
    delete[] endIndex;
    delete[] sArgs;
    delete[] threads;
}

static void* sortDataThread(void* args)
{
    SortThreadArg* arg = static_cast<SortThreadArg*>(args);

    SortComparator compObj(arg->index);

    for(int k=2; k<=arg->size; k<<=1)
    {
        for (int j=k>>1; j>0; j=j>>1)
        {
            for (int i=arg->beginIdx; i<arg->endIdx; i++)
            {
                int mask=i^j;
                if (mask>i)
                {
                    Dataset::iterator it1 = arg->out + i;
                    Dataset::iterator it2 = arg->out + mask;
                    if((i&k)==0 && compObj(*it2, *it1))
                    {
                        std::iter_swap(it1, it2);
                    }
                    if((i&k)!=0 && compObj(*it1, *it2))
                    {
                        std::iter_swap(it1, it2);
                    }
                }
            }
            int rc = pthread_barrier_wait(&barr);
            if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
            {
                printf("Could not wait on barrier\n");
                exit(-1);
            }
        }
    }

    pthread_exit(NULL);
}
