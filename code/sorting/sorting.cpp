#include "sorting.h"

static void* sortDataThread(void* args);
static void* mergeDataThread(void* args);

class SortComparator
{
public:
    SortComparator(int idx)
    {
        index = idx;
    }
    bool operator() (const Record& rec1, const Record& rec2)
    {
        return rec1.at(index) < rec2.at(index);
    }
private:
    int index;
};

struct SortThreadArg
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

struct MergeThreadArg
{
public:
    Dataset::iterator beginIt1;
    Dataset::iterator endIt1;
    Dataset::iterator beginIt2;
    Dataset::iterator endIt2;
    int index;
public:
    void setArgs(Dataset::iterator bIt1, 
                 Dataset::iterator eIt1,
                 Dataset::iterator bIt2, 
                 Dataset::iterator eIt2,
                 int idx)
    {
        beginIt1 = bIt1;
        endIt1 = eIt1;
        beginIt2 = bIt2;
        endIt2 = eIt2;
        index = idx;
    }
};

void sortData(Dataset& out, Dataset& in, int index, int numThreads)
{
    out = in;

    pthread_t* threads = new pthread_t[numThreads];
    SortThreadArg* sArgs = new SortThreadArg[numThreads];
    MergeThreadArg* mArgs = new MergeThreadArg[numThreads];
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

    // Sort Phase
    for(int i=0; i<numThreads; i++)
    {
        sArgs[i].setArgs(out.begin() + beginIndex[i], 
                        out.begin() + endIndex[i], 
                        index);
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

    // Merge Phase
    int numChunks = numThreads >> 1;
    for(int stride = 1; stride < numThreads; stride <<= 1, numChunks >>= 1)
    {
        for(int i=0; i<numChunks; i++)
        {
            int idx1 = stride*(2*i);
            int idx2 = idx1 + stride - 1;
            int idx3 = idx2 + 1;
            int idx4 = idx3 + stride - 1;

            mArgs[i].setArgs(out.begin() + beginIndex[idx1], 
                             out.begin() + endIndex[idx2],
                             out.begin() + beginIndex[idx3],
                             out.begin() + endIndex[idx4],
                             index);
            rc = pthread_create(&threads[i], NULL, 
                                mergeDataThread, (void*)&mArgs[i]);
            if(rc)
            {
                std::cerr << "Error: Return code from pthread_create on threadId: " 
                          << i << " is " << rc << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        for(int i=0; i<numChunks; i++)
        {
            rc = pthread_join(threads[i], NULL);
            if(rc)
            {
                std::cerr << "Error: Return code from pthread_create on threadId: " 
                          << i << " is " << rc << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    delete[] beginIndex;
    delete[] endIndex;
    delete[] mArgs;
    delete[] sArgs;
    delete[] threads;
}

static void* sortDataThread(void* args)
{
    SortThreadArg* arg = static_cast<SortThreadArg*>(args);

    SortComparator compObj(arg->index);
    std::sort(arg->beginIt, arg->endIt, compObj);

    pthread_exit(NULL);
}

static void* mergeDataThread(void* args)
{
    MergeThreadArg* arg = static_cast<MergeThreadArg*>(args);

    SortComparator compObj(arg->index);
    std::inplace_merge(arg->beginIt1, arg->beginIt2, arg->endIt2, compObj);

    pthread_exit(NULL);
}
