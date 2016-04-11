#include "sorting.cuh"
#include <sstream>

__device__
bool compFoo(Record* rec1, Record* rec2, int fieldIdx)
{
    switch(fieldIdx)
    {
        case 0:
            return rec1->userId < rec2->userId;
        case 1:
            return rec1->movieId < rec2->movieId;
        case 2:
            return rec1->rating < rec2->rating;
        case 3:
            return rec1->timestamp < rec2->timestamp;
        default:
            return rec1->rating < rec2->rating;
    }
}

__global__
void sortDataKernel(Record* out,
                    int fieldIdx,
                    size_t size)
{
    int blockId = blockIdx.x
                  + blockIdx.y*gridDim.x
                  + blockIdx.z*gridDim.y*gridDim.x;
    int threadId = threadIdx.x
                   + threadIdx.y*blockDim.x
                   + threadIdx.z*blockDim.x*blockDim.y
                   + blockId*blockDim.x*blockDim.y*blockDim.z;
    int numThreads = blockDim.x*blockDim.y*blockDim.z
                        *gridDim.x*gridDim.y*gridDim.z;

    int numComp = ((size>>1) + numThreads-1)/numThreads;
    
    for(int ostep = 2; ostep <= size; ostep <<= 1)
    {
        int halfStep = ostep >> 1;
        for(int istep = ostep; istep > 1; istep >>= 1)
        {
            int stride = istep >> 1;
            for(int i=0; i<numComp; i++)
            {
                int compId = threadId*numComp + i;
                if(compId >= size>>1)
                {
                    break;
                }
                int idx1 = (compId/stride)*istep + (compId%stride);
                int idx2 = idx1 + stride;
                Record* it1 = out + idx1;
                Record* it2 = out + idx2;
                bool dir = (compId%ostep) < halfStep;
                if(dir)
                {
                    if(compFoo(it2, it1, fieldIdx))
                    {
                        Record temp = *it1;
                        *it1 = *it2;
                        *it2 = temp;
                    }
                }
                else
                {
                    if(compFoo(it1, it2, fieldIdx))
                    {
                        Record temp = *it1;
                        *it1 = *it2;
                        *it2 = temp;
                    }
                }
            }
            __syncthreads();
        }
    }
}

void sortData(Dataset& out, Dataset& in, int index, int numThreads)
{
    Dataset_d out_d = in;
    size_t size = in.size();

    dim3 block(numThreads,1,1);

    Record* out_begin = thrust::raw_pointer_cast(out_d.data());

    sortDataKernel<<<1,block>>>(out_begin, index, size);
    checkCudaErrorKernel("sortDataKernel");

    out.resize(size);
    thrust::copy(out_d.begin(), out_d.end(), out.begin());
}

