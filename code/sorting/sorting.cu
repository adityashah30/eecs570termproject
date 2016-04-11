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

// __global__
// void sortDataKernel(Record* out,
//                     int ostep, int istep,
//                     int fieldIdx,
//                     size_t size,
//                     size_t numComp)
// {
//     size_t blockId = blockIdx.x
//                   + blockIdx.y*gridDim.x
//                   + blockIdx.z*gridDim.y*gridDim.x;
//     size_t threadId = threadIdx.x
//                    + threadIdx.y*blockDim.x
//                    + threadIdx.z*blockDim.x*blockDim.y
//                    + blockId*blockDim.x*blockDim.y*blockDim.z;
//     // int blockId = blockIdx.x;
//     // int threadId = blockIdx.x*blockDim.x + threadIdx.x;

//     // size_t numThreads = blockDim.x*blockDim.y*blockDim.z*
//     //                     gridDim.x*gridDim.y*gridDim.z;

//     // size_t numComp = (size+2*numThreads-1)/(2*numThreads);

//     int halfStep = ostep >> 1;
//     int stride = istep >> 1;
    
//     for(int i=0; i<numComp; i++)
//     {
//         int compId = threadId*numComp + i;

//         if(compId >= size>>1)
//         {
//             break;
//         }

//         int idx1 = (compId/stride)*istep + (compId%stride);
//         int idx2 = idx1 + stride;
//         Record* it1 = out + idx1;
//         Record* it2 = out + idx2;

//         printf("%d %d; %d %d %d; %x %x\n", blockId, threadId, compId, idx1, idx2, it1, it2);

//         bool dir = (compId%ostep) < halfStep;
//         if(dir)
//         {
//             if(compFoo(it2, it1, fieldIdx))
//             {
//                 thrust::swap(*it1, *it2);
//             }
//         }
//         else
//         {
//             if(compFoo(it1, it2, fieldIdx))
//             {
//                 thrust::swap(*it1, *it2);
//             }
//         }
//     }
// }

// void sortData(Dataset& out, Dataset& in, int index, int numThreads)
// {
//     Dataset_d out_d = in;
//     // checkCudaErrorKernel("Allocating device_vector");
//     size_t size = in.size();

//     dim3 block(64,1,1);
//     dim3 grid(1,1,1);

//     size_t numThreads1 = block.x*block.y*block.z*grid.x*grid.y*grid.z;

//     size_t numComp = (size+2*numThreads1-1)/(2*numThreads1);

//     std::cout << size << " " << numThreads1 << " " << numComp << std::endl;

//     Record* out_begin = thrust::raw_pointer_cast(out_d.data());

//     for(int ostep = 2; ostep <= size; ostep <<= 1)
//     {
//         for(int istep = ostep; istep > 1; istep >>= 1)
//         {
//             sortDataKernel<<<grid,block>>>(out_begin, ostep, istep, index, size, numComp);
//             std::stringstream s;
//             s << "sortDataKernel " << ostep << ", " << istep;
//             checkCudaErrorKernel(s.str());
//         }
//     }

//     out.resize(in.size());
//     thrust::copy(out_d.begin(), out_d.end(), out.begin());
//     checkCudaErrorKernel("Copying from deivce to host");
// }
