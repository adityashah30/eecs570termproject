#include "selection.cuh"

__global__
void selDataKernel(Record* out,
                   int* cnt_d,
                   double ratingVal,
                   size_t size)
{
    size_t blockId = blockIdx.x
                   + blockIdx.y*gridDim.x
                   + blockIdx.z*gridDim.y*gridDim.x;

    size_t threadId = threadIdx.x
                    + threadIdx.y*blockDim.x
                    + threadIdx.z*blockDim.x*blockDim.y
                    + blockId*blockDim.x*blockDim.y*blockDim.z;

    size_t numThreads = blockDim.x*blockDim.y*blockDim.z
                        *gridDim.x*gridDim.y*gridDim.z;

    if(threadId < numThreads)
    {
        int chunkSize = (size + numThreads-1)/numThreads;
        int idx1 = threadId*chunkSize;
        int idx2 = (threadId+1)*chunkSize;

        int cnt = 0;

        if(threadId == numThreads-1)
        {
            idx2 = size;
        }

        Record* selIt = out + idx1;

        for (int i=idx1; i<idx2; i++)
        {
            if (out[i].rating == ratingVal)
            {
                *selIt = out[i];
                selIt++; 
                cnt++;
            }
        }
        cnt_d[threadId] = cnt;
    }
}

void selData(Dataset& out, Dataset& in, double ratingVal, int numThreads)
{
    Dataset_d out_d = in;
    size_t size = in.size();

    Dataset out_temp(size);

    vecint_d cnt_d(numThreads);
    std::vector<int> cnt(numThreads);
    int chunkSize = (size + numThreads-1)/numThreads;

    int tpb = (numThreads<1024)?numThreads:1024;
    dim3 block(tpb,1,1);
    int numBlocks = (numThreads+tpb-1)/tpb;
    dim3 grid(numBlocks,1,1);
 
    Record* out_begin = thrust::raw_pointer_cast(out_d.data());
    int* cnt_begin = thrust::raw_pointer_cast(cnt_d.data());

    selDataKernel<<<grid,block>>>(out_begin, cnt_begin, ratingVal, size);
    checkCudaErrorKernel("selDataKernel");

    thrust::copy(out_d.begin(), out_d.end(), out_temp.begin());
    thrust::copy(cnt_d.begin(), cnt_d.end(), cnt.begin());

    size_t osize = std::accumulate(cnt.begin(), cnt.end(), 0);
    out.resize(osize);

    Dataset::iterator out_it = out.begin();

    for(int i=0; i<numThreads; i++)
    {
        Dataset::iterator out_temp_it = out_temp.begin() + i*chunkSize;
        std::copy(out_temp_it, out_temp_it+cnt[i], out_it);
        out_it += cnt[i];
    }
}
