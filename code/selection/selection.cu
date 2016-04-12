#include "selection.cuh"

__global__
void selDataKernel( Record* in,
                    int beginIndex[],
		    int endIndex[],
		    int *cnt_d,
                    double constraint)
{
	size_t blockId = blockIdx.x
 	               + blockIdx.y*gridDim.x
       	               + blockIdx.z*gridDim.y*gridDim.x;

    	size_t threadId = threadIdx.x
                        + threadIdx.y*blockDim.x
                        + threadIdx.z*blockDim.x*blockDim.y
                        + blockId*blockDim.x*blockDim.y*blockDim.z;
/*
    	size_t numThreads = blockDim.x*blockDim.y*blockDim.z*
                            gridDim.x*gridDim.y*gridDim.z;
*/
	int idx1 = beginIndex[threadId];
	int idx2 = endIndex[threadId]; 

	Record* bIt = in + idx1;
       	Record* eIt = in + idx2;	
	Record* selIt = bIt;

  	for (Record* it = bIt; it != eIt; it++) {
		if (it->rating == constraint) {
			*selIt = *it;
			selIt++;
			cnt_d[threadId] = cnt_d[threadId] + 1;
		}
	}
    
}

void selData(Dataset& out, Dataset& in, double constraint, int numThreads)
{
	out.clear();
    	Dataset_d in_d = in;
	Dataset out_tmp;

    	size_t size = in.size();	
	size_t o_size;

	int  *cnt, *cnt_d;
	int k = 0;

	cnt = (int*)malloc(numThreads*sizeof(int));
	cudaMalloc(&cnt_d, numThreads*sizeof(int));
	cudaMemset(cnt_d, 0, numThreads*sizeof(int));

    	int beginIndex[numThreads];
    	int endIndex[numThreads];

    	dim3 block(1024,1,1);
	int numBlocks = (numThreads+1023)/1024;
    	dim3 grid(numBlocks,1,1);

    	int chunkSize = size/numThreads;
	
    	for(int i=0; i < numThreads - 1; i++) {
	    beginIndex[i] = chunkSize * i;
	    endIndex[i] = chunkSize * (i+1);
	}
	beginIndex[numThreads-1] = chunkSize*(numThreads-1);
	endIndex[numThreads-1] = size;

    	Record* in_begin = thrust::raw_pointer_cast(in_d.data());

  	selDataKernel<<<grid,block>>>(in_begin, beginIndex, endIndex, cnt_d, constraint);
       	checkCudaErrorKernel("selDataKernel");

	out_tmp.resize(in.size());
    	thrust::copy(in_d.begin(), in_d.end(), out_tmp.begin());
	cudaMemcpy(cnt, cnt_d, numThreads*sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0; i < numThreads; i++) {
		o_size = o_size + cnt[i];
	}

	out.resize(o_size);

	for(int i = 0; i < numThreads; i++) {
		for(int j = 0; j < cnt[i]; j++) { 
			out[k++] = out_tmp[chunkSize*i+j];
		}
	}
}
