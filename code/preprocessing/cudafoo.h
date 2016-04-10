#ifndef _CUDAFOO_H_
#define _CUDAFOO_H_

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/swap.h>

#define checkCudaErrors(a, b)             \
{                                         \
 cudaError_t _rc;                         \
 if((_rc=a)!=cudaSuccess)                 \
 {                                        \
    std::cerr << "Error in "              \
              << b << std::endl;          \
    std::cerr << "Reason: "               \
              << cudaGetErrorString(_rc)  \
              << std::endl;               \
    exit(1);                              \
 }                                        \
}

#define checkCudaErrorKernel(b)           \
 cudaDeviceSynchronize();                 \
 checkCudaErrors(cudaGetLastError(), b)

#endif
