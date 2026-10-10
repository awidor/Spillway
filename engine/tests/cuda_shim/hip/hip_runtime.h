#pragma once
// The HIP runtime names a shared kernel test uses, on CUDA (tests only: cuda_router_fast builds tests/hip/router_fast.cpp).
#include <cuda_runtime.h>
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipError_t cudaError_t
#define hipEventCreate cudaEventCreate
#define hipEventElapsedTime cudaEventElapsedTime
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipEvent_t cudaEvent_t
#define hipFree cudaFree
#define hipGetErrorString cudaGetErrorString
#define hipMalloc cudaMalloc
#define hipMemcpy cudaMemcpy
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemset cudaMemset
#define hipSuccess cudaSuccess
