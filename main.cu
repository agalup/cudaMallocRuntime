#include <cstdio>
#include "cuda.h"
#include "cuda_device_runtime_api.h"
#include <cassert>

#define GUARD_CU(a) if (a != 0) printf("[%s:%d] err(%d) %s\n", __FILE__, __LINE__, a, cudaGetErrorString(a));

__global__
void cudaMallocRuntimeTest(size_t size){
    int thid = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    int* tab = NULL;
    GUARD_CU(cudaMalloc((void**)&tab, size*sizeof(int)));
    assert(tab);
    for (size_t i = 0; i<size; ++i)
        tab[i] = thid;

    GUARD_CU(cudaFree(tab));
}

int main(int argn, char* arg[]){
    GUARD_CU(cudaDeviceReset());

    int* dev_tmp;
//    GUARD_CU(cudaMalloc((void**)&dev_tmp, sizeof(int)));

    //https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=cudalimitmallocheapsize#heap-memory-allocation
    size_t size_to_alloc = 1024*7;
    size_t size = 0u;
    //default 8MiB
    GUARD_CU(cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize));
    printf("cudaLimitMallocHeapSize: %lu\n", size);
  
    //aligned to MiB, not less than 4MiB
    size = (size_t)7*1024*1024*1024; //7GiB
    //size = (size_t)5*1024*1024 - 500; //5MiB-500 does not work
    //size = (size_t)3*1024*1024; //3MiB does not work
    printf("set limit to %lu\n", size);
    GUARD_CU(cudaDeviceSetLimit(cudaLimitMallocHeapSize, size));
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    
    size = 0;
    GUARD_CU(cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize));
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    printf("new cudaLimitMallocHeapSize: %lu\n", size);
    
    size_t bl = 36;
    size_t th = 1024;
    size_t total_size = bl*th*size_to_alloc*sizeof(int);
    printf("total to allocate: %lu\n", total_size);
   
    cudaMallocRuntimeTest<<<dim3(bl), dim3(th)>>>(size_to_alloc);
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());

    return 0;
}
