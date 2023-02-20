#include <cstdio>
#include "cuda.h"

#define GUARD_CU(a) if (a != 0) printf("[%s:%d] err %d\n", __FILE__, __LINE__, a);

__global__
void cudaMallocRuntimeTest(int size){
    int thid = (blockDim.x * blockIdx.x) + threadIdx.x;
    int* tab;
    GUARD_CU(cudaMalloc((void**)&tab, size));
    tab[thid] = thid;
}


int main(int argn, char* arg[]){
    int size_to_alloc = 10;
    cudaMallocRuntimeTest<<<1, 2>>>(size_to_alloc);
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());
    return 0;
}
