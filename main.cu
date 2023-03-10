#include <cstdio>
#include <cuda.h>

#define GUARD_CU(a) if (a != cudaSuccess) printf("[%s:%d] err(%d) %s\n", __FILE__, __LINE__, a, cudaGetErrorString(a));

__global__
void cudaMallocRuntimeTest(int** tab){
    int thid = (blockDim.x * blockIdx.x) + threadIdx.x;
    tab[thid] = (int*)malloc(sizeof(int));
}

__global__
void cudaWriteRuntimeTest(int** tab){
    int thid = (blockDim.x * blockIdx.x) + threadIdx.x;
    tab[thid][0] = 12345;
}

__global__
void cudaReadRuntimeTest(int** tab){
    int thid = (blockDim.x * blockIdx.x) + threadIdx.x;
    printf("%d\n", tab[thid][0]);
}

__global__
void cudaFreeRuntimeTest(int** tab){
    int thid = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    free(tab[thid]);
}

int main(int argn, char* arg[]){
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());

    int size = 1;

    int** tab = NULL; GUARD_CU(cudaMalloc((void**)&tab, sizeof(int*)*size));
    
    cudaMallocRuntimeTest<<<1, size>>>(tab);
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU((cudaError_t)cuCtxSynchronize());
    
    cudaWriteRuntimeTest<<<1, size>>>(tab);
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU((cudaError_t)cuCtxSynchronize());
    
    cudaReadRuntimeTest<<<1, size>>>(tab);
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU((cudaError_t)cuCtxSynchronize());
    
    cudaFreeRuntimeTest<<<1, size>>>(tab);
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU((cudaError_t)cuCtxSynchronize());

    GUARD_CU(cudaFree(tab));
    return 0;
}
