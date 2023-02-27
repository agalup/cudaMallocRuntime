#include <cstdio>
#include <cassert>

#define GUARD_CU(a) if (a != 0) printf("[%s:%d] err(%d) %s\n", __FILE__, __LINE__, a, cudaGetErrorString(a));

__global__
void cudaMallocRuntimeTest(){
    int thid = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    int* tab = NULL;
    GUARD_CU(cudaMalloc((void**)&tab, sizeof(int)));
    assert(tab);
    *tab = thid;
    printf("%d\n", *tab);

    GUARD_CU(cudaFree(tab));
}

int main(int argn, char* arg[]){

    cudaMallocRuntimeTest<<<1, 32>>>();
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());

    return 0;
}
