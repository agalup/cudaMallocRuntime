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

    unsigned int mm_grid_size = 1;
    unsigned int app_grid_size = 1;
    int size = 1;
    int device = 0;

    CUexecAffinityParam_v1 mm_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, mm_grid_size};
    CUexecAffinityParam_v1 app_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, app_grid_size};
    auto affinity_flags = CUctx_flags::CU_CTX_SCHED_AUTO;

    CUcontext mm_ctx, app_ctx;
    GUARD_CU((cudaError_t)cuCtxCreate_v3(&mm_ctx, &mm_param, 1, affinity_flags, device));
    GUARD_CU((cudaError_t)cuCtxPopCurrent(&mm_ctx));

    GUARD_CU((cudaError_t)cuCtxCreate_v3(&app_ctx, &app_param, 1, affinity_flags, device));
    GUARD_CU((cudaError_t)cuCtxPopCurrent(&app_ctx));

    int** tab = NULL; GUARD_CU(cudaMalloc((void**)&tab, sizeof(int*)*size));
    
    GUARD_CU((cudaError_t)cuCtxPushCurrent(mm_ctx));
    cudaMallocRuntimeTest<<<1, size>>>(tab);
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU((cudaError_t)cuCtxSynchronize());
    GUARD_CU((cudaError_t)cuCtxPopCurrent(&mm_ctx));
    
    GUARD_CU((cudaError_t)cuCtxPushCurrent(mm_ctx));
    cudaWriteRuntimeTest<<<1, size>>>(tab);
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU((cudaError_t)cuCtxSynchronize());
    GUARD_CU((cudaError_t)cuCtxPopCurrent(&mm_ctx));
    
    GUARD_CU((cudaError_t)cuCtxPushCurrent(mm_ctx));
    cudaReadRuntimeTest<<<1, size>>>(tab);
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU((cudaError_t)cuCtxSynchronize());
    GUARD_CU((cudaError_t)cuCtxPopCurrent(&mm_ctx));
    
    GUARD_CU((cudaError_t)cuCtxPushCurrent(mm_ctx));
    cudaFreeRuntimeTest<<<1, size>>>(tab);
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU((cudaError_t)cuCtxSynchronize());
    GUARD_CU((cudaError_t)cuCtxPopCurrent(&mm_ctx));

    GUARD_CU((cudaError_t)cuCtxDestroy(mm_ctx));
    GUARD_CU((cudaError_t)cuCtxDestroy(app_ctx));

    GUARD_CU(cudaFree(tab));

    return 0;
}
