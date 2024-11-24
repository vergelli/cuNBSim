#include <cuda_runtime.h>
#include "body.cuh"
#include "cuda_utils.cuh"

void bodyForceMalloc(int bytes, Body *p, Body *&p_device) {
    CHECK_CUDA_ERROR(cudaMalloc(&p_device, bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(p_device, p, bytes, cudaMemcpyHostToDevice));
}

void cudaFreeMemRoutines(Body *p_device){
    cudaFree(p_device);
}

