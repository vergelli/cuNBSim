// leapFrogPositionUpdateWrapper.cu
#include <cuda_runtime.h>
#include "body.cuh"
#include "cuda_utils.cuh"
#include "integratorLeapFrog.cuh"


void execLeapFrogPositionUpdate(int nBodies, float dt, Body *p_device, int gridDimX, int blockDimX) {
    dim3 dimGrid(gridDimX, 1, 1);
    dim3 dimBlock(blockDimX, 1, 1);
    leapFrogPositionCUDA<<<dimGrid, dimBlock>>>(p_device, dt, nBodies);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void execLeapFrogVelocityUpdate(int nBodies, float dt, Body *p_device, int gridDimX, int blockDimX) {
    dim3 dimGrid(gridDimX, 1, 1);
    dim3 dimBlock(blockDimX, 1, 1);
    leapFrogVelocityCUDA<<<dimGrid, dimBlock>>>(p_device, dt, nBodies);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
