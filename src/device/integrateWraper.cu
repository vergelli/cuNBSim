#include <cuda_runtime.h>
#include <iostream>
#include "body.cuh"
#include "integrate.cuh"
#include "cuda_utils.cuh"
#include "deviceProps.cuh"

void execIntegrate(
    int nBodies, 
    float dt, 
    Body *p_device, 
    int gridDimX, 
    int integrateBlockDimX, 
    int stride) {

    dim3 dimGrid(gridDimX, 1, 1);
    dim3 IntegrationDimBlock(integrateBlockDimX, 1, 1);

    integrateCUDA<<<dimGrid, IntegrationDimBlock>>>(p_device, dt, nBodies, stride);

    CHECK_CUDA_ERROR(
        cudaDeviceSynchronize());

};
