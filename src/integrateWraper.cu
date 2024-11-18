#include <cuda_runtime.h>
#include <iostream>
#include "body.cuh"
#include "integrate.cuh"
#include "cuda_utils.cuh"

void integrateWraper( int nBodies, float dt, Body *p_device){

    int warpDim;
    int deviceId;
    int numberOfSMs;
    int gridDimX;
    int blockDimX;
    int stride;

    CHECK_CUDA_ERROR( 
        cudaGetDevice(&deviceId));

    CHECK_CUDA_ERROR( 
        cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId));

    CHECK_CUDA_ERROR( 
        cudaDeviceGetAttribute(&warpDim, cudaDevAttrWarpSize, deviceId));


    gridDimX = warpDim*numberOfSMs;
    blockDimX =  sqrt(warpDim*warpDim); 
    stride = gridDimX * blockDimX;

    dim3 dimGrid(gridDimX, 1, 1);
    dim3 IntegrationDimBlock(blockDimX, 1, 1);

    integrateCUDA<<<dimGrid, IntegrationDimBlock>>>(p_device, dt, nBodies, stride);

    CHECK_CUDA_ERROR(
        cudaDeviceSynchronize());

};
