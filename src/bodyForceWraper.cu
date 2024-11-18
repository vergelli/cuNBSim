#include <cuda_runtime.h>
#include <iostream>
#include "body.cuh"
#include "bodyForce.cuh"
#include "cuda_utils.cuh"

void bodyForceWraper( int nBodies, float dt, Body *p_device){

    int warpDim;
    int deviceId;
    int numberOfSMs;
    int gridDimX;
    int blockDimX;
    int blockDimY;
    int stride;

    CHECK_CUDA_ERROR( 
        cudaGetDevice(&deviceId));

    CHECK_CUDA_ERROR( 
        cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId));

    CHECK_CUDA_ERROR( 
        cudaDeviceGetAttribute(&warpDim, cudaDevAttrWarpSize, deviceId));


    gridDimX = warpDim*numberOfSMs;
    blockDimX =  sqrt(warpDim*warpDim); 
    blockDimY =  sqrt(warpDim*warpDim);
    stride = gridDimX * blockDimX;

    dim3 dimGrid(gridDimX, 1, 1);
    dim3 BodyForceDimBlock(blockDimX, blockDimY, 1);

    bodyForceCUDA<<<dimGrid, BodyForceDimBlock>>>(p_device, dt, nBodies, stride);

    CHECK_CUDA_ERROR(
        cudaDeviceSynchronize());

};
