#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include "body.cuh"
#include "bodyForce.cuh"
#include "cuda_utils.cuh"
#include "deviceProps.cuh"

void initBodyForce(
    int &gridDimX, 
    int &bodyForceBlockDimX, 
    DeviceProperties deviceProps) {

    gridDimX = deviceProps.warpDim * deviceProps.numberOfSMs;
    bodyForceBlockDimX = deviceProps.warpDim * deviceProps.warpDim;
}

void execBodyForce(
    int nBodies, 
    float dt, 
    Body *p_device, 
    int gridDimX, 
    int blockDimX) {

    dim3 dimGrid(gridDimX, 1, 1);
    dim3 BodyForceDimBlock(blockDimX, 1, 1);
    bodyForceCUDA<<<dimGrid, BodyForceDimBlock>>>(p_device, dt, nBodies);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}



























void bodyForceWraper( int nBodies, float dt, Body *p_device){

    int warpDim;
    int deviceId;
    int numberOfSMs;
    int gridDimX;
    int blockDimX;

    CHECK_CUDA_ERROR( 
        cudaGetDevice(&deviceId));

    CHECK_CUDA_ERROR( 
        cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId));

    CHECK_CUDA_ERROR( 
        cudaDeviceGetAttribute(&warpDim, cudaDevAttrWarpSize, deviceId));

    gridDimX = warpDim * numberOfSMs;

    blockDimX = warpDim * warpDim; 

    //*    stride = gridDimX * blockDimX;

    dim3 dimGrid(gridDimX, 1, 1);

    dim3 BodyForceDimBlock(blockDimX, 1, 1);

    // Llamada al kernel con los valores calculados
    bodyForceCUDA<<<dimGrid, BodyForceDimBlock>>>(p_device, dt, nBodies);

    CHECK_CUDA_ERROR(
        cudaDeviceSynchronize());

};
