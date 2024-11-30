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

    gridDimX = 2*deviceProps.warpDim; //& deviceProps.numberOfSMs;
    bodyForceBlockDimX = deviceProps.warpDim * deviceProps.warpDim;
    printf("INFO - gridDimX: %d, bodyForceBlockDimX: %d\n", gridDimX, bodyForceBlockDimX);
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
