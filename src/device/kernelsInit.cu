#include <stdio.h>
#include <cuda_runtime.h>
#include "deviceProps.cuh"

void kernelsLaunchParamsInit(
    int &gridDimX, 
    int &blockDimX, 
    int &integrateStride, 
    DeviceProperties deviceProps) {

    //* Esto espera cambios en el futuro.
    gridDimX = 2*deviceProps.warpDim; //& deviceProps.numberOfSMs;
    blockDimX = deviceProps.warpDim * deviceProps.warpDim;
    integrateStride = gridDimX * blockDimX;

    printf("INFO - gridDimX: %d, blockDimX: %d Integration Stride: %d", gridDimX, blockDimX, integrateStride);
}