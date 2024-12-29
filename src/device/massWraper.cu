#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include "body.cuh"
#include "mass.cuh"
#include "cuda_utils.cuh"
#include "deviceProps.cuh"

void massKernelLaunch(
    int nBodies, 
    Body *p_device, 
    int gridDimX, 
    int blockDimX) {

    dim3 dimGrid(gridDimX, 1, 1);
    dim3 massDimBlock(blockDimX, 1, 1);
    initialize_mass<<<dimGrid, massDimBlock>>>(p_device, nBodies);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
