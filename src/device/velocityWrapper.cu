#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include "body.cuh"
#include "velocity.cuh"
#include "cuda_utils.cuh"
#include "deviceProps.cuh"

void velocityKernelLaunch(
    int nBodies, 
    Body *p_device, 
    int gridDimX, 
    int blockDimX,
    float max_particles_speed) {

    dim3 dimGrid(gridDimX, 1, 1);
    dim3 VelocityDimBlock(blockDimX, 1, 1);
    initialize_velocity<<<dimGrid, VelocityDimBlock>>>(p_device, nBodies, max_particles_speed);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
