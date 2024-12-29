#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include "body.cuh"
#include "boxMuller.cuh"
#include "boxMullerWraper.cuh"
#include "cuda_utils.cuh"
#include "deviceProps.cuh"

void execBoxMuller(
    int nBodies, 
    curandState* d_state,
    Body* p_device,
    int gridDimX, 
    int blockDimX) {

    dim3 dimGrid(gridDimX, 1, 1);
    dim3 boxMullerDimBlock(blockDimX, 1, 1);

    init_curand_states<<<dimGrid, boxMullerDimBlock>>>(d_state, time(NULL), nBodies);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    box_muller_kernel<<<dimGrid, boxMullerDimBlock>>>(p_device, d_state, nBodies);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
