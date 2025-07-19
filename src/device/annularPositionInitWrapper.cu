// annularPositionInitWrapper.cu
#include <curand_kernel.h>
#include <cstdlib>
#include <iostream>
#include "cuda_utils.cuh"      // Para CHECK_CUDA_ERROR y otras utilidades
#include "body.cuh"            // Para la estructura Body
#include "annularDisk.cuh"      // Incluye las definiciones de los kernels (init_curand_states_annular y annular_disk_kernel)
#include "curandKernels.cuh"    // Para init_curand_states

// Función wrapper para inicializar posiciones en un disco anular
// Wrapper para la inicialización con z uniforme
void execAnnularDiskInitUniform(
    int nBodies,
    curandState* d_state,
    Body* p_device,
    int gridDimX,
    int blockDimX,
    float r1,
    float r2,
    float h
) {
    dim3 dimGrid(gridDimX, 1, 1);
    dim3 dimBlock(blockDimX, 1, 1);
    init_curand_states<<<dimGrid, dimBlock>>>(d_state, time(NULL), nBodies);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    annular_disk_kernel_uniform<<<dimGrid, dimBlock>>>(p_device, d_state, nBodies, r1, r2, h);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// Wrapper para la inicialización con z gaussiana "lomada interna"
void execAnnularDiskInitGaussianZIntern(
    int nBodies,
    curandState* d_state,
    Body* p_device,
    int gridDimX,
    int blockDimX,
    float r1,
    float r2,
    float h,
    float mu,    // media para z
    float sigma  // sigma base para z
) {
    dim3 dimGrid(gridDimX, 1, 1);
    dim3 dimBlock(blockDimX, 1, 1);
    init_curand_states<<<dimGrid, dimBlock>>>(d_state, time(NULL), nBodies);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    annular_disk_kernel_gaussianZ_intern<<<dimGrid, dimBlock>>>(p_device, d_state, nBodies, r1, r2, h, mu, sigma);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// Wrapper para la inicialización con z gaussiana "lomada externa"
void execAnnularDiskInitGaussianZExtern(
    int nBodies,
    curandState* d_state,
    Body* p_device,
    int gridDimX,
    int blockDimX,
    float r1,
    float r2,
    float h,
    float mu,    // media para z
    float sigma  // sigma base para z
) {
    dim3 dimGrid(gridDimX, 1, 1);
    dim3 dimBlock(blockDimX, 1, 1);
    init_curand_states<<<dimGrid, dimBlock>>>(d_state, time(NULL), nBodies);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    annular_disk_kernel_gaussianZ_extern<<<dimGrid, dimBlock>>>(p_device, d_state, nBodies, r1, r2, h, mu, sigma);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
