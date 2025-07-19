// annularDisk.cu
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Kernel para inicializar los estados de CURAND
/**
 * Kernel to initialize CURAND states.
 * 
 * @param state Pointer to the array of CURAND states.
 * @param seed Seed for random number generation.
 * @param nBodies Number of bodies (threads) to initialize.
 */
__global__ void init_curand_states(curandState* state, unsigned long seed, int nBodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nBodies) {
        curand_init(seed, i, 0, &state[i]);
    }
}

