#ifndef CURRAND_STATE_CUH
#define CURRAND_STATE_CUH
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void init_curand_states(curandState* state, unsigned long seed, int nBodies);

#endif // CURRAND_STATE_CUH