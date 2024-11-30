#ifndef BOXMULLER_CUH
#define BOXMULLER_CUH
#include <curand_kernel.h>
#include "body.cuh"

#define M_PI 3.14159265358979323846

__global__ void init_curand_states(curandState* state, unsigned long seed, int nBodies);

__global__ void box_muller_kernel(Body* p_device, curandState* state, int nBodies);

#endif // BOXMULLER_CUH




