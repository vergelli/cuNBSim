#ifndef BOXMULLER_CUH
#define BOXMULLER_CUH
#include <curand_kernel.h>
#include "body.cuh"

__global__ void box_muller_kernel(Body* p_device, curandState* state, int nBodies);

#endif // BOXMULLER_CUH




