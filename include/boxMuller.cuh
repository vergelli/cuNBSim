#ifndef BOXMULLER_CUH
#define BOXMULLER_CUH
#include <curand_kernel.h>
#include "body.cuh"

#define M_PI 3.14159265358979323846
#define SIGMA_X 1.0f
#define SIGMA_Y 1.0f
#define SIGMA_Z 0.05f  // Este valor aplana la distribución en el eje z

__global__ void init_curand_states(curandState* state, unsigned long seed, int nBodies);

__global__ void box_muller_kernel(Body* p_device, curandState* state, int nBodies);

#endif // BOXMULLER_CUH




