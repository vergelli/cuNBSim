#ifndef BODYFORCE_CUH
#define BODYFORCE_CUH
#include "body.cuh"

#define SOFTENING 1e-9f

__global__ void bodyForceCUDA(Body *p, float dt, int nBodies);

#endif // INTEGRATE_CUH
