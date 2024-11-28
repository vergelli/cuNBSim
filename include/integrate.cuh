#ifndef INTEGRATE_CUH
#define INTEGRATE_CUH
#include "body.cuh"

__global__ void integrateCUDA(Body *p, float dt, int n, int stride);

#endif // INTEGRATE_CUH
