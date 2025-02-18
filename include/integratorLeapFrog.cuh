#ifndef INTEGRATOR_LEAP_FROG_CUH
#define INTEGRATOR_LEAP_FROG_CUH
#include "body.cuh"
#include "deviceProps.cuh"

__global__ void leapFrogPositionCUDA(Body *p_device, float dt, int nBodies);

__global__ void leapFrogVelocityCUDA(Body *p_device, float dt, int nBodies);

#endif // INTEGRATOR_LEAP_FROG_CUH