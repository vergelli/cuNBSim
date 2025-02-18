#ifndef INTEGRATOR_LEAP_FROG_WRAPER_CUH
#define INTEGRATOR_LEAP_FROG_WRAPER_CUH
#include "body.cuh"
#include "deviceProps.cuh"

void execLeapFrogPositionUpdate(int nBodies, float dt, Body *p_device, int gridDimX, int blockDimX);

void execLeapFrogVelocityUpdate(int nBodies, float dt, Body *p_device, int gridDimX, int blockDimX);

#endif // INTEGRATOR_LEAP_FROG_WRAPER_CUH