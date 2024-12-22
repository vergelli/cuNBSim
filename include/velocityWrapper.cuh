#ifndef VELOCITY_WRAPPER_CUH
#define VELOCITY_WRAPPER_CUH
#include "body.cuh"
#include "deviceProps.cuh"

void initVelocityKernelLaunch(int &gridDimX, int &VelocityDimX, DeviceProperties deviceProps);
void velocityKernelLaunch(int nBodies, Body *p_device, int gridDimX, int blockDimX, float max_particles_speed);

#endif // VELOCITY_WRAPPER_CUH


