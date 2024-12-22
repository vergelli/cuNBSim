#ifndef MASS_WRAPPER_CUH
#define MASS_WRAPPER_CUH
#include "body.cuh"
#include "deviceProps.cuh"

void initMassKernelLaunch(int &gridDimX, int &massDimX, DeviceProperties deviceProps);
void massKernelLaunch(int nBodies, Body *p_device, int gridDimX, int blockDimX);

#endif // MASS_WRAPPER_CUH


