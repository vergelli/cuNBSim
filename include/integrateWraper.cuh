#ifndef INTEGRATE_WRAPPER_CUH
#define INTEGRATE_WRAPPER_CUH
#include "body.cuh"
#include "deviceProps.cuh"

void initIntegrate(int &gridDimX, int &integrateBlockDimX, int &integrateStride, DeviceProperties deviceProps);

void execIntegrate(int nBodies, float dt, Body *p_device, int gridDimX, int integrateBlockDimX, int stride);

#endif // INTEGRATE_WRAPPER_CUH
