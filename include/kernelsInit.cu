#ifndef KERNELLAUNCHPARAMSINIT_CUH
#define KERNELLAUNCHPARAMSINIT_CUH
#include "deviceProps.cuh"

void kernelsLaunchParamsInit(int &gridDimX, int &BlockDimX, int &integrateStride, DeviceProperties deviceProps);

#endif // KERNELLAUNCHPARAMSINIT_CUH


