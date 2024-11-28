#ifndef BODYFORCEWRAPER_CUH
#define BODYFORCEWRAPER_CUH
#include "body.cuh"
#include "deviceProps.cuh"

void initBodyForce(int &gridDimX, int &bodyForceBlockDimX, DeviceProperties deviceProps);

void execBodyForce(int nBodies, float dt, Body *p_device, int gridDimX, int blockDimX);

#endif // BODYFORCEWRAPER_CUH
