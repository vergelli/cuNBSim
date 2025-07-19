#ifndef BODYFORCEWRAPER_CUH
#define BODYFORCEWRAPER_CUH
#include "body.cuh"
#include "deviceProps.cuh"
#include "centralBody.cuh"

void initBodyForce(int &gridDimX, int &bodyForceBlockDimX, DeviceProperties deviceProps);

void execBodyForce(
    int nBodies, 
    float dt, 
    Body* p_device, 
    int gridDimX, 
    int blockDimX,
    std::string simulation_type,
    const CentralBody* central_device = nullptr) ;

#endif // BODYFORCEWRAPER_CUH
