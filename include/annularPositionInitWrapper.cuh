#ifndef ANNULARDISKINIT_CUH
#define ANNULARDISKINIT_CUH
#include "body.cuh"
#include <curand_kernel.h>
#include "deviceProps.cuh"

void execAnnularDiskInitUniform(
    int nBodies,
    curandState* d_state,
    Body* p_device,
    int gridDimX,
    int blockDimX,
    float r1,
    float r2,
    float h
);

void execAnnularDiskInitGaussianZIntern(
    int nBodies,
    curandState* d_state,
    Body* p_device,
    int gridDimX,
    int blockDimX,
    float r1,
    float r2,
    float h,
    float mu,    // media para z
    float sigma  // sigma base para z
);

void execAnnularDiskInitGaussianZExtern(
    int nBodies,
    curandState* d_state,
    Body* p_device,
    int gridDimX,
    int blockDimX,
    float r1,
    float r2,
    float h,
    float mu,    // media para z
    float sigma  // sigma base para z
);

#endif // ANNULARDISKINIT_CUH
