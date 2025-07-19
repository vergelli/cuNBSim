#ifndef ANNULARDISKINIT_DISPATCHER_CUH
#define ANNULARDISKINIT_DISPATCHER_CUH
#include "body.cuh"
#include <curand_kernel.h>
#include "deviceProps.cuh"

enum AnnularDiskInitType {
    UNIFORM_Z = 0,
    GAUSSIAN_Z_INTERN = 1,
    GAUSSIAN_Z_EXTERN = 2
};

void execAnnularDiskInitDispatcher(
    int nBodies,
    curandState* d_state,
    Body* p_device,
    int gridDimX,
    int blockDimX,
    float r1,
    float r2,
    float h,
    AnnularDiskInitType initType,
    float mu,
    float sigma);

#endif // ANNULARDISKINIT_DISPATCHER_CUH