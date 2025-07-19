
#ifndef ANULAR_DISK_CUH
#define ANULAR_DISK_CUH
#include "body.cuh"
#include <curand_kernel.h>

__global__ void annular_disk_kernel_uniform(Body* p_device, curandState* state, int nBodies, float r1, float r2, float h);

__global__ void annular_disk_kernel_gaussianZ_intern(Body* p_device, curandState* state, int nBodies, 
                                                        float r1, float r2, float h, float mu, float sigma);

__global__ void annular_disk_kernel_gaussianZ_extern(Body* p_device, curandState* state, int nBodies, 
                                                        float r1, float r2, float h, float mu, float sigma);

#endif // ANNULARDISKINIT_CUH
