#ifndef MASS_CUH
#define MASS_CUH
#include "body.cuh"
#define MASS_SOFTENING 0.1

__global__ void initialize_mass(Body* p_device, int nBodies);

#endif // MASS_CUH
