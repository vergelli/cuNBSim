#ifndef VELOCITY_CUH
#define VELOCITY_CUH
#include "body.cuh"
#define VELOCITY_SOFTENING 0.1

__global__ void initialize_velocity(Body* p_device, int nBodies, float max_particles_speed);

#endif // VELOCITY_CUH

