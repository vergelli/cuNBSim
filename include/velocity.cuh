#ifndef VELOCITY_CUH
#define VELOCITY_CUH
#include "body.cuh"
#define VELOCITY_SOFTENING 0.1

__global__ void initialize_velocity(Body* p_device, int nBodies, float max_particles_speed);

__global__ void initialize_velocity_radial(Body* p_device, int nBodies, float max_particles_speed);

__global__ void initialize_velocity_random(Body* p_device, int nBodies, float max_particles_speed);

__global__ void initialize_velocity_perpendicular(Body* p_device, int nBodies, float max_particles_speed);


#endif // VELOCITY_CUH

