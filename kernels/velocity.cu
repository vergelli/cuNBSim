#include "body.cuh"
#include "mass.cuh"

__global__ void initialize_velocity(Body* p_device, int nBodies, float max_particles_speed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nBodies) {
        float dist = sqrtf(
            p_device[i].x * p_device[i].x +
            p_device[i].y * p_device[i].y +
            p_device[i].z * p_device[i].z
        );
        float speed = max_particles_speed * sqrtf(dist);
        p_device[i].vx = -speed * p_device[i].y / dist;
        p_device[i].vy = speed * p_device[i].x / dist;
        p_device[i].vz = 0.0f;
    }
}
