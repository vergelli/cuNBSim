#include "body.cuh"
#include "mass.cuh"
#include "device_config.cuh"

__global__ void initialize_mass(Body* p_device, int nBodies) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nBodies) {
        float dist = sqrtf(
            (p_device[i].x) * (p_device[i].x) +
            (p_device[i].y) * (p_device[i].y) +
            (p_device[i].z) * (p_device[i].z)
        );

        // Por ejemplo, podrías definir la masa como inversamente proporcional a la distancia
        p_device[i].mass = 1.0f / (dist + d_MASS_SOFTENING);

        // p_device[i].mass = some_base_mass * exp(-dist * decay_factor);
    }
}
