#include "body.cuh"
#include "mass.cuh"
#include "device_config.cuh"
#include <curand_kernel.h>

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

__global__ void initialize_velocity_radial(Body* p_device, int nBodies, float max_particles_speed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nBodies) {
        float dist = sqrtf(
            p_device[i].x * p_device[i].x +
            p_device[i].y * p_device[i].y +
            p_device[i].z * p_device[i].z
        );
        // Velocidad radial hacia afuera
        float speed = max_particles_speed * (dist / 100.0f);  // Velocidad escalada según la distancia
        p_device[i].vx = speed * (p_device[i].x / dist);
        p_device[i].vy = speed * (p_device[i].y / dist);
        p_device[i].vz = speed * (p_device[i].z / dist);
    }
}

__global__ void initialize_velocity_random(Body* p_device, int nBodies, float max_particles_speed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long seed = 1234;  // Puedes cambiar la semilla
    if (i < nBodies) {
        // Inicializar el estado de cuRAND
        curandState state;
        curand_init(seed, i, 0, &state);

        // Generar valores aleatorios para theta y phi
        float theta = 2.0f * d_PI_VALUE * curand_uniform(&state);  // valor entre 0 y 2*PI
        float phi = acosf(2.0f * curand_uniform(&state) - 1.0f);  // valor entre 0 y PI

        // Asignar velocidades aleatorias
        float vx = max_particles_speed * sinf(phi) * cosf(theta);
        float vy = max_particles_speed * sinf(phi) * sinf(theta);
        float vz = max_particles_speed * cosf(phi);

        p_device[i].vx = vx;
        p_device[i].vy = vy;
        p_device[i].vz = vz;
    }
}

__global__ void initialize_velocity_perpendicular(Body* p_device, int nBodies, float max_particles_speed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nBodies) {
        float dist = sqrtf(
            p_device[i].x * p_device[i].x +
            p_device[i].y * p_device[i].y +
            p_device[i].z * p_device[i].z
        );
        // Velocidad perpendicular, similar a la inicialización original pero normalizada
        float speed = max_particles_speed * sqrtf(dist);
        p_device[i].vx = -speed * p_device[i].y / dist;
        p_device[i].vy = speed * p_device[i].x / dist;
        p_device[i].vz = 0.0f;  // Consideramos en 2D en este ejemplo
    }
}
