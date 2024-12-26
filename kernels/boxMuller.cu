#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "body.cuh"
#include "boxMuller.cuh"

__global__ void init_curand_states(curandState* state, unsigned long seed, int nBodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nBodies) {
        //* Inicializa el estado PRNG con el seed
        curand_init(seed, i, 0, &state[i]);
    }
}

__global__ void box_muller_kernel(Body* p_device, curandState* state, int nBodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nBodies) {
        //* Generación de 2 pares de números uniformes usando curand_uniform
        float u1 = curand_uniform(&state[i]);
        float u2 = curand_uniform(&state[i]);
        float R = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * M_PI * u2;

        float u3 = curand_uniform(&state[i]);
        float u4 = curand_uniform(&state[i]);
        float R2 = sqrtf(-2.0f * logf(u3));
        float theta2 = 2.0f * M_PI * u4;

        //* Asignar a las posiciones
        p_device[i].x = R * cosf(theta);
        p_device[i].y = R * sinf(theta);
        p_device[i].z = R2 * cosf(theta2);
    }
}
