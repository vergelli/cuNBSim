// integratorLeapFrog.cu
#include <cuda_runtime.h>
#include "body.cuh"

__global__ void leapFrogPositionCUDA(Body *p_device, float dt, int nBodies) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nBodies) {
        //* Actualizamos la posición usando la velocidad en medio paso.
        p_device[index].x += p_device[index].vx * dt;
        p_device[index].y += p_device[index].vy * dt;
        p_device[index].z += p_device[index].vz * dt;
    }
}

__global__ void leapFrogVelocityCUDA(Body *p_device, float dt, int nBodies) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nBodies) {
        //* Actualizamos la velocidad utilizando la fuerza calculada.
        p_device[index].vx += (p_device[index].fx / p_device[index].mass) * dt;
        p_device[index].vy += (p_device[index].fy / p_device[index].mass) * dt;
        p_device[index].vz += (p_device[index].fz / p_device[index].mass) * dt;
    }
}
