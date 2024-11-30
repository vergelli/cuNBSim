#include <cuda_runtime.h>
#include "bodyForce.cuh"
#include "body.cuh"


__global__ void bodyForceCUDA(Body *p_device, float dt, int nBodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nBodies) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int j = 0; j < nBodies; j++) {
            float dx = p_device[j].x - p_device[i].x;
            float dy = p_device[j].y - p_device[i].y;
            float dz = p_device[j].z - p_device[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p_device[i].vx += dt * Fx;
        p_device[i].vy += dt * Fy;
        p_device[i].vz += dt * Fz;
    }
}
