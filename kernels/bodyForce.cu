#include <cuda_runtime.h>
#include "bodyForce.cuh"
#include "centralBody.cuh"
#include "body.cuh"
#include "device_config.cuh"

__global__ void bodyForceCUDA(Body *p_device, float dt, int nBodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nBodies) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
        for (int j = 0; j < nBodies; j++) {
            if (i != j) {
                float dx = p_device[j].x - p_device[i].x;
                float dy = p_device[j].y - p_device[i].y;
                float dz = p_device[j].z - p_device[i].z;
                float distSqr = dx*dx + dy*dy + dz*dz + d_FORCE_SOFTENING;
                if (distSqr > d_MIN_DISTANCE_THRESHOLD * d_MIN_DISTANCE_THRESHOLD) {
                    float distSixth = distSqr * sqrtf(distSqr);
                    float forceMag = d_G * p_device[i].mass * p_device[j].mass / distSixth;
                    Fx += forceMag * dx;
                    Fy += forceMag * dy;
                    Fz += forceMag * dz;
                }
            }
        }
        p_device[i].fx = Fx;
        p_device[i].fy = Fy;
        p_device[i].fz = Fz;
    }
}

__global__ void annularBodyForceCUDA(Body *p_device, float dt, int nBodies, const CentralBody *centralObj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nBodies) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
        // Contribución del objeto centralObj (asumiendo centralObj en (0,0,0))
        float dx = -p_device[i].x;
        float dy = -p_device[i].y;
        float dz = -p_device[i].z;
        float distSqr = dx * dx + dy * dy + dz * dz + centralObj->epsilon;
        float dist = sqrtf(distSqr);
        float distCubed = distSqr * dist;
        // La fuerza gravitatoria centralObj:
        Fx += d_G * centralObj->mass * p_device[i].mass * dx / distCubed;
        Fy += d_G * centralObj->mass * p_device[i].mass * dy / distCubed;
        Fz += d_G * centralObj->mass * p_device[i].mass * dz / distCubed;
        // Cálculo de fuerzas entre partículas (como en el kernel original)
        for (int j = 0; j < nBodies; j++) {
            if (i != j) {
                float dx2 = p_device[j].x - p_device[i].x;
                float dy2 = p_device[j].y - p_device[i].y;
                float dz2 = p_device[j].z - p_device[i].z;
                float distSqr2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + d_FORCE_SOFTENING;
                if (distSqr2 > d_MIN_DISTANCE_THRESHOLD * d_MIN_DISTANCE_THRESHOLD) {
                    float distSixth = distSqr2 * sqrtf(distSqr2);
                    float forceMag = d_G * p_device[i].mass * p_device[j].mass / distSixth;
                    Fx += forceMag * dx2;
                    Fy += forceMag * dy2;
                    Fz += forceMag * dz2;
                }
            }
        }

        p_device[i].fx = Fx;
        p_device[i].fy = Fy;
        p_device[i].fz = Fz;
    }
}

