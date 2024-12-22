#include <cuda_runtime.h>
#include "bodyForce.cuh"
#include "body.cuh"


__global__ void bodyForceCUDA(Body *p_device, float dt, int nBodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nBodies) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int j = 0; j < nBodies; j++) {
            if (i != j) {
                float dx = p_device[j].x - p_device[i].x;
                float dy = p_device[j].y - p_device[i].y;
                float dz = p_device[j].z - p_device[i].z;
                //& Δx^2 + Δy^2 +Δz^2 +ε
                float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
                //& r^3
                float distSixth = distSqr * sqrtf(distSqr);

                // Calculamos la magnitud de la fuerza gravitacional
                float forceMag = G * p_device[i].mass * p_device[j].mass / distSixth;

                // Ahora sumamos la contribución de la fuerza en cada componente
                Fx += forceMag * dx;
                Fy += forceMag * dy;
                Fz += forceMag * dz;
            }
        }
        // Guardamos las fuerzas acumuladas en cada componente
        p_device[i].fx = Fx;
        p_device[i].fy = Fy;
        p_device[i].fz = Fz;
    }
}
