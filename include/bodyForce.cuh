#ifndef BODYFORCE_CUH
#define BODYFORCE_CUH
#include "body.cuh"
#include "centralBody.cuh"

// Función que calcula la fuerza entre las partículas
__global__ void bodyForceCUDA(Body *p_device, float dt, int nBodies);

__global__ void annularBodyForceCUDA(Body *p_device, float dt, int nBodies, const CentralBody *centralObj);

#endif // INTEGRATE_CUH
