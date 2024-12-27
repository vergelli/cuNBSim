#ifndef BODYFORCE_CUH
#define BODYFORCE_CUH
#include "body.cuh"

// Define un umbral mínimo de distancia
#define MIN_DISTANCE 1.0f
// Define un valor de amortiguacion
#define SOFTENING 1e-9f
// Define la constante gravitacional
#define G 9.807

// Función que calcula la fuerza entre las partículas
__global__ void bodyForceCUDA(Body *p, float dt, int nBodies);

#endif // INTEGRATE_CUH
