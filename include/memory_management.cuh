#ifndef MEMORY_MANAGEMENT_CUH
#define MEMORY_MANAGEMENT_CUH
#include "body.cuh"

void bodyForceMalloc(int bytes, Body *p, Body *&p_device);
void cudaFreeMemRoutines(Body *p_device, float *buf);

#endif // MEMORY_MANAGEMENT_CUH
