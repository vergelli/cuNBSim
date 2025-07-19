#ifndef MEMORY_MANAGEMENT_CUH
#define MEMORY_MANAGEMENT_CUH
#include "body.cuh"
#include "centralBody.cuh"

void allocateMemoryForParticles(int bytes, Body *p, Body *&p_device, curandState* &d_states, 
                                int nBodies, CentralBody*& central_host, CentralBody*& central_device) ;

void copyConfigToDevice();

void cudaFreeMemRoutines(Body *p_device, curandState *d_states, float *buf);

#endif // MEMORY_MANAGEMENT_CUH
