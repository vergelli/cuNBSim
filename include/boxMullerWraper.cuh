#ifndef BOXMULLERWRAPER_CUH
#define BOXMULLERWRAPER_CUH
#include "body.cuh"
#include "deviceProps.cuh"

void execBoxMuller( int nBodies, curandState* d_state, Body* p_device, int gridDimX, int blockDimX);

#endif // BOXMULLERWRAPER_CUH
