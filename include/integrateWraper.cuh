#ifndef INTEGRATE_CUH
#define INTEGRATE_CUH
#include "body.cuh"

void integrateWraper( int nBodies, float dt, Body *p_device);

#endif // INTEGRATE_CUH
