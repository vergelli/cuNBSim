#include <cstdlib> 
#include "body.cuh"
#include "config.hpp"
#include "memory_management.cuh"
#include "bodyForceWraper.cuh"
#include "integrateWraper.cuh"

int main() {

    int bytes = nBodies * sizeof(Body);
    float *buf;
    buf = (float *)malloc(bytes);
    Body *p = (Body*)buf;
    Body *p_device = nullptr;

    bodyForceMalloc(bytes, p, p_device);

    for (int iter = 0; iter < nIters; iter++) {

        bodyForceWraper( nBodies, dt, p_device);
        integrateWraper( nBodies, dt, p_device);
    };

    cudaFreeMemRoutines(p_device);
    return 0;
}
