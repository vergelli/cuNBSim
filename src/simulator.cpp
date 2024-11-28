#include <stdio.h>
#include <cstdlib> 
#include "body.cuh"
#include "config.hpp"
#include "memory_management.cuh"
#include "bodyForceWraper.cuh"
#include "integrateWraper.cuh"
#include "deviceProps.cuh"

int main() {

    int bytes = nBodies * sizeof(Body);
    printf("%d Bytes for %d particles\n", bytes, nBodies);
    float *buf;
    buf = (float *)malloc(bytes);
    Body *p = (Body*)buf;
    Body *p_device;

    DeviceProperties deviceProps = getDeviceProps();

    bodyForceMalloc(bytes, p, p_device);

    //* All kernels
    int gridDimX;

    //* bodyForce kernel
    int bodyForceBlockDimX;

    //* Integrate Kernel
    int integrateBlockDimX, integrateStride;

    //~ Inicialización de parametros de configuracion de lanzamiento
    initBodyForce(gridDimX, bodyForceBlockDimX, deviceProps);
    initIntegrate(gridDimX, integrateBlockDimX, integrateStride, deviceProps);

    //~ Ciclo principal: solo ejecuta los kernels
    for (int iter = 0; iter < nIters; iter++) {
        execBodyForce(nBodies, dt, p_device, gridDimX, bodyForceBlockDimX);
        execIntegrate(nBodies, dt, p_device, gridDimX, integrateBlockDimX, integrateStride);
    }

    printf("Simulation terminated\n");

    //~ Rutinas de liberacion de memoria
    cudaFreeMemRoutines(p_device);
    return 0;
}
