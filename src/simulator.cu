#include <curand_kernel.h>
#include <iostream>
#include <cstdlib>
#include "body.cuh"
#include "utils.hpp"
#include "config.hpp"
#include "memory_management.cuh"
#include "deviceProps.cuh"
#include "kernelsInit.cu"
#include "boxMullerWraper.cuh"
#include "massWraper.cuh"
#include "velocityWrapper.cuh"
#include "bodyForceWraper.cuh"
#include "integrateWraper.cuh"
#include "data_collector.cuh"

int main(int argc, char* argv[]) {

    std::string config_path = (argc > 1) ? argv[1] : default_config_path;

    DeviceProperties deviceProps = getDeviceProps();
    load_config_from_file(config_path, deviceProps);

    int bytes = nBodies * sizeof(Body);
    float *buf;
    buf = (float *)malloc(bytes);
    Body *p = (Body*)buf;
    Body *p_device;
    curandState *d_states;

    allocateMemoryForParticles(bytes, p, p_device, d_states, nBodies);

    //~ Inicializacion de los parametros de lanzamiento de los kernels
    kernelsLaunchParamsInit(gridDimX, blockDimX, integrateStride, deviceProps);

    //~ Definiendo posicion inicial de las particulas
    execBoxMuller( nBodies, d_states, p_device, gridDimX, blockDimX);

    //~ Definiendo la masa inicial de las particulas
    massKernelLaunch( nBodies, p_device, gridDimX, blockDimX);

    //~ Definiendo la velocidad inicial de las particulas
    velocityKernelLaunch( nBodies, p_device, gridDimX, blockDimX, max_particles_speed);

    //~ Ciclo principal
    for (int iter = 0; iter < nIters; iter++) {
        execBodyForce(nBodies, dt, p_device, gridDimX, blockDimX);
        execIntegrate(nBodies, dt, p_device, gridDimX, blockDimX, integrateStride);
        simulationDataCollection(p, p_device, nBodies, bytes, iter);
        printProgress(iter + 1, nIters);
    }

    printf("INFO - Simulation terminated\n");

    //~ Rutinas de liberacion de memoria
    cudaFreeMemRoutines(p_device,d_states, buf);
    return 0;
}
