// cloud.cu
#include <curand_kernel.h>
#include <iostream>
#include <cstdlib>

// Incluye específicos del proyecto
#include "utils.hpp"
#include "body.cuh"
#include "config.hpp"
#include "memory_management.cuh"
#include "deviceProps.cuh"
#include "boxMullerWraper.cuh"
#include "massWraper.cuh"
#include "velocityWrapper.cuh"
#include "bodyForceWraper.cuh"
#include "integrateWraper.cuh"
#include "integratorLeapFrogWraper.cuh"
#include "data_collector.cuh"

// Definición de la función de simulación tipo "cloud"
int simulate_cloud(
    Body* p_device,
    Body* p,
    curandState* d_states,
    int nBodies,
    int nIters,
    float dt,
    int gridDimX,
    int blockDimX,
    int integrateStride,
    int bytes,
    const std::string& numerical_integrator,
    float max_particles_speed
) {

    std::string SIMULATION_TYPE = "cloud";

    // Inicialización de partículas usando Box-Muller
    execBoxMuller(nBodies, d_states, p_device, gridDimX, blockDimX);

    // Inicialización de masa
    massKernelLaunch(nBodies, p_device, gridDimX, blockDimX);

    // Inicialización de velocidades
    velocityKernelLaunch(nBodies, p_device, gridDimX, blockDimX, max_particles_speed);

    // Selección del integrador
    if (numerical_integrator == "euler-explicit") {
        for (int iter = 0; iter < nIters; iter++) {
            execBodyForce(nBodies, dt, p_device, gridDimX, blockDimX, SIMULATION_TYPE);
            execIntegrate(nBodies, dt, p_device, gridDimX, blockDimX, integrateStride);
            simulationDataCollection(p, p_device, nBodies, bytes, iter, numerical_integrator);
            printProgress(iter + 1, nIters);
        }
    }
    else if (numerical_integrator == "leap-frog") {
        execBodyForce(nBodies, dt, p_device, gridDimX, blockDimX, SIMULATION_TYPE);
        execLeapFrogVelocityUpdate(nBodies, 0.5f * dt, p_device, gridDimX, blockDimX);
        for (int iter = 0; iter < nIters; iter++) {
            execLeapFrogPositionUpdate(nBodies, dt, p_device, gridDimX, blockDimX);
            execBodyForce(nBodies, dt, p_device, gridDimX, blockDimX, SIMULATION_TYPE);
            execLeapFrogVelocityUpdate(nBodies, dt, p_device, gridDimX, blockDimX);
            simulationDataCollection(p, p_device, nBodies, bytes, iter, numerical_integrator);
            printProgress(iter + 1, nIters);
        }
    }
    else {
        std::cerr << "ERROR: Unknown integrator: " << numerical_integrator << std::endl;
        return -1;
    }

    std::cout << "INFO - Simulation terminated\n";
    cudaFreeMemRoutines(p_device, d_states, (float*)p);
    return 0;
}
