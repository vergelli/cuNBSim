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
#include "annular.cuh"
#include "annularDiskInitDispatcher.cuh"
#include "annularPositionInitWrapper.cuh"
#include "centralBody.cuh"

// Definición de la función de simulación tipo "annular"
int simulate_annular(
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
    float max_particles_speed,
    float r1,
    float r2,
    float h,
    CentralBody*& central_device
    ){


    //TODO: Tiene que haber una mejor forma para decirle a execBodyForce que es un disco annular
    //TODO: este aproach me parece un poco feo, se puede mejorar.
    std::string SIMULATION_TYPE = "annular";

    AnnularDiskInitType initType = GAUSSIAN_Z_EXTERN;
    float mu = 0.0f;
    float sigma = h/8.0f;

    execAnnularDiskInitDispatcher(nBodies, d_states, p_device, gridDimX, blockDimX, r1, r2, h, initType, mu, sigma);

    // Inicialización de masa para disco annular
    massKernelLaunch(nBodies, p_device, gridDimX, blockDimX);

    // Inicialización de velocidades para disco annular
    velocityKernelLaunch(nBodies, p_device, gridDimX, blockDimX, max_particles_speed);

    // Selección del integrador
    if (numerical_integrator == "euler-explicit") {
        for (int iter = 0; iter < nIters; iter++) {
            execBodyForce(nBodies, dt, p_device, gridDimX, blockDimX, SIMULATION_TYPE, central_device);
            execIntegrate(nBodies, dt, p_device, gridDimX, blockDimX, integrateStride);
            simulationDataCollection(p, p_device, nBodies, bytes, iter, numerical_integrator);
            printProgress(iter + 1, nIters);
        }
    }
    else if (numerical_integrator == "leap-frog") {
        execBodyForce(nBodies, dt, p_device, gridDimX, blockDimX, SIMULATION_TYPE, central_device);
        execLeapFrogVelocityUpdate(nBodies, 0.5f * dt, p_device, gridDimX, blockDimX);
        for (int iter = 0; iter < nIters; iter++) {
            execLeapFrogPositionUpdate(nBodies, dt, p_device, gridDimX, blockDimX);
            execBodyForce(nBodies, dt, p_device, gridDimX, blockDimX, SIMULATION_TYPE, central_device);
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
