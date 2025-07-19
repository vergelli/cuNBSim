#ifndef ANULAR_CUH
#define ANULAR_CUH

#include <string>
#include "body.cuh"         // Definition of the Body structure
#include <curand_kernel.h>  // For curandState
#include "centralBody.cuh"  // Definition of the CentralBody structure

/**
 * @brief Executes the annular disk simulation, initializing particles within an annular region,
 *        assigning mass, velocity, and performing integration using the specified method.
 * 
 * @param p_device Pointer to the particle memory on the device.
 * @param p Pointer to the particle memory on the host.
 * @param d_states Pointer to the CURAND states.
 * @param nBodies Number of particles.
 * @param nIters Number of simulation iterations.
 * @param dt Time step.
 * @param gridDimX Grid dimension for kernel launch.
 * @param blockDimX Block dimension for kernel launch.
 * @param integrateStride Stride for the integration kernel.
 * @param bytes Size in bytes of the memory for the particles.
 * @param numerical_integrator String defining the integrator method ("euler-explicit" or "leap-frog").
 * @param max_particles_speed Maximum speed value for initialization.
 * @param r1 Inner radius of the annular region.
 * @param r2 Outer radius of the annular region.
 * @param h Height of the annular region.
 * 
 * @return int Exit code (0 if everything goes well).
 */
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
    );
#endif // ANULAR_CUH
