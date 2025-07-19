#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "body.cuh"
#include "cuda_utils.cuh"
#include "centralBody.cuh"
#include "config.hpp"
#include "device_config.cuh"

// Función que gestiona la reserva de memoria para el array de cuerpos y los estados de curand
void allocateMemoryForParticles(int bytes, Body *p, Body *&p_device, curandState* &d_states, 
                                int nBodies, CentralBody*& central_host, CentralBody*& central_device) {

    //TODO: bytes es una variable que viene de afuera. Podriamos pasar tambien los bytes del objeto central
    //TODO: esto tambien me llevaria a renombrar este argumento de forma mas precisa, y no solo "bytes"
    printf("INFO - Allocating central body memory on GPU\n");
    CHECK_CUDA_ERROR(cudaMalloc((void**)&central_device, sizeof(CentralBody)));

    printf("INFO - Initializing values on central body\n");
    initCentralBody(central_host);

    printf("INFO - starting memory migrate operation to GPU (Central Body)\n");
    CHECK_CUDA_ERROR(cudaMemcpy(central_device, central_host, sizeof(CentralBody), cudaMemcpyHostToDevice));

    printf("INFO - Allocating particles state on GPU\n");
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_states, nBodies * sizeof(curandState)));

    printf("INFO - Allocating particles on GPU\n");
    CHECK_CUDA_ERROR(cudaMalloc(&p_device, bytes));

    printf("INFO - starting memory migrate operation to GPU (Particles)\n");
    CHECK_CUDA_ERROR(cudaMemcpy(p_device, p, bytes, cudaMemcpyHostToDevice));
}

void copyConfigToDevice() {
    std::cout << "INFO - Copying configuration parameters to device constant memory..." << std::endl;

    std::cout << "INFO - Copying MASS_SOFTENING = " << MASS_SOFTENING << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_MASS_SOFTENING, &MASS_SOFTENING, sizeof(float)));

    std::cout << "INFO - Copying FORCE_SOFTENING = " << SOFTENING << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_FORCE_SOFTENING, &SOFTENING, sizeof(float)));

    std::cout << "INFO - Copying MIN_DISTANCE_THRESHOLD = " << MIN_DISTANCE_THRESHOLD << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_MIN_DISTANCE_THRESHOLD, &MIN_DISTANCE_THRESHOLD, sizeof(float)));

    std::cout << "INFO - Copying G = " << G << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_G, &G, sizeof(float)));

    std::cout << "INFO - Copying MAX_PARTICLES_SPEED = " << max_particles_speed << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_MAX_PARTICLES_SPEED, &max_particles_speed, sizeof(float)));

    std::cout << "INFO - Copying PI_VALUE = " << pi_value << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_PI_VALUE, &pi_value, sizeof(float)));

    std::cout << "INFO - Configuration copied successfully." << std::endl;
}

void cudaFreeMemRoutines(Body *p_device, curandState *d_states, float *buf){

    printf("INFO - Freeing memory on GPU\n");
    CHECK_CUDA_ERROR(cudaFree(d_states));
    CHECK_CUDA_ERROR(cudaFree(p_device));
    printf("INFO - Freeing memory on CPU\n");
    free(buf);

}
