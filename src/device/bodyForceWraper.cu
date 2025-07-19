#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include "body.cuh"
#include "bodyForce.cuh"
#include "cuda_utils.cuh"
#include "deviceProps.cuh"
#include "centralBody.cuh"

void execBodyForce(
    int nBodies, 
    float dt, 
    Body *p_device, 
    int gridDimX, 
    int blockDimX,
    std::string simulation_type,
    const CentralBody* central_device = nullptr) {

    //! ____________________________________________________________
    //TODO: Analizar este patron. Se esta utilizando para definir la 
    //TODO: cantidad de bloques y threads por bloque
    //TODO: Se dispara en cada iteracion, habria que sacarlo del bucle.
    dim3 dimGrid(gridDimX, 1, 1);
    dim3 BodyForceDimBlock(blockDimX, 1, 1);
    //! ____________________________________________________________

    if (simulation_type == "annular") {
        //TODO: Pasar resto de parametros necesartios para la simulación de disco annular
        annularBodyForceCUDA<<<dimGrid, BodyForceDimBlock>>>(p_device, dt, nBodies, central_device);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    else if (simulation_type == "cloud") {
        bodyForceCUDA<<<dimGrid, BodyForceDimBlock>>>(p_device, dt, nBodies);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
}
