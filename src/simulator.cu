#include <curand_kernel.h>
#include <iostream>
#include <cstdlib>
#include "body.cuh"
#include "config.hpp"
#include "memory_management.cuh"
#include "data_collector.cuh"
#include "deviceProps.cuh"
#include "bodyForceWraper.cuh"
#include "integrateWraper.cuh"
#include "boxMullerWraper.cuh"

int main(int argc, char* argv[]) {

    //~ Si se pasa un argumento, lo usamos como path de la configuración
    std::string config_path = (argc > 1) ? argv[1] : default_config_path;

    //~ Cargar la configuración desde el archivo (usando el path que corresponda)
    load_config_from_file(config_path);

    int bytes = nBodies * sizeof(Body);
    printf("INFO - Configuration file path: %s \n", config_path.c_str());
    printf("INFO - %d Bytes for %d particles\n", bytes, nBodies);
    printf("INFO - Iterations for simulation = %d and dt = %f \n", nIters, dt);

    float *buf;
    buf = (float *)malloc(bytes);
    Body *p = (Body*)buf;
    Body *p_device;

    DeviceProperties deviceProps = getDeviceProps();

    // Variables para la memoria del dispositivo
    curandState *d_states;

    // Llamada a la función para reservar memoria en el dispositivo
    bodyForceMalloc(bytes, p, p_device, d_states, nBodies);

    //* GridsDim
    int gridDimX;

    //* BlocksDim
    int BlockDimX;

    //* Strides
    int integrateStride;

    //~ Definiendo estado inicial de las particulas
    initBoxMuller( gridDimX, BlockDimX, deviceProps);
    execBoxMuller( nBodies, d_states, p_device, gridDimX, BlockDimX);

    //~ Inicialización de parametros de configuracion de lanzamiento
    initBodyForce(gridDimX, BlockDimX, deviceProps);
    initIntegrate(gridDimX, BlockDimX, integrateStride, deviceProps);

    //~ Ciclo principal: solo ejecuta los kernels
    for (int iter = 0; iter < nIters; iter++) {
        execBodyForce(nBodies, dt, p_device, gridDimX, BlockDimX);
        execIntegrate(nBodies, dt, p_device, gridDimX, BlockDimX, integrateStride);

        //~ Data recollection routines
        simulationDataCollection(
            p, 
            p_device, 
            nBodies, 
            bytes, 
            iter);
    }

    printf("INFO - Simulation terminated\n");

    //~ Rutinas de liberacion de memoria
    cudaFreeMemRoutines(p_device,d_states, buf);
    return 0;
}
