#include <iostream>
#include <cstdlib> 
#include "body.cuh"
#include "config.hpp"
#include "memory_management.cuh"
#include "bodyForceWraper.cuh"
#include "integrateWraper.cuh"
#include "deviceProps.cuh"
#include "data_collector.cuh"

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
    cudaFreeMemRoutines(p_device, buf);
    return 0;
}
