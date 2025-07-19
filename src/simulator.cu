#include <curand_kernel.h>
#include <iostream>
#include <cstdlib>
#include "body.cuh"
#include "utils.hpp"
#include "config.hpp"
#include "memory_management.cuh"
#include "deviceProps.cuh"
#include "kernelsInit.cuh"
#include "boxMullerWraper.cuh"
#include "massWraper.cuh"
#include "velocityWrapper.cuh"
#include "bodyForceWraper.cuh"
#include "integrateWraper.cuh"
#include "integratorLeapFrogWraper.cuh"
#include "data_collector.cuh"
#include "cloud.cuh"   // Incluimos el header para la simulación "cloud"
#include "annular.cuh" // Incluimos el header para la simulación "annular"
#include "centralBody.cuh"


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
    CentralBody *central_host = (CentralBody*)malloc(sizeof(CentralBody));
    CentralBody *central_device;
    if (!central_host) {
        std::cerr << "ERROR - Could not allocate memory for central body on host." << std::endl;
        exit(EXIT_FAILURE);
    }

    //! ===============================================================================================
    //TODO: Implementar la lectura de los parametros de la simulacion desde un archivo de configuracion
    //! Por el momento, se declara un FLAG que define el tipo de simulacion.
    //! En el futuro, se podria implementar una variable en el archivo de configuracion
    //! que defina el tipo de simulacion a realizar. Por el momento, se define en el codigo.

    std::string simulation_type = "annular";// 'annular' o 'cloud'

    float r2 = 2.0f;
    float r1 = 0.1f;
    float h = 2.0f;

    //! ===============================================================================================

    //TODO: La funcion tiene que saber si inicializar el objeto central ono
    //TODO: Esto solo lo puede saber si se le pasa simulation_type
    allocateMemoryForParticles(bytes, p, p_device, d_states, nBodies, central_host, central_device);

    copyConfigToDevice();

    //~ Inicializacion de los parametros de lanzamiento de los kernels
    kernelsLaunchParamsInit(gridDimX, blockDimX, integrateStride, deviceProps);

    //~ _______________________________________________________________________

    if (simulation_type == "annular") {
        std::cout << "INFO - Starting annular disk simulation" << std::endl;

        simulate_annular(
            p_device,
            p,
            d_states,
            nBodies,
            nIters,
            dt,
            gridDimX,
            blockDimX,
            integrateStride,
            bytes,
            numerical_integrator,
            max_particles_speed,
            r1,
            r2,
            h,
            central_device
            );
    }
    else if (simulation_type == "cloud") {

        std::cout << "INFO - Starting cloud simulation" << std::endl;
        simulate_cloud(
            p_device,
            p,
            d_states,
            nBodies,
            nIters,
            dt,
            gridDimX,
            blockDimX,
            integrateStride,
            bytes,
            numerical_integrator,
            max_particles_speed
        );
    }

    return 0;
}
