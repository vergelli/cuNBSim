#ifndef CLOUD_CUH
#define CLOUD_CUH

#include <string>
#include "body.cuh"         // Definición de la estructura Body
#include <curand_kernel.h>  // Para curandState

//TODO: Traducir al inglés la docstring de la función
/**
 * @brief Ejecuta la simulación del tipo "cloud", inicializando partículas con el método Box-Muller,
 *        asignando masa, velocidad, y realizando la integración usando el método indicado.
 * 
 * @param p_device Puntero a la memoria de partículas en el dispositivo.
 * @param p Puntero a la memoria de partículas en el host.
 * @param d_states Puntero a los estados de CURAND.
 * @param nBodies Número de partículas.
 * @param nIters Número de iteraciones de la simulación.
 * @param dt Paso de tiempo.
 * @param gridDimX Dimensión de la grilla para el lanzamiento de kernels.
 * @param blockDimX Dimensión del bloque para el lanzamiento de kernels.
 * @param integrateStride Stride para el kernel de integración.
 * @param bytes Tamaño en bytes de la memoria para las partículas.
 * @param numerical_integrator Cadena que define el método integrador ("euler-explicit" o "leap-frog").
 * @param max_particles_speed Valor máximo de velocidad para la inicialización.
 * 
 * @return int Código de salida (0 si todo sale bien).
 */
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
);

#endif // CLOUD_CUH
