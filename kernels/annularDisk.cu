// annularDisk.cu
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "body.cuh"    // Define la estructura Body
#include "config.hpp"  // Puede incluir definiciones como SIGMA, etc.
#include "mathUtils.cuh"   // Puede incluir definiciones como M_PI, etc.


// Kernel para generar posiciones de partículas en un disco anular

// annular_disk_kernel_uniformZ.cu
__global__ void annular_disk_kernel_uniform(Body* p_device, curandState* state, int nBodies, float r1, float r2, float h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nBodies) {
        float theta = 2.0f * M_PI * curand_uniform(&state[i]);
        // Distribución uniforme en área para el radio:
        float u = curand_uniform(&state[i]);
        float r = sqrtf(u * (r2 * r2 - r1 * r1) + r1 * r1);
        // Distribución uniforme en z
        float z = -h / 2.0f + h * curand_uniform(&state[i]);

        p_device[i].x = r * cosf(theta);
        p_device[i].y = r * sinf(theta);
        p_device[i].z = z;
    }
}

// annular_disk_kernel_gaussianZ_interna.cu
__global__ void annular_disk_kernel_gaussianZ_intern(Body* p_device, curandState* state, int nBodies, 
                                                        float r1, float r2, float h, float mu, float sigma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nBodies) {
        float theta = 2.0f * M_PI * curand_uniform(&state[i]);
        // Distribución radial uniforme en área (puedes reemplazarla por otra si lo deseas)
        float u = curand_uniform(&state[i]);
        float r = sqrtf(u * (r2 * r2 - r1 * r1) + r1 * r1);

        // Para z: calculamos la desviación estándar adaptativa
        // "Lomada interna": mayor dispersión cuando r es cercano a r1
        float z_sigma = sigma * (1.0f - (r - r1) / (r2 - r1));
        // Generar un valor normal usando el método Box-Muller
        float u1 = curand_uniform(&state[i]);
        float u2 = curand_uniform(&state[i]);
        float R = sqrtf(-2.0f * logf(u1));
        float theta2 = 2.0f * M_PI * u2;
        float z_normal = R * cosf(theta2); // variable normal estándar
        float z = mu + z_sigma * z_normal;
        // Opcional: limitar z a [-h/2, h/2]
        z = fmaxf(z, -h / 2.0f);
        z = fminf(z, h / 2.0f);

        p_device[i].x = r * cosf(theta);
        p_device[i].y = r * sinf(theta);
        p_device[i].z = z;
    }
}

// annular_disk_kernel_gaussianZ_externa.cu
__global__ void annular_disk_kernel_gaussianZ_extern(Body* p_device, curandState* state, int nBodies, 
                                                        float r1, float r2, float h, float mu, float sigma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nBodies) {
        float theta = 2.0f * M_PI * curand_uniform(&state[i]);
        float u = curand_uniform(&state[i]);
        float r = sqrtf(u * (r2 * r2 - r1 * r1) + r1 * r1);

        // Para z: calculamos la desviación estándar adaptativa
        // "Lomada externa": mayor dispersión cuando r es cercano a r2
        float z_sigma = sigma * ((r - r1) / (r2 - r1));
        // Generar un valor normal
        float u1 = curand_uniform(&state[i]);
        float u2 = curand_uniform(&state[i]);
        float R = sqrtf(-2.0f * logf(u1));
        float theta2 = 2.0f * M_PI * u2;
        float z_normal = R * cosf(theta2);
        float z = mu + z_sigma * z_normal;
        // Opcional: limitar z a [-h/2, h/2]
        z = fmaxf(z, -h / 2.0f);
        z = fminf(z, h / 2.0f);

        p_device[i].x = r * cosf(theta);
        p_device[i].y = r * sinf(theta);
        p_device[i].z = z;
    }
}
