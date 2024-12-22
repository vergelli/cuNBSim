#include "body.cuh"
#include "mass.cuh"


__global__ void initialize_velocity(Body* p_device, int nBodies, float max_particles_speed) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nBodies) {
        float dist = sqrtf(
            p_device[i].x * p_device[i].x +
            p_device[i].y * p_device[i].y +
            p_device[i].z * p_device[i].z
        );

        // Definir la velocidad en función de la distancia al centro (para dar mayor velocidad a partículas cercanas)
        float speed = max_particles_speed * (1.0f / (dist + 0.1f));  // +0.1f para evitar divisiones por cero

        // Crear un vector tangencial para la velocidad
        p_device[i].vx = -speed * p_device[i].y / dist;  // Velocidad tangente al eje Z
        p_device[i].vy = speed * p_device[i].x / dist;   // Velocidad tangente al eje Z
        p_device[i].vz = 0.0f;  // Queremos que las partículas formen un disco en el plano X-Y
    }
}
