#ifndef DEVICE_CONFIG_CUH
#define DEVICE_CONFIG_CUH

// Declaramos una variable en memoria constante
__constant__ float d_MASS_SOFTENING;
__constant__ float d_FORCE_SOFTENING;
__constant__ float d_MIN_DISTANCE_THRESHOLD;
__constant__ float d_G;
__constant__ float d_MAX_PARTICLES_SPEED;
__constant__ float d_PI_VALUE;

#endif // DEVICE_CONFIG_CUH
