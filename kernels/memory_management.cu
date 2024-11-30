#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "body.cuh"
#include "cuda_utils.cuh"

// Función que gestiona la reserva de memoria para el array de cuerpos y los estados de curand
void bodyForceMalloc(int bytes, Body *p, Body *&p_device, curandState* &d_states, int nBodies) {

    printf("INFO - Allocating particles state on GPU\n");
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_states, nBodies * sizeof(curandState)));

    printf("INFO - Allocating particles on GPU\n");
    CHECK_CUDA_ERROR(cudaMalloc(&p_device, bytes));

    printf("INFO - starting memory migrate operation to GPU\n");
    CHECK_CUDA_ERROR(cudaMemcpy(p_device, p, bytes, cudaMemcpyHostToDevice));
}

void cudaFreeMemRoutines(Body *p_device, curandState *d_states, float *buf){

    printf("INFO - Freeing memory on GPU\n");
    CHECK_CUDA_ERROR(cudaFree(d_states));
    CHECK_CUDA_ERROR(cudaFree(p_device));
    printf("INFO - Freeing memory on CPU\n");
    free(buf);

}
