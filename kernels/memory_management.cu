#include <cuda_runtime.h>
#include"body.cuh"


void bodyForceMalloc( int bytes, Body *p, Body *p_device){

    cudaMalloc(&p_device,bytes);
    cudaMemcpy(p_device, p, bytes, cudaMemcpyHostToDevice);

}

void cudaFreeMemRoutines(Body *p_device){
    cudaFree(p_device);
}

