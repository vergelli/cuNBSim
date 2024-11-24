#include <cuda_runtime.h>
#include "body.cuh"

__global__ void integrateCUDA(Body *p_device, float dt, int nBody, int stride)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  for(int i = index; i<nBody ; i+=stride)
  {
      p_device[i].x += p_device[i].vx*dt;
      p_device[i].y += p_device[i].vy*dt;
      p_device[i].z += p_device[i].vz*dt;
  }
}