#include <cuda_runtime.h>
#include "body.cuh"

__global__ void integrateCUDA(Body *p, float dt, int n, int stride)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  for(int i = index; i<n ; i+=stride)
  {
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
  }
}