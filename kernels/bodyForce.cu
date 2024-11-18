#include <cuda_runtime.h>
#include "bodyForce.cuh"

__global__ void bodyForceCUDA(Body *p, float dt, int nBodies, int stride) {
  int indexIthBody = threadIdx.x + blockIdx.x * blockDim.x;
  for(int i = indexIthBody; i<nBodies ; i+=stride)
  {
    float Fx = 0.0f; 
    float Fy = 0.0f; 
    float Fz = 0.0f;

    int indexJthBody = threadIdx.y + blockIdx.y * blockDim.y;
    for (int j = indexJthBody; j < nBodies; j+=stride) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; 
      Fy += dy * invDist3; 
      Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; 
    p[i].vy += dt*Fy; 
    p[i].vz += dt*Fz;
  }
}
