#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
    #include <windows.h>  // Para Windows
#else
    #include <unistd.h>   // Para Unix/Linux
#endif

#define SOFTENING 1e-9f

/*
* Each body contains x, y, and z coordinate positions,
* as well as velocities in the x, y, and z directions.
*/

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
* Calculate the gravitational impact of all bodies in the system
* on all others.
*/

__global__ void bodyForceCUDA(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

__global__ void integrateCUDA(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

int main(const int argc, const char** argv) {
int nBodies = 2<<11;
if (argc > 1) nBodies = 2<<atoi(argv[1]);

const float dt = 0.01f; // Time step
const int nIters = 10;  // Simulation iterations

int bytes = nBodies * sizeof(Body);
float *buf;
buf = (float *)malloc(bytes);
Body *p = (Body*)buf;

Body *p_device;
cudaMalloc((void**)&p_device, bytes);
cudaMemcpy(p_device, p, bytes, cudaMemcpyHostToDevice);

int warpDim;
int deviceId;
int numberOfSMs;
int numberOfBlocks;
int threadsPerBlock;

cudaGetDevice(&deviceId);
cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
cudaDeviceGetAttribute(&warpDim, cudaDevAttrWarpSize, deviceId);

threadsPerBlock = (warpDim*warpDim)/4; //? 1024 maximizamos los hilos por bloque
numberOfBlocks = warpDim*numberOfSMs;
//numberOfBlocks = (nBodies + threadsPerBlock - 1) / threadsPerBlock;


for (int iter = 0; iter < nIters; iter++) {

    bodyForceCUDA<<<numberOfBlocks, threadsPerBlock>>>(p_device, dt, nBodies);
    cudaDeviceSynchronize();

    integrateCUDA<<<numberOfBlocks, threadsPerBlock>>>(p_device, dt, nBodies);

}
cudaMemcpy(p, p_device, bytes, cudaMemcpyDeviceToHost);
cudaFree(p_device);
free(buf);

printf("The end\n");
}
