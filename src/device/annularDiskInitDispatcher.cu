#include    <iostream>
#include    <curand_kernel.h>

#include    "annularDisk.cuh"
#include    "annularDiskInitDispatcher.cuh"
#include    "annularPositionInitWrapper.cuh"

void execAnnularDiskInitDispatcher(
    int nBodies,
    curandState* d_state,
    Body* p_device,
    int gridDimX,
    int blockDimX,
    float r1,
    float r2,
    float h,
    AnnularDiskInitType initType,
    float mu = 0.0f,
    float sigma = 1.0f
) {
    switch (initType) {
        case UNIFORM_Z:
            std::cout << "INFO - Using Uniform Z distribution" << std::endl;
            execAnnularDiskInitUniform(nBodies, d_state, p_device, gridDimX, blockDimX, r1, r2, h);
            break;
        case GAUSSIAN_Z_INTERN:
            std::cout << "INFO - Using Gaussian Z distribution (internal hump)" << std::endl;
            execAnnularDiskInitGaussianZIntern(nBodies, d_state, p_device, gridDimX, blockDimX, r1, r2, h, mu, sigma);
            break;
        case GAUSSIAN_Z_EXTERN:
            std::cout << "INFO - Using Gaussian Z distribution (external hump)" << std::endl;
            execAnnularDiskInitGaussianZExtern(nBodies, d_state, p_device, gridDimX, blockDimX, r1, r2, h, mu, sigma);
            break;
        default:
            std::cerr << "ERROR: Unknown annular disk init type" << std::endl;
            break;
    }
}
