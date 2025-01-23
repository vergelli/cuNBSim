#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "deviceProps.cuh"

// Función para obtener las propiedades del dispositivo
DeviceProperties getDeviceProps() {
    DeviceProperties props;

    // Obtener el ID del dispositivo
    CHECK_CUDA_ERROR(cudaGetDevice(&props.deviceId));

    // Obtener las propiedades del dispositivo
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, props.deviceId));

    // Asignar las propiedades
    props.warpDim = deviceProp.warpSize;
    props.numberOfSMs = deviceProp.multiProcessorCount;
    props.maxGridDimX = deviceProp.maxGridSize[0];
    props.maxGridDimY = deviceProp.maxGridSize[1];
    props.maxGridDimZ = deviceProp.maxGridSize[2];
    props.maxBlockDimX = deviceProp.maxThreadsDim[0];
    props.maxBlockDimY = deviceProp.maxThreadsDim[1];
    props.maxBlockDimZ = deviceProp.maxThreadsDim[2];
    props.maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    // Mejorar los mensajes de logging
    std::cout << "INFO - CUDA Device Properties Detected:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Device ID                : " << props.deviceId << std::endl;
    std::cout << "Device Name              : " << deviceProp.name << std::endl;
    std::cout << "Compute Capability       : " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Warp Size                : " << props.warpDim << std::endl;
    std::cout << "Number of SMs            : " << props.numberOfSMs << std::endl;
    std::cout << "Max Grid Dimensions      : X=" << props.maxGridDimX 
              << ", Y=" << props.maxGridDimY 
              << ", Z=" << props.maxGridDimZ << std::endl;
    std::cout << "Max Block Dimensions     : X=" << props.maxBlockDimX 
              << ", Y=" << props.maxBlockDimY 
              << ", Z=" << props.maxBlockDimZ << std::endl;
    std::cout << "Max Threads Per Block    : " << props.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Per SM       : " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Total Global Memory (MB) : " << (deviceProp.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Memory Clock Rate (kHz)  : " << deviceProp.memoryClockRate << " kHz" << std::endl;
    std::cout << "Memory Bus Width (bits)  : " << deviceProp.memoryBusWidth << " bits" << std::endl;
    std::cout << "L2 Cache Size (KB)       : " << deviceProp.l2CacheSize / 1024 << " KB" << std::endl;
    std::cout << "Shared Memory Per Block  : " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    return props;
}
