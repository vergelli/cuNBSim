#include <cuda_runtime.h>
#include "cuda_utils.cuh" // Asegúrate de tener esta cabecera para CHECK_CUDA_ERROR
#include "deviceProps.cuh"

// Función para obtener las propiedades del dispositivo
DeviceProperties getDeviceProps() {

    DeviceProperties props;

    CHECK_CUDA_ERROR(cudaGetDevice(&props.deviceId));
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&props.numberOfSMs,
        cudaDevAttrMultiProcessorCount, 
        props.deviceId));
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&props.warpDim, 
        cudaDevAttrWarpSize, 
        props.deviceId));

    return props;
}
