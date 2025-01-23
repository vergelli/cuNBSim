#ifndef DEVICE_PROPS_CUH
#define DEVICE_PROPS_CUH

struct DeviceProperties {
    int warpDim;
    int deviceId;
    int numberOfSMs;
    int maxGridDimX;
    int maxGridDimY;
    int maxGridDimZ;
    int maxBlockDimX;
    int maxBlockDimY;
    int maxBlockDimZ;
    int maxThreadsPerBlock;
};

DeviceProperties getDeviceProps();

#endif // DEVICE_PROPS_CUH
