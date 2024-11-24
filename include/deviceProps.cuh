#ifndef DEVICE_PROPS_CUH
#define DEVICE_PROPS_CUH

struct DeviceProperties {
    int warpDim;
    int deviceId;
    int numberOfSMs;
};

DeviceProperties getDeviceProps();

#endif // DEVICE_PROPS_CUH


