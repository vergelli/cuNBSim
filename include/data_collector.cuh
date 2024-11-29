#ifndef LOG_SIMULATION_DATA
#define LOG_SIMULATION_DATA
#include "body.cuh"

void simulationDataCollection(Body* p, Body* p_device, int nBodies, int bytes, int iter);

void logSimulationData(Body* p, int nBodies, int iter);

#endif // LOG_SIMULATION_DATA
