#ifndef LOG_SIMULATION_DATA
#define LOG_SIMULATION_DATA
#include "body.cuh"
#include <string> 

void simulationDataCollection(Body* p, Body* p_device, int nBodies, int bytes, int iter, std::string numerical_integrator);

void logSimulationData(Body* p, int nBodies, int iter, std::string numerical_integrator);

#endif // LOG_SIMULATION_DATA
