#include "config.hpp"
#include <iostream>

std::string data_directory;
std::string simulation_data_file_name;
const std::string default_config_path = "../config/config.json";
int nIters;
int nBodies;
int gridDimX;
int blockDimX;
int integrateStride;
float dt;
float max_particles_speed;
std::string position_initializer;
std::string numerical_integrator;
int gridDimY;
int blockDimY;
int gridDimZ;
int blockDimZ;
float position_std_dev_x;
float position_std_dev_y;
float position_std_dev_z;
float pi_value;
float G;
float SOFTENING;
float MIN_DISTANCE_THRESHOLD;
float MASS_SOFTENING;
bool launch_params_automatic;