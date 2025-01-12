#ifndef CONFIG_HPP
#define CONFIG_HPP
#include <string> 

extern const std::string default_config_path;

extern std::string data_directory;
extern std::string simulation_data_file_name;
extern std::string position_initializer;
extern std::string numerical_integrator;
extern int nIters;
extern int nBodies;
extern int gridDimX;
extern int blockDimX;
extern int gridDimY;
extern int blockDimY;
extern int gridDimZ;
extern int blockDimZ;
extern int integrateStride;
extern float position_std_dev_x;
extern float position_std_dev_y;
extern float position_std_dev_z;
extern float pi_value;
extern float dt;
extern float max_particles_speed;
extern float G;
extern float SOFTENING;
extern float MIN_DISTANCE_THRESHOLD;
extern float MASS_SOFTENING;
extern bool launch_params_automatic;

void load_config_from_file(const std::string& config_file, DeviceProperties deviceProps);

#endif // CONFIG_HPP