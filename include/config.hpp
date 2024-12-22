#ifndef CONFIG_HPP
#define CONFIG_HPP
#include <string> 

extern float dt;
extern int nIters;
extern int nBodies;
extern float max_particles_speed;
extern const std::string data_directory;
extern const std::string default_config_path;
extern const std::string simulation_data_file_name;

void load_config_from_file(const std::string& config_file);

#endif // CONFIG_HPP