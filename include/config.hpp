#ifndef CONFIG_HPP
#define CONFIG_HPP

const float dt = 0.01f;
const int nIters = 100;
const int nBodies = 256;

//^ const int nBodies = 2<<11;

//~ Declarar las variables como 'extern' (esto dice que estas variables están definidas en otro archivo)
extern const char* data_directory;
extern const char* simulation_data_file_name;
#endif // CONFIG_HPP


