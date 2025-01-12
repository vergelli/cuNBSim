#include "config.hpp"
#include "validation.hpp"
#include "json.hpp"
#include <iostream>
#include <fstream>
#include <string> 

//~ Definir las variables globales aquí (una única vez)
const  std::string data_directory = "../data/";
const  std::string simulation_data_file_name = "simulation_data.csv";
const  std::string default_config_path = "../config/config.json";

float dt = 0.001f;
float max_particles_speed = 1;
int nBodies = 256;
int nIters = 10;


void load_config_from_file(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "ERROR - No se pudo abrir el archivo de configuración." << std::endl;
        return;
    }

    nlohmann::json config;
    file >> config;

    config_file_validation_routines(config);

    // Leer los valores desde el archivo JSON
    if (config.contains("nBodies")) {
        nBodies = config["nBodies"];
    }
    if (config.contains("nIters")) {
        nIters = config["nIters"];
    }
    if (config.contains("dt")) {
        dt = config["dt"];
    }
    if (config.contains("max_particles_speed")) {
        max_particles_speed = config["max_particles_speed"];
    }
}
