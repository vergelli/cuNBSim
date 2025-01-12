#include "config.hpp"
#include "validation.hpp"
#include "json.hpp"
#include "deviceProps.cuh"
#include <iostream>
#include <fstream>
#include <string> 

void initialize_default_values() {

    std::cout << "INFO - Initializing default variables" << std::endl;

    const  std::string default_config_path = "../config/config.json";
    data_directory = "../data/";
    simulation_data_file_name = "simulation_data.csv";
    std::cout << "INFO - Default values initialized:" << std::endl;
    std::cout << "INFO - data_directory: " << data_directory << std::endl;
    std::cout << "INFO - simulation_data_file_name: " << simulation_data_file_name << std::endl;
}

void initialize_configuration_variables(const nlohmann::json& config) {

    std::cout << "INFO - Initializing configuration variables" << std::endl;

    nBodies = config["simulation"]["nBodies"];
    nIters = config["simulation"]["nIters"];
    dt = config["simulation"]["dt"];
    //* position_initializer ------------------------------------------
    for (const auto& item : config["simulation"]["position"].items()) {
        if (item.value().contains("on") && item.value()["on"]) {
            position_initializer = item.key();
            break;
        }
    }
    if (position_initializer == "box-muller") {
        position_std_dev_x = config["simulation"]["position"]["box-muller"]["position_std_dev_x"];
        position_std_dev_y = config["simulation"]["position"]["box-muller"]["position_std_dev_y"];
        position_std_dev_z = config["simulation"]["position"]["box-muller"]["position_std_dev_z"];
        pi_value = config["simulation"]["position"]["box-muller"]["pi_value"];
    }
    max_particles_speed = config["simulation"]["velocity"]["max_particles_speed"];
    MASS_SOFTENING = config["simulation"]["mass"]["mass_softening"];
    G = config["simulation"]["force"]["G"];
    SOFTENING = config["simulation"]["force"]["SOFTENING"];
    MIN_DISTANCE_THRESHOLD = config["simulation"]["force"]["MIN_DISTANCE_TRESHOLD"];
    //* numeric integration ------------------------------------------
    for (const auto& item : config["simulation"]["numeric-integration"].items()) {
        if (item.value().contains("on") && item.value()["on"]) {
            numerical_integrator = item.key();
            break;
        }
    }
    //* device --------------------------------------------------------
    launch_params_automatic = config["device"]["launch-params-automatic"];
    if (!launch_params_automatic) {
        gridDimX = config["device"]["launch-params-manual"]["gridDimX"];
        blockDimX = config["device"]["launch-params-manual"]["blockDimX"];
        gridDimY = config["device"]["launch-params-manual"]["gridDimY"];
        blockDimY = config["device"]["launch-params-manual"]["blockDimY"];
        gridDimZ = config["device"]["launch-params-manual"]["gridDimZ"];
        blockDimZ = config["device"]["launch-params-manual"]["blockDimZ"];
        integrateStride = config["device"]["launch-params-manual"]["IntegrateStride"];
    }

    std::cout << "INFO - Configuration values initialized:" << std::endl;
    std::cout << "INFO - nBodies: " << nBodies << std::endl;
    std::cout << "INFO - nIters: " << nIters << std::endl;
    std::cout << "INFO - dt: " << dt << std::endl;
    std::cout << "INFO - position_initializer: " << position_initializer << std::endl;
    if (position_initializer == "box-muller") {
        std::cout << "INFO - position_std_dev_x: " << position_std_dev_x << std::endl;
        std::cout << "INFO - position_std_dev_y: " << position_std_dev_y << std::endl;
        std::cout << "INFO - position_std_dev_z: " << position_std_dev_z << std::endl;
        std::cout << "INFO - pi_value: " << pi_value << std::endl;
    }
    std::cout << "INFO - max_particles_speed: " << max_particles_speed << std::endl;
    std::cout << "INFO - MASS_SOFTENING: " << MASS_SOFTENING << std::endl;
    std::cout << "INFO - G: " << G << std::endl;
    std::cout << "INFO - SOFTENING: " << SOFTENING << std::endl;
    std::cout << "INFO - MIN_DISTANCE_THRESHOLD: " << MIN_DISTANCE_THRESHOLD << std::endl;
    std::cout << "INFO - numerical_integrator: " << numerical_integrator << std::endl;
    std::cout << "INFO - launch_params_automatic: " << launch_params_automatic << std::endl;
    if (!launch_params_automatic) {
        std::cout << "INFO - gridDimX: " << gridDimX << std::endl;
        std::cout << "INFO - blockDimX: " << blockDimX << std::endl;
        std::cout << "INFO - gridDimY: " << gridDimY << std::endl;
        std::cout << "INFO - blockDimY: " << blockDimY << std::endl;
        std::cout << "INFO - gridDimZ: " << gridDimZ << std::endl;
        std::cout << "INFO - blockDimZ: " << blockDimZ << std::endl;
        std::cout << "INFO - integrateStride: " << integrateStride << std::endl;
    }
}

void load_config_from_file(const std::string& config_file, DeviceProperties deviceProps) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "ERROR - No se pudo abrir el archivo de configuración." << std::endl;
        return;
    }
    nlohmann::json config;
    file >> config;
    config_file_validation_routines(config, deviceProps);
    initialize_default_values();
    initialize_configuration_variables(config);
    std::cout << "INFO - Configuration file loaded." << std::endl;

}
