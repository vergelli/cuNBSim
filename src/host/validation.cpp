#include "config.hpp"
#include "json.hpp"
#include "deviceProps.cuh"
#include <iostream>
#include <fstream>
#include <string>

//& Fields validations =================================================================================

void io_fields_validations(const nlohmann::json& config) {
    std::cout << "INFO - Checking IO fields." << std::endl;
    if (!config.contains("io")) {
        std::cerr << "ERROR - Configuration file does not contain 'io' section." << std::endl;
        exit(1);
    }
    if (!config["io"].contains("data_directory")) {
        std::cerr << "ERROR - 'io' section does not contain 'data_directory' field." << std::endl;
        exit(1);
    }
    if (!config["io"].contains("simulation_data_file_name")) {
        std::cerr << "ERROR - 'io' section does not contain 'simulation_data_file_name' field." << std::endl;
        exit(1);
    }
}

void simulation_fields_validations(const nlohmann::json& config) {
    std::cout << "INFO - Checking Simulation fields." << std::endl;
    if (!config.contains("simulation")) {
        std::cerr << "ERROR - Configuration file does not contain 'simulation' section." << std::endl;
        exit(1);
    }
    if (!config["simulation"].contains("nBodies")) {
        std::cerr << "ERROR - 'simulation' section does not contain 'nBodies' field." << std::endl;
        exit(1);
    }
    if (!config["simulation"].contains("nIters")) {
        std::cerr << "ERROR - 'simulation' section does not contain 'nIters' field." << std::endl;
        exit(1);
    }
    if (!config["simulation"].contains("dt")) {
        std::cerr << "ERROR - 'simulation' section does not contain 'dt' field." << std::endl;
        exit(1);
    }
    if (!config["simulation"].contains("velocity") || !config["simulation"]["velocity"].contains("max_particles_speed")) {
        std::cerr << "ERROR - 'simulation' section does not contain 'max_particles_speed' field in 'velocity'." << std::endl;
        exit(1);
    }

    //* POSITION =================================================================================
    if (!config["simulation"].contains("position")) {
        std::cerr << "ERROR - 'simulation' section does not contain 'position' field." << std::endl;
        exit(1);
    } else {
        if (!config["simulation"]["position"].contains("box-muller")) {
            std::cerr << "ERROR - 'position' section does not contain 'box-muller' field or 'on' field in 'box-muller'." << std::endl;
            exit(1);
        }
        if (!config["simulation"]["position"].contains("marsaglia-bray")) {
            std::cerr << "ERROR - 'position' section does not contain 'marsaglia-bray' field or 'on' field in 'marsaglia-bray'." << std::endl;
            exit(1);
        }
    }
    if (!config["simulation"].contains("mass") || !config["simulation"]["mass"].contains("MASS_SOFTENING")) {
        std::cerr << "ERROR - 'simulation' section does not contain 'mass' field or 'MASS_SOFTENING' field in 'mass'." << std::endl;
        exit(1);
    }

    //* FORCE ===================================================================================
    if (!config["simulation"].contains("force")) {
        std::cerr << "ERROR - 'simulation' section does not contain 'force' field." << std::endl;
        exit(1);
    } else {
        if (!config["simulation"]["force"].contains("G")) {
            std::cerr << "ERROR - 'force' section does not contain 'G' field." << std::endl;
            exit(1);
        }
        if (!config["simulation"]["force"].contains("SOFTENING")) {
            std::cerr << "ERROR - 'force' section does not contain 'SOFTENING' field." << std::endl;
            exit(1);
        }
        if (!config["simulation"]["force"].contains("MIN_DISTANCE_TRESHOLD")) {
            std::cerr << "ERROR - 'force' section does not contain 'MIN_DISTANCE_TRESHOLD' field." << std::endl;
            exit(1);
        }
    }

    //* NUMERIC INTEGRATION ==========================================================================
    if (!config["simulation"].contains("numeric-integration")) {
        std::cerr << "ERROR - 'simulation' section does not contain 'numeric-integration' field." << std::endl;
        exit(1);
    } else {
        if (!config["simulation"]["numeric-integration"].contains("euler-explicit")) {
            std::cerr << "ERROR - 'numeric-integration' section does not contain 'euler-explicit' field or 'on' field in 'euler-explicit'." << std::endl;
            exit(1);
        }
        if (!config["simulation"]["numeric-integration"].contains("leap-frog")) {
            std::cerr << "ERROR - 'numeric-integration' section does not contain 'leap-frog' field or 'on' field in 'leap-frog'." << std::endl;
            exit(1);
        }
    }
}

void device_fields_validations(const nlohmann::json& config) {
    std::cout << "INFO - Checking device fields." << std::endl;
    if (!config.contains("device")) {
        std::cerr << "ERROR - Configuration file does not contain 'device' section." << std::endl;
        exit(1);
    }
    if (!config["device"].contains("launch-params-automatic")) {
        std::cerr << "ERROR - 'device' section does not contain 'launch-params-automatic' field." << std::endl;
        exit(1);
    }
    if (!config["device"].contains("launch-params-manual")) {
        std::cerr << "ERROR - 'device' section does not contain 'launch-params-manual' field." << std::endl;
        exit(1);
    }
    if (!config["device"]["launch-params-manual"].contains("gridDimX")) {
        std::cerr << "ERROR - 'launch-params-manual' does not contain 'gridDimX' field." << std::endl;
        exit(1);
    }
    if (!config["device"]["launch-params-manual"].contains("blockDimX")) {
        std::cerr << "ERROR - 'launch-params-manual' does not contain 'blockDimX' field." << std::endl;
        exit(1);
    };
    if (!config["device"]["launch-params-manual"].contains("gridDimY")) {
        std::cerr << "ERROR - 'launch-params-manual' does not contain 'gridDimY' field." << std::endl;
        exit(1);
    }
    if (!config["device"]["launch-params-manual"].contains("blockDimY")) {
        std::cerr << "ERROR - 'launch-params-manual' does not contain 'blockDimY' field." << std::endl;
        exit(1);
    };
    if (!config["device"]["launch-params-manual"].contains("gridDimZ")) {
        std::cerr << "ERROR - 'launch-params-manual' does not contain 'gridDimZ' field." << std::endl;
        exit(1);
    }
    if (!config["device"]["launch-params-manual"].contains("blockDimZ")) {
        std::cerr << "ERROR - 'launch-params-manual' does not contain 'blockDimZ' field." << std::endl;
        exit(1);
    };
    if (!config["device"]["launch-params-manual"].contains("integrateStride")) {
        std::cerr << "ERROR - 'launch-params-manual' does not contain 'integrateStride' field." << std::endl;
        exit(1);
    };
    

}

//& Values validations =================================================================================

void io_values_validations(const nlohmann::json& config) {
    std::cout << "INFO - Checking IO values." << std::endl;
    if (!config["io"]["data_directory"].is_string()) {
        std::cerr << "ERROR - 'data_directory' should be a string." << std::endl;
        exit(1);
    }
    if (!config["io"]["simulation_data_file_name"].is_string()) {
        std::cerr << "ERROR - 'simulation_data_file_name' should be a string." << std::endl;
        exit(1);
    }
}

void simulation_values_validations(const nlohmann::json& config) {
    std::cout << "INFO - Checking Simulation values." << std::endl;

    //* General =================================================================================
    std::cout << "INFO - Checking simulation general values." << std::endl;
    if (!config["simulation"]["nBodies"].is_number_integer()) {
        std::cerr << "ERROR - 'nBodies' should be an integer." << std::endl;
        exit(1);
    }
    if (!config["simulation"]["nIters"].is_number_integer()) {
        std::cerr << "ERROR - 'nIters' should be an integer." << std::endl;
        exit(1);
    }
    if (!config["simulation"]["dt"].is_number()) {
        std::cerr << "ERROR - 'dt' should be a number." << std::endl;
        exit(1);
    }

    //* VELOCITY =================================================================================

    std::cout << "INFO - Checking simulation velocity values." << std::endl;

    if (!config["simulation"]["velocity"]["max_particles_speed"].is_number()) {
        std::cerr << "ERROR - 'max_particles_speed' should be a number." << std::endl;
        exit(1);
    }

    //* POSITION =================================================================================

    std::cout << "INFO - Checking simulation position values." << std::endl;

    if (!config["simulation"]["position"]["box-muller"]["on"].is_boolean()) {
        std::cerr << "ERROR - 'on' in 'box-muller' should be a boolean." << std::endl;
        exit(1);
    }
    if (!config["simulation"]["position"]["box-muller"]["position_std_dev_x"].is_number()) {
        std::cerr << "ERROR - 'position_std_dev_x' in 'box-muller' should be a number." << std::endl;
        exit(1);
    }
    if (!config["simulation"]["position"]["box-muller"]["position_std_dev_y"].is_number()) {
        std::cerr << "ERROR - 'position_std_dev_y' in 'box-muller' should be a number." << std::endl;
        exit(1);
    }
    if (!config["simulation"]["position"]["box-muller"]["position_std_dev_z"].is_number()) {
        std::cerr << "ERROR - 'position_std_dev_z' in 'box-muller' should be a number." << std::endl;
        exit(1);
    }
    if (!config["simulation"]["position"]["box-muller"]["pi_value"].is_number_float()) {
        std::cerr << "ERROR - 'pi_value' in 'box-muller' should be a float." << std::endl;
        exit(1);
    } else {
        float pi_value = config["simulation"]["position"]["box-muller"]["pi_value"];
        const float MIN_FLOAT16 = -65504.0f;
        const float MAX_FLOAT16 = 65504.0f;
        if (pi_value < MIN_FLOAT16 || pi_value > MAX_FLOAT16) {
            std::cerr << "WARNING - 'pi_value' in 'box-muller' is not a 16-bit float (float16). Using higher precision (e.g., float32 or float64) may result in slower performance. Consider using float16 for optimal performance when possible." << std::endl;
        }
        if (sizeof(pi_value) == 8) {
            std::cerr << "WARNING - 'pi_value' is a 64-bit float (float64). Using float64 may result in significantly slower performance. Consider using float16 or float32 for better GPU performance." << std::endl;
        }
    }

    //* MASS =====================================================================================
    std::cout << "INFO - Checking simulation mass values." << std::endl;
    if (!config["simulation"]["mass"]["MASS_SOFTENING"].is_number()) {
        std::cerr << "ERROR - 'MASS_SOFTENING' in 'mass' should be a number." << std::endl;
        exit(1);
    }

    //* FORCE ====================================================================================
    std::cout << "INFO - Checking simulation force values." << std::endl;
    if (!config["simulation"]["force"]["G"].is_number()) {
        std::cerr << "ERROR - 'G' in 'force' should be a number." << std::endl;
        exit(1);
    }
    if (!config["simulation"]["force"]["SOFTENING"].is_number()) {
        std::cerr << "ERROR - 'SOFTENING' in 'force' should be a number." << std::endl;
        exit(1);
    }
    if (!config["simulation"]["force"]["MIN_DISTANCE_TRESHOLD"].is_number()) {
        std::cerr << "ERROR - 'MIN_DISTANCE_TRESHOLD' in 'force' should be a number." << std::endl;
        exit(1);
    }

    //* NUMERIC-INTEGRATION ======================================================================
    std::cout << "INFO - Checking simulation numeric-integration values." << std::endl;
    if (!config["simulation"]["numeric-integration"]["euler-explicit"]["on"].is_boolean()) {
        std::cerr << "ERROR - 'on' in 'euler-explicit' should be a boolean." << std::endl;
        exit(1);
    }
    if (!config["simulation"]["numeric-integration"]["leap-frog"]["on"].is_boolean()) {
        std::cerr << "ERROR - 'on' in 'leap-frog' should be a boolean." << std::endl;
        exit(1);
    }
}

void device_values_validations(const nlohmann::json& config, DeviceProperties deviceProps) {
    std::cout << "INFO - Checking device values." << std::endl;

    if (!config["device"]["launch-params-automatic"].is_boolean()) {
        std::cerr << "ERROR - 'launch-params-automatic' should be a boolean." << std::endl;
        exit(1);
    }
    if (!config["device"]["launch-params-manual"]["gridDimX"].is_number_integer()) {
        std::cerr << "ERROR - 'gridDimX' should be an integer." << std::endl;
        exit(1);
    }
    if (!config["device"]["launch-params-manual"]["blockDimX"].is_number_integer()) {
        std::cerr << "ERROR - 'blockDimX' should be an integer." << std::endl;
        exit(1);
    }
    if (!config["device"]["launch-params-manual"]["gridDimY"].is_number_integer()) {
        std::cerr << "ERROR - 'gridDimY' should be an integer." << std::endl;
        exit(1);
    }
    if (!config["device"]["launch-params-manual"]["blockDimY"].is_number_integer()) {
        std::cerr << "ERROR - 'blockDimY' should be an integer." << std::endl;
        exit(1);
    }
        if (!config["device"]["launch-params-manual"]["gridDimZ"].is_number_integer()) {
        std::cerr << "ERROR - 'gridDimZ' should be an integer." << std::endl;
        exit(1);
    }
    if (!config["device"]["launch-params-manual"]["blockDimZ"].is_number_integer()) {
        std::cerr << "ERROR - 'blockDimZ' should be an integer." << std::endl;
        exit(1);
    }
    if (!config["device"]["launch-params-manual"]["integrateStride"].is_number_integer()) {
        std::cerr << "ERROR - 'integrateStride' should be an integer." << std::endl;
        exit(1);
    }

    std::cout << "INFO - Checking device values against detected CUDA device" << std::endl;
    {
        int gridDimX = config["device"]["launch-params-manual"]["gridDimX"];
        int gridDimY = config["device"]["launch-params-manual"]["gridDimY"];
        int gridDimZ = config["device"]["launch-params-manual"]["gridDimZ"];
        int blockDimX = config["device"]["launch-params-manual"]["blockDimX"];
        int blockDimY = config["device"]["launch-params-manual"]["blockDimY"];
        int blockDimZ = config["device"]["launch-params-manual"]["blockDimZ"];
        int totalThreads = blockDimX * blockDimY * blockDimZ;

        if ((gridDimX <= 1 && gridDimY <= 1 && gridDimZ <= 1) || 
            (blockDimX <= 1 && blockDimY <= 1 && blockDimZ <= 1)) {
            std::cerr << "ERROR - At least one of the grid dimensions and block dimensions must be greater than 1." << std::endl;
            exit(1);
        }

        if (gridDimX > deviceProps.maxGridDimX || gridDimY > deviceProps.maxGridDimY || gridDimZ > deviceProps.maxGridDimZ) {
            std::cerr << "ERROR - Grid dimensions exceed the maximum allowed values." << std::endl;
            exit(1);
        }

        if (blockDimX > deviceProps.maxBlockDimX || blockDimY > deviceProps.maxBlockDimY || blockDimZ > deviceProps.maxBlockDimZ) {
            std::cerr << "ERROR - Block dimensions exceed the maximum allowed values." << std::endl;
            exit(1);
        }

        if (gridDimX == 0 || gridDimY == 0 || gridDimZ == 0 || 
            blockDimX == 0 || blockDimY == 0 || blockDimZ == 0) {
            std::cerr << "ERROR - None of the grid or block dimensions can be zero." << std::endl;
            exit(1);
        }

        //~ Dimensionality validation
        //& 1D
        if (blockDimY == 1 && blockDimZ == 1) {
            if (blockDimX > 1024) {
                throw std::runtime_error("ERROR - Exceeding max threads for 1D block (1024 threads max).");
            }
            std::cout << "INFO - Valid 1D block. Total threads: " << totalThreads << std::endl;
        }
        //& 2D
        else if (blockDimZ == 1) {
            if (blockDimX > 32 || blockDimY > 32 || totalThreads > 1024) {
                throw std::runtime_error("ERROR - Exceeding max threads for 2D block (32x32 threads max, 1024 threads total).");
            }
            std::cout << "INFO - Valid 2D block. Total threads: " << totalThreads << std::endl;
        }
        //& 3D
        else {
            if (blockDimX > 10 || blockDimY > 10 || blockDimZ > 10 || totalThreads > 1024) {
                throw std::runtime_error("ERROR - Exceeding max threads for 3D block (10x10x10 threads max, 1024 threads total).");
            }
            std::cout << "INFO - Valid 3D block. Total threads: " << totalThreads << std::endl;
        }
    }
}

//& Switches validations ===============================================================================

void validate_switches(const nlohmann::json& switches) {
    std::cout << "DEBUG - Starting switch validation." << std::endl;
    int true_count = 0;
    for (const auto& item : switches.items()) {
        std::cout << "DEBUG - Checking switch: " << item.key() << " with value: " << item.value() << std::endl;
        if (item.value()) {
            true_count++;
        }
    }
    std::cout << "DEBUG - Number of true switches: " << true_count << std::endl;
    if (true_count != 1) {
        std::cerr << "ERROR - Exactly one switch must be true in the following options: ";
        for (const auto& item : switches.items()) {
            std::cerr << "'" << item.key() << "' ";
        }
        std::cerr << std::endl;
        exit(1);
    }
    std::cout << "DEBUG - Switch validation completed successfully." << std::endl;
}

void switches_validations(const nlohmann::json& config) {

    validate_switches(config["simulation"]["position"]);
    std::cout << "DEBUG - validate_switches PASSED." << std::endl;

    validate_switches(config["simulation"]["numeric-integration"]);
    std::cout << "DEBUG - validate_switches PASSED." << std::endl;

    std::cout << "INFO - All switches valid." << std::endl;

}

//& Validation routines main function ===============================================================================

void config_file_validation_routines(const nlohmann::json& config, DeviceProperties deviceProps) {

    std::cout << "INFO - Starting configuration file validations." << std::endl;
    std::cout << "INFO - Checking mandatory fields." << std::endl;

    io_fields_validations(config);
    simulation_fields_validations(config);
    device_fields_validations(config);

    std::cout << "INFO - Checking configuration values." << std::endl;

    io_values_validations(config);
    simulation_values_validations(config);
    device_values_validations(config, deviceProps);

    std::cout << "INFO - Checking configuration switches." << std::endl;

    switches_validations(config);

    std::cout << "INFO - Configuration file is valid." << std::endl;

}