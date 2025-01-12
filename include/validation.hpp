#ifndef VALIDATION_HPP
#define VALIDATION_HPP
#include "json.hpp"

void io_fields_validations(const nlohmann::json& config);
void simulation_fields_validations(const nlohmann::json& config);
void device_fields_validations(const nlohmann::json& config);
void io_values_validations(const nlohmann::json& config);
void simulation_values_validations(const nlohmann::json& config);
void device_values_validations(const nlohmann::json& config, DeviceProperties deviceProps);
void validate_switches(const nlohmann::json& switches);
void switches_validations(const nlohmann::json& config);
void config_file_validation_routines(const nlohmann::json& config, DeviceProperties deviceProps);

#endif // VALIDATION_HPP



