#include <filesystem>  // Para crear directorios (C++17 en adelante)
#include <fstream>
#include <iostream>
#include <string>
#include "body.cuh"
#include "config.hpp"
#include "cuda_utils.cuh"
namespace fs = std::filesystem;

void logSimulationData(Body* p, int nBodies, int iter) {
    
    //* Definir la ruta del archivo
    std::string file_path = std::string(data_directory) + simulation_data_file_name;

    //* Verificar si la carpeta 'data' existe, si no, crearla
    std::string data_directory_str = std::string(data_directory); 
    if (!fs::exists(data_directory_str)) {
        if (!fs::create_directories(data_directory_str)) {
            std::cerr << "ERROR: Could not create the directory: " << data_directory_str << std::endl;
            return;
        }
    }

    //* Verificar si el archivo ya existe
    bool fileExists = fs::exists(file_path);

    //* Abrir el archivo en modo append
    std::ofstream csvFile(file_path, std::ios_base::app);
    
    //* Validar si el archivo se abrió correctamente
    if (!csvFile.is_open()) {
        std::cerr << "ERROR: Can't open the file " << file_path << " to write data." << std::endl;
        return;
    }

    //* Si el archivo es nuevo, escribir la cabecera
    if (!fileExists) {
        csvFile << "Iteration,BodyID,PosX,PosY,PosZ,VelX,VelY,VelZ\n";
    }

    //* Escribir los datos de la simulación en el archivo
    for (int i = 0; i < nBodies; i++) {
        csvFile << iter << "," << i << "," << p[i].x << "," << p[i].y << "," << p[i].z << "," 
                << p[i].vx << "," << p[i].vy << "," << p[i].vz << "\n";
    }

    //* Cerrar el archivo
    csvFile.close();

    //* Validar si el archivo se cerró correctamente
    if (csvFile.fail()) {
        std::cerr << "ERROR: File could not close properly." << std::endl;
    } else {
        printf("Simulation data gracefully saved on %s\n", file_path.c_str());
    }
}

void simulationDataCollection(
    Body* p, 
    Body* p_device, 
    int nBodies, 
    int bytes, 
    int iter) {

    CHECK_CUDA_ERROR(cudaMemcpy(p, p_device, bytes, cudaMemcpyDeviceToHost));
    logSimulationData(p, nBodies, iter);
}
