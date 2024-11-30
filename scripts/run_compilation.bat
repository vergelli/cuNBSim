@echo off
:: Compilar todos los archivos relevantes del proyecto

:: Definir las rutas de los directorios
set SRC_DIR=..\src
set KERNEL_DIR=..\kernels
set OBJ_DIR=..\bin
set INCLUDE_DIR=..\include
set DEVICE_DIR= %SRC_DIR%\device
set HOST_DIR= %SRC_DIR%\host

:: Compilar los archivos
nvcc -std=c++17 ^
%DEVICE_DIR%\deviceProps.cu ^
%DEVICE_DIR%\integrateWraper.cu ^
%DEVICE_DIR%\bodyForceWraper.cu ^
%DEVICE_DIR%\boxMullerWraper.cu ^
%DEVICE_DIR%\data_collector.cu ^
%SRC_DIR%\simulator.cu ^
%KERNEL_DIR%\integrate.cu ^
%KERNEL_DIR%\bodyForce.cu ^
%KERNEL_DIR%\boxMuller.cu ^
%KERNEL_DIR%\memory_management.cu ^
%HOST_DIR%\utils.cpp ^
%HOST_DIR%\config.cpp ^
-o %OBJ_DIR%\cuNBSim.exe ^
-lcurand ^
-I %INCLUDE_DIR% -O3 -arch=sm_80 -lineinfo -diag-suppress=611

pause
