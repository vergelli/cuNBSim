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
%DEVICE_DIR%\kernelsInit.cu ^
%DEVICE_DIR%\boxMullerWraper.cu ^
%DEVICE_DIR%\massWraper.cu^
%DEVICE_DIR%\velocityWrapper.cu ^
%DEVICE_DIR%\bodyForceWraper.cu ^
%DEVICE_DIR%\integrateWraper.cu ^
%DEVICE_DIR%\integratorLeapFrogWraper.cu ^
%DEVICE_DIR%\data_collector.cu ^
%SRC_DIR%\simulator.cu ^
%KERNEL_DIR%\boxMuller.cu ^
%KERNEL_DIR%\mass.cu ^
%KERNEL_DIR%\velocity.cu ^
%KERNEL_DIR%\bodyForce.cu ^
%KERNEL_DIR%\integrate.cu ^
%KERNEL_DIR%\integratorLeapFrog.cu ^
%KERNEL_DIR%\memory_management.cu ^
%HOST_DIR%\utils.cpp ^
%HOST_DIR%\config.cpp ^
%HOST_DIR%\globals.cpp ^
%HOST_DIR%\validation.cpp ^
-o %OBJ_DIR%\cuNBSim.exe ^
-lcurand ^
-I %INCLUDE_DIR% -O3 -arch=sm_80 -lineinfo -diag-suppress=611

pause
