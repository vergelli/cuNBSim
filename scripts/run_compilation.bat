@echo off
nvcc -std=c++17 ..\src\simulatorToRefactor.cu -o ..\bin\cuNBSim.exe -O3 -arch=sm_80 -lineinfo -diag-suppress=611
pause
