@echo off
REM Verifica si se ha pasado un argumento
if "%1"=="" (
    echo No se ha proporcionado ningun argumento.
    echo Uso: run_profiling.bat [basic|complete]
    pause
    exit /b
)

REM Selecciona el comando basado en el argumento
if "%1"=="basic" (
    echo Ejecutando perfilado basico...
    nsys profile --stats=true -o ..\profiling\cuNBSim_basic ..\bin\cuNBSim.exe
) else if "%1"=="complete" (
    echo Ejecutando perfilado completo...
    nsys profile --stats=true --gpu-metrics-devices=all --cuda-memory-usage=true --show-output=true -o ..\profiling\cuNBSim_complete ..\bin\cuNBSim.exe
) else (
    echo Argumento no valido. Uso: run_profiling.bat [basic|complete]
)

pause
