@echo off
set OBJ_DIR=..\bin

REM Verifica si se ha pasado un argumento
if "%1"=="" (
    echo No se ha proporcionado ningun argumento.
    echo Uso: run_sanitizer.bat [memcheck|racecheck|initcheck|synccheck]
    echo Por defecto se correra memcheck
    compute-sanitizer --tool memcheck %OBJ_DIR%\cuNBSim.exe
)

REM Selecciona el comando basado en el argumento
if "%1"=="memcheck" (
    echo Ejecutando memcheck...
    compute-sanitizer --tool memcheck %OBJ_DIR%\cuNBSim.exe
) else if "%1"=="racecheck" (
    echo Ejecutando racecheck...
    compute-sanitizer --tool racecheck %OBJ_DIR%\cuNBSim.exe
) else if "%1"=="initcheck" (
    echo Ejecutando initcheck...
    compute-sanitizer --tool initcheck %OBJ_DIR%\cuNBSim.exe
) else if "%1"=="synccheck" (
    echo Ejecutando synccheck...
    compute-sanitizer --tool synccheck %OBJ_DIR%\cuNBSim.exe
) else (
    echo Argumento no valido. Uso: run_profiling.bat [basic|complete]
)

pause
