@echo off
:: Si se proporciona un argumento, usarlo como el path del archivo de configuración
if "%1"=="" (
    echo INFO - No configuration file path provided, using default configuration path: config\default.cfg
    ..\bin\cuNBSim.exe
) else (
    echo INFO - Using configuration file: %1
    ..\bin\cuNBSim.exe %1
)
pause
