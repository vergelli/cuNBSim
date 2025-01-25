::! ESPERA MUCHOS CAMBIOS EN EL FUTURO !::
REM Este script ejecuta el script de Julia que renderiza la animacion de la simulacion 
REM NBody en sus proyecciones solamente

@echo off
echo Running renderNBodySimAnimation3.jl script with Julia...
julia ..\postprocess\renderNBodySimAnimation3.jl
echo Script execution finished.
pause
