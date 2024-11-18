@echo off
set OBJ_DIR=..\bin
compute-sanitizer --tool memcheck %OBJ_DIR%\cuNBSim.exe
pause




