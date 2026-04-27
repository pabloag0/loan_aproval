@echo off
CALL C:\Users\pablo\miniforge3\Scripts\activate.bat montecarlo
cd /d "D:\onedrive\OneDrive - Universidad Complutense de Madrid (UCM)\Uni\3\2\AA\workspace\Proyecto"

:loop
cls
python main.py

echo.
echo ======================================
set /p choice=Pulsa ENTER para volver a ejecutar (o escribe exit y ENTER para salir):

if /i "%choice%"=="exit" exit
goto loop