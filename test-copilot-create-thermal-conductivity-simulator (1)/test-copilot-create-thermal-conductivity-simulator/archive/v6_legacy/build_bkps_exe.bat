@echo off
REM Build script for BKPS NFL Thermal v6.0 Windows executable
REM Dedicated to: Brijesh Kumar Pandey

echo ================================================================================
echo BKPS NFL Thermal v6.0 - Windows Executable Builder
echo Dedicated to: Brijesh Kumar Pandey
echo ================================================================================
echo.

echo [1/5] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo.
echo [2/5] Installing dependencies...
pip install --upgrade pip
pip install numpy scipy matplotlib PyQt6
pip install pyinstaller

echo.
echo [3/5] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist BKPS_NFL_Thermal_v6.0.exe del BKPS_NFL_Thermal_v6.0.exe

echo.
echo [4/5] Building executable with PyInstaller...
pyinstaller --clean --noconfirm bkps_nfl_thermal.spec

echo.
echo [5/5] Finalizing...
if exist dist\BKPS_NFL_Thermal_v6.0.exe (
    move dist\BKPS_NFL_Thermal_v6.0.exe .
    echo.
    echo ================================================================================
    echo SUCCESS! Executable created: BKPS_NFL_Thermal_v6.0.exe
    echo ================================================================================
    echo.
    echo File size:
    dir BKPS_NFL_Thermal_v6.0.exe | find "BKPS_NFL_Thermal"
    echo.
    echo You can now run: BKPS_NFL_Thermal_v6.0.exe
    echo.
    echo Cleaning up temporary files...
    rmdir /s /q build
    rmdir /s /q dist
    del BKPS_NFL_Thermal_v6.0.spec 2>nul
    echo.
    echo DONE! Your standalone application is ready.
    echo.
) else (
    echo.
    echo ERROR: Build failed! Check the output above for errors.
    echo.
    pause
    exit /b 1
)

pause
