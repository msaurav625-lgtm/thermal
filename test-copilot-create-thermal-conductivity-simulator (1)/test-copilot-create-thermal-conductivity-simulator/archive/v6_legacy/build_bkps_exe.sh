#!/bin/bash
# Build script for BKPS NFL Thermal v6.0 executable (Linux/Mac)
# Dedicated to: Brijesh Kumar Pandey

echo "================================================================================"
echo "BKPS NFL Thermal v6.0 - Executable Builder"
echo "Dedicated to: Brijesh Kumar Pandey"
echo "================================================================================"
echo ""

echo "[1/5] Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python not found! Please install Python 3.8+"
    exit 1
fi

echo ""
echo "[2/5] Installing dependencies..."
pip3 install --upgrade pip
pip3 install numpy scipy matplotlib PyQt6
pip3 install pyinstaller

echo ""
echo "[3/5] Cleaning previous builds..."
rm -rf build dist
rm -f BKPS_NFL_Thermal_v6.0

echo ""
echo "[4/5] Building executable with PyInstaller..."
pyinstaller --clean --noconfirm bkps_nfl_thermal.spec

echo ""
echo "[5/5] Finalizing..."
if [ -f "dist/BKPS_NFL_Thermal_v6.0" ]; then
    mv dist/BKPS_NFL_Thermal_v6.0 .
    chmod +x BKPS_NFL_Thermal_v6.0
    echo ""
    echo "================================================================================"
    echo "SUCCESS! Executable created: BKPS_NFL_Thermal_v6.0"
    echo "================================================================================"
    echo ""
    echo "File size:"
    ls -lh BKPS_NFL_Thermal_v6.0 | awk '{print $5, $9}'
    echo ""
    echo "You can now run: ./BKPS_NFL_Thermal_v6.0"
    echo ""
    echo "Cleaning up temporary files..."
    rm -rf build dist
    rm -f BKPS_NFL_Thermal_v6.0.spec
    echo ""
    echo "DONE! Your standalone application is ready."
    echo ""
else
    echo ""
    echo "ERROR: Build failed! Check the output above for errors."
    echo ""
    exit 1
fi
