@echo off
REM Quick build script for Nanofluid Simulator Desktop App (Windows)

echo.
echo ╔═══════════════════════════════════════════════════════════╗
echo ║   Building Nanofluid Simulator Desktop Application       ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.

REM Check Python
echo ✓ Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found! Install Python 3.9+ first.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo ✓ Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo ✓ Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo ✓ Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements-desktop.txt

REM Clean previous builds
echo ✓ Cleaning previous builds...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist

REM Build application
echo ✓ Building application with PyInstaller...
pyinstaller nanofluid_app.spec

REM Check if build successful
if exist "dist\NanofluidSimulator\NanofluidSimulator.exe" (
    echo.
    echo ╔═══════════════════════════════════════════════════════════╗
    echo ║              ✅ BUILD SUCCESSFUL!                        ║
    echo ╚═══════════════════════════════════════════════════════════╝
    echo.
    echo 📦 Your application is ready in:
    echo    dist\NanofluidSimulator\
    echo.
    echo 🚀 To run:
    echo    dist\NanofluidSimulator\NanofluidSimulator.exe
    echo.
    echo 📚 Next steps:
    echo    1. Test the application
    echo    2. Create installer with NSIS/Inno Setup (see BUILD_DESKTOP_APP.md)
    echo    3. Distribute to users!
    echo.
    pause
) else (
    echo.
    echo ╔═══════════════════════════════════════════════════════════╗
    echo ║              ❌ BUILD FAILED!                            ║
    echo ╚═══════════════════════════════════════════════════════════╝
    echo.
    echo Check the build log above for errors.
    echo Common issues:
    echo   • Missing dependencies → pip install -r requirements-desktop.txt
    echo   • Import errors → Ensure nanofluid_simulator is in PYTHONPATH
    echo   • Permission errors → Run as administrator
    echo.
    pause
    exit /b 1
)
