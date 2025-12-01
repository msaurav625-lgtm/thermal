#!/bin/bash
# Quick build script for Nanofluid Simulator Desktop App

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   Building Nanofluid Simulator Desktop Application       ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check Python
echo "✓ Checking Python installation..."
python --version || { echo "Error: Python not found!"; exit 1; }

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "✓ Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Install/upgrade dependencies
echo "✓ Installing dependencies..."
pip install --upgrade pip
pip install -r requirements-desktop.txt

# Clean previous builds
echo "✓ Cleaning previous builds..."
rm -rf build/ dist/

# Build application
echo "✓ Building application with PyInstaller..."
pyinstaller nanofluid_app.spec

# Check if build successful
if [ -f "dist/NanofluidSimulator/NanofluidSimulator.exe" ] || [ -f "dist/NanofluidSimulator/NanofluidSimulator" ]; then
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║              ✅ BUILD SUCCESSFUL!                        ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    echo "📦 Your application is ready in:"
    echo "   dist/NanofluidSimulator/"
    echo ""
    echo "🚀 To run:"
    echo "   Windows: dist\\NanofluidSimulator\\NanofluidSimulator.exe"
    echo "   Linux:   dist/NanofluidSimulator/NanofluidSimulator"
    echo ""
    echo "📚 Next steps:"
    echo "   1. Test the application"
    echo "   2. Create installer with NSIS/Inno Setup (see BUILD_DESKTOP_APP.md)"
    echo "   3. Distribute to users!"
    echo ""
else
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║              ❌ BUILD FAILED!                            ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    echo "Check the build log above for errors."
    echo "Common issues:"
    echo "  • Missing dependencies → pip install -r requirements-desktop.txt"
    echo "  • Import errors → Ensure nanofluid_simulator is in PYTHONPATH"
    echo "  • Permission errors → Run as administrator"
    exit 1
fi
