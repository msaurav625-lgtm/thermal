# Installation Guide for Nanofluid Simulator v3.0

## Windows Installation

### Method 1: Use Pre-built Executable (Recommended)
**No installation needed!**

1. Download `NanofluidSimulator-Windows-x64.exe` from:
   https://github.com/msaurav625-lgtm/test/actions
   
2. Double-click the .exe file to run
3. Done! No Python or packages needed.

### Method 2: Run from Python Source

**Step 1: Install Python**
- Download Python 3.10 or 3.11 from https://www.python.org/downloads/
- During installation, check "Add Python to PATH"
- Verify: Open Command Prompt and type `python --version`

**Step 2: Install Core Packages Only**
```cmd
cd path\to\test
pip install numpy scipy matplotlib PyQt6
```

**Step 3: Run the simulator**
```cmd
python run_gui_v3.py
```

**Optional: Install All Features**
```cmd
pip install pandas openpyxl seaborn
```

---

## Linux Installation

### Method 1: Use Pre-built Binary (Recommended)
1. Download `NanofluidSimulator-Linux-x64` from GitHub Actions
2. Make it executable:
   ```bash
   chmod +x NanofluidSimulator-Linux-x64
   ./NanofluidSimulator-Linux-x64
   ```

### Method 2: Run from Python Source

**Step 1: Install Python and dependencies**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Fedora/RHEL
sudo dnf install python3 python3-pip

# Arch
sudo pacman -S python python-pip
```

**Step 2: Create virtual environment (recommended)**
```bash
cd /path/to/test
python3 -m venv venv
source venv/bin/activate
```

**Step 3: Install core packages**
```bash
pip install --upgrade pip
pip install numpy scipy matplotlib PyQt6
```

**Step 4: Run the simulator**
```bash
python run_gui_v3.py
```

---

## macOS Installation

### Method 1: Use Pre-built App
1. Download `NanofluidSimulator-macOS-x64` from GitHub Actions
2. Make executable and run:
   ```bash
   chmod +x NanofluidSimulator-macOS-x64
   ./NanofluidSimulator-macOS-x64
   ```

### Method 2: Run from Python Source

**Step 1: Install Homebrew (if not installed)**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Step 2: Install Python**
```bash
brew install python@3.11
```

**Step 3: Install packages**
```bash
cd /path/to/test
pip3 install numpy scipy matplotlib PyQt6
```

**Step 4: Run the simulator**
```bash
python3 run_gui_v3.py
```

---

## Troubleshooting Installation Issues

### Issue: "Operation cancelled by user"
**Solution:** The installation was interrupted. Try installing packages one by one:
```bash
pip install numpy
pip install scipy
pip install matplotlib
pip install PyQt6
```

### Issue: "pip not found"
**Solution:** 
```bash
# Linux/Mac
python3 -m ensurepip --upgrade

# Windows
python -m ensurepip --upgrade
```

### Issue: "Permission denied"
**Solution:** Use virtual environment or `--user` flag:
```bash
pip install --user numpy scipy matplotlib PyQt6
```

### Issue: PyQt6 installation fails
**Solutions:**

**On Linux:**
```bash
# Install Qt dependencies first
sudo apt install python3-pyqt6  # Ubuntu/Debian
# OR
sudo dnf install python3-qt6    # Fedora
```

**On macOS:**
```bash
# Use Homebrew
brew install pyqt@6
```

**On Windows:**
- Ensure Visual C++ Redistributable is installed
- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Issue: Large download size
**Solution:** Install minimal requirements only:
```bash
pip install -r requirements-minimal.txt
```

This installs only essential packages (~150 MB) instead of full requirements (~500 MB).

---

## Quick Installation Commands

### Absolute Minimum (Core functionality only)
```bash
pip install numpy scipy matplotlib PyQt6
```
**Download size:** ~150 MB  
**Features:** All visualization, flow analysis, thermal calculations

### Full Installation (All features)
```bash
pip install -r requirements.txt
```
**Download size:** ~500 MB  
**Features:** Everything + Excel export, advanced plots, PDF generation

### Recommended Installation
```bash
pip install -r requirements-minimal.txt
```
**Download size:** ~200 MB  
**Features:** All core features + data export

---

## Verification

After installation, verify it works:
```bash
python -c "import numpy, scipy, matplotlib, PyQt6; print('All packages OK!')"
```

Then run the simulator:
```bash
python run_gui_v3.py
```

You should see:
```
============================================================
Nanofluid Thermal Analyzer v3.0 - Professional Edition
============================================================
Features:
  ✓ Popup windows for advanced configuration
  ✓ Multiple particle shapes (sphere, rod, cube, platelet)
  ✓ Temperature range input and analysis
  ...
```

---

## Package Size Reference

| Package | Size | Required? | Purpose |
|---------|------|-----------|---------|
| numpy | ~50 MB | ✅ YES | Numerical computations |
| scipy | ~40 MB | ✅ YES | Scientific algorithms |
| matplotlib | ~40 MB | ✅ YES | Plotting |
| PyQt6 | ~30 MB | ✅ YES | GUI framework |
| pandas | ~30 MB | 🔶 OPTIONAL | Data export |
| openpyxl | ~5 MB | 🔶 OPTIONAL | Excel files |
| seaborn | ~5 MB | 🔶 OPTIONAL | Enhanced plots |
| vtk | ~100 MB | ❌ NOT NEEDED | 3D visualization (unused) |
| pyvista | ~50 MB | ❌ NOT NEEDED | 3D plots (unused) |

**Total minimal install:** ~160 MB  
**Total recommended:** ~200 MB  
**Total full (unnecessary):** ~500 MB

---

## Alternative: Use Anaconda/Miniconda

If pip installation fails, try Conda:

```bash
# Install Miniconda from https://docs.conda.io/en/latest/miniconda.html

# Create environment
conda create -n nanofluid python=3.11
conda activate nanofluid

# Install packages
conda install numpy scipy matplotlib
pip install PyQt6

# Run simulator
python run_gui_v3.py
```

---

## Docker Installation (Advanced)

For a completely isolated environment:

```bash
# Clone repository
git clone https://github.com/msaurav625-lgtm/test.git
cd test

# Build Docker image
docker build -t nanofluid-simulator .

# Run with X11 forwarding (Linux)
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix nanofluid-simulator
```

---

## Getting Help

If installation still fails:

1. **Check Python version:** Must be 3.9 - 3.11
   ```bash
   python --version
   ```

2. **Update pip:**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

3. **Use virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

4. **Report issue:**
   https://github.com/msaurav625-lgtm/test/issues

---

## Post-Installation

Once installed successfully:

1. Read `QUICK_REFERENCE_v3.md` for feature guide
2. Check `examples/` folder for sample scripts
3. See `docs/USER_GUIDE.md` for detailed documentation

**Enjoy your nanofluid simulations! 🎉**
