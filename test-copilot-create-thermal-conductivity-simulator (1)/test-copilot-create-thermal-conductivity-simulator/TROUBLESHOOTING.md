# Troubleshooting Guide

## App Won't Start / No Window Appears

### Quick Diagnostic Steps

#### 1. **Check if it's Running in Background**
- Open Task Manager (Ctrl + Shift + Esc)
- Look for `NanofluidSimulator.exe` process
- If found, end the process and try again

#### 2. **Run from Command Line to See Errors**
```cmd
# Open Command Prompt in the folder with the .exe
# Run:
NanofluidSimulator.exe
```
This will show any error messages that might be hidden.

#### 3. **Check Windows Defender / Antivirus**
- Windows might block unsigned executables
- Right-click the .exe → Properties → Check "Unblock" if present
- Temporarily disable antivirus and try again
- Add the folder to Windows Defender exclusions

#### 4. **Missing Visual C++ Redistributables**
The app requires Microsoft Visual C++ Redistributable:
- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Install and restart your computer

#### 5. **Run as Administrator**
- Right-click `NanofluidSimulator.exe`
- Select "Run as administrator"

### Common Error Messages and Solutions

#### "VCRUNTIME140.dll was not found"
**Solution:** Install Visual C++ Redistributable (see step 4 above)

#### "The application was unable to start correctly (0xc000007b)"
**Solution:** 
- Install both x64 and x86 Visual C++ Redistributables
- Update Windows to latest version

#### "Entry point not found"
**Solution:**
- Update Windows 10/11 to latest version
- Install .NET Framework 4.8 from Microsoft

#### No error, just closes immediately
**Solution:**
- Run from command prompt to see error messages
- Check if Python dependencies are embedded correctly
- Try the Python source version instead (see below)

### Alternative: Run from Python Source

If the .exe doesn't work, you can run from source code:

#### Step 1: Install Python 3.10 or 3.11
Download from: https://www.python.org/downloads/

#### Step 2: Install Dependencies
```bash
pip install numpy scipy matplotlib PyQt6
```

#### Step 3: Run the App
```bash
python run_gui_v3.py
```

### Getting More Help

If none of the above works, please provide:

1. **Windows Version**: 
   - Press Win + R, type `winver`, press Enter
   - Take a screenshot

2. **Error Message**:
   - Run from Command Prompt
   - Copy the full error message

3. **Event Viewer Logs**:
   - Press Win + R, type `eventvwr`, press Enter
   - Go to: Windows Logs → Application
   - Look for recent errors related to the app
   - Copy the error details

4. **System Info**:
   - Press Win + R, type `msinfo32`, press Enter
   - File → Export → Save as text file
   - Share the file

### Test if Dependencies Are Working

Create a file called `test_app.py` with this content:

```python
print("Testing imports...")

try:
    import numpy
    print("✓ NumPy OK")
except Exception as e:
    print(f"✗ NumPy FAILED: {e}")

try:
    import scipy
    print("✓ SciPy OK")
except Exception as e:
    print(f"✗ SciPy FAILED: {e}")

try:
    import matplotlib
    print("✓ Matplotlib OK")
except Exception as e:
    print(f"✗ Matplotlib FAILED: {e}")

try:
    from PyQt6 import QtWidgets
    print("✓ PyQt6 OK")
except Exception as e:
    print(f"✗ PyQt6 FAILED: {e}")

try:
    import nanofluid_simulator
    print("✓ Nanofluid Simulator OK")
except Exception as e:
    print(f"✗ Nanofluid Simulator FAILED: {e}")

print("\nAll imports successful! App should work.")
input("Press Enter to exit...")
```

Run it:
```bash
python test_app.py
```

### Known Issues

1. **High DPI Displays**: App might appear tiny or text might be blurred
   - Solution: Right-click .exe → Properties → Compatibility
   - Check "Override high DPI scaling behavior"
   - Select "System (Enhanced)"

2. **Multiple Monitors**: Window might open on wrong screen
   - Solution: Move it to desired screen, close properly (File → Exit)
   - Next time it should remember the position

3. **Dark Mode**: Some elements might not respect Windows dark mode
   - This is a PyQt6 limitation on Windows

### Performance Issues

If the app starts but runs slowly:

1. **First Launch**: Building cache, wait 30 seconds
2. **Large Calculations**: Increase Reynolds/Prandtl gradually
3. **Memory**: Close other applications
4. **Graphics**: Update graphics drivers

### Success Indicators

When the app starts correctly, you should see:

1. Main window with tabs: Thermal Contours, Velocity Field, etc.
2. Left sidebar with nanofluid properties
3. Menu bar: File, Tools, Help
4. Status bar at bottom showing "Ready"

### Still Not Working?

The executable was built with:
- Python 3.10.11
- PyQt6 (latest)
- Windows 10/11 x64

If you're on Windows 7 or 32-bit Windows, the .exe won't work.
Use the Python source method instead.

### File Locations

After successful start, the app creates:
- Settings: `%APPDATA%\NanofluidSimulator\settings.ini`
- Cache: `%TEMP%\nanofluid_cache\`
- Exports: User-selected folders

You can safely delete these if you want to reset the app.
