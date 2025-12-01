#!/usr/bin/env python3
"""
Diagnostic script for Nanofluid Simulator
Run this to check if all dependencies are working
"""

import sys
import os
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python():
    print_header("Python Environment")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    print(f"Architecture: {sys.maxsize > 2**32 and '64-bit' or '32-bit'}")

def check_imports():
    print_header("Checking Dependencies")
    
    deps = [
        ('numpy', 'NumPy (numerical computing)'),
        ('scipy', 'SciPy (scientific computing)'),
        ('matplotlib', 'Matplotlib (plotting)'),
        ('PyQt6', 'PyQt6 (GUI framework)'),
        ('pandas', 'Pandas (data handling) - Optional'),
        ('openpyxl', 'OpenPyXL (Excel support) - Optional'),
    ]
    
    results = []
    for module_name, description in deps:
        try:
            mod = __import__(module_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {description}")
            print(f"  Version: {version}")
            results.append(True)
        except ImportError as e:
            print(f"✗ {description}")
            print(f"  Error: {e}")
            results.append(False)
        except Exception as e:
            print(f"⚠ {description}")
            print(f"  Warning: {e}")
            results.append(None)
    
    return results

def check_gui():
    print_header("Checking GUI Components")
    
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
        
        print(f"✓ PyQt6 imports successful")
        print(f"  Qt Version: {QT_VERSION_STR}")
        print(f"  PyQt Version: {PYQT_VERSION_STR}")
        
        # Try to create QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        print(f"✓ QApplication created successfully")
        
        return True
    except Exception as e:
        print(f"✗ GUI components failed: {e}")
        return False

def check_nanofluid_modules():
    print_header("Checking Nanofluid Simulator Modules")
    
    modules = [
        'nanofluid_simulator',
        'nanofluid_simulator.gui.main_window_v3',
        'nanofluid_simulator.models',
        'nanofluid_simulator.simulator',
        'nanofluid_simulator.visualization',
    ]
    
    results = []
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
            results.append(True)
        except ImportError as e:
            print(f"✗ {module_name}")
            print(f"  Error: {e}")
            results.append(False)
        except Exception as e:
            print(f"⚠ {module_name}")
            print(f"  Warning: {e}")
            results.append(None)
    
    return results

def check_files():
    print_header("Checking Required Files")
    
    files_to_check = [
        'run_gui_v3.py',
        'app_icon.png',
        'README.md',
        'nanofluid_simulator/__init__.py',
        'nanofluid_simulator/gui/main_window_v3.py',
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✓ {file_path} ({size:,} bytes)")
        else:
            print(f"✗ {file_path} (missing)")

def check_display():
    print_header("Checking Display Environment")
    
    # Check environment variables
    display_vars = ['DISPLAY', 'QT_QPA_PLATFORM', 'QT_SCALE_FACTOR']
    for var in display_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

def run_quick_test():
    print_header("Running Quick GUI Test")
    
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Try to show a simple message box
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText("Diagnostic test successful!")
        msg.setInformativeText("If you can see this window, GUI is working.")
        msg.setWindowTitle("Nanofluid Simulator - Diagnostic")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        
        print("✓ Showing test dialog...")
        print("  (Close the dialog to continue)")
        
        msg.exec()
        
        print("✓ GUI test successful!")
        return True
        
    except Exception as e:
        print(f"✗ GUI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*15 + "NANOFLUID SIMULATOR" + " "*24 + "║")
    print("║" + " "*18 + "Diagnostic Tool" + " "*25 + "║")
    print("╚" + "="*58 + "╝")
    
    check_python()
    
    import_results = check_imports()
    
    check_files()
    
    check_display()
    
    if all(r is not False for r in import_results[:4]):  # Check core deps
        gui_ok = check_gui()
        
        if gui_ok:
            nanofluid_results = check_nanofluid_modules()
            
            if all(r is not False for r in nanofluid_results):
                print_header("Summary")
                print("✓ All core components are working!")
                print("\nAttempting to launch GUI test...")
                
                test_ok = run_quick_test()
                
                if test_ok:
                    print("\n" + "="*60)
                    print("SUCCESS! Your system can run Nanofluid Simulator.")
                    print("\nTo start the app, run:")
                    print("  python run_gui_v3.py")
                    print("="*60)
                else:
                    print("\n" + "="*60)
                    print("GUI test failed. Check errors above.")
                    print("="*60)
            else:
                print_header("Summary")
                print("⚠ Some nanofluid modules failed to load.")
                print("  Make sure you're running from the correct directory.")
        else:
            print_header("Summary")
            print("✗ GUI components are not working properly.")
            print("  This could be a PyQt6 installation issue.")
    else:
        print_header("Summary")
        print("✗ Some core dependencies are missing.")
        print("\nTo install missing dependencies, run:")
        print("  pip install numpy scipy matplotlib PyQt6")
    
    print("\nFor more help, see TROUBLESHOOTING.md")
    input("\nPress Enter to exit...")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiagnostic cancelled by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
