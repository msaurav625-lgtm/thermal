"""
Launch CFD GUI

Convenience script to launch the CFD simulation GUI interface.

Usage:
    python run_cfd_gui.py

Author: Nanofluid Simulator v4.0
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from PyQt6.QtWidgets import QApplication
    from nanofluid_simulator.gui.cfd_window import CFDWindow
    
    def main():
        """Launch CFD GUI"""
        print("="*70)
        print("NANOFLUID CFD SIMULATOR - GUI")
        print("="*70)
        print("\nLaunching interactive CFD interface...")
        print("\nFeatures:")
        print("  • Visual geometry editor")
        print("  • Interactive mesh configuration")
        print("  • Boundary condition setup")
        print("  • Solver parameter control")
        print("  • Real-time convergence monitoring")
        print("  • Result visualization")
        print("\n" + "="*70)
        
        app = QApplication(sys.argv)
        window = CFDWindow()
        window.show()
        sys.exit(app.exec())
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print("="*70)
    print("ERROR: GUI dependencies not available")
    print("="*70)
    print(f"\nMissing: {str(e)}")
    print("\nTo use the GUI, install PyQt6:")
    print("  pip install PyQt6")
    print("\nOr use the command-line interface with the example scripts.")
    print("="*70)
    sys.exit(1)
