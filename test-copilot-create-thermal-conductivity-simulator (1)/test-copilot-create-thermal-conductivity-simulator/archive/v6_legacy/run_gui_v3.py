#!/usr/bin/env python3
"""
Nanofluid Thermal Analyzer v3.0 - Professional Edition
Advanced launcher with all features enabled
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from nanofluid_simulator.gui.main_window_v3 import AdvancedNanofluidGUI
    
    def main():
        """Launch the application."""
        # Enable high DPI scaling
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        app.setApplicationName("Nanofluid Thermal Analyzer v3.0")
        app.setOrganizationName("Research Lab")
        
        # Create and show main window
        window = AdvancedNanofluidGUI()
        window.show()
        
        print("="*60)
        print("Nanofluid Thermal Analyzer v3.0 - Professional Edition")
        print("="*60)
        print("Features:")
        print("  ✓ Popup windows for advanced configuration")
        print("  ✓ Multiple particle shapes (sphere, rod, cube, platelet)")
        print("  ✓ Temperature range input and analysis")
        print("  ✓ Scientific-grade graphs")
        print("  ✓ Nanoparticle observer")
        print("  ✓ Surface interaction analysis")
        print("  ✓ Refresh/Reset functionality")
        print("  ✓ Save results (TXT, JSON)")
        print("  ✓ Export all data and plots")
        print("="*60)
        
        # Start event loop
        sys.exit(app.exec())
    
    if __name__ == '__main__':
        main()

except ImportError as e:
    print(f"Error: Required package not installed")
    print(f"Details: {e}")
    print("\nPlease install required packages:")
    print("pip install PyQt6 matplotlib numpy scipy pandas")
    sys.exit(1)
except Exception as e:
    print(f"Error launching application: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
