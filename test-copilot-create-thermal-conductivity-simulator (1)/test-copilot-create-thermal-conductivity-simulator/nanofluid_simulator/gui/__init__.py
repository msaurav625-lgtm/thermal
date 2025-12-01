"""
GUI Package for Nanofluid Simulator

Professional PyQt6-based graphical user interface with
solver mode selection (Static vs Flow-Enhanced) and CFD simulation interface.
"""

from .main_window import NanofluidSimulatorGUI, main
from .solver_dialog import SolverSelectionDialog

try:
    from .cfd_window import CFDWindow, MeshCanvas
    __all__ = ['NanofluidSimulatorGUI', 'SolverSelectionDialog', 'CFDWindow', 'MeshCanvas', 'main']
except ImportError:
    # Full CFD GUI requires additional dependencies
    __all__ = ['NanofluidSimulatorGUI', 'SolverSelectionDialog', 'main']
