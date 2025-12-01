"""
CFD Simulation GUI Window

Interactive interface for setting up and running CFD simulations with nanofluids.

Features:
- Visual geometry editor
- Mesh configuration
- Boundary condition setup
- Solver parameter control
- Real-time convergence monitoring
- Result visualization

Author: Nanofluid Simulator v4.0
"""

import sys
import os
from typing import Optional, Dict, List, Tuple
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
        QGroupBox, QTabWidget, QTextEdit, QProgressBar, QTableWidget,
        QTableWidgetItem, QMessageBox, QFileDialog, QCheckBox, QSplitter
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
    from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("Warning: PyQt6 not available. GUI will not work.")

# Import simulator components
try:
    from ..cfd_mesh import StructuredMesh2D
    from ..cfd_solver import NavierStokesSolver, SolverSettings
    from ..cfd_linear_solvers import SolverType
    from ..simulator import NanofluidSimulator
    from ..models import Nanoparticle
    from ..cfd_postprocess import FlowPostProcessor
except ImportError:
    # Fallback for direct execution
    pass


class MeshCanvas(QWidget):
    """Canvas for displaying mesh geometry"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = None
        self.bc_regions = {}  # {region_name: [cell_indices]}
        self.setMinimumSize(400, 300)
        
    def set_mesh(self, mesh: 'StructuredMesh2D'):
        """Set mesh to display"""
        self.mesh = mesh
        self.update()
    
    def set_boundary_regions(self, regions: Dict[str, List[int]]):
        """Set boundary condition regions for visualization"""
        self.bc_regions = regions
        self.update()
    
    def paintEvent(self, event):
        """Draw mesh"""
        if self.mesh is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get canvas dimensions
        w = self.width()
        h = self.height()
        margin = 40
        
        # Calculate scaling
        x_min, x_max = self.mesh.x_min, self.mesh.x_max
        y_min, y_max = self.mesh.y_min, self.mesh.y_max
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        scale_x = (w - 2 * margin) / x_range
        scale_y = (h - 2 * margin) / y_range
        scale = min(scale_x, scale_y)
        
        def to_screen(x, y):
            """Convert mesh coordinates to screen coordinates"""
            sx = margin + (x - x_min) * scale
            sy = h - margin - (y - y_min) * scale
            return int(sx), int(sy)
        
        # Draw mesh cells
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        
        for i in range(self.mesh.nx + 1):
            x = x_min + i * self.mesh.dx
            x1, y1 = to_screen(x, y_min)
            x2, y2 = to_screen(x, y_max)
            painter.drawLine(x1, y1, x2, y2)
        
        for j in range(self.mesh.ny + 1):
            y = y_min + j * self.mesh.dy
            x1, y1 = to_screen(x_min, y)
            x2, y2 = to_screen(x_max, y)
            painter.drawLine(x1, y1, x2, y2)
        
        # Draw boundary condition regions
        colors = {
            'inlet': QColor(0, 0, 255, 100),      # Blue
            'outlet': QColor(255, 0, 0, 100),     # Red
            'wall': QColor(128, 128, 128, 100),   # Gray
            'heated': QColor(255, 165, 0, 100),   # Orange
            'symmetry': QColor(0, 255, 0, 100),   # Green
        }
        
        for region_name, cell_indices in self.bc_regions.items():
            color = colors.get(region_name, QColor(200, 200, 200, 100))
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(), 2))
            
            for idx in cell_indices:
                if idx < len(self.mesh.cell_centers):
                    cx, cy = self.mesh.cell_centers[idx]
                    sx, sy = to_screen(cx, cy)
                    painter.drawRect(
                        sx - int(self.mesh.dx * scale / 2),
                        sy - int(self.mesh.dy * scale / 2),
                        int(self.mesh.dx * scale),
                        int(self.mesh.dy * scale)
                    )
        
        # Draw domain outline
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        x1, y1 = to_screen(x_min, y_min)
        x2, y2 = to_screen(x_max, y_max)
        painter.drawRect(x1, y2, x2 - x1, y1 - y2)


class SolverThread(QThread):
    """Thread for running CFD solver"""
    
    progress_update = pyqtSignal(int, str)  # iteration, message
    finished_signal = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, solver, max_iterations):
        super().__init__()
        self.solver = solver
        self.max_iterations = max_iterations
        self.running = True
    
    def run(self):
        """Run solver in separate thread"""
        try:
            # Run solver with verbose=False for GUI
            success = self.solver.solve(
                max_iterations=self.max_iterations,
                verbose=False
            )
            
            if success:
                self.finished_signal.emit(True, "Converged successfully!")
            else:
                self.finished_signal.emit(False, "Maximum iterations reached")
                
        except Exception as e:
            self.finished_signal.emit(False, f"Error: {str(e)}")
    
    def stop(self):
        """Stop solver"""
        self.running = False


class CFDWindow(QMainWindow):
    """Main CFD simulation window"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CFD Simulation - Nanofluid Thermal Simulator")
        self.setGeometry(100, 100, 1200, 800)
        
        # State
        self.mesh = None
        self.solver = None
        self.solver_thread = None
        self.nanofluid_sim = NanofluidSimulator()
        self.bc_regions = {}
        
        self.setup_ui()
        self.create_default_mesh()
    
    def setup_ui(self):
        """Setup user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(400)
        
        # Tabs for different settings
        tabs = QTabWidget()
        tabs.addTab(self.create_geometry_tab(), "Geometry & Mesh")
        tabs.addTab(self.create_nanofluid_tab(), "Nanofluid")
        tabs.addTab(self.create_boundary_tab(), "Boundary Conditions")
        tabs.addTab(self.create_solver_tab(), "Solver Settings")
        
        left_layout.addWidget(tabs)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("▶ Run Simulation")
        self.run_btn.clicked.connect(self.run_simulation)
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.stop_btn.setEnabled(False)
        
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.stop_btn)
        left_layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        left_layout.addWidget(QLabel("Status:"))
        left_layout.addWidget(self.status_text)
        
        main_layout.addWidget(left_panel)
        
        # Right panel - visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Mesh canvas
        right_layout.addWidget(QLabel("Mesh Preview:"))
        self.mesh_canvas = MeshCanvas()
        right_layout.addWidget(self.mesh_canvas)
        
        # Results display
        self.results_tabs = QTabWidget()
        self.results_tabs.addTab(self.create_residuals_tab(), "Convergence")
        self.results_tabs.addTab(self.create_results_tab(), "Results")
        right_layout.addWidget(self.results_tabs)
        
        main_layout.addWidget(right_panel)
        
        self.log("CFD GUI initialized. Configure your simulation and click Run.")
    
    def create_geometry_tab(self):
        """Create geometry and mesh configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Geometry group
        geom_group = QGroupBox("Domain Geometry")
        geom_layout = QGridLayout()
        
        geom_layout.addWidget(QLabel("Length (m):"), 0, 0)
        self.length_input = QDoubleSpinBox()
        self.length_input.setRange(0.001, 10.0)
        self.length_input.setValue(1.0)
        self.length_input.setSingleStep(0.1)
        self.length_input.valueChanged.connect(self.update_mesh)
        geom_layout.addWidget(self.length_input, 0, 1)
        
        geom_layout.addWidget(QLabel("Height (m):"), 1, 0)
        self.height_input = QDoubleSpinBox()
        self.height_input.setRange(0.001, 10.0)
        self.height_input.setValue(0.1)
        self.height_input.setSingleStep(0.01)
        self.height_input.valueChanged.connect(self.update_mesh)
        geom_layout.addWidget(self.height_input, 1, 1)
        
        geom_group.setLayout(geom_layout)
        layout.addWidget(geom_group)
        
        # Mesh group
        mesh_group = QGroupBox("Mesh Resolution")
        mesh_layout = QGridLayout()
        
        mesh_layout.addWidget(QLabel("Cells in X:"), 0, 0)
        self.nx_input = QSpinBox()
        self.nx_input.setRange(5, 500)
        self.nx_input.setValue(50)
        self.nx_input.valueChanged.connect(self.update_mesh)
        mesh_layout.addWidget(self.nx_input, 0, 1)
        
        mesh_layout.addWidget(QLabel("Cells in Y:"), 1, 0)
        self.ny_input = QSpinBox()
        self.ny_input.setRange(5, 500)
        self.ny_input.setValue(20)
        self.ny_input.valueChanged.connect(self.update_mesh)
        mesh_layout.addWidget(self.ny_input, 1, 1)
        
        self.cell_count_label = QLabel("Total cells: 1000")
        mesh_layout.addWidget(self.cell_count_label, 2, 0, 1, 2)
        
        mesh_group.setLayout(mesh_layout)
        layout.addWidget(mesh_group)
        
        layout.addStretch()
        return tab
    
    def create_nanofluid_tab(self):
        """Create nanofluid properties tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        nano_group = QGroupBox("Nanofluid Properties")
        nano_layout = QGridLayout()
        
        nano_layout.addWidget(QLabel("Base Fluid:"), 0, 0)
        self.base_fluid_combo = QComboBox()
        self.base_fluid_combo.addItems(["water", "ethylene_glycol"])
        nano_layout.addWidget(self.base_fluid_combo, 0, 1)
        
        nano_layout.addWidget(QLabel("Nanoparticle:"), 1, 0)
        self.nanoparticle_combo = QComboBox()
        self.nanoparticle_combo.addItems(["AL2O3", "CUO", "TIO2", "SIO2", "FE3O4"])
        nano_layout.addWidget(self.nanoparticle_combo, 1, 1)
        
        nano_layout.addWidget(QLabel("Volume Fraction φ (%):"), 2, 0)
        self.phi_input = QDoubleSpinBox()
        self.phi_input.setRange(0.0, 10.0)
        self.phi_input.setValue(3.0)
        self.phi_input.setSingleStep(0.5)
        self.phi_input.setSuffix(" %")
        nano_layout.addWidget(self.phi_input, 2, 1)
        
        nano_layout.addWidget(QLabel("Temperature (K):"), 3, 0)
        self.temp_input = QDoubleSpinBox()
        self.temp_input.setRange(273.0, 400.0)
        self.temp_input.setValue(300.0)
        self.temp_input.setSingleStep(5.0)
        self.temp_input.setSuffix(" K")
        nano_layout.addWidget(self.temp_input, 3, 1)
        
        calc_btn = QPushButton("Calculate Properties")
        calc_btn.clicked.connect(self.calculate_nanofluid_properties)
        nano_layout.addWidget(calc_btn, 4, 0, 1, 2)
        
        self.props_text = QTextEdit()
        self.props_text.setReadOnly(True)
        self.props_text.setMaximumHeight(120)
        nano_layout.addWidget(self.props_text, 5, 0, 1, 2)
        
        nano_group.setLayout(nano_layout)
        layout.addWidget(nano_group)
        
        layout.addStretch()
        return tab
    
    def create_boundary_tab(self):
        """Create boundary conditions tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        bc_group = QGroupBox("Boundary Conditions")
        bc_layout = QVBoxLayout()
        
        # Inlet
        inlet_group = QGroupBox("Inlet (Left)")
        inlet_layout = QGridLayout()
        inlet_layout.addWidget(QLabel("Velocity (m/s):"), 0, 0)
        self.inlet_velocity = QDoubleSpinBox()
        self.inlet_velocity.setRange(0.001, 10.0)
        self.inlet_velocity.setValue(0.1)
        self.inlet_velocity.setSingleStep(0.01)
        inlet_layout.addWidget(self.inlet_velocity, 0, 1)
        
        inlet_layout.addWidget(QLabel("Temperature (K):"), 1, 0)
        self.inlet_temp = QDoubleSpinBox()
        self.inlet_temp.setRange(273.0, 400.0)
        self.inlet_temp.setValue(300.0)
        self.inlet_temp.setSingleStep(5.0)
        inlet_layout.addWidget(self.inlet_temp, 1, 1)
        inlet_group.setLayout(inlet_layout)
        bc_layout.addWidget(inlet_group)
        
        # Outlet
        outlet_group = QGroupBox("Outlet (Right)")
        outlet_layout = QGridLayout()
        outlet_layout.addWidget(QLabel("Pressure (Pa):"), 0, 0)
        self.outlet_pressure = QDoubleSpinBox()
        self.outlet_pressure.setRange(-1000.0, 10000.0)
        self.outlet_pressure.setValue(0.0)
        self.outlet_pressure.setSingleStep(100.0)
        outlet_layout.addWidget(self.outlet_pressure, 0, 1)
        outlet_group.setLayout(outlet_layout)
        bc_layout.addWidget(outlet_group)
        
        # Walls
        wall_group = QGroupBox("Walls (Top & Bottom)")
        wall_layout = QGridLayout()
        
        self.wall_type_combo = QComboBox()
        self.wall_type_combo.addItems(["No-Slip (u=0)", "Heated Wall", "Adiabatic"])
        self.wall_type_combo.currentTextChanged.connect(self.on_wall_type_changed)
        wall_layout.addWidget(QLabel("Type:"), 0, 0)
        wall_layout.addWidget(self.wall_type_combo, 0, 1)
        
        wall_layout.addWidget(QLabel("Wall Temp (K):"), 1, 0)
        self.wall_temp = QDoubleSpinBox()
        self.wall_temp.setRange(273.0, 400.0)
        self.wall_temp.setValue(320.0)
        self.wall_temp.setSingleStep(5.0)
        self.wall_temp.setEnabled(False)
        wall_layout.addWidget(self.wall_temp, 1, 1)
        
        wall_group.setLayout(wall_layout)
        bc_layout.addWidget(wall_group)
        
        bc_group.setLayout(bc_layout)
        layout.addWidget(bc_group)
        
        layout.addStretch()
        return tab
    
    def create_solver_tab(self):
        """Create solver settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        solver_group = QGroupBox("Solver Parameters")
        solver_layout = QGridLayout()
        
        solver_layout.addWidget(QLabel("Max Iterations:"), 0, 0)
        self.max_iter_input = QSpinBox()
        self.max_iter_input.setRange(10, 2000)
        self.max_iter_input.setValue(200)
        self.max_iter_input.setSingleStep(10)
        solver_layout.addWidget(self.max_iter_input, 0, 1)
        
        solver_layout.addWidget(QLabel("Tolerance:"), 1, 0)
        self.tolerance_input = QComboBox()
        self.tolerance_input.addItems(["1e-3 (Fast)", "1e-4 (Standard)", "1e-5 (Accurate)", "1e-6 (Precise)"])
        self.tolerance_input.setCurrentIndex(1)
        solver_layout.addWidget(self.tolerance_input, 1, 1)
        
        solver_layout.addWidget(QLabel("Linear Solver:"), 2, 0)
        self.linear_solver_combo = QComboBox()
        self.linear_solver_combo.addItems(["Direct", "BiCGSTAB", "Gauss-Seidel"])
        self.linear_solver_combo.setCurrentIndex(0)
        solver_layout.addWidget(self.linear_solver_combo, 2, 1)
        
        solver_layout.addWidget(QLabel("Turbulence Model:"), 3, 0)
        self.turbulence_combo = QComboBox()
        self.turbulence_combo.addItems(["Laminar", "k-epsilon"])
        solver_layout.addWidget(self.turbulence_combo, 3, 1)
        
        solver_group.setLayout(solver_layout)
        layout.addWidget(solver_group)
        
        # Under-relaxation
        relax_group = QGroupBox("Under-Relaxation Factors")
        relax_layout = QGridLayout()
        
        relax_layout.addWidget(QLabel("Velocity (u, v):"), 0, 0)
        self.relax_u = QDoubleSpinBox()
        self.relax_u.setRange(0.1, 1.0)
        self.relax_u.setValue(0.7)
        self.relax_u.setSingleStep(0.05)
        relax_layout.addWidget(self.relax_u, 0, 1)
        
        relax_layout.addWidget(QLabel("Pressure (p):"), 1, 0)
        self.relax_p = QDoubleSpinBox()
        self.relax_p.setRange(0.1, 1.0)
        self.relax_p.setValue(0.3)
        self.relax_p.setSingleStep(0.05)
        relax_layout.addWidget(self.relax_p, 1, 1)
        
        relax_layout.addWidget(QLabel("Temperature (T):"), 2, 0)
        self.relax_T = QDoubleSpinBox()
        self.relax_T.setRange(0.1, 1.0)
        self.relax_T.setValue(0.8)
        self.relax_T.setSingleStep(0.05)
        relax_layout.addWidget(self.relax_T, 2, 1)
        
        relax_group.setLayout(relax_layout)
        layout.addWidget(relax_group)
        
        layout.addStretch()
        return tab
    
    def create_residuals_tab(self):
        """Create convergence monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.residuals_text = QTextEdit()
        self.residuals_text.setReadOnly(True)
        layout.addWidget(self.residuals_text)
        
        return tab
    
    def create_results_tab(self):
        """Create results display tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
        
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        layout.addWidget(export_btn)
        
        return tab
    
    def create_default_mesh(self):
        """Create default mesh on startup"""
        self.update_mesh()
    
    def update_mesh(self):
        """Update mesh based on current settings"""
        try:
            length = self.length_input.value()
            height = self.height_input.value()
            nx = self.nx_input.value()
            ny = self.ny_input.value()
            
            self.mesh = StructuredMesh2D(
                x_range=(0.0, length),
                y_range=(0.0, height),
                nx=nx,
                ny=ny
            )
            
            # Update cell count
            self.cell_count_label.setText(f"Total cells: {self.mesh.n_cells}")
            
            # Update canvas
            self.identify_boundary_regions()
            self.mesh_canvas.set_mesh(self.mesh)
            self.mesh_canvas.set_boundary_regions(self.bc_regions)
            
            self.log(f"Mesh updated: {nx}×{ny} = {self.mesh.n_cells} cells")
            
        except Exception as e:
            self.log(f"Error updating mesh: {str(e)}")
    
    def identify_boundary_regions(self):
        """Identify cells for each boundary region"""
        if self.mesh is None:
            return
        
        self.bc_regions = {}
        
        # Inlet (left)
        self.bc_regions['inlet'] = [i for i in range(self.mesh.n_cells) 
                                     if self.mesh.cell_centers[i, 0] < 1e-6]
        
        # Outlet (right)
        self.bc_regions['outlet'] = [i for i in range(self.mesh.n_cells)
                                      if abs(self.mesh.cell_centers[i, 0] - self.mesh.x_max) < 1e-6]
        
        # Bottom wall
        self.bc_regions['wall'] = [i for i in range(self.mesh.n_cells)
                                    if self.mesh.cell_centers[i, 1] < 1e-6 or 
                                    abs(self.mesh.cell_centers[i, 1] - self.mesh.y_max) < 1e-6]
    
    def calculate_nanofluid_properties(self):
        """Calculate nanofluid properties"""
        try:
            nanoparticle = getattr(Nanoparticle, self.nanoparticle_combo.currentText())
            phi = self.phi_input.value() / 100.0  # Convert % to fraction
            T = self.temp_input.value()
            base_fluid = self.base_fluid_combo.currentText()
            
            props = self.nanofluid_sim.calculate_properties(
                nanoparticle=nanoparticle,
                phi=phi,
                T=T,
                base_fluid=base_fluid
            )
            
            # Display properties
            text = f"""
Nanofluid Properties:

• Density: {props['density']:.2f} kg/m³
• Dynamic Viscosity: {props['dynamic_viscosity']*1000:.3f} mPa·s
• Thermal Conductivity: {props['thermal_conductivity']:.3f} W/m·K
• Specific Heat: {props['specific_heat']:.1f} J/kg·K

Enhancement vs Base Fluid:
• Conductivity: +{(props['thermal_conductivity']/props.get('k_bf', props['thermal_conductivity'])-1)*100:.1f}%
• Viscosity: +{(props['dynamic_viscosity']/props.get('mu_bf', props['dynamic_viscosity'])-1)*100:.1f}%
            """
            
            self.props_text.setText(text.strip())
            self.log("Nanofluid properties calculated")
            
        except Exception as e:
            self.log(f"Error calculating properties: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to calculate properties:\n{str(e)}")
    
    def on_wall_type_changed(self, text):
        """Handle wall type change"""
        self.wall_temp.setEnabled("Heated" in text)
    
    def run_simulation(self):
        """Run CFD simulation"""
        try:
            # Validate inputs
            if self.mesh is None:
                QMessageBox.warning(self, "Error", "No mesh defined!")
                return
            
            self.log("="*60)
            self.log("Starting CFD simulation...")
            
            # Get nanofluid properties
            nanoparticle = getattr(Nanoparticle, self.nanoparticle_combo.currentText())
            phi = self.phi_input.value() / 100.0
            T = self.temp_input.value()
            base_fluid = self.base_fluid_combo.currentText()
            
            props = self.nanofluid_sim.calculate_properties(
                nanoparticle=nanoparticle,
                phi=phi,
                T=T,
                base_fluid=base_fluid
            )
            
            # Setup solver
            tolerance_map = {
                "1e-3 (Fast)": 1e-3,
                "1e-4 (Standard)": 1e-4,
                "1e-5 (Accurate)": 1e-5,
                "1e-6 (Precise)": 1e-6
            }
            
            solver_map = {
                "Direct": SolverType.DIRECT,
                "BiCGSTAB": SolverType.BICGSTAB,
                "Gauss-Seidel": SolverType.GAUSS_SEIDEL
            }
            
            settings = SolverSettings(
                max_iterations=self.max_iter_input.value(),
                convergence_tol=tolerance_map[self.tolerance_input.currentText()],
                linear_solver=solver_map[self.linear_solver_combo.currentText()],
                turbulence_model='laminar' if self.turbulence_combo.currentText() == "Laminar" else 'k-epsilon',
                under_relaxation_u=self.relax_u.value(),
                under_relaxation_v=self.relax_u.value(),
                under_relaxation_p=self.relax_p.value(),
                under_relaxation_T=self.relax_T.value()
            )
            
            self.solver = NavierStokesSolver(self.mesh, settings)
            
            # Set fluid properties
            self.solver.set_fluid_properties(
                rho=props['density'],
                mu=props['dynamic_viscosity'],
                cp=props['specific_heat'],
                k=props['thermal_conductivity']
            )
            
            # Apply boundary conditions
            self.apply_boundary_conditions()
            
            # Disable controls
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setValue(0)
            
            # Run in thread
            self.solver_thread = SolverThread(self.solver, self.max_iter_input.value())
            self.solver_thread.progress_update.connect(self.on_progress_update)
            self.solver_thread.finished_signal.connect(self.on_solver_finished)
            self.solver_thread.start()
            
            self.log("Solver started...")
            
        except Exception as e:
            self.log(f"Error starting simulation: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to start simulation:\n{str(e)}")
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def apply_boundary_conditions(self):
        """Apply boundary conditions to solver"""
        # Inlet
        u_inlet = self.inlet_velocity.value()
        for idx in self.bc_regions.get('inlet', []):
            self.solver.apply_velocity_bc(idx, u_inlet, 0.0)
            self.solver.apply_temperature_bc(idx, self.inlet_temp.value())
        
        # Outlet
        for idx in self.bc_regions.get('outlet', []):
            self.solver.apply_pressure_bc(idx, self.outlet_pressure.value())
        
        # Walls
        for idx in self.bc_regions.get('wall', []):
            self.solver.apply_velocity_bc(idx, 0.0, 0.0)  # No-slip
            
            if "Heated" in self.wall_type_combo.currentText():
                self.solver.apply_temperature_bc(idx, self.wall_temp.value())
        
        self.log("Boundary conditions applied")
    
    def stop_simulation(self):
        """Stop running simulation"""
        if self.solver_thread is not None:
            self.solver_thread.stop()
            self.log("Stopping solver...")
    
    def on_progress_update(self, iteration, message):
        """Update progress"""
        progress = int(100 * iteration / self.max_iter_input.value())
        self.progress_bar.setValue(min(progress, 100))
        self.log(message)
    
    def on_solver_finished(self, success, message):
        """Handle solver completion"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100 if success else 0)
        
        self.log("="*60)
        self.log(f"Simulation finished: {message}")
        
        if success:
            self.display_results()
            QMessageBox.information(self, "Success", "Simulation completed successfully!")
        else:
            QMessageBox.warning(self, "Warning", f"Simulation incomplete:\n{message}")
    
    def display_results(self):
        """Display simulation results"""
        if self.solver is None:
            return
        
        try:
            field = self.solver.get_results()
            
            # Calculate statistics
            u_max = np.max(np.abs(field.u))
            v_max = np.max(np.abs(field.v))
            p_max = np.max(field.p)
            p_min = np.min(field.p)
            T_max = np.max(field.T)
            T_min = np.min(field.T)
            
            results_text = f"""
SIMULATION RESULTS

Velocity Field:
• Max u-velocity: {u_max:.4f} m/s
• Max v-velocity: {v_max:.4f} m/s
• Max velocity magnitude: {np.max(np.sqrt(field.u**2 + field.v**2)):.4f} m/s

Pressure Field:
• Max pressure: {p_max:.2f} Pa
• Min pressure: {p_min:.2f} Pa
• Pressure drop: {p_max - p_min:.2f} Pa

Temperature Field:
• Max temperature: {T_max:.2f} K
• Min temperature: {T_min:.2f} K
• Temperature difference: {T_max - T_min:.2f} K

Residuals:
"""
            
            # Add residual info
            for var in ['u', 'v', 'continuity', 'T']:
                if var in self.solver.residuals and self.solver.residuals[var]:
                    final_res = self.solver.residuals[var][-1]
                    results_text += f"• {var}: {final_res:.2e}\n"
            
            self.results_text.setText(results_text.strip())
            
            # Display convergence
            conv_text = "CONVERGENCE HISTORY\n\n"
            conv_text += f"{'Iteration':<12} {'u':<12} {'v':<12} {'p':<12} {'T':<12}\n"
            conv_text += "-" * 60 + "\n"
            
            n_iter = len(self.solver.residuals.get('u', []))
            for i in range(0, n_iter, max(1, n_iter // 20)):  # Show ~20 lines
                line = f"{i:<12} "
                for var in ['u', 'v', 'continuity', 'T']:
                    if var in self.solver.residuals and i < len(self.solver.residuals[var]):
                        line += f"{self.solver.residuals[var][i]:<12.2e} "
                    else:
                        line += f"{'N/A':<12} "
                conv_text += line + "\n"
            
            self.residuals_text.setText(conv_text)
            
            self.log("Results displayed")
            
        except Exception as e:
            self.log(f"Error displaying results: {str(e)}")
    
    def export_results(self):
        """Export simulation results"""
        if self.solver is None:
            QMessageBox.warning(self, "Error", "No simulation results to export!")
            return
        
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Results",
                "",
                "Text Files (*.txt);;All Files (*)"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.results_text.toPlainText())
                    f.write("\n\n")
                    f.write(self.residuals_text.toPlainText())
                
                self.log(f"Results exported to {filename}")
                QMessageBox.information(self, "Success", f"Results exported to:\n{filename}")
                
        except Exception as e:
            self.log(f"Error exporting results: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to export results:\n{str(e)}")
    
    def log(self, message):
        """Add message to status log"""
        self.status_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


# Main execution
if __name__ == "__main__":
    if not PYQT_AVAILABLE:
        print("Error: PyQt6 is required for GUI")
        print("Install with: pip install PyQt6")
        sys.exit(1)
    
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = CFDWindow()
    window.show()
    sys.exit(app.exec())
