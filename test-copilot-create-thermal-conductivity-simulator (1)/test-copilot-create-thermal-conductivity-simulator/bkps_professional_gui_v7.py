#!/usr/bin/env python3
"""
BKPS NFL Thermal Pro 7.0 - Next-Generation Professional GUI
Dedicated to: Brijesh Kumar Pandey

ENHANCED FEATURES:
✓ Dockable panels with flexible layout
✓ Professional dark theme option
✓ Auto-save with crash recovery
✓ Recent projects with quick open
✓ Progress overlay with cancellation
✓ Responsive resizing
✓ Unified engine integration
✓ Validation Center tab
✓ Advanced visualization tools
✓ PDF report generation
"""

import sys
import os
from pathlib import Path
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QGroupBox,
    QFormLayout, QMessageBox, QSplitter, QTextEdit, QCheckBox,
    QSpinBox, QFileDialog, QApplication, QProgressBar, QProgressDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QSlider,
    QGridLayout, QFrame, QScrollArea, QLineEdit, QDockWidget,
    QMenu, QMenuBar, QStatusBar, QToolBar, QStyle
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QThread, pyqtSlot, QSettings,
    QSize, QPoint, QByteArray
)
from PyQt6.QtGui import (
    QFont, QIcon, QAction, QColor, QPalette, QKeySequence,
    QBrush, QPen
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.use('Qt5Agg')

# Import v7.0 unified engine
from nanofluid_simulator import BKPSNanofluidEngine, UnifiedConfig, SimulationMode
from nanofluid_simulator.pdf_report import PDFReportGenerator
from nanofluid_simulator.validation_center import ValidationCenter, get_validation_summary
from nanofluid_simulator.advanced_visualization import AdvancedVisualizer, create_sample_cfd_field


class ComputationThread(QThread):
    """Thread for non-blocking computation with cancellation"""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int, str)  # progress, message
    error = pyqtSignal(str)
    
    def __init__(self, engine: BKPSNanofluidEngine):
        super().__init__()
        self.engine = engine
        self.cancelled = False
    
    def run(self):
        try:
            def progress_callback(value):
                if self.cancelled:
                    raise RuntimeError("Computation cancelled by user")
                self.progress.emit(int(value), "Computing...")
            
            result = self.engine.run(progress_callback=progress_callback)
            
            # Run UQ if enabled
            if not self.cancelled and getattr(self.engine.config, 'enable_uq', False):
                try:
                    from nanofluid_simulator.uq import Distribution, monte_carlo
                    self.progress.emit(80, "Running uncertainty quantification...")
                    
                    # Extract base values
                    if 'static' in result:
                        res = result['static']
                    else:
                        res = result
                    
                    k_bf = res.get('k_base', 0.613)
                    k_np = 40.0  # Al2O3 default
                    phi = self.engine.config.nanoparticles[0].volume_fraction if self.engine.config.nanoparticles else 0.02
                    
                    # Define uncertainty distributions (±2% for k_bf, ±5% for k_np, ±10% for phi)
                    inputs = {
                        'k_bf': Distribution('normal', (k_bf, k_bf * 0.02)),
                        'k_np': Distribution('normal', (k_np, k_np * 0.05)),
                        'phi': Distribution('uniform', (phi * 0.9, phi * 1.1))
                    }
                    
                    # Maxwell model for UQ
                    def k_eff_model(k_bf, k_np, phi):
                        numerator = k_np + 2*k_bf + 2*phi*(k_np - k_bf)
                        denominator = k_np + 2*k_bf - phi*(k_np - k_bf)
                        return k_bf * (numerator/denominator)
                    
                    uq_result = monte_carlo(k_eff_model, inputs, n_samples=1000, random_state=42)
                    
                    result['uq'] = {
                        'mean': uq_result['mean'],
                        'std': uq_result['std'],
                        'ci95': uq_result['ci95'],
                        'samples': 1000
                    }
                    self.progress.emit(95, "UQ complete")
                except ImportError:
                    pass  # UQ module not available
                except Exception as e:
                    print(f"UQ failed: {e}")
            
            if not self.cancelled:
                self.finished.emit(result)
        except Exception as e:
            if not self.cancelled:
                self.error.emit(str(e))
    
    def cancel(self):
        """Cancel computation"""
        self.cancelled = True


class BKPSProfessionalGUI_V7(QMainWindow):
    """
    Next-generation professional GUI for BKPS NFL Thermal Pro 7.0
    
    Features:
    - Unified engine integration
    - Dockable panels
    - Dark/light theme
    - Auto-save
    - Recent projects
    - Progress overlay
    - Validation center
    - Advanced visualization
    - PDF export
    """
    
    def __init__(self):
        super().__init__()
        
        # Settings persistence
        self.settings = QSettings("BKPS", "NFLThermalPro7")
        
        # Recent files
        self.recent_files: List[str] = self._load_recent_files()
        self.max_recent_files = 10
        
        # Current state
        self.current_project_path: Optional[str] = None
        self.config: Optional[UnifiedConfig] = None
        self.results: Optional[Dict] = None
        self.computation_thread: Optional[ComputationThread] = None
        self.autosave_timer = QTimer()
        self.autosave_timer.timeout.connect(self._autosave)
        
        # Validation center
        self.validation_center = ValidationCenter()
        self.validation_results: Optional[Dict] = None
        
        # Initialize UI
        self.init_ui()
        
        # Restore window state
        self._restore_window_state()
        
        # Start autosave (every 2 minutes)
        self.autosave_timer.start(120000)
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("BKPS NFL Thermal Pro 7.0 — Professional Research Platform")
        self.setGeometry(100, 100, 1600, 900)
        
        # Menu bar
        self._create_menus()
        
        # Toolbar
        self._create_toolbar()
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready | Dedicated to: Brijesh Kumar Pandey")
        
        # Central widget with tabs
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setMovable(True)
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self._create_config_tab()
        self._create_parametric_tab()
        self._create_results_tab()
        self._create_visualization_tab()
        self._create_validation_tab()
        self._create_advanced_tab()
        
        # Initialize custom materials storage
        self.custom_fluids = {}
        self.custom_nanoparticles = {}
        
        # Dockable panels
        self._create_dock_panels()
        
        # Apply theme
        self._apply_theme(self.settings.value("theme", "light"))
    
    def _create_menus(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New Project", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open Project...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_project)
        file_menu.addAction(open_action)
        
        # Recent files submenu
        self.recent_menu = file_menu.addMenu("Recent Projects")
        self._update_recent_menu()
        
        file_menu.addSeparator()
        
        save_action = QAction("&Save Project", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_project)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save Project &As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self._save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        export_pdf_action = QAction("Export PDF Report...", self)
        export_pdf_action.triggered.connect(self._export_pdf_report)
        file_menu.addAction(export_pdf_action)
        
        export_json_action = QAction("Export Results (JSON)...", self)
        export_json_action.triggered.connect(self._export_results_json)
        file_menu.addAction(export_json_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        theme_menu = view_menu.addMenu("Theme")
        
        light_action = QAction("Light Theme", self)
        light_action.triggered.connect(lambda: self._apply_theme("light"))
        theme_menu.addAction(light_action)
        
        dark_action = QAction("Dark Theme", self)
        dark_action.triggered.connect(lambda: self._apply_theme("dark"))
        theme_menu.addAction(dark_action)
        
        view_menu.addSeparator()
        
        # Dock visibility toggles (will be populated)
        self.dock_menu = view_menu.addMenu("Dock Panels")
        
        # Compute menu
        compute_menu = menubar.addMenu("&Compute")
        
        run_action = QAction("&Run Simulation", self)
        run_action.setShortcut("F5")
        run_action.triggered.connect(self._run_simulation)
        compute_menu.addAction(run_action)
        
        cancel_action = QAction("&Cancel Computation", self)
        cancel_action.setShortcut("Esc")
        cancel_action.triggered.connect(self._cancel_computation)
        compute_menu.addAction(cancel_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        docs_action = QAction("&Documentation", self)
        docs_action.setShortcut("F1")
        docs_action.triggered.connect(self._show_documentation)
        help_menu.addAction(docs_action)
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """Create toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # New project
        new_btn = QAction("New", self)
        new_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        new_btn.triggered.connect(self._new_project)
        toolbar.addAction(new_btn)
        
        # Open project
        open_btn = QAction("Open", self)
        open_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        open_btn.triggered.connect(self._open_project)
        toolbar.addAction(open_btn)
        
        # Save project
        save_btn = QAction("Save", self)
        save_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        save_btn.triggered.connect(self._save_project)
        toolbar.addAction(save_btn)
        
        toolbar.addSeparator()
        
        # Run simulation
        run_btn = QAction("Run", self)
        run_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        run_btn.triggered.connect(self._run_simulation)
        toolbar.addAction(run_btn)
        
        # Cancel
        cancel_btn = QAction("Cancel", self)
        cancel_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        cancel_btn.triggered.connect(self._cancel_computation)
        toolbar.addAction(cancel_btn)
        
        toolbar.addSeparator()
        
        # Export PDF
        pdf_btn = QAction("PDF Report", self)
        pdf_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        pdf_btn.triggered.connect(self._export_pdf_report)
        toolbar.addAction(pdf_btn)
    
    def _create_config_tab(self):
        """Create configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Simulation mode
        mode_group = QGroupBox("Simulation Mode")
        mode_layout = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Static", "Flow", "CFD", "Hybrid"])
        mode_layout.addRow("Mode:", self.mode_combo)
        mode_group.setLayout(mode_layout)
        scroll_layout.addWidget(mode_group)
        
        # Base fluid
        fluid_group = QGroupBox("Base Fluid")
        fluid_layout = QFormLayout()
        
        # Fluid selector with custom fluid option
        fluid_material_layout = QHBoxLayout()
        self.fluid_combo = QComboBox()
        self.fluid_combo.addItems(["Water", "EG", "EG-Water-50/50", "Custom..."])
        self.fluid_combo.currentTextChanged.connect(self._on_fluid_changed)
        fluid_material_layout.addWidget(self.fluid_combo)
        
        self.add_fluid_btn = QPushButton("➕ Custom")
        self.add_fluid_btn.setToolTip("Add custom base fluid")
        self.add_fluid_btn.clicked.connect(self._add_custom_fluid)
        self.add_fluid_btn.setMaximumWidth(80)
        fluid_material_layout.addWidget(self.add_fluid_btn)
        
        fluid_layout.addRow("Fluid:", fluid_material_layout)
        
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(273, 373)
        self.temp_spin.setValue(300)
        self.temp_spin.setSuffix(" K")
        fluid_layout.addRow("Temperature:", self.temp_spin)
        
        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setRange(50, 500)
        self.pressure_spin.setValue(101.325)
        self.pressure_spin.setSuffix(" kPa")
        fluid_layout.addRow("Pressure:", self.pressure_spin)
        
        fluid_group.setLayout(fluid_layout)
        scroll_layout.addWidget(fluid_group)
        
        # Nanoparticle
        np_group = QGroupBox("Nanoparticle")
        np_layout = QFormLayout()
        
        # Material selector with custom material option
        np_material_layout = QHBoxLayout()
        self.np_combo = QComboBox()
        self.np_combo.addItems([
            "Al2O3", "CuO", "TiO2", "SiO2", "ZnO", "Fe3O4",
            "Cu", "Ag", "Au", "CNT", "Graphene", "Custom..."
        ])
        self.np_combo.currentTextChanged.connect(self._on_np_changed)
        np_material_layout.addWidget(self.np_combo)
        
        self.add_np_btn = QPushButton("➕ Custom")
        self.add_np_btn.setToolTip("Add custom nanoparticle material")
        self.add_np_btn.clicked.connect(self._add_custom_nanoparticle)
        self.add_np_btn.setMaximumWidth(80)
        np_material_layout.addWidget(self.add_np_btn)
        
        np_layout.addRow("Material:", np_material_layout)
        
        self.phi_spin = QDoubleSpinBox()
        self.phi_spin.setRange(0.001, 0.10)
        self.phi_spin.setValue(0.02)
        self.phi_spin.setDecimals(4)
        self.phi_spin.setSuffix(" (vol.)")
        np_layout.addRow("Volume Fraction:", self.phi_spin)
        
        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(1, 200)
        self.diameter_spin.setValue(30)
        self.diameter_spin.setSuffix(" nm")
        np_layout.addRow("Diameter:", self.diameter_spin)
        
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["spherical", "cylindrical", "platelet"])
        np_layout.addRow("Shape:", self.shape_combo)
        
        np_group.setLayout(np_layout)
        scroll_layout.addWidget(np_group)
        
        # Advanced options
        adv_group = QGroupBox("Advanced Options")
        adv_layout = QVBoxLayout()
        self.dlvo_check = QCheckBox("Enable DLVO Analysis")
        self.dlvo_check.setChecked(True)
        adv_layout.addWidget(self.dlvo_check)
        
        self.non_newt_check = QCheckBox("Enable Non-Newtonian Effects")
        adv_layout.addWidget(self.non_newt_check)
        
        self.ai_check = QCheckBox("Enable AI Recommendations")
        self.ai_check.setChecked(True)
        adv_layout.addWidget(self.ai_check)
        
        self.uq_check = QCheckBox("Enable Uncertainty Quantification")
        self.uq_check.setToolTip("Run Monte Carlo uncertainty analysis on thermal conductivity")
        adv_layout.addWidget(self.uq_check)
        
        adv_group.setLayout(adv_layout)
        scroll_layout.addWidget(adv_group)
        
        # CFD Mesh Configuration (visible when CFD/Hybrid mode)
        self.mesh_group = QGroupBox("CFD Mesh Configuration")
        mesh_layout = QFormLayout()
        
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(10, 500)
        self.nx_spin.setValue(50)
        self.nx_spin.setToolTip("Grid points in x-direction")
        mesh_layout.addRow("Grid X (nx):", self.nx_spin)
        
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(10, 500)
        self.ny_spin.setValue(50)
        self.ny_spin.setToolTip("Grid points in y-direction")
        mesh_layout.addRow("Grid Y (ny):", self.ny_spin)
        
        self.mesh_group.setLayout(mesh_layout)
        self.mesh_group.setVisible(False)
        scroll_layout.addWidget(self.mesh_group)
        
        # CFD Solver Settings
        self.solver_group = QGroupBox("Solver Settings")
        solver_layout = QFormLayout()
        
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(10, 10000)
        self.max_iter_spin.setValue(1000)
        self.max_iter_spin.setToolTip("Maximum solver iterations")
        solver_layout.addRow("Max Iterations:", self.max_iter_spin)
        
        self.conv_tol_spin = QDoubleSpinBox()
        self.conv_tol_spin.setRange(1e-10, 1e-3)
        self.conv_tol_spin.setValue(1e-6)
        self.conv_tol_spin.setDecimals(10)
        self.conv_tol_spin.setSingleStep(1e-7)
        self.conv_tol_spin.setToolTip("Convergence tolerance")
        solver_layout.addRow("Convergence Tol:", self.conv_tol_spin)
        
        self.relax_spin = QDoubleSpinBox()
        self.relax_spin.setRange(0.1, 1.0)
        self.relax_spin.setValue(0.7)
        self.relax_spin.setDecimals(2)
        self.relax_spin.setToolTip("Under-relaxation factor for momentum")
        solver_layout.addRow("Relaxation Factor:", self.relax_spin)
        
        self.solver_group.setLayout(solver_layout)
        self.solver_group.setVisible(False)
        scroll_layout.addWidget(self.solver_group)
        
        # Flow Conditions
        self.flow_group = QGroupBox("Flow Conditions")
        flow_layout = QFormLayout()
        
        self.velocity_spin = QDoubleSpinBox()
        self.velocity_spin.setRange(0.001, 100)
        self.velocity_spin.setValue(1.0)
        self.velocity_spin.setSuffix(" m/s")
        self.velocity_spin.setToolTip("Inlet flow velocity")
        flow_layout.addRow("Velocity:", self.velocity_spin)
        
        self.reynolds_spin = QDoubleSpinBox()
        self.reynolds_spin.setRange(1, 1e6)
        self.reynolds_spin.setValue(100)
        self.reynolds_spin.setReadOnly(True)
        self.reynolds_spin.setToolTip("Reynolds number (calculated)")
        flow_layout.addRow("Reynolds #:", self.reynolds_spin)
        
        self.turb_check = QCheckBox("Enable Turbulence Model")
        self.turb_check.setToolTip("Enable RANS turbulence modeling")
        flow_layout.addRow("", self.turb_check)
        
        self.turb_model_combo = QComboBox()
        self.turb_model_combo.addItems(["k-epsilon", "k-omega-sst", "spalart-allmaras"])
        self.turb_model_combo.setEnabled(False)
        self.turb_model_combo.setToolTip("Turbulence model selection")
        flow_layout.addRow("Turb. Model:", self.turb_model_combo)
        
        self.turb_check.toggled.connect(self.turb_model_combo.setEnabled)
        
        self.flow_group.setLayout(flow_layout)
        self.flow_group.setVisible(False)
        scroll_layout.addWidget(self.flow_group)
        
        # Geometry Configuration
        self.geom_group = QGroupBox("Geometry")
        geom_layout = QFormLayout()
        
        self.geom_type_combo = QComboBox()
        self.geom_type_combo.addItems(["channel", "pipe", "custom"])
        self.geom_type_combo.setToolTip("Flow geometry type")
        geom_layout.addRow("Type:", self.geom_type_combo)
        
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.001, 10)
        self.length_spin.setValue(0.1)
        self.length_spin.setSuffix(" m")
        self.length_spin.setDecimals(4)
        self.length_spin.setToolTip("Domain length")
        geom_layout.addRow("Length:", self.length_spin)
        
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.001, 1)
        self.height_spin.setValue(0.01)
        self.height_spin.setSuffix(" m")
        self.height_spin.setDecimals(4)
        self.height_spin.setToolTip("Domain height")
        geom_layout.addRow("Height:", self.height_spin)
        
        self.geom_group.setLayout(geom_layout)
        self.geom_group.setVisible(False)
        scroll_layout.addWidget(self.geom_group)
        
        # Connect mode change to show/hide CFD options
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Run button
        run_btn = QPushButton("▶ Run Simulation")
        run_btn.setStyleSheet("QPushButton { font-size: 14pt; padding: 10px; }")
        run_btn.clicked.connect(self._run_simulation)
        layout.addWidget(run_btn)
        
        self.tabs.addTab(tab, "⚙ Configuration")
    
    def _on_mode_changed(self, mode: str):
        """Show/hide CFD-specific controls based on mode"""
        is_cfd = mode in ["CFD", "Hybrid"]
        is_flow = mode in ["Flow", "CFD", "Hybrid"]
        
        self.mesh_group.setVisible(is_cfd)
        self.solver_group.setVisible(is_cfd)
        self.flow_group.setVisible(is_flow)
        self.geom_group.setVisible(is_flow)
        
        self._log(f"Mode changed to {mode}")
    
    def _create_parametric_tab(self):
        """Create parametric sweep tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        label = QLabel("<h2>🔄 Parametric Study</h2><p>Run sweeps across parameter ranges for optimization</p>")
        layout.addWidget(label)
        
        # Sweep configuration
        sweep_group = QGroupBox("Sweep Configuration")
        sweep_layout = QFormLayout()
        
        # Sweep parameter selector
        self.sweep_param_combo = QComboBox()
        self.sweep_param_combo.addItems([
            "Volume Fraction",
            "Temperature",
            "Particle Diameter",
            "Reynolds Number",
            "Pressure"
        ])
        self.sweep_param_combo.currentTextChanged.connect(self._update_sweep_units)
        sweep_layout.addRow("Sweep Parameter:", self.sweep_param_combo)
        
        # Start value
        self.sweep_start_spin = QDoubleSpinBox()
        self.sweep_start_spin.setRange(0.001, 10000)
        self.sweep_start_spin.setValue(0.01)
        self.sweep_start_spin.setDecimals(4)
        self.sweep_start_spin.setSuffix(" (vol.)")
        sweep_layout.addRow("Start Value:", self.sweep_start_spin)
        
        # End value
        self.sweep_end_spin = QDoubleSpinBox()
        self.sweep_end_spin.setRange(0.001, 10000)
        self.sweep_end_spin.setValue(0.05)
        self.sweep_end_spin.setDecimals(4)
        self.sweep_end_spin.setSuffix(" (vol.)")
        sweep_layout.addRow("End Value:", self.sweep_end_spin)
        
        # Number of steps
        self.sweep_steps_spin = QSpinBox()
        self.sweep_steps_spin.setRange(2, 100)
        self.sweep_steps_spin.setValue(10)
        self.sweep_steps_spin.setToolTip("Number of points in sweep")
        sweep_layout.addRow("Steps:", self.sweep_steps_spin)
        
        # Options
        self.sweep_plot_check = QCheckBox("Auto-generate comparison plots")
        self.sweep_plot_check.setChecked(True)
        sweep_layout.addRow("", self.sweep_plot_check)
        
        self.sweep_export_check = QCheckBox("Auto-export sweep data to CSV")
        self.sweep_export_check.setChecked(True)
        sweep_layout.addRow("", self.sweep_export_check)
        
        sweep_group.setLayout(sweep_layout)
        layout.addWidget(sweep_group)
        
        # Run button
        run_sweep_btn = QPushButton("▶ Run Parametric Sweep")
        run_sweep_btn.setStyleSheet("QPushButton { font-size: 14pt; padding: 10px; background-color: #4CAF50; color: white; }")
        run_sweep_btn.clicked.connect(self._run_parametric_sweep)
        layout.addWidget(run_sweep_btn)
        
        # Results table
        results_label = QLabel("<b>Sweep Results:</b>")
        layout.addWidget(results_label)
        
        self.sweep_table = QTableWidget()
        self.sweep_table.setColumnCount(3)
        self.sweep_table.setHorizontalHeaderLabels(["Parameter Value", "Thermal Conductivity (W/m·K)", "Enhancement (%)"])
        self.sweep_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.sweep_table)
        
        # Plot area
        self.sweep_figure = Figure(figsize=(10, 5))
        self.sweep_canvas = FigureCanvas(self.sweep_figure)
        layout.addWidget(self.sweep_canvas)
        
        # Export button
        export_sweep_btn = QPushButton("💾 Export Sweep Data")
        export_sweep_btn.clicked.connect(self._export_sweep_data)
        layout.addWidget(export_sweep_btn)
        
        self.tabs.addTab(tab, "🔄 Parametric")
    
    def _create_results_tab(self):
        """Create results tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results text
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 10))
        layout.addWidget(self.results_text)
        
        # Export buttons
        button_layout = QHBoxLayout()
        
        export_json_btn = QPushButton("Export JSON")
        export_json_btn.clicked.connect(self._export_results_json)
        button_layout.addWidget(export_json_btn)
        
        export_pdf_btn = QPushButton("Export PDF Report")
        export_pdf_btn.clicked.connect(self._export_pdf_report)
        button_layout.addWidget(export_pdf_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        self.tabs.addTab(tab, "📊 Results")
    
    def _create_visualization_tab(self):
        """Create visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Matplotlib canvas
        self.viz_figure = Figure(figsize=(8, 6))
        self.viz_canvas = FigureCanvas(self.viz_figure)
        self.viz_toolbar = NavigationToolbar(self.viz_canvas, tab)
        
        layout.addWidget(self.viz_toolbar)
        layout.addWidget(self.viz_canvas)
        
        # Visualization controls
        controls_layout = QHBoxLayout()
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Enhancement vs φ",
            "Viscosity Comparison",
            "Property Summary",
            "3D Surface"
        ])
        self.plot_type_combo.currentTextChanged.connect(self._update_visualization)
        controls_layout.addWidget(QLabel("Plot Type:"))
        controls_layout.addWidget(self.plot_type_combo)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._update_visualization)
        controls_layout.addWidget(refresh_btn)
        
        save_fig_btn = QPushButton("Save Figure (600 DPI)")
        save_fig_btn.clicked.connect(self._save_figure)
        controls_layout.addWidget(save_fig_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        self.tabs.addTab(tab, "📈 Visualization")
    
    def _create_validation_tab(self):
        """Create validation center tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header
        header = QLabel("<h2>Validation Center</h2><p>Compare against published experimental data</p>")
        layout.addWidget(header)
        
        # Dataset selector
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Dataset:"))
        self.dataset_combo = QComboBox()
        
        # Populate with real dataset names
        dataset_names = [
            "pak_cho_1998",
            "lee_1999", 
            "eastman_2001",
            "xuan_li_2000",
            "das_2003",
            "cuo_water"
        ]
        display_names = [
            "Pak & Cho (1998) - Al2O3/Water",
            "Lee et al. (1999) - Al2O3/Water",
            "Eastman et al. (2001) - Cu/EG",
            "Xuan & Li (2000) - Cu/Water",
            "Das et al. (2003) - Al2O3/Water (T-dep)",
            "CuO/Water (Multiple Sources)"
        ]
        
        for name in display_names:
            self.dataset_combo.addItem(name)
        
        dataset_layout.addWidget(self.dataset_combo)
        
        validate_btn = QPushButton("▶ Run Validation")
        validate_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 5px 15px; }")
        validate_btn.clicked.connect(self._run_validation)
        dataset_layout.addWidget(validate_btn)
        
        validate_all_btn = QPushButton("Run All Datasets")
        validate_all_btn.clicked.connect(self._run_validation_all)
        dataset_layout.addWidget(validate_all_btn)
        
        dataset_layout.addStretch()
        layout.addLayout(dataset_layout)
        
        # Validation canvas
        self.val_figure = Figure(figsize=(8, 6))
        self.val_canvas = FigureCanvas(self.val_figure)
        self.val_toolbar = NavigationToolbar(self.val_canvas, tab)
        
        layout.addWidget(self.val_toolbar)
        layout.addWidget(self.val_canvas)
        
        # Results summary with color-coded badge
        summary_layout = QHBoxLayout()
        
        self.val_badge_label = QLabel("⏸ READY")
        self.val_badge_label.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                padding: 10px;
                border: 2px solid #888;
                border-radius: 5px;
                background-color: #f0f0f0;
            }
        """)
        summary_layout.addWidget(self.val_badge_label)
        
        self.val_summary = QTextEdit()
        self.val_summary.setReadOnly(True)
        self.val_summary.setMaximumHeight(150)
        self.val_summary.setText("Select a dataset and click 'Run Validation' to compare against experimental data.")
        summary_layout.addWidget(self.val_summary, stretch=1)
        
        layout.addLayout(summary_layout)
        
        # Export validation button
        export_layout = QHBoxLayout()
        export_val_pdf = QPushButton("Export Validation PDF")
        export_val_pdf.clicked.connect(self._export_validation_pdf)
        export_layout.addWidget(export_val_pdf)
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
        self.tabs.addTab(tab, "✓ Validation")
    
    def _create_advanced_tab(self):
        """Create advanced visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        label = QLabel("<h2>Advanced Visualization</h2><p>CFD post-processing and sensitivity analysis</p>")
        layout.addWidget(label)
        
        # Control panel
        controls = QHBoxLayout()
        
        # Visualization type selector
        controls.addWidget(QLabel("Visualization Type:"))
        self.viz_type = QComboBox()
        self.viz_type.addItems([
            "Temperature Contour",
            "Velocity Vectors", 
            "Q-Criterion (Vortex)",
            "Streamlines",
            "Sobol Sensitivity",
            "Morris Screening"
        ])
        controls.addWidget(self.viz_type)
        
        # Generate button
        self.viz_generate_btn = QPushButton("Generate Visualization")
        self.viz_generate_btn.clicked.connect(self._generate_advanced_viz)
        controls.addWidget(self.viz_generate_btn)
        
        # Export button
        self.viz_export_btn = QPushButton("Export High-Res")
        self.viz_export_btn.clicked.connect(self._export_high_res_viz)
        controls.addWidget(self.viz_export_btn)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # Matplotlib figure
        self.adv_figure = Figure(figsize=(10, 6))
        self.adv_canvas = FigureCanvas(self.adv_figure)
        layout.addWidget(self.adv_canvas)
        
        # Info text
        self.adv_text = QTextEdit()
        self.adv_text.setReadOnly(True)
        self.adv_text.setMaximumHeight(100)
        self.adv_text.setText("Advanced visualization ready. Run CFD simulation first, then generate visualizations.")
        layout.addWidget(self.adv_text)
        
        self.tabs.addTab(tab, "🔬 Advanced")
    
    def _create_dock_panels(self):
        """Create dockable panels"""
        # Properties panel
        self.props_dock = QDockWidget("Properties", self)
        props_widget = QWidget()
        props_layout = QVBoxLayout(props_widget)
        
        self.props_table = QTableWidget()
        self.props_table.setColumnCount(2)
        self.props_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.props_table.horizontalHeader().setStretchLastSection(True)
        props_layout.addWidget(self.props_table)
        
        self.props_dock.setWidget(props_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.props_dock)
        
        # Log panel
        self.log_dock = QDockWidget("Log", self)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_dock.setWidget(self.log_text)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        
        # Update dock menu
        self.dock_menu.addAction(self.props_dock.toggleViewAction())
        self.dock_menu.addAction(self.log_dock.toggleViewAction())
        
        self._log("BKPS NFL Thermal Pro 7.0 initialized")
        self._log("Dedicated to: Brijesh Kumar Pandey")
    
    def _apply_theme(self, theme: str):
        """Apply light or dark theme"""
        self.settings.setValue("theme", theme)
        
        if theme == "dark":
            palette = QPalette()
            palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
            palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
            palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
            self.setPalette(palette)
            self._log(f"Applied dark theme")
        else:
            self.setPalette(self.style().standardPalette())
            self._log(f"Applied light theme")
    
    def _new_project(self):
        """Create new project"""
        self.current_project_path = None
        self.config = None
        self.results = None
        self.results_text.clear()
        self._update_properties_table()
        self._log("New project created")
    
    def _open_project(self):
        """Open existing project"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "BKPS Project (*.bkps);;JSON Config (*.json)"
        )
        
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Load config
                self.config = UnifiedConfig.load(filepath)
                
                # Update UI controls
                self._config_to_ui(self.config)
                
                # Update recent files
                self._add_recent_file(filepath)
                self.current_project_path = filepath
                
                self._log(f"Opened project: {Path(filepath).name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open project:\n{e}")
                self._log(f"ERROR: {e}")
    
    def _save_project(self):
        """Save current project"""
        if self.current_project_path:
            self._save_project_to(self.current_project_path)
        else:
            self._save_project_as()
    
    def _save_project_as(self):
        """Save project with new name"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "BKPS Project (*.bkps);;JSON Config (*.json)"
        )
        
        if filepath:
            self._save_project_to(filepath)
            self.current_project_path = filepath
            self._add_recent_file(filepath)
    
    def _save_project_to(self, filepath: str):
        """Save project to file"""
        try:
            config = self._ui_to_config()
            config.save(filepath)
            self._log(f"Saved project: {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project:\n{e}")
            self._log(f"ERROR: {e}")
    
    def _autosave(self):
        """Auto-save current state"""
        if self.config:
            autosave_path = Path.home() / ".bkps_autosave.json"
            try:
                config = self._ui_to_config()
                config.save(str(autosave_path))
                self._log("Auto-saved")
            except:
                pass  # Silent failure for autosave
    
    def _ui_to_config(self) -> UnifiedConfig:
        """Convert UI controls to UnifiedConfig"""
        from nanofluid_simulator import (
            SimulationMode, BaseFluidConfig, NanoparticleConfig,
            UnifiedConfig
        )
        
        mode_map = {
            "Static": SimulationMode.STATIC,
            "Flow": SimulationMode.FLOW,
            "CFD": SimulationMode.CFD,
            "Hybrid": SimulationMode.HYBRID
        }
        
        config = UnifiedConfig(
            mode=mode_map[self.mode_combo.currentText()],
            base_fluid=BaseFluidConfig(
                name=self.fluid_combo.currentText(),
                temperature=self.temp_spin.value(),
                pressure=self.pressure_spin.value() * 1000  # kPa to Pa
            ),
            nanoparticles=[
                NanoparticleConfig(
                    material=self.np_combo.currentText(),
                    volume_fraction=self.phi_spin.value(),
                    diameter=self.diameter_spin.value() * 1e-9,  # nm to m
                    shape=self.shape_combo.currentText()
                )
            ],
            enable_dlvo=self.dlvo_check.isChecked(),
            enable_non_newtonian=self.non_newt_check.isChecked(),
            enable_ai_recommendations=self.ai_check.isChecked(),
            enable_uq=self.uq_check.isChecked(),
            geometry=self._get_geometry_config() if mode_map[self.mode_combo.currentText()] in [SimulationMode.FLOW, SimulationMode.CFD, SimulationMode.HYBRID] else None,
            mesh=self._get_mesh_config() if mode_map[self.mode_combo.currentText()] in [SimulationMode.CFD, SimulationMode.HYBRID] else None,
            solver=self._get_solver_config(),
            flow=self._get_flow_config() if mode_map[self.mode_combo.currentText()] in [SimulationMode.FLOW, SimulationMode.CFD, SimulationMode.HYBRID] else None
        )
        
        return config
    
    def _config_to_ui(self, config: UnifiedConfig):
        """Update UI controls from config"""
        mode_map = {
            SimulationMode.STATIC: "Static",
            SimulationMode.FLOW: "Flow",
            SimulationMode.CFD: "CFD",
            SimulationMode.HYBRID: "Hybrid"
        }
        
        self.mode_combo.setCurrentText(mode_map[config.mode])
        self.fluid_combo.setCurrentText(config.base_fluid.name)
        self.temp_spin.setValue(config.base_fluid.temperature)
        self.pressure_spin.setValue(config.base_fluid.pressure / 1000)
        
        if config.nanoparticles:
            np = config.nanoparticles[0]
            self.np_combo.setCurrentText(np.material)
            self.phi_spin.setValue(np.volume_fraction)
            self.diameter_spin.setValue(np.diameter * 1e9)
            self.shape_combo.setCurrentText(np.shape)
        
        self.dlvo_check.setChecked(config.enable_dlvo)
        self.non_newt_check.setChecked(config.enable_non_newtonian)
        self.ai_check.setChecked(config.enable_ai_recommendations)
        if hasattr(self, 'uq_check'):
            self.uq_check.setChecked(getattr(config, 'enable_uq', False))
        
        # Load CFD-specific configs
        if config.mesh:
            self.nx_spin.setValue(config.mesh.nx)
            self.ny_spin.setValue(config.mesh.ny)
        
        if config.solver:
            self.max_iter_spin.setValue(config.solver.max_iterations)
            self.conv_tol_spin.setValue(config.solver.convergence_tolerance)
            self.relax_spin.setValue(config.solver.relaxation_factor)
            if hasattr(config.solver, 'enable_turbulence'):
                self.turb_check.setChecked(config.solver.enable_turbulence)
        
        if config.flow:
            self.velocity_spin.setValue(config.flow.velocity)
        
        if config.geometry:
            self.geom_type_combo.setCurrentText(config.geometry.geometry_type)
            self.length_spin.setValue(config.geometry.length)
            self.height_spin.setValue(config.geometry.height)
    
    def _get_geometry_config(self) -> 'GeometryConfig':
        """Build geometry configuration from UI"""
        from nanofluid_simulator import GeometryConfig
        return GeometryConfig(
            geometry_type=self.geom_type_combo.currentText(),
            length=self.length_spin.value(),
            height=self.height_spin.value()
        )
    
    def _get_mesh_config(self) -> 'MeshConfig':
        """Build mesh configuration from UI"""
        from nanofluid_simulator import MeshConfig
        return MeshConfig(
            nx=self.nx_spin.value(),
            ny=self.ny_spin.value(),
            mesh_type="structured"
        )
    
    def _get_solver_config(self) -> 'SolverConfig':
        """Build solver configuration from UI"""
        from nanofluid_simulator import SolverConfig, SolverBackend
        
        turb_model_map = {
            "k-epsilon": "k-epsilon",
            "k-omega-sst": "k-omega-sst",
            "spalart-allmaras": "spalart-allmaras"
        }
        
        return SolverConfig(
            max_iterations=self.max_iter_spin.value(),
            convergence_tolerance=self.conv_tol_spin.value(),
            relaxation_factor=self.relax_spin.value(),
            enable_turbulence=self.turb_check.isChecked(),
            turbulence_model=turb_model_map.get(self.turb_model_combo.currentText(), "k-epsilon"),
            backend=SolverBackend.NUMPY
        )
    
    def _get_flow_config(self) -> 'FlowConfig':
        """Build flow configuration from UI"""
        from nanofluid_simulator import FlowConfig
        return FlowConfig(
            velocity=self.velocity_spin.value(),
            inlet_temperature=self.temp_spin.value()
        )
    
    def _run_simulation(self):
        """Run simulation with progress dialog"""
        try:
            # Build config
            self.config = self._ui_to_config()
            
            # Validate
            valid, errors = self.config.validate()
            if not valid:
                QMessageBox.warning(self, "Validation Error", "\n".join(errors))
                return
            
            # Create engine
            engine = BKPSNanofluidEngine(self.config)
            
            # Run UQ if enabled
            if self.uq_check.isChecked():
                try:
                    from nanofluid_simulator.uq import Distribution, monte_carlo
                    self._log("UQ enabled - will run Monte Carlo after main simulation")
                except ImportError:
                    self._log("WARNING: UQ module not available, skipping uncertainty analysis")
            
            # Progress dialog
            self.progress_dialog = QProgressDialog(
                "Running simulation...", "Cancel", 0, 100, self
            )
            self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self.progress_dialog.setMinimumDuration(0)
            
            # Start computation thread
            self.computation_thread = ComputationThread(engine)
            self.computation_thread.progress.connect(self._update_progress)
            self.computation_thread.finished.connect(self._computation_finished)
            self.computation_thread.error.connect(self._computation_error)
            self.progress_dialog.canceled.connect(self._cancel_computation)
            
            self.computation_thread.start()
            self._log("Simulation started")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start simulation:\n{e}")
            self._log(f"ERROR: {e}")
    
    def _update_progress(self, value: int, message: str):
        """Update progress dialog"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(message)
    
    def _cancel_computation(self):
        """Cancel running computation"""
        if self.computation_thread:
            self.computation_thread.cancel()
            self._log("Computation cancelled")
    
    def _computation_finished(self, results: Dict):
        """Handle computation completion"""
        self.results = results
        
        # Close progress dialog
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        
        # Display results
        self._display_results()
        
        # Update visualization
        self._update_visualization()
        
        # Update properties
        self._update_properties_table()
        
        self._log("Simulation completed successfully")
        self.statusBar.showMessage("Simulation completed", 5000)
    
    def _computation_error(self, error_msg: str):
        """Handle computation error"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        
        QMessageBox.critical(self, "Computation Error", error_msg)
        self._log(f"ERROR: {error_msg}")
    
    def _display_results(self):
        """Display simulation results"""
        if not self.results:
            return
        
        # Extract results
        if 'static' in self.results:
            res = self.results['static']
        else:
            res = self.results
        
        # Format output
        output = "═══ BKPS NFL THERMAL PRO 7.0 — RESULTS ═══\n\n"
        output += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        output += f"Mode: {self.config.mode.value.upper()}\n\n"
        
        # Base results
        if 'k_base' in res:
            output += f"Base Fluid k: {res['k_base']:.6f} W/m·K\n"
        if 'k_static' in res:
            output += f"Nanofluid k:  {res['k_static']:.6f} W/m·K\n"
        if 'enhancement_k' in res:
            output += f"Enhancement:  {res['enhancement_k']:.2f}%\n"
        
        output += "\n"
        
        if 'mu_base' in res:
            output += f"Base Fluid μ: {res['mu_base']*1000:.6f} mPa·s\n"
        if 'mu_nf' in res:
            output += f"Nanofluid μ:  {res['mu_nf']*1000:.6f} mPa·s\n"
        if 'viscosity_ratio' in res:
            output += f"Viscosity Ratio: {res['viscosity_ratio']:.4f}\n"
        
        # Uncertainty Quantification results
        if 'uq' in self.results:
            uq = self.results['uq']
            output += "\n═══ UNCERTAINTY QUANTIFICATION ═══\n\n"
            output += f"Monte Carlo Samples: {uq.get('samples', 'N/A')}\n"
            output += f"Mean k_eff: {uq['mean']:.6f} W/m·K\n"
            output += f"Std dev:    {uq['std']:.6f} W/m·K\n"
            if 'ci95' in uq and len(uq['ci95']) == 2:
                output += f"95% CI:     [{uq['ci95'][0]:.6f}, {uq['ci95'][1]:.6f}] W/m·K\n"
        
        self.results_text.setText(output)
        output += f"Dedicated to: Brijesh Kumar Pandey\n\n"
        
        output += "── CONFIGURATION ──\n"
        output += f"Base Fluid: {self.config.base_fluid.name}\n"
        output += f"Temperature: {self.config.base_fluid.temperature} K\n"
        output += f"Nanoparticle: {self.config.nanoparticles[0].material}\n"
        output += f"Volume Fraction: {self.config.nanoparticles[0].volume_fraction*100:.2f}%\n"
        output += f"Diameter: {self.config.nanoparticles[0].diameter*1e9:.1f} nm\n\n"
        
        output += "── THERMAL CONDUCTIVITY ──\n"
        if 'k_base' in res:
            output += f"k_base:  {res['k_base']:.6f} W/m·K\n"
        if 'k_static' in res:
            output += f"k_nf:    {res['k_static']:.6f} W/m·K\n"
        if 'enhancement_k' in res:
            output += f"Enhancement: {res['enhancement_k']:.2f}%\n\n"
        
        output += "── VISCOSITY ──\n"
        if 'mu_base' in res:
            output += f"μ_base:  {res['mu_base']*1000:.6f} mPa·s\n"
        if 'mu_nf' in res:
            output += f"μ_nf:    {res['mu_nf']*1000:.6f} mPa·s\n"
        if 'viscosity_ratio' in res:
            output += f"Ratio:   {res['viscosity_ratio']:.4f}\n\n"
        
        output += "── OTHER PROPERTIES ──\n"
        if 'rho_nf' in res:
            output += f"Density:      {res['rho_nf']:.2f} kg/m³\n"
        if 'cp_nf' in res:
            output += f"Specific Heat: {res['cp_nf']:.2f} J/kg·K\n"
        
        self.results_text.setText(output)
        self.tabs.setCurrentIndex(1)  # Switch to results tab
    
    def _update_visualization(self):
        """Update visualization plots"""
        if not self.results:
            return
        
        self.viz_figure.clear()
        
        plot_type = self.plot_type_combo.currentText()
        
        if plot_type == "Enhancement vs φ":
            self._plot_enhancement()
        elif plot_type == "Viscosity Comparison":
            self._plot_viscosity()
        elif plot_type == "Property Summary":
            self._plot_summary()
        elif plot_type == "3D Surface":
            self._plot_3d_surface()
        
        self.viz_canvas.draw()
    
    def _plot_enhancement(self):
        """Plot enhancement bar chart"""
        if 'static' in self.results:
            res = self.results['static']
        else:
            res = self.results
        
        ax = self.viz_figure.add_subplot(111)
        
        if 'enhancement_k' in res:
            ax.bar(['Base Fluid', 'Nanofluid'],
                  [0, res['enhancement_k']],
                  color=['#2E86AB', '#06A77D'],
                  edgecolor='black')
            ax.set_ylabel('Enhancement (%)', fontsize=12)
            ax.set_title('Thermal Conductivity Enhancement', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def _plot_viscosity(self):
        """Plot viscosity comparison"""
        if 'static' in self.results:
            res = self.results['static']
        else:
            res = self.results
        
        ax = self.viz_figure.add_subplot(111)
        
        if 'mu_base' in res and 'mu_nf' in res:
            ax.bar(['Base Fluid', 'Nanofluid'],
                  [res['mu_base']*1000, res['mu_nf']*1000],
                  color=['#2E86AB', '#F18F01'],
                  edgecolor='black')
            ax.set_ylabel('Viscosity (mPa·s)', fontsize=12)
            ax.set_title('Viscosity Comparison', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def _plot_summary(self):
        """Plot properties summary"""
        ax = self.viz_figure.add_subplot(111)
        ax.axis('off')
        
        if 'static' in self.results:
            res = self.results['static']
        else:
            res = self.results
        
        table_data = []
        if 'k_static' in res:
            table_data.append(['k_eff', f"{res['k_static']:.6f} W/m·K"])
        if 'mu_nf' in res:
            table_data.append(['μ_eff', f"{res['mu_nf']*1000:.6f} mPa·s"])
        if 'rho_nf' in res:
            table_data.append(['ρ_eff', f"{res['rho_nf']:.2f} kg/m³"])
        if 'cp_nf' in res:
            table_data.append(['c_p', f"{res['cp_nf']:.2f} J/kg·K"])
        
        if table_data:
            table = ax.table(cellText=table_data,
                           colLabels=['Property', 'Value'],
                           cellLoc='left',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2)
    
    def _plot_3d_surface(self):
        """Plot 3D surface (placeholder)"""
        ax = self.viz_figure.add_subplot(111, projection='3d')
        
        # Placeholder data
        X = np.linspace(0, 5, 30)
        Y = np.linspace(0, 5, 30)
        X, Y = np.meshgrid(X, Y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_title('3D Surface Plot (Placeholder)', fontsize=14)
    
    def _save_figure(self):
        """Save current figure at 600 DPI"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        
        if filepath:
            self.viz_figure.savefig(filepath, dpi=600, bbox_inches='tight')
            self._log(f"Figure saved: {Path(filepath).name}")
    
    def _run_validation(self):
        """Run validation against selected dataset"""
        if not self.config:
            QMessageBox.warning(self, "No Configuration", 
                              "Please configure and run a simulation first.")
            return
        
        # Get selected dataset
        idx = self.dataset_combo.currentIndex()
        dataset_keys = ['pak_cho_1998', 'lee_1999', 'eastman_2001', 
                       'xuan_li_2000', 'das_2003', 'cuo_water']
        
        if idx >= len(dataset_keys):
            return
        
        dataset_key = dataset_keys[idx]
        
        try:
            self._log(f"Running validation: {dataset_key}")
            self.statusBar.showMessage(f"Running validation against {dataset_key}...")
            
            # Create engine from current config
            engine = BKPSNanofluidEngine(self.config)
            
            # Run validation
            self.validation_results = self.validation_center.validate_simulation(
                engine, dataset_key
            )
            
            # Update display
            self._display_validation_results()
            
            self.statusBar.showMessage("Validation complete", 3000)
            self._log(f"Validation complete: MAPE={self.validation_results['mape']:.2f}%")
            
        except Exception as e:
            QMessageBox.critical(self, "Validation Error", f"Validation failed:\n{e}")
            self._log(f"ERROR: {e}")
    
    def _run_validation_all(self):
        """Run validation against all datasets"""
        if not self.config:
            QMessageBox.warning(self, "No Configuration",
                              "Please configure and run a simulation first.")
            return
        
        try:
            self._log("Running validation against all datasets...")
            self.statusBar.showMessage("Running validation against all datasets...")
            
            # Create engine
            engine = BKPSNanofluidEngine(self.config)
            
            # Run all validations
            all_results = self.validation_center.validate_all(engine)
            
            # Display summary
            summary_text = "═══ ALL DATASETS VALIDATION ═══\n\n"
            
            for key, results in all_results.items():
                if key == 'overall':
                    continue
                if 'mape' in results:
                    summary_text += f"{results['dataset'].name}:\n"
                    summary_text += f"  MAPE: {results['mape']:.2f}%\n"
                    summary_text += f"  Badge: {results['badge']}\n"
                    summary_text += f"  Rating: {results['rating']}\n\n"
            
            if 'overall' in all_results:
                summary_text += f"\nOVERALL MEAN MAPE: {all_results['overall']['mean_mape']:.2f}%\n"
                summary_text += f"Datasets Validated: {all_results['overall']['datasets_validated']}"
            
            self.val_summary.setText(summary_text)
            
            # Update badge for overall performance
            overall_mape = all_results.get('overall', {}).get('mean_mape', 100)
            if overall_mape < 10:
                self.val_badge_label.setText("✓ EXCELLENT")
                self.val_badge_label.setStyleSheet("""
                    QLabel {
                        font-size: 14pt; font-weight: bold; padding: 10px;
                        border: 2px solid #06A77D; border-radius: 5px;
                        background-color: #D4EDDA; color: #155724;
                    }
                """)
            elif overall_mape < 20:
                self.val_badge_label.setText("✓ GOOD")
                self.val_badge_label.setStyleSheet("""
                    QLabel {
                        font-size: 14pt; font-weight: bold; padding: 10px;
                        border: 2px solid #F18F01; border-radius: 5px;
                        background-color: #FFF3CD; color: #856404;
                    }
                """)
            else:
                self.val_badge_label.setText("⚠ REVIEW")
                self.val_badge_label.setStyleSheet("""
                    QLabel {
                        font-size: 14pt; font-weight: bold; padding: 10px;
                        border: 2px solid #C1121F; border-radius: 5px;
                        background-color: #F8D7DA; color: #721C24;
                    }
                """)
            
            self.statusBar.showMessage("All datasets validated", 3000)
            self._log(f"All datasets validated: Overall MAPE={overall_mape:.2f}%")
            
        except Exception as e:
            QMessageBox.critical(self, "Validation Error", f"Validation failed:\n{e}")
            self._log(f"ERROR: {e}")
    
    def _display_validation_results(self):
        """Display validation results with plot and summary"""
        if not self.validation_results:
            return
        
        results = self.validation_results
        dataset = results['dataset']
        
        # Clear figure
        self.val_figure.clear()
        ax = self.val_figure.add_subplot(111)
        
        # Plot experimental vs simulated
        x_vals = np.array(results['phi_values'])
        exp_vals = np.array(results['k_enhancement_exp'])
        sim_vals = np.array(results['k_enhancement_sim'])
        
        # Scatter plots
        ax.scatter(x_vals * 100 if not dataset.temp_dependent else x_vals, 
                  exp_vals, 
                  s=100, marker='o', c='#2E86AB', label='Experimental', 
                  edgecolors='black', linewidth=1.5, zorder=3)
        
        ax.scatter(x_vals * 100 if not dataset.temp_dependent else x_vals,
                  sim_vals,
                  s=100, marker='^', c='#06A77D', label='Simulated (BKPS)',
                  edgecolors='black', linewidth=1.5, zorder=3)
        
        # Connect points
        ax.plot(x_vals * 100 if not dataset.temp_dependent else x_vals,
               exp_vals, 'o-', color='#2E86AB', alpha=0.3, linewidth=2)
        ax.plot(x_vals * 100 if not dataset.temp_dependent else x_vals,
               sim_vals, '^-', color='#06A77D', alpha=0.3, linewidth=2)
        
        # Error bands
        for i, (x, exp, sim) in enumerate(zip(x_vals, exp_vals, sim_vals)):
            x_plot = x * 100 if not dataset.temp_dependent else x
            ax.plot([x_plot, x_plot], [exp, sim], 
                   'r--', alpha=0.5, linewidth=1)
        
        # Labels
        if dataset.temp_dependent:
            ax.set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
        else:
            ax.set_xlabel('Volume Fraction (%)', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Thermal Conductivity Enhancement (%)', 
                     fontsize=12, fontweight='bold')
        ax.set_title(f'Validation: {dataset.name}', 
                    fontsize=14, fontweight='bold')
        
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        self.val_canvas.draw()
        
        # Update summary text
        summary = get_validation_summary(results)
        self.val_summary.setText(summary)
        
        # Update badge
        if results['rating'] == 'EXCELLENT':
            self.val_badge_label.setText("✓ EXCELLENT")
            self.val_badge_label.setStyleSheet("""
                QLabel {
                    font-size: 14pt; font-weight: bold; padding: 10px;
                    border: 2px solid #06A77D; border-radius: 5px;
                    background-color: #D4EDDA; color: #155724;
                }
            """)
        elif results['rating'] == 'GOOD':
            self.val_badge_label.setText("✓ GOOD")
            self.val_badge_label.setStyleSheet("""
                QLabel {
                    font-size: 14pt; font-weight: bold; padding: 10px;
                    border: 2px solid #F18F01; border-radius: 5px;
                    background-color: #FFF3CD; color: #856404;
                }
            """)
        else:
            self.val_badge_label.setText("⚠ REVIEW")
            self.val_badge_label.setStyleSheet("""
                QLabel {
                    font-size: 14pt; font-weight: bold; padding: 10px;
                    border: 2px solid #C1121F; border-radius: 5px;
                    background-color: #F8D7DA; color: #721C24;
                }
            """)
    
    def _export_validation_pdf(self):
        """Export validation results to PDF"""
        if not self.validation_results:
            QMessageBox.warning(self, "No Results", 
                              "No validation results to export. Run validation first.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Validation PDF", "", "PDF (*.pdf)"
        )
        
        if filepath:
            try:
                # Save current figure
                self.val_figure.savefig(filepath, dpi=300, bbox_inches='tight')
                self._log(f"Validation results exported: {Path(filepath).name}")
                QMessageBox.information(self, "Success", 
                                      f"Validation results saved to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")
                self._log(f"ERROR: {e}")
    
    def _generate_advanced_viz(self):
        """Generate advanced visualization"""
        viz_type = self.viz_type.currentText()
        
        try:
            # Create visualizer
            visualizer = AdvancedVisualizer(dpi=300)
            
            self.adv_figure.clear()
            ax = self.adv_figure.add_subplot(111)
            
            # Generate sample CFD field for demonstration
            # In real usage, this would use actual simulation results
            if not self.results or 'cfd' not in self.results:
                self.adv_text.setText(
                    "⚠️ No CFD results available. Using sample data for demonstration.\n"
                    "Run a CFD simulation to visualize actual results."
                )
                field = create_sample_cfd_field()
            else:
                # Use actual CFD results (to be implemented)
                field = create_sample_cfd_field()
            
            # Generate visualization based on type
            if viz_type == "Temperature Contour":
                visualizer.plot_2d_contour(
                    field['T'], field['X'], field['Y'],
                    title="Temperature Distribution",
                    xlabel="x (m)", ylabel="y (m)",
                    colorbar_label="Temperature (K)",
                    fig=self.adv_figure, ax=ax
                )
                info = "Temperature contour plot showing spatial distribution."
                
            elif viz_type == "Velocity Vectors":
                visualizer.plot_velocity_vectors(
                    field['u'], field['v'], field['X'], field['Y'],
                    title="Velocity Vector Field",
                    xlabel="x (m)", ylabel="y (m)",
                    fig=self.adv_figure, ax=ax
                )
                info = "Velocity vectors with magnitude background."
                
            elif viz_type == "Q-Criterion (Vortex)":
                visualizer.plot_q_criterion(
                    field['u'], field['v'], field['X'], field['Y'],
                    field['dx'], field['dy'],
                    title="Q-Criterion Vortex Identification",
                    xlabel="x (m)", ylabel="y (m)",
                    fig=self.adv_figure, ax=ax
                )
                info = "Q-criterion for vortex identification: Q = 0.5(Ω² - S²)"
                
            elif viz_type == "Streamlines":
                visualizer.plot_streamlines(
                    field['u'], field['v'], field['X'], field['Y'],
                    title="Flow Streamlines",
                    xlabel="x (m)", ylabel="y (m)",
                    fig=self.adv_figure, ax=ax
                )
                info = "Streamline visualization of flow field."
                
            elif viz_type == "Sobol Sensitivity":
                # Placeholder - requires sensitivity analysis run
                info = ("Sobol sensitivity analysis requires SALib package.\n"
                       "Install with: pip install SALib\n"
                       "Feature coming soon in full release.")
                ax.text(0.5, 0.5, "Sensitivity Analysis\n(Requires SALib)",
                       ha='center', va='center', transform=ax.transAxes)
                
            elif viz_type == "Morris Screening":
                # Placeholder - requires sensitivity analysis run
                info = ("Morris screening requires SALib package.\n"
                       "Install with: pip install SALib\n"
                       "Feature coming soon in full release.")
                ax.text(0.5, 0.5, "Morris Screening\n(Requires SALib)",
                       ha='center', va='center', transform=ax.transAxes)
            
            self.adv_text.setText(info)
            self.adv_canvas.draw()
            self._log(f"Generated {viz_type} visualization")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Visualization failed:\n{e}")
            self._log(f"ERROR: {e}")
    
    def _export_high_res_viz(self):
        """Export high-resolution visualization"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export High-Res Visualization", "", 
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        
        if filepath:
            try:
                # Determine DPI based on format
                dpi = 600 if filepath.endswith('.png') else 300
                
                # Save figure
                self.adv_figure.savefig(filepath, dpi=dpi, bbox_inches='tight')
                self._log(f"High-res visualization exported: {Path(filepath).name}")
                QMessageBox.information(self, "Success", 
                                      f"Visualization saved at {dpi} DPI to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{e}")
                self._log(f"ERROR: {e}")
    
    def _update_properties_table(self):
        """Update properties dock panel"""
        self.props_table.setRowCount(0)
        
        if self.results:
            if 'static' in self.results:
                res = self.results['static']
            else:
                res = self.results
            
            for key, value in res.items():
                row = self.props_table.rowCount()
                self.props_table.insertRow(row)
                self.props_table.setItem(row, 0, QTableWidgetItem(str(key)))
                
                if isinstance(value, float):
                    self.props_table.setItem(row, 1, QTableWidgetItem(f"{value:.6f}"))
                else:
                    self.props_table.setItem(row, 1, QTableWidgetItem(str(value)))
    
    def _export_results_json(self):
        """Export results to JSON"""
        if not self.results:
            QMessageBox.warning(self, "No Results", "No results to export. Run simulation first.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "JSON (*.json)"
        )
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2)
            self._log(f"Results exported: {Path(filepath).name}")
    
    def _export_pdf_report(self):
        """Export PDF report"""
        if not self.results or not self.config:
            QMessageBox.warning(self, "No Results", "No results to export. Run simulation first.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export PDF Report", "", "PDF (*.pdf)"
        )
        
        if filepath:
            try:
                generator = PDFReportGenerator()
                generator.generate_report(self.results, self.config, filepath)
                self._log(f"PDF report generated: {Path(filepath).name}")
                QMessageBox.information(self, "Success", f"PDF report saved to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to generate PDF:\n{e}")
                self._log(f"ERROR: {e}")
    
    def _load_recent_files(self) -> List[str]:
        """Load recent files from settings"""
        recent = self.settings.value("recent_files", [])
        if isinstance(recent, str):
            recent = [recent]
        return recent if recent else []
    
    def _add_recent_file(self, filepath: str):
        """Add file to recent files"""
        if filepath in self.recent_files:
            self.recent_files.remove(filepath)
        self.recent_files.insert(0, filepath)
        self.recent_files = self.recent_files[:self.max_recent_files]
        self.settings.setValue("recent_files", self.recent_files)
        self._update_recent_menu()
    
    def _update_recent_menu(self):
        """Update recent files menu"""
        self.recent_menu.clear()
        
        for filepath in self.recent_files:
            if Path(filepath).exists():
                action = QAction(Path(filepath).name, self)
                action.setToolTip(filepath)
                action.triggered.connect(lambda checked, f=filepath: self._open_recent(f))
                self.recent_menu.addAction(action)
    
    def _open_recent(self, filepath: str):
        """Open recent file"""
        if Path(filepath).exists():
            self.current_project_path = filepath
            self._open_project()
    
    def _restore_window_state(self):
        """Restore window geometry and state"""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
    
    def _save_window_state(self):
        """Save window geometry and state"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
    
    def _log(self, message: str):
        """Append message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def _show_documentation(self):
        """Show documentation"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Documentation")
        msg.setText("BKPS NFL Thermal Pro 7.0")
        msg.setInformativeText(
            "For documentation, see:\n\n"
            "• CHANGELOG_V7.md\n"
            "• V7_DELIVERY_SUMMARY.md\n"
            "• docs/USER_GUIDE.md\n\n"
            "Dedicated to: Brijesh Kumar Pandey"
        )
        msg.exec()
    
    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About BKPS NFL Thermal Pro",
            "<h2>BKPS NFL Thermal Pro 7.0</h2>"
            "<p>Professional Nanofluid Simulation Platform</p>"
            "<p><b>Dedicated to:</b> Brijesh Kumar Pandey</p>"
            "<p><b>Release Date:</b> 2025-11-30</p>"
            "<p><b>License:</b> MIT</p>"
            "<p><b>Repository:</b> <a href='https://github.com/msaurav625-lgtm/test'>"
            "github.com/msaurav625-lgtm/test</a></p>"
        )
    
    def _on_fluid_changed(self, fluid: str):
        """Handle fluid selection change"""
        if fluid == "Custom...":
            self._add_custom_fluid()
    
    def _on_np_changed(self, material: str):
        """Handle nanoparticle selection change"""
        if material == "Custom...":
            self._add_custom_nanoparticle()
    
    def _add_custom_fluid(self):
        """Add custom fluid material"""
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Custom Base Fluid")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("<h3>Custom Fluid Properties</h3>"))
        
        form = QFormLayout()
        
        # Name
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("e.g., Glycerol, Thermal Oil")
        form.addRow("Name:", name_edit)
        
        # Thermal conductivity
        k_spin = QDoubleSpinBox()
        k_spin.setRange(0.01, 1000)
        k_spin.setValue(0.613)
        k_spin.setDecimals(3)
        k_spin.setSuffix(" W/m·K")
        form.addRow("Thermal Conductivity:", k_spin)
        
        # Density
        rho_spin = QDoubleSpinBox()
        rho_spin.setRange(100, 20000)
        rho_spin.setValue(997)
        rho_spin.setDecimals(1)
        rho_spin.setSuffix(" kg/m³")
        form.addRow("Density:", rho_spin)
        
        # Specific heat
        cp_spin = QDoubleSpinBox()
        cp_spin.setRange(100, 10000)
        cp_spin.setValue(4179)
        cp_spin.setDecimals(1)
        cp_spin.setSuffix(" J/kg·K")
        form.addRow("Specific Heat:", cp_spin)
        
        # Viscosity
        mu_spin = QDoubleSpinBox()
        mu_spin.setRange(0.0001, 10)
        mu_spin.setValue(0.001)
        mu_spin.setDecimals(6)
        mu_spin.setSuffix(" Pa·s")
        form.addRow("Viscosity:", mu_spin)
        
        layout.addLayout(form)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = name_edit.text().strip()
            if name:
                # Store custom fluid properties
                self.custom_fluids[name] = {
                    'k': k_spin.value(),
                    'rho': rho_spin.value(),
                    'cp': cp_spin.value(),
                    'mu': mu_spin.value()
                }
                
                # Add to combo box (insert before "Custom...")
                index = self.fluid_combo.count() - 1
                self.fluid_combo.insertItem(index, name)
                self.fluid_combo.setCurrentText(name)
                
                self._log(f"Added custom fluid: {name}")
                QMessageBox.information(self, "Success", f"Custom fluid '{name}' added successfully!")
            else:
                QMessageBox.warning(self, "Warning", "Fluid name cannot be empty!")
    
    def _add_custom_nanoparticle(self):
        """Add custom nanoparticle material"""
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Custom Nanoparticle")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("<h3>Custom Nanoparticle Properties</h3>"))
        
        form = QFormLayout()
        
        # Name
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("e.g., Diamond, ZrO2, MgO")
        form.addRow("Name:", name_edit)
        
        # Thermal conductivity
        k_spin = QDoubleSpinBox()
        k_spin.setRange(1, 5000)
        k_spin.setValue(40.0)
        k_spin.setDecimals(1)
        k_spin.setSuffix(" W/m·K")
        form.addRow("Thermal Conductivity:", k_spin)
        
        # Density
        rho_spin = QDoubleSpinBox()
        rho_spin.setRange(1000, 25000)
        rho_spin.setValue(3970)
        rho_spin.setDecimals(1)
        rho_spin.setSuffix(" kg/m³")
        form.addRow("Density:", rho_spin)
        
        # Specific heat
        cp_spin = QDoubleSpinBox()
        cp_spin.setRange(100, 2000)
        cp_spin.setValue(765)
        cp_spin.setDecimals(1)
        cp_spin.setSuffix(" J/kg·K")
        form.addRow("Specific Heat:", cp_spin)
        
        layout.addLayout(form)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = name_edit.text().strip()
            if name:
                # Store custom nanoparticle properties
                self.custom_nanoparticles[name] = {
                    'k': k_spin.value(),
                    'rho': rho_spin.value(),
                    'cp': cp_spin.value()
                }
                
                # Add to combo box (insert before "Custom...")
                index = self.np_combo.count() - 1
                self.np_combo.insertItem(index, name)
                self.np_combo.setCurrentText(name)
                
                self._log(f"Added custom nanoparticle: {name}")
                QMessageBox.information(self, "Success", f"Custom nanoparticle '{name}' added successfully!")
            else:
                QMessageBox.warning(self, "Warning", "Nanoparticle name cannot be empty!")
    
    def _update_sweep_units(self, param: str):
        """Update sweep spin box units based on parameter"""
        if param == "Volume Fraction":
            self.sweep_start_spin.setRange(0.001, 0.10)
            self.sweep_start_spin.setValue(0.01)
            self.sweep_end_spin.setRange(0.001, 0.10)
            self.sweep_end_spin.setValue(0.05)
            self.sweep_start_spin.setSuffix(" (vol.)")
            self.sweep_end_spin.setSuffix(" (vol.)")
            self.sweep_start_spin.setDecimals(4)
            self.sweep_end_spin.setDecimals(4)
        elif param == "Temperature":
            self.sweep_start_spin.setRange(273, 373)
            self.sweep_start_spin.setValue(280)
            self.sweep_end_spin.setRange(273, 373)
            self.sweep_end_spin.setValue(360)
            self.sweep_start_spin.setSuffix(" K")
            self.sweep_end_spin.setSuffix(" K")
            self.sweep_start_spin.setDecimals(1)
            self.sweep_end_spin.setDecimals(1)
        elif param == "Particle Diameter":
            self.sweep_start_spin.setRange(1, 200)
            self.sweep_start_spin.setValue(10)
            self.sweep_end_spin.setRange(1, 200)
            self.sweep_end_spin.setValue(100)
            self.sweep_start_spin.setSuffix(" nm")
            self.sweep_end_spin.setSuffix(" nm")
            self.sweep_start_spin.setDecimals(1)
            self.sweep_end_spin.setDecimals(1)
        elif param == "Reynolds Number":
            self.sweep_start_spin.setRange(1, 100000)
            self.sweep_start_spin.setValue(100)
            self.sweep_end_spin.setRange(1, 100000)
            self.sweep_end_spin.setValue(10000)
            self.sweep_start_spin.setSuffix("")
            self.sweep_end_spin.setSuffix("")
            self.sweep_start_spin.setDecimals(0)
            self.sweep_end_spin.setDecimals(0)
        elif param == "Pressure":
            self.sweep_start_spin.setRange(50, 1000)
            self.sweep_start_spin.setValue(101.325)
            self.sweep_end_spin.setRange(50, 1000)
            self.sweep_end_spin.setValue(500)
            self.sweep_start_spin.setSuffix(" kPa")
            self.sweep_end_spin.setSuffix(" kPa")
            self.sweep_start_spin.setDecimals(2)
            self.sweep_end_spin.setDecimals(2)
    
    def _run_parametric_sweep(self):
        """Run parametric sweep"""
        try:
            param = self.sweep_param_combo.currentText()
            start = self.sweep_start_spin.value()
            end = self.sweep_end_spin.value()
            steps = self.sweep_steps_spin.value()
            
            if start >= end:
                QMessageBox.warning(self, "Invalid Range", "Start value must be less than end value!")
                return
            
            self._log(f"Starting parametric sweep: {param} from {start} to {end} ({steps} steps)")
            
            # Create progress dialog
            progress = QProgressDialog(f"Running {param} sweep...", "Cancel", 0, steps, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            
            # Generate parameter values
            param_values = np.linspace(start, end, steps)
            results_k = []
            results_enh = []
            
            # Run sweep
            for i, value in enumerate(param_values):
                if progress.wasCanceled():
                    self._log("Parametric sweep cancelled by user")
                    return
                
                progress.setValue(i)
                
                # Update parameter
                if param == "Volume Fraction":
                    self.phi_spin.setValue(value)
                elif param == "Temperature":
                    self.temp_spin.setValue(value)
                elif param == "Particle Diameter":
                    self.diameter_spin.setValue(value)
                elif param == "Reynolds Number":
                    self.velocity_spin.setValue(value / 1000.0)  # Approximate
                elif param == "Pressure":
                    self.pressure_spin.setValue(value)
                
                # Build config and run
                config = self._build_config()
                engine = BKPSNanofluidEngine(config)
                result = engine.run()
                
                # Extract results
                if 'static' in result:
                    k_eff = result['static']['k_static']
                    k_base = result['static']['k_base']
                else:
                    k_eff = result.get('k_nf', result.get('k_static', 0))
                    k_base = result.get('k_base', 0.613)
                
                enhancement = ((k_eff - k_base) / k_base) * 100 if k_base > 0 else 0
                
                results_k.append(k_eff)
                results_enh.append(enhancement)
            
            progress.setValue(steps)
            
            # Update table
            self.sweep_table.setRowCount(steps)
            for i, (pval, k_val, enh_val) in enumerate(zip(param_values, results_k, results_enh)):
                self.sweep_table.setItem(i, 0, QTableWidgetItem(f"{pval:.4f}"))
                self.sweep_table.setItem(i, 1, QTableWidgetItem(f"{k_val:.6f}"))
                self.sweep_table.setItem(i, 2, QTableWidgetItem(f"{enh_val:.2f}"))
            
            # Plot results if requested
            if self.sweep_plot_check.isChecked():
                self.sweep_figure.clear()
                ax = self.sweep_figure.add_subplot(111)
                
                ax.plot(param_values, results_k, 'o-', linewidth=2, markersize=6, label='k_eff')
                ax.set_xlabel(param, fontsize=12)
                ax.set_ylabel('Thermal Conductivity (W/m·K)', fontsize=12)
                ax.set_title(f'Parametric Sweep: {param}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                self.sweep_canvas.draw()
            
            # Store sweep data
            self.sweep_data = {
                'parameter': param,
                'values': param_values.tolist(),
                'k_eff': results_k,
                'enhancement': results_enh
            }
            
            # Auto-export if requested
            if self.sweep_export_check.isChecked():
                self._export_sweep_data(auto=True)
            
            self._log(f"Parametric sweep completed: {steps} points")
            QMessageBox.information(self, "Success", 
                                  f"Parametric sweep completed!\n{steps} points calculated.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Parametric sweep failed:\n{e}")
            self._log(f"ERROR in parametric sweep: {e}")
    
    def _export_sweep_data(self, auto=False):
        """Export parametric sweep data to CSV"""
        if not hasattr(self, 'sweep_data'):
            QMessageBox.warning(self, "No Data", "Run a parametric sweep first!")
            return
        
        if auto:
            filepath = f"sweep_{self.sweep_data['parameter'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Export Sweep Data", "", "CSV Files (*.csv)"
            )
        
        if filepath:
            try:
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.sweep_data['parameter'], 'k_eff (W/m·K)', 'Enhancement (%)'])
                    for val, k, enh in zip(self.sweep_data['values'], 
                                          self.sweep_data['k_eff'], 
                                          self.sweep_data['enhancement']):
                        writer.writerow([val, k, enh])
                
                self._log(f"Sweep data exported: {Path(filepath).name}")
                if not auto:
                    QMessageBox.information(self, "Success", f"Sweep data exported to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{e}")
                self._log(f"ERROR: {e}")
    
    def closeEvent(self, event):
        """Handle window close"""
        # Save window state
        self._save_window_state()
        
        # Cancel any running computation
        if self.computation_thread and self.computation_thread.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Computation is running. Cancel and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._cancel_computation()
                self.computation_thread.wait()
            else:
                event.ignore()
                return
        
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("BKPS NFL Thermal Pro")
    app.setOrganizationName("BKPS")
    app.setOrganizationDomain("bkps.nfl")
    
    window = BKPSProfessionalGUI_V7()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
