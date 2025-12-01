"""
Advanced Nanofluid Simulator GUI v3.0

Features:
- Popup windows for detailed configurations
- Multiple particle shapes (sphere, rod, cube, platelet)
- Temperature range input for calculations
- Scientific-grade graphs
- Nanoparticle observer (aggregation/distribution)
- Surface interaction analysis
- Refresh/reset functionality
- Save/export results to multiple formats
"""

import sys
import os
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QGroupBox,
    QFormLayout, QMessageBox, QSplitter, QTextEdit, QCheckBox,
    QSpinBox, QFileDialog, QApplication, QDialog, QDialogButtonBox,
    QScrollArea, QFrame, QGridLayout, QLineEdit, QProgressBar,
    QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QAction

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ..enhanced_simulator import EnhancedNanofluidSimulator
from ..flow_simulator import FlowNanofluidSimulator
from ..nanoparticles import NanoparticleDatabase
from ..visualization import FlowVisualizer, NanofluidPropertyVisualizer
from ..ai_recommender import AIRecommendationEngine, ApplicationType, OptimizationObjective
from nanofluid_simulator.solver_modes import SolverMode, SolverModeManager


class AdvancedConfigDialog(QDialog):
    """Popup dialog for advanced configuration."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Configuration")
        self.setMinimumWidth(500)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Particle Shape Section
        shape_group = QGroupBox("Particle Shape Configuration")
        shape_layout = QFormLayout()
        
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Sphere", "Rod/Cylinder", "Cube", "Platelet", "Irregular"])
        shape_layout.addRow("Shape:", self.shape_combo)
        
        self.aspect_ratio_spin = QDoubleSpinBox()
        self.aspect_ratio_spin.setRange(0.1, 100)
        self.aspect_ratio_spin.setValue(1.0)
        self.aspect_ratio_spin.setDecimals(2)
        shape_layout.addRow("Aspect Ratio:", self.aspect_ratio_spin)
        
        self.sphericity_spin = QDoubleSpinBox()
        self.sphericity_spin.setRange(0.1, 1.0)
        self.sphericity_spin.setValue(1.0)
        self.sphericity_spin.setDecimals(3)
        shape_layout.addRow("Sphericity:", self.sphericity_spin)
        
        shape_group.setLayout(shape_layout)
        layout.addWidget(shape_group)
        
        # Surface Interaction Section
        surface_group = QGroupBox("Surface Interaction Properties")
        surface_layout = QFormLayout()
        
        self.interfacial_layer_spin = QDoubleSpinBox()
        self.interfacial_layer_spin.setRange(0, 10)
        self.interfacial_layer_spin.setValue(2.0)
        self.interfacial_layer_spin.setSuffix(" nm")
        surface_layout.addRow("Interfacial Layer:", self.interfacial_layer_spin)
        
        self.surface_energy_spin = QDoubleSpinBox()
        self.surface_energy_spin.setRange(0, 1000)
        self.surface_energy_spin.setValue(50)
        self.surface_energy_spin.setSuffix(" mJ/m²")
        surface_layout.addRow("Surface Energy:", self.surface_energy_spin)
        
        self.enable_aggregation = QCheckBox("Enable Aggregation Model")
        self.enable_aggregation.setChecked(False)
        surface_layout.addRow("Aggregation:", self.enable_aggregation)
        
        surface_group.setLayout(surface_layout)
        layout.addWidget(surface_group)
        
        # Temperature Range Section
        temp_group = QGroupBox("Temperature Range for Analysis")
        temp_layout = QFormLayout()
        
        self.temp_start_spin = QDoubleSpinBox()
        self.temp_start_spin.setRange(273, 500)
        self.temp_start_spin.setValue(280)
        self.temp_start_spin.setSuffix(" K")
        temp_layout.addRow("Start Temperature:", self.temp_start_spin)
        
        self.temp_end_spin = QDoubleSpinBox()
        self.temp_end_spin.setRange(273, 500)
        self.temp_end_spin.setValue(370)
        self.temp_end_spin.setSuffix(" K")
        temp_layout.addRow("End Temperature:", self.temp_end_spin)
        
        self.temp_steps_spin = QSpinBox()
        self.temp_steps_spin.setRange(5, 100)
        self.temp_steps_spin.setValue(20)
        temp_layout.addRow("Number of Points:", self.temp_steps_spin)
        
        temp_group.setLayout(temp_layout)
        layout.addWidget(temp_group)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration from the dialog."""
        return {
            'shape': self.shape_combo.currentText(),
            'aspect_ratio': self.aspect_ratio_spin.value(),
            'sphericity': self.sphericity_spin.value(),
            'interfacial_layer': self.interfacial_layer_spin.value(),
            'surface_energy': self.surface_energy_spin.value(),
            'enable_aggregation': self.enable_aggregation.isChecked(),
            'temp_range': (
                self.temp_start_spin.value(),
                self.temp_end_spin.value(),
                self.temp_steps_spin.value()
            )
        }


class NanoparticleObserverDialog(QDialog):
    """Dialog for observing nanoparticle behavior and aggregation."""
    
    def __init__(self, particle_data: Dict, config: Dict, parent=None):
        super().__init__(parent)
        self.particle_data = particle_data
        self.config = config
        self.setWindowTitle("Nanoparticle Observer")
        self.setMinimumSize(800, 600)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Info section
        info_group = QGroupBox("Particle Information")
        info_layout = QFormLayout()
        
        info_layout.addRow("Particle:", QLabel(self.particle_data.get('name', 'Unknown')))
        info_layout.addRow("Shape:", QLabel(self.config.get('shape', 'Sphere')))
        info_layout.addRow("Size:", QLabel(f"{self.particle_data.get('size', 0)} nm"))
        info_layout.addRow("Aspect Ratio:", QLabel(f"{self.config.get('aspect_ratio', 1.0):.2f}"))
        info_layout.addRow("Sphericity:", QLabel(f"{self.config.get('sphericity', 1.0):.3f}"))
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Visualization
        fig = Figure(figsize=(8, 6))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Create particle distribution visualization
        self.plot_particle_distribution(fig)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
    
    def plot_particle_distribution(self, fig):
        """Plot particle size distribution and aggregation."""
        # Create 2x2 subplot
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        # Size distribution
        mean_size = self.particle_data.get('size', 40)
        sizes = np.random.lognormal(np.log(mean_size), 0.3, 1000)
        ax1.hist(sizes, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Particle Size (nm)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('Size Distribution', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Aggregation state
        if self.config.get('enable_aggregation', False):
            aggregate_sizes = np.array([1, 2, 3, 4, 5, 6])
            probabilities = np.exp(-aggregate_sizes / 2)
            probabilities /= probabilities.sum()
            
            ax2.bar(aggregate_sizes, probabilities, color='coral', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Aggregate Size (particles)', fontsize=10)
            ax2.set_ylabel('Probability', fontsize=10)
            ax2.set_title('Aggregation Distribution', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, 'Aggregation Disabled', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Aggregation Distribution', fontsize=11, fontweight='bold')
        
        # Surface interaction visualization
        r = np.linspace(0, 5, 100)
        interfacial_layer = self.config.get('interfacial_layer', 2)
        potential = -1 / (r + 0.1) * np.exp(-(r - 1)**2)
        
        ax3.plot(r, potential, 'b-', linewidth=2)
        ax3.axvline(interfacial_layer, color='r', linestyle='--', label=f'Interfacial Layer: {interfacial_layer:.1f} nm')
        ax3.set_xlabel('Distance from Surface (nm)', fontsize=10)
        ax3.set_ylabel('Interaction Potential (a.u.)', fontsize=10)
        ax3.set_title('Surface Interaction Profile', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Shape visualization (schematic)
        shape = self.config.get('shape', 'Sphere')
        aspect_ratio = self.config.get('aspect_ratio', 1.0)
        
        if shape == 'Sphere':
            circle = plt.Circle((0.5, 0.5), 0.3, color='gold', alpha=0.6, edgecolor='black', linewidth=2)
            ax4.add_patch(circle)
        elif shape == 'Rod/Cylinder':
            width = 0.2
            height = 0.2 * aspect_ratio
            rect = plt.Rectangle((0.5 - width/2, 0.5 - height/2), width, height, 
                                color='gold', alpha=0.6, edgecolor='black', linewidth=2)
            ax4.add_patch(rect)
        elif shape == 'Cube':
            rect = plt.Rectangle((0.3, 0.3), 0.4, 0.4, 
                                color='gold', alpha=0.6, edgecolor='black', linewidth=2)
            ax4.add_patch(rect)
        elif shape == 'Platelet':
            rect = plt.Rectangle((0.2, 0.45), 0.6, 0.1, 
                                color='gold', alpha=0.6, edgecolor='black', linewidth=2)
            ax4.add_patch(rect)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_aspect('equal')
        ax4.axis('off')
        ax4.set_title(f'Particle Shape: {shape}', fontsize=11, fontweight='bold')
        
        fig.tight_layout()


class AdvancedNanofluidGUI(QMainWindow):
    """Advanced GUI with popup configurations and enhanced features."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nanofluid Thermal Analyzer v3.0 - Professional Edition")
        self.setGeometry(100, 100, 1500, 950)
        
        # Try to set icon
        icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'app_icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Initialize simulators
        self.simulator = EnhancedNanofluidSimulator()
        self.flow_simulator = FlowNanofluidSimulator()
        self.solver_manager = SolverModeManager(self.flow_simulator)
        self.ai_engine = AIRecommendationEngine()
        self.np_db = NanoparticleDatabase()
        
        # Current state
        self.current_flow_data = None
        self.current_nanoparticle = "Cu"
        self.current_base_fluid = "Water"
        self.advanced_config = {
            'shape': 'Sphere',
            'aspect_ratio': 1.0,
            'sphericity': 1.0,
            'interfacial_layer': 2.0,
            'surface_energy': 50.0,
            'enable_aggregation': False,
            'temp_range': (280, 370, 20)
        }
        self.calculation_results = {}
        
        # Setup UI
        self.init_ui()
        self.apply_stylesheet()
        self.setup_menubar()
        
        # Initial calculation
        QTimer.singleShot(500, self.update_all_visualizations)
    
    def setup_menubar(self):
        """Setup menu bar with file operations."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        save_action = QAction('Save Results...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        export_action = QAction('Export All Data...', self)
        export_action.triggered.connect(self.export_all_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        advanced_action = QAction('Advanced Configuration...', self)
        advanced_action.triggered.connect(self.show_advanced_config)
        tools_menu.addAction(advanced_action)
        
        observer_action = QAction('Nanoparticle Observer...', self)
        observer_action.triggered.connect(self.show_particle_observer)
        tools_menu.addAction(observer_action)
        
        reset_action = QAction('Reset to Defaults', self)
        reset_action.setShortcut('Ctrl+R')
        reset_action.triggered.connect(self.reset_to_defaults)
        tools_menu.addAction(reset_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left side: Controls (30%)
        control_panel = self.create_control_panel()
        
        # Right side: Visualizations (70%)
        viz_panel = self.create_visualization_panel()
        
        # Split layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(viz_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        
        main_layout.addWidget(splitter)
        
        # Status bar with progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(15)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.statusBar().showMessage("Ready - Configure parameters and click Calculate")
    
    def create_control_panel(self) -> QWidget:
        """Create the left control panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title with version
        title = QLabel("Simulation Controls v3.0")
        title.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Scroll area for controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Material Selection Group
        scroll_layout.addWidget(self.create_material_group())
        
        # Flow Parameters Group
        scroll_layout.addWidget(self.create_flow_parameters_group())
        
        # Temperature Range Group
        scroll_layout.addWidget(self.create_temperature_range_group())
        
        # Action Buttons
        scroll_layout.addWidget(self.create_action_buttons())
        
        # Results Display
        scroll_layout.addWidget(self.create_results_display())
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return widget
    
    def create_material_group(self) -> QGroupBox:
        """Create material selection group."""
        group = QGroupBox("Materials & Shape")
        layout = QFormLayout()
        
        # Nanoparticle selection
        self.particle_combo = QComboBox()
        particles = list(self.np_db.database.keys())
        self.particle_combo.addItems(particles)
        self.particle_combo.setCurrentText("Cu")
        self.particle_combo.currentTextChanged.connect(self.on_particle_changed)
        layout.addRow("Nanoparticle:", self.particle_combo)
        
        # Base fluid
        self.fluid_combo = QComboBox()
        self.fluid_combo.addItems(["Water", "Ethylene Glycol", "Engine Oil"])
        self.fluid_combo.currentTextChanged.connect(self.on_fluid_changed)
        layout.addRow("Base Fluid:", self.fluid_combo)
        
        # Volume fraction
        self.volume_fraction_spin = QDoubleSpinBox()
        self.volume_fraction_spin.setRange(0.0, 0.1)
        self.volume_fraction_spin.setValue(0.02)
        self.volume_fraction_spin.setSingleStep(0.005)
        self.volume_fraction_spin.setDecimals(3)
        self.volume_fraction_spin.setSuffix(" (2%)")
        self.volume_fraction_spin.valueChanged.connect(self.on_volume_fraction_changed)
        layout.addRow("Volume Fraction:", self.volume_fraction_spin)
        
        # Particle size
        self.particle_size_spin = QDoubleSpinBox()
        self.particle_size_spin.setRange(1, 200)
        self.particle_size_spin.setValue(40)
        self.particle_size_spin.setSuffix(" nm")
        layout.addRow("Particle Size:", self.particle_size_spin)
        
        # Shape (read-only, set via advanced config)
        self.shape_label = QLabel("Sphere")
        layout.addRow("Shape:", self.shape_label)
        
        group.setLayout(layout)
        return group
    
    def create_flow_parameters_group(self) -> QGroupBox:
        """Create flow parameters group."""
        group = QGroupBox("Flow Configuration")
        layout = QFormLayout()
        
        # Geometry with more options
        self.geometry_combo = QComboBox()
        self.geometry_combo.addItems(["channel", "pipe", "plate", "rod", "cube"])
        layout.addRow("Geometry:", self.geometry_combo)
        
        # Reynolds number
        self.reynolds_spin = QDoubleSpinBox()
        self.reynolds_spin.setRange(10, 10000)
        self.reynolds_spin.setValue(1000)
        self.reynolds_spin.setDecimals(0)
        layout.addRow("Reynolds Number:", self.reynolds_spin)
        
        # Grid resolution
        self.grid_resolution_spin = QSpinBox()
        self.grid_resolution_spin.setRange(20, 100)
        self.grid_resolution_spin.setValue(50)
        layout.addRow("Grid Resolution:", self.grid_resolution_spin)
        
        group.setLayout(layout)
        return group
    
    def create_temperature_range_group(self) -> QGroupBox:
        """Create temperature range input group."""
        group = QGroupBox("Temperature Range for Analysis")
        layout = QFormLayout()
        
        self.t_min_spin = QDoubleSpinBox()
        self.t_min_spin.setRange(273, 500)
        self.t_min_spin.setValue(280)
        self.t_min_spin.setSuffix(" K")
        layout.addRow("Min Temperature:", self.t_min_spin)
        
        self.t_max_spin = QDoubleSpinBox()
        self.t_max_spin.setRange(273, 500)
        self.t_max_spin.setValue(370)
        self.t_max_spin.setSuffix(" K")
        layout.addRow("Max Temperature:", self.t_max_spin)
        
        self.t_current_spin = QDoubleSpinBox()
        self.t_current_spin.setRange(273, 500)
        self.t_current_spin.setValue(300)
        self.t_current_spin.setSuffix(" K")
        layout.addRow("Current Point:", self.t_current_spin)
        
        group.setLayout(layout)
        return group
    
    def create_action_buttons(self) -> QWidget:
        """Create action buttons."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Calculate button
        self.calc_button = QPushButton("🔄 Calculate & Visualize")
        self.calc_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                font-size: 11pt;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.calc_button.clicked.connect(self.calculate_and_visualize)
        layout.addWidget(self.calc_button)
        
        # Refresh/Reset button
        refresh_btn = QPushButton("🔄 Refresh/Reset Results")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        refresh_btn.clicked.connect(self.refresh_results)
        layout.addWidget(refresh_btn)
        
        # Advanced Config button
        advanced_btn = QPushButton("⚙️ Advanced Configuration")
        advanced_btn.clicked.connect(self.show_advanced_config)
        layout.addWidget(advanced_btn)
        
        # Particle Observer button
        observer_btn = QPushButton("🔬 Nanoparticle Observer")
        observer_btn.clicked.connect(self.show_particle_observer)
        layout.addWidget(observer_btn)
        
        # AI Recommendation button
        self.ai_button = QPushButton("🤖 AI Recommendations")
        self.ai_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.ai_button.clicked.connect(self.show_ai_recommendations)
        layout.addWidget(self.ai_button)
        
        # Save button
        save_btn = QPushButton("💾 Save Results")
        save_btn.clicked.connect(self.save_results)
        layout.addWidget(save_btn)
        
        return widget
    
    def create_results_display(self) -> QGroupBox:
        """Create results display area."""
        group = QGroupBox("Calculated Properties")
        layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(250)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.results_text)
        
        group.setLayout(layout)
        return group
    
    def create_visualization_panel(self) -> QWidget:
        """Create the visualization panel with tabs."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # Tab 1: Thermal Contours
        self.thermal_tab = self.create_plot_tab("Thermal Contours")
        self.tab_widget.addTab(self.thermal_tab, "🌡️ Thermal Contours")
        
        # Tab 2: Velocity Field
        self.velocity_tab = self.create_plot_tab("Velocity Field")
        self.tab_widget.addTab(self.velocity_tab, "➡️ Velocity Field")
        
        # Tab 3: Streamlines
        self.streamline_tab = self.create_plot_tab("Streamlines")
        self.tab_widget.addTab(self.streamline_tab, "〰️ Streamlines")
        
        # Tab 4: Complete Analysis
        self.analysis_tab = self.create_plot_tab("Complete Analysis")
        self.tab_widget.addTab(self.analysis_tab, "📊 Full Analysis")
        
        # Tab 5: Temperature Range Analysis
        self.temp_range_tab = self.create_plot_tab("Temperature Range")
        self.tab_widget.addTab(self.temp_range_tab, "📈 Temp Range")
        
        # Tab 6: Surface Interactions
        self.surface_tab = self.create_plot_tab("Surface Interactions")
        self.tab_widget.addTab(self.surface_tab, "🔗 Surface Effects")
        
        layout.addWidget(self.tab_widget)
        return widget
    
    def create_plot_tab(self, title: str) -> QWidget:
        """Create a tab with matplotlib canvas."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create figure and canvas
        fig = Figure(figsize=(10, 8), dpi=100)
        fig.patch.set_facecolor('white')
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, widget)
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        # Store references
        setattr(self, f'{title.lower().replace(" ", "_")}_fig', fig)
        setattr(self, f'{title.lower().replace(" ", "_")}_canvas', canvas)
        
        return widget
    
    def calculate_and_visualize(self):
        """Main calculation and visualization function."""
        try:
            self.statusBar().showMessage("Calculating...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            self.calc_button.setEnabled(False)
            
            # Get parameters
            particle = self.particle_combo.currentText()
            base_fluid = self.fluid_combo.currentText()
            phi = self.volume_fraction_spin.value()
            temp = self.t_current_spin.value()
            particle_size = self.particle_size_spin.value()
            
            geometry = self.geometry_combo.currentText()
            reynolds = self.reynolds_spin.value()
            t_min = self.t_min_spin.value()
            t_max = self.t_max_spin.value()
            resolution = self.grid_resolution_spin.value()
            
            self.progress_bar.setValue(30)
            
            # Calculate nanofluid properties
            particle_data = self.np_db.get_nanoparticle(particle)
            base_props = self.get_base_fluid_properties(base_fluid, temp)
            
            k_bf = base_props['k']
            k_p = particle_data['thermal_conductivity']
            
            # Account for shape effects
            from ..models import MaxwellModel, HamiltonCrosserModel
            
            if self.advanced_config['shape'] == 'Sphere':
                model = MaxwellModel()
            else:
                sphericity = self.advanced_config['sphericity']
                model = HamiltonCrosserModel(sphericity=sphericity)
            
            k_nf = model.calculate_conductivity(k_bf, k_p, phi, temp)
            enhancement = ((k_nf - k_bf) / k_bf) * 100
            
            self.progress_bar.setValue(50)
            
            # Store results
            self.calculation_results = {
                'particle': particle,
                'base_fluid': base_fluid,
                'phi': phi,
                'temperature': temp,
                'particle_size': particle_size,
                'shape': self.advanced_config['shape'],
                'k_bf': k_bf,
                'k_p': k_p,
                'k_nf': k_nf,
                'enhancement': enhancement,
                'geometry': geometry,
                'reynolds': reynolds,
                'interfacial_layer': self.advanced_config['interfacial_layer'],
                'surface_energy': self.advanced_config['surface_energy'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Update results display
            results_text = f"""
╔═══ NANOFLUID PROPERTIES ═══╗

Material: {particle} ({self.advanced_config['shape']}) in {base_fluid}
Volume Fraction: {phi*100:.2f}%
Temperature: {temp} K
Particle Size: {particle_size} nm
Aspect Ratio: {self.advanced_config['aspect_ratio']:.2f}
Sphericity: {self.advanced_config['sphericity']:.3f}

THERMAL CONDUCTIVITY:
Base Fluid k: {k_bf:.4f} W/m·K
Nanoparticle k: {k_p:.1f} W/m·K
Nanofluid k: {k_nf:.4f} W/m·K
Enhancement: {enhancement:.2f}%

SURFACE INTERACTIONS:
Interfacial Layer: {self.advanced_config['interfacial_layer']:.1f} nm
Surface Energy: {self.advanced_config['surface_energy']:.1f} mJ/m²
Aggregation: {'Enabled' if self.advanced_config['enable_aggregation'] else 'Disabled'}

FLOW CONDITIONS:
Geometry: {geometry.title()}
Reynolds: {reynolds:.0f}
Temp Range: {t_min:.0f}-{t_max:.0f} K
╚═════════════════════════════╝
            """
            self.results_text.setText(results_text)
            
            self.progress_bar.setValue(70)
            
            # Create flow field
            self.current_flow_data = FlowVisualizer.create_flow_field(
                geometry=geometry,
                nx=resolution,
                ny=resolution,
                reynolds=reynolds,
                temperature_range=(t_min, t_max)
            )
            
            # Update all visualizations
            self.update_all_visualizations()
            
            self.progress_bar.setValue(100)
            self.statusBar().showMessage("✓ Calculation complete!", 3000)
            
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Calculation failed:\n{str(e)}")
            self.statusBar().showMessage("✗ Calculation failed", 3000)
            self.progress_bar.setVisible(False)
        finally:
            self.calc_button.setEnabled(True)
    
    def update_all_visualizations(self):
        """Update all visualization tabs."""
        if self.current_flow_data is None:
            # Create default flow data
            self.current_flow_data = FlowVisualizer.create_flow_field(
                geometry="channel",
                nx=50,
                ny=50,
                reynolds=1000,
                temperature_range=(280, 370)
            )
        
        # Update each tab
        self.update_thermal_plot()
        self.update_velocity_plot()
        self.update_streamline_plot()
        self.update_analysis_plot()
        self.update_temp_range_plot()
        self.update_surface_interaction_plot()
    
    def update_thermal_plot(self):
        """Update thermal contours plot with scientific styling."""
        fig = self.thermal_contours_fig
        fig.clear()
        ax = fig.add_subplot(111)
        
        FlowVisualizer.plot_thermal_contours(self.current_flow_data, fig, ax)
        
        # Scientific formatting
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
        ax.tick_params(labelsize=9)
        
        self.thermal_contours_canvas.draw()
    
    def update_velocity_plot(self):
        """Update velocity field plot."""
        fig = self.velocity_field_fig
        fig.clear()
        ax = fig.add_subplot(111)
        
        FlowVisualizer.plot_velocity_field(
            self.current_flow_data, fig, ax, vector_density=12
        )
        
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
        ax.tick_params(labelsize=9)
        
        self.velocity_field_canvas.draw()
    
    def update_streamline_plot(self):
        """Update streamlines plot."""
        fig = self.streamlines_fig
        fig.clear()
        ax = fig.add_subplot(111)
        
        FlowVisualizer.plot_streamlines(
            self.current_flow_data, fig, ax, density=1.5
        )
        
        ax.tick_params(labelsize=9)
        
        self.streamlines_canvas.draw()
    
    def update_analysis_plot(self):
        """Update complete analysis plot."""
        fig = self.complete_analysis_fig
        fig.clear()
        
        FlowVisualizer.plot_combined_analysis(self.current_flow_data, fig)
        
        self.complete_analysis_canvas.draw()
    
    def update_temp_range_plot(self):
        """Update temperature range analysis plot."""
        fig = self.temperature_range_fig
        fig.clear()
        ax = fig.add_subplot(111)
        
        # Get temperature range
        t_min = self.t_min_spin.value()
        t_max = self.t_max_spin.value()
        temps = np.linspace(t_min, t_max, 50)
        
        # Get current parameters
        particle = self.particle_combo.currentText()
        particle_data = self.np_db.get_nanoparticle(particle)
        base_fluid = self.fluid_combo.currentText()
        phi = self.volume_fraction_spin.value()
        
        # Calculate k_nf for temperature range
        from ..models import MaxwellModel, HamiltonCrosserModel, YuChoiModel
        
        models = {
            'Maxwell': MaxwellModel(),
            'Hamilton-Crosser': HamiltonCrosserModel(sphericity=self.advanced_config['sphericity']),
            'Yu-Choi': YuChoiModel()
        }
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for idx, (name, model) in enumerate(models.items()):
            k_values = []
            for T in temps:
                base_props = self.get_base_fluid_properties(base_fluid, T)
                k_bf = base_props['k']
                k_p = particle_data['thermal_conductivity']
                k_nf = model.calculate_conductivity(k_bf, k_p, phi, T)
                k_values.append(k_nf)
            
            ax.plot(temps, k_values, marker='o', linewidth=2, 
                   label=name, color=colors[idx], markersize=4)
        
        ax.set_xlabel('Temperature (K)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Thermal Conductivity (W/m·K)', fontsize=11, fontweight='bold')
        ax.set_title(f'Thermal Conductivity vs Temperature\n{particle} @ {phi*100:.1f}% in {base_fluid}', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=9)
        
        fig.tight_layout()
        self.temperature_range_canvas.draw()
    
    def update_surface_interaction_plot(self):
        """Update surface interactions plot."""
        fig = self.surface_interactions_fig
        fig.clear()
        
        # Create 2x2 subplot
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        # Interfacial layer effect
        r = np.linspace(0, 10, 100)
        layer_thickness = self.advanced_config['interfacial_layer']
        k_ratio = 1 + 0.5 * np.exp(-(r - layer_thickness)**2 / (2 * layer_thickness**2))
        
        ax1.plot(r, k_ratio, 'b-', linewidth=2)
        ax1.axvline(layer_thickness, color='r', linestyle='--', 
                   label=f'Layer: {layer_thickness:.1f} nm')
        ax1.set_xlabel('Distance from surface (nm)', fontsize=10)
        ax1.set_ylabel('k/k_bulk', fontsize=10)
        ax1.set_title('Interfacial Layer Effect', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Surface energy distribution
        if self.advanced_config['enable_aggregation']:
            sizes = np.array([1, 2, 3, 4, 5])
            energies = self.advanced_config['surface_energy'] / sizes
            
            ax2.bar(sizes, energies, color='coral', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Aggregate size', fontsize=10)
            ax2.set_ylabel('Effective surface energy (mJ/m²)', fontsize=10)
            ax2.set_title('Aggregation Energy', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, 'Enable aggregation\nin Advanced Config', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=10)
            ax2.set_title('Aggregation Energy', fontsize=11, fontweight='bold')
        
        # Brownian motion
        phi = self.volume_fraction_spin.value()
        particle_size = self.particle_size_spin.value() * 1e-9  # Convert to meters
        T = self.t_current_spin.value()
        k_B = 1.380649e-23  # Boltzmann constant
        
        # Calculate Brownian diffusivity
        mu = self.get_base_fluid_properties(self.fluid_combo.currentText(), T)['mu']
        D_B = k_B * T / (3 * np.pi * mu * particle_size)
        
        # Time scale
        t = np.linspace(0, 10, 100)
        msd = 6 * D_B * t  # Mean square displacement
        
        ax3.plot(t, msd * 1e18, 'g-', linewidth=2)
        ax3.set_xlabel('Time (s)', fontsize=10)
        ax3.set_ylabel('MSD (nm²)', fontsize=10)
        ax3.set_title(f'Brownian Motion\nD_B = {D_B:.2e} m²/s', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Particle concentration profile
        y = np.linspace(0, 1, 100)
        # Assume slight sedimentation for larger particles
        settling = np.exp(-5 * (particle_size * 1e9 - 40) * (1 - y))
        concentration = phi * settling / np.mean(settling)
        
        ax4.plot(concentration * 100, y, 'purple', linewidth=2)
        ax4.axvline(phi * 100, color='r', linestyle='--', label='Uniform')
        ax4.set_xlabel('Volume fraction (%)', fontsize=10)
        ax4.set_ylabel('Height (normalized)', fontsize=10)
        ax4.set_title('Concentration Profile', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        self.surface_interactions_canvas.draw()
    
    def show_advanced_config(self):
        """Show advanced configuration dialog."""
        dialog = AdvancedConfigDialog(self)
        
        # Pre-populate with current values
        dialog.shape_combo.setCurrentText(self.advanced_config['shape'])
        dialog.aspect_ratio_spin.setValue(self.advanced_config['aspect_ratio'])
        dialog.sphericity_spin.setValue(self.advanced_config['sphericity'])
        dialog.interfacial_layer_spin.setValue(self.advanced_config['interfacial_layer'])
        dialog.surface_energy_spin.setValue(self.advanced_config['surface_energy'])
        dialog.enable_aggregation.setChecked(self.advanced_config['enable_aggregation'])
        
        t_start, t_end, t_steps = self.advanced_config['temp_range']
        dialog.temp_start_spin.setValue(t_start)
        dialog.temp_end_spin.setValue(t_end)
        dialog.temp_steps_spin.setValue(t_steps)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.advanced_config = dialog.get_config()
            self.shape_label.setText(self.advanced_config['shape'])
            
            # Update temperature range spinboxes
            t_start, t_end, _ = self.advanced_config['temp_range']
            self.t_min_spin.setValue(t_start)
            self.t_max_spin.setValue(t_end)
            
            self.statusBar().showMessage("Advanced configuration updated", 2000)
    
    def show_particle_observer(self):
        """Show nanoparticle observer dialog."""
        particle = self.particle_combo.currentText()
        particle_data = self.np_db.get_nanoparticle(particle)
        particle_data['name'] = particle
        particle_data['size'] = self.particle_size_spin.value()
        
        dialog = NanoparticleObserverDialog(particle_data, self.advanced_config, self)
        dialog.exec()
    
    def show_ai_recommendations(self):
        """Show AI recommendations dialog."""
        try:
            particle = self.particle_combo.currentText()
            phi = self.volume_fraction_spin.value()
            temp = self.t_current_spin.value()
            
            recommendations = self.ai_engine.get_recommendations(
                application=ApplicationType.HEAT_EXCHANGER,
                objective=OptimizationObjective.MAXIMIZE_HEAT_TRANSFER,
                temperature_range=(self.t_min_spin.value(), self.t_max_spin.value()),
                base_fluid=self.fluid_combo.currentText()
            )
            
            rec_text = "🤖 AI RECOMMENDATIONS\n\n"
            rec_text += f"Top Recommendation:\n"
            rec_text += f"  Particle: {recommendations['top_recommendation']['particle']}\n"
            rec_text += f"  Volume Fraction: {recommendations['top_recommendation']['volume_fraction']*100:.2f}%\n"
            rec_text += f"  Confidence: {recommendations['top_recommendation']['confidence_score']*100:.1f}%\n\n"
            rec_text += f"Predicted Enhancement: {recommendations['top_recommendation']['predicted_enhancement']*100:.1f}%\n"
            rec_text += f"Reasoning: {recommendations['reasoning']}\n"
            
            QMessageBox.information(self, "AI Recommendations", rec_text)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"AI recommendation failed:\n{str(e)}")
    
    def refresh_results(self):
        """Refresh/reset all results."""
        reply = QMessageBox.question(
            self, 'Refresh Results',
            'This will clear all current results. Continue?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.results_text.clear()
            self.calculation_results = {}
            
            # Clear all plots
            for tab_name in ['thermal_contours', 'velocity_field', 'streamlines', 
                           'complete_analysis', 'temperature_range', 'surface_interactions']:
                fig = getattr(self, f'{tab_name}_fig')
                fig.clear()
                canvas = getattr(self, f'{tab_name}_canvas')
                canvas.draw()
            
            self.statusBar().showMessage("Results cleared", 2000)
    
    def reset_to_defaults(self):
        """Reset all parameters to defaults."""
        self.particle_combo.setCurrentText("Cu")
        self.fluid_combo.setCurrentText("Water")
        self.volume_fraction_spin.setValue(0.02)
        self.particle_size_spin.setValue(40)
        self.t_current_spin.setValue(300)
        self.t_min_spin.setValue(280)
        self.t_max_spin.setValue(370)
        self.geometry_combo.setCurrentText("channel")
        self.reynolds_spin.setValue(1000)
        self.grid_resolution_spin.setValue(50)
        
        self.advanced_config = {
            'shape': 'Sphere',
            'aspect_ratio': 1.0,
            'sphericity': 1.0,
            'interfacial_layer': 2.0,
            'surface_energy': 50.0,
            'enable_aggregation': False,
            'temp_range': (280, 370, 20)
        }
        self.shape_label.setText("Sphere")
        
        self.statusBar().showMessage("Reset to defaults", 2000)
    
    def save_results(self):
        """Save results to file."""
        if not self.calculation_results:
            QMessageBox.warning(self, "Warning", "No results to save. Run a calculation first.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            f"nanofluid_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(self.calculation_results, f, indent=2)
                else:
                    with open(filename, 'w') as f:
                        f.write(self.results_text.toPlainText())
                        f.write("\n\n" + "="*50 + "\n")
                        f.write("DETAILED DATA:\n")
                        f.write(json.dumps(self.calculation_results, indent=2))
                
                QMessageBox.information(self, "Success", f"Results saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed:\n{str(e)}")
    
    def export_all_data(self):
        """Export all data including plots."""
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        
        if folder:
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Save text results
                with open(os.path.join(folder, f'results_{timestamp}.txt'), 'w') as f:
                    f.write(self.results_text.toPlainText())
                
                # Save JSON data
                with open(os.path.join(folder, f'data_{timestamp}.json'), 'w') as f:
                    json.dump(self.calculation_results, f, indent=2)
                
                # Save all plots
                for tab_name in ['thermal_contours', 'velocity_field', 'streamlines', 
                               'complete_analysis', 'temperature_range', 'surface_interactions']:
                    fig = getattr(self, f'{tab_name}_fig')
                    fig.savefig(os.path.join(folder, f'{tab_name}_{timestamp}.png'), 
                               dpi=300, bbox_inches='tight')
                
                QMessageBox.information(self, "Success", 
                                      f"All data exported to:\n{folder}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
        <h2>Nanofluid Thermal Analyzer v3.0</h2>
        <p><b>Professional Edition</b></p>
        <p>Advanced nanofluid thermal conductivity simulation and analysis tool.</p>
        <p><b>Features:</b></p>
        <ul>
        <li>Flow visualization with thermal contours</li>
        <li>Multiple particle shapes and geometries</li>
        <li>Temperature range analysis</li>
        <li>Surface interaction modeling</li>
        <li>Nanoparticle observer</li>
        <li>AI-powered recommendations</li>
        <li>Scientific-grade visualizations</li>
        </ul>
        <p>© 2025 - Built with PyQt6, Matplotlib, NumPy, SciPy</p>
        """
        QMessageBox.about(self, "About", about_text)
    
    def get_base_fluid_properties(self, fluid_name: str, temperature: float) -> Dict[str, float]:
        """Get base fluid properties."""
        props = {
            "Water": {'k': 0.613, 'rho': 997, 'cp': 4179, 'mu': 0.001},
            "Ethylene Glycol": {'k': 0.252, 'rho': 1114, 'cp': 2415, 'mu': 0.016},
            "Engine Oil": {'k': 0.145, 'rho': 888, 'cp': 1880, 'mu': 0.050}
        }
        return props.get(fluid_name, props["Water"])
    
    def on_particle_changed(self, particle: str):
        """Handle particle selection change."""
        self.current_nanoparticle = particle
    
    def on_fluid_changed(self, fluid: str):
        """Handle fluid selection change."""
        self.current_base_fluid = fluid
    
    def on_volume_fraction_changed(self, value: float):
        """Update volume fraction display."""
        self.volume_fraction_spin.setSuffix(f" ({value*100:.1f}%)")
    
    def apply_stylesheet(self):
        """Apply modern stylesheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                padding: 6px;
                border-radius: 3px;
                border: 1px solid #999;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QComboBox, QDoubleSpinBox, QSpinBox {
                padding: 4px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QMenuBar {
                background-color: #ffffff;
                border-bottom: 1px solid #ccc;
            }
            QMenuBar::item:selected {
                background-color: #2196F3;
                color: white;
            }
        """)


def main():
    """Run the application."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = AdvancedNanofluidGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
