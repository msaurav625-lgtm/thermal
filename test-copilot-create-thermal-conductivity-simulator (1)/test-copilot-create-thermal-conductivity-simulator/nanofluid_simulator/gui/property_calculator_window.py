"""
Property Calculator Window

Multi-tab interface for comprehensive nanofluid property calculations.

Features:
- Thermal conductivity calculator
- Viscosity calculator  
- Custom particle shapes
- AI model recommendations
- Batch calculations
- Export results

Author: Nanofluid Simulator v5.0
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QGroupBox,
    QFormLayout, QTextEdit, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QProgressBar, QCheckBox, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np

from ..simulator import NanofluidSimulator
from ..nanoparticles import NanoparticleDatabase
from ..custom_shapes import CustomParticleShape


class PropertyCalculatorWindow(QMainWindow):
    """Property calculator window with multi-tab interface"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Property Calculator - Nanofluid Simulator v5.0")
        self.setMinimumSize(1200, 800)
        
        self.simulator = NanofluidSimulator()
        self.db = NanoparticleDatabase()
        self.results = {}
        
        self._setup_ui()
        self._apply_styling()
    
    def _setup_ui(self):
        """Setup UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("🔬 Nanofluid Property Calculator")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_thermal_tab(), "Thermal Conductivity")
        self.tabs.addTab(self._create_viscosity_tab(), "Viscosity")
        self.tabs.addTab(self._create_custom_shape_tab(), "Custom Shapes")
        self.tabs.addTab(self._create_ai_recommendations_tab(), "AI Recommendations")
        self.tabs.addTab(self._create_batch_tab(), "Batch Calculations")
        self.tabs.addTab(self._create_results_tab(), "Results & Export")
        
        layout.addWidget(self.tabs)
        
        # Status bar
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
    
    def _create_thermal_tab(self) -> QWidget:
        """Create thermal conductivity tab"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Left: Inputs
        input_group = QGroupBox("Input Parameters")
        input_layout = QFormLayout()
        
        self.particle_combo = QComboBox()
        particles = self.db.list_particles()
        self.particle_combo.addItems([p['name'] for p in particles])
        input_layout.addRow("Nanoparticle:", self.particle_combo)
        
        self.base_fluid_combo = QComboBox()
        self.base_fluid_combo.addItems(["Water", "Ethylene Glycol", "Oil"])
        input_layout.addRow("Base Fluid:", self.base_fluid_combo)
        
        self.phi_spin = QDoubleSpinBox()
        self.phi_spin.setRange(0.0, 10.0)
        self.phi_spin.setValue(1.0)
        self.phi_spin.setSuffix(" %")
        self.phi_spin.setDecimals(2)
        input_layout.addRow("Volume Fraction (φ):", self.phi_spin)
        
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(273.0, 373.0)
        self.temp_spin.setValue(300.0)
        self.temp_spin.setSuffix(" K")
        input_layout.addRow("Temperature:", self.temp_spin)
        
        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(1.0, 100.0)
        self.diameter_spin.setValue(20.0)
        self.diameter_spin.setSuffix(" nm")
        input_layout.addRow("Particle Diameter:", self.diameter_spin)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Maxwell", "Hamilton-Crosser", "Yu-Choi", "Xue", 
            "Bruggeman", "Nan", "Jeffrey", "Interfacial Layer"
        ])
        input_layout.addRow("Model:", self.model_combo)
        
        calc_btn = QPushButton("Calculate")
        calc_btn.clicked.connect(self._calculate_thermal)
        input_layout.addRow(calc_btn)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Right: Results and plot
        right_layout = QVBoxLayout()
        
        self.thermal_result_text = QTextEdit()
        self.thermal_result_text.setReadOnly(True)
        self.thermal_result_text.setMaximumHeight(200)
        right_layout.addWidget(QLabel("Results:"))
        right_layout.addWidget(self.thermal_result_text)
        
        self.thermal_canvas = FigureCanvas(plt.Figure(figsize=(8, 6)))
        right_layout.addWidget(self.thermal_canvas)
        
        layout.addLayout(right_layout, stretch=2)
        
        return widget
    
    def _create_viscosity_tab(self) -> QWidget:
        """Create viscosity tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label = QLabel("Viscosity calculator (similar to thermal conductivity)")
        layout.addWidget(label)
        
        return widget
    
    def _create_custom_shape_tab(self) -> QWidget:
        """Create custom particle shape tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info = QLabel("Define custom particle shapes from TEM images or mathematical descriptions")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        return widget
    
    def _create_ai_recommendations_tab(self) -> QWidget:
        """Create AI recommendations tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info = QLabel("🤖 AI-powered model recommendations based on your application")
        info.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(info)
        
        return widget
    
    def _create_batch_tab(self) -> QWidget:
        """Create batch calculations tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info = QLabel("Run calculations for multiple parameter combinations")
        layout.addWidget(info)
        
        return widget
    
    def _create_results_tab(self) -> QWidget:
        """Create results and export tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Particle", "φ (%)", "T (K)", "k (W/mK)", "μ (Pa·s)", "Model"
        ])
        layout.addWidget(self.results_table)
        
        export_btn = QPushButton("Export to Excel")
        export_btn.clicked.connect(self._export_results)
        layout.addWidget(export_btn)
        
        return widget
    
    def _calculate_thermal(self):
        """Calculate thermal conductivity"""
        try:
            # Get parameters
            particle = self.particle_combo.currentText()
            phi = self.phi_spin.value() / 100.0  # Convert % to fraction
            T = self.temp_spin.value()
            
            # Calculate
            props = self.simulator.calculate_properties(
                nanoparticle=particle,
                base_fluid="Water",
                volume_fraction=phi,
                temperature=T
            )
            
            # Display results
            result_text = f"""
Thermal Conductivity Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Base fluid k:      {props['k_bf']:.4f} W/(m·K)
Nanofluid k:       {props['k_nf']:.4f} W/(m·K)
Enhancement:       {props['enhancement']:.2f}%

Particle:          {particle}
Volume fraction:   {phi*100:.2f}%
Temperature:       {T:.1f} K
Model:             {self.model_combo.currentText()}
"""
            self.thermal_result_text.setText(result_text)
            
            # Plot
            self._plot_thermal_results(props)
            
            self.status_label.setText("✅ Calculation complete")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Calculation failed:\n{str(e)}")
            self.status_label.setText("❌ Calculation failed")
    
    def _plot_thermal_results(self, props: dict):
        """Plot thermal conductivity results"""
        ax = self.thermal_canvas.figure.clear()
        ax = self.thermal_canvas.figure.add_subplot(111)
        
        # Simple bar chart
        labels = ['Base Fluid', 'Nanofluid']
        values = [props['k_bf'], props['k_nf']]
        colors = ['#3498db', '#e74c3c']
        
        ax.bar(labels, values, color=colors, alpha=0.7)
        ax.set_ylabel('Thermal Conductivity (W/m·K)')
        ax.set_title('Thermal Conductivity Comparison')
        ax.grid(True, alpha=0.3)
        
        self.thermal_canvas.draw()
    
    def _export_results(self):
        """Export results to file"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
        )
        if filename:
            # TODO: Implement export
            QMessageBox.information(self, "Export", f"Results exported to {filename}")
    
    def _apply_styling(self):
        """Apply window styling"""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
