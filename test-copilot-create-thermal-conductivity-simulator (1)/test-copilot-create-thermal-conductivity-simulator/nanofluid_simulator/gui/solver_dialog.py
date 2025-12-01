"""
Solver Selection Dialog for Nanofluid Simulator

This module provides the startup solver mode selection dialog.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QGroupBox, QTextEdit, QWidget,
    QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon

from nanofluid_simulator.solver_modes import SolverMode, SolverModeConfig


class SolverSelectionDialog(QDialog):
    """
    Dialog for selecting solver mode at startup.
    """
    
    mode_selected = pyqtSignal(SolverMode)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_mode = SolverMode.STATIC
        self.init_ui()
    
    def init_ui(self):
        """Initialize the dialog UI."""
        self.setWindowTitle("Nanofluid Simulator - Solver Mode Selection")
        self.setModal(True)
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("🌊 Welcome to Nanofluid Simulator v2.1.0")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("World's Most Advanced Flow-Integrated Nanofluid Modeling Software")
        subtitle_font = QFont()
        subtitle_font.setPointSize(11)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #666;")
        layout.addWidget(subtitle)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)
        
        # Instruction
        instruction = QLabel("Please select your solver mode:")
        instruction_font = QFont()
        instruction_font.setPointSize(12)
        instruction_font.setBold(True)
        instruction.setFont(instruction_font)
        layout.addWidget(instruction)
        
        # Mode selection buttons
        self.button_group = QButtonGroup(self)
        
        # Static mode option
        static_widget = self.create_mode_option(SolverMode.STATIC)
        layout.addWidget(static_widget)
        
        # Flow mode option
        flow_widget = self.create_mode_option(SolverMode.FLOW)
        layout.addWidget(flow_widget)
        
        layout.addStretch()
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        select_button = QPushButton("Start Simulation")
        select_button.setDefault(True)
        select_button.clicked.connect(self.accept_selection)
        select_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 12pt;
                font-weight: bold;
                padding: 10px 30px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        button_layout.addWidget(select_button)
        
        layout.addLayout(button_layout)
        
        # Set default selection
        self.button_group.button(0).setChecked(True)
    
    def create_mode_option(self, mode: SolverMode) -> QWidget:
        """Create a mode selection option widget."""
        info = SolverModeConfig.get_mode_description(mode)
        
        group_box = QGroupBox()
        group_box.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ddd;
                border-radius: 10px;
                margin-top: 10px;
                padding: 15px;
                background-color: #f9f9f9;
            }
            QGroupBox:hover {
                border-color: #2196F3;
                background-color: #f0f8ff;
            }
        """)
        
        layout = QVBoxLayout(group_box)
        
        # Radio button with title
        radio_layout = QHBoxLayout()
        
        radio = QRadioButton()
        radio_id = 0 if mode == SolverMode.STATIC else 1
        self.button_group.addButton(radio, radio_id)
        radio.toggled.connect(lambda checked: self.on_mode_selected(mode) if checked else None)
        radio_layout.addWidget(radio)
        
        title_label = QLabel(f"{info['icon']} {info['name']}")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        radio_layout.addWidget(title_label)
        radio_layout.addStretch()
        
        layout.addLayout(radio_layout)
        
        # Description
        desc_label = QLabel(info['description'])
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #444; margin-left: 30px; font-size: 10pt;")
        layout.addWidget(desc_label)
        
        # Best for
        best_for_label = QLabel(f"<b>Best for:</b> {info['best_for']}")
        best_for_label.setWordWrap(True)
        best_for_label.setStyleSheet("color: #666; margin-left: 30px; margin-top: 5px; font-size: 9pt;")
        layout.addWidget(best_for_label)
        
        # Features
        if mode == SolverMode.FLOW:
            features_text = "<b>Key Features:</b><ul style='margin-top: 2px; margin-bottom: 2px;'>"
            for feature in info['features'][:5]:  # Show first 5
                features_text += f"<li>{feature}</li>"
            features_text += "</ul>"
            
            features_label = QLabel(features_text)
            features_label.setWordWrap(True)
            features_label.setStyleSheet("color: #555; margin-left: 30px; font-size: 9pt;")
            layout.addWidget(features_label)
        
        return group_box
    
    def on_mode_selected(self, mode: SolverMode):
        """Handle mode selection."""
        self.selected_mode = mode
    
    def accept_selection(self):
        """Accept the selection and close dialog."""
        self.mode_selected.emit(self.selected_mode)
        self.accept()
    
    @staticmethod
    def get_solver_mode(parent=None) -> SolverMode:
        """
        Show the dialog and return the selected solver mode.
        Returns None if cancelled.
        """
        dialog = SolverSelectionDialog(parent)
        result = dialog.exec()
        
        if result == QDialog.DialogCode.Accepted:
            return dialog.selected_mode
        return None
