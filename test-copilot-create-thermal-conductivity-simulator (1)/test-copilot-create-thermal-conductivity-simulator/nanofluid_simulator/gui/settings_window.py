"""
Settings Window

Application configuration and preferences.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget, QLabel,
    QCheckBox, QComboBox, QFormLayout, QPushButton,
    QDialogButtonBox
)

class SettingsWindow(QDialog):
    """Settings dialog"""
    
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(600, 400)
        self.settings = settings.copy()
        
        layout = QVBoxLayout(self)
        
        tabs = QTabWidget()
        tabs.addTab(self._create_general_tab(), "General")
        tabs.addTab(self._create_ai_tab(), "AI")
        tabs.addTab(self._create_advanced_tab(), "Advanced")
        
        layout.addWidget(tabs)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _create_general_tab(self) -> QWidget:
        """General settings"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        auto_save = QCheckBox()
        auto_save.setChecked(self.settings.get('auto_save', True))
        layout.addRow("Auto-save:", auto_save)
        
        return widget
    
    def _create_ai_tab(self) -> QWidget:
        """AI settings"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        ai_enabled = QCheckBox()
        ai_enabled.setChecked(self.settings.get('ai_enabled', True))
        layout.addRow("Enable AI:", ai_enabled)
        
        return widget
    
    def _create_advanced_tab(self) -> QWidget:
        """Advanced settings"""
        widget = QWidget()
        return widget
    
    def get_settings(self) -> dict:
        """Get updated settings"""
        return self.settings
