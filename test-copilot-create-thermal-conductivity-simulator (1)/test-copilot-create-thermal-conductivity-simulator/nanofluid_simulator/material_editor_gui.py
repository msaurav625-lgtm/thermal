#!/usr/bin/env python3
"""
Material Database Editor GUI for BKPS NFL Thermal Pro 7.0
CRUD interface for custom nanoparticles and base fluids

Dedicated to: Brijesh Kumar Pandey
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel,
    QLineEdit, QDoubleSpinBox, QTextEdit, QMessageBox,
    QFormLayout, QGroupBox, QHeaderView, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from pathlib import Path
from nanofluid_simulator.material_database import (
    get_material_database, NanoparticleMaterial, BaseFluidMaterial
)


class MaterialDatabaseEditor(QDialog):
    """
    Material Database Editor Dialog
    
    Provides full CRUD operations for:
    - Nanoparticle materials
    - Base fluid materials
    
    With validation and persistence to user directory
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.database = get_material_database()
        self.setWindowTitle("Material Database Editor — BKPS NFL Thermal Pro 7.0")
        self.setGeometry(200, 200, 1000, 700)
        self.setModal(True)
        
        self.init_ui()
        self.load_data()
    
    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("<h2>Material Database Editor</h2><p>Create and manage custom materials</p>")
        layout.addWidget(header)
        
        # Tabs for nanoparticles and base fluids
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_nanoparticle_tab(), "🔬 Nanoparticles")
        self.tabs.addTab(self._create_base_fluid_tab(), "💧 Base Fluids")
        layout.addWidget(self.tabs)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        export_btn = QPushButton("📦 Export Database")
        export_btn.clicked.connect(self._export_database)
        button_layout.addWidget(export_btn)
        
        import_btn = QPushButton("📥 Import Database")
        import_btn.clicked.connect(self._import_database)
        button_layout.addWidget(import_btn)
        
        button_layout.addStretch()
        
        help_btn = QPushButton("❓ Help")
        help_btn.clicked.connect(self._show_help)
        button_layout.addWidget(help_btn)
        
        close_btn = QPushButton("✓ Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def _create_nanoparticle_tab(self):
        """Create nanoparticle materials tab"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Left: Table
        left_layout = QVBoxLayout()
        
        left_layout.addWidget(QLabel("<b>Nanoparticle Materials</b>"))
        
        self.np_table = QTableWidget()
        self.np_table.setColumnCount(5)
        self.np_table.setHorizontalHeaderLabels([
            "Name", "k (W/m·K)", "ρ (kg/m³)", "cp (J/kg·K)", "Description"
        ])
        self.np_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.np_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.np_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.np_table.itemSelectionChanged.connect(self._on_np_selection_changed)
        left_layout.addWidget(self.np_table)
        
        table_buttons = QHBoxLayout()
        add_np_btn = QPushButton("➕ New")
        add_np_btn.clicked.connect(self._add_nanoparticle)
        table_buttons.addWidget(add_np_btn)
        
        delete_np_btn = QPushButton("🗑 Delete")
        delete_np_btn.clicked.connect(self._delete_nanoparticle)
        table_buttons.addWidget(delete_np_btn)
        
        table_buttons.addStretch()
        left_layout.addLayout(table_buttons)
        
        layout.addLayout(left_layout, stretch=2)
        
        # Right: Editor
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("<b>Material Properties</b>"))
        
        editor_group = QGroupBox("Edit Selected Material")
        editor_layout = QFormLayout()
        
        self.np_name_edit = QLineEdit()
        self.np_name_edit.setPlaceholderText("e.g., Al2O3, CuO, Custom1")
        editor_layout.addRow("Material Name*:", self.np_name_edit)
        
        self.np_k_spin = QDoubleSpinBox()
        self.np_k_spin.setRange(0.1, 10000)
        self.np_k_spin.setDecimals(2)
        self.np_k_spin.setSuffix(" W/m·K")
        self.np_k_spin.setToolTip("Thermal conductivity at room temperature")
        editor_layout.addRow("Thermal Conductivity*:", self.np_k_spin)
        
        self.np_rho_spin = QDoubleSpinBox()
        self.np_rho_spin.setRange(100, 50000)
        self.np_rho_spin.setDecimals(0)
        self.np_rho_spin.setSuffix(" kg/m³")
        self.np_rho_spin.setToolTip("Density at room temperature")
        editor_layout.addRow("Density*:", self.np_rho_spin)
        
        self.np_cp_spin = QDoubleSpinBox()
        self.np_cp_spin.setRange(100, 10000)
        self.np_cp_spin.setDecimals(0)
        self.np_cp_spin.setSuffix(" J/kg·K")
        self.np_cp_spin.setToolTip("Specific heat capacity at room temperature")
        editor_layout.addRow("Specific Heat*:", self.np_cp_spin)
        
        self.np_desc_edit = QTextEdit()
        self.np_desc_edit.setMaximumHeight(80)
        self.np_desc_edit.setPlaceholderText("Optional description...")
        editor_layout.addRow("Description:", self.np_desc_edit)
        
        editor_group.setLayout(editor_layout)
        right_layout.addWidget(editor_group)
        
        # Validation info
        validation_label = QLabel(
            "<b>Validation Ranges:</b><br>"
            "• k: 0.1 - 10,000 W/m·K<br>"
            "• ρ: 100 - 50,000 kg/m³<br>"
            "• cp: 100 - 10,000 J/kg·K"
        )
        validation_label.setStyleSheet("QLabel { background-color: #f0f8ff; padding: 10px; border-radius: 5px; }")
        right_layout.addWidget(validation_label)
        
        # Save/Update button
        save_np_btn = QPushButton("💾 Save Changes")
        save_np_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        save_np_btn.clicked.connect(self._save_nanoparticle)
        right_layout.addWidget(save_np_btn)
        
        right_layout.addStretch()
        layout.addLayout(right_layout, stretch=1)
        
        return widget
    
    def _create_base_fluid_tab(self):
        """Create base fluid materials tab"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Left: Table
        left_layout = QVBoxLayout()
        
        left_layout.addWidget(QLabel("<b>Base Fluid Materials</b>"))
        
        self.bf_table = QTableWidget()
        self.bf_table.setColumnCount(6)
        self.bf_table.setHorizontalHeaderLabels([
            "Name", "k (W/m·K)", "ρ (kg/m³)", "cp (J/kg·K)", "μ (mPa·s)", "T_ref (K)"
        ])
        self.bf_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.bf_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.bf_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.bf_table.itemSelectionChanged.connect(self._on_bf_selection_changed)
        left_layout.addWidget(self.bf_table)
        
        table_buttons = QHBoxLayout()
        add_bf_btn = QPushButton("➕ New")
        add_bf_btn.clicked.connect(self._add_base_fluid)
        table_buttons.addWidget(add_bf_btn)
        
        delete_bf_btn = QPushButton("🗑 Delete")
        delete_bf_btn.clicked.connect(self._delete_base_fluid)
        table_buttons.addWidget(delete_bf_btn)
        
        table_buttons.addStretch()
        left_layout.addLayout(table_buttons)
        
        layout.addLayout(left_layout, stretch=2)
        
        # Right: Editor
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("<b>Fluid Properties</b>"))
        
        editor_group = QGroupBox("Edit Selected Fluid")
        editor_layout = QFormLayout()
        
        self.bf_name_edit = QLineEdit()
        self.bf_name_edit.setPlaceholderText("e.g., Water, EG, Custom_Oil")
        editor_layout.addRow("Fluid Name*:", self.bf_name_edit)
        
        self.bf_k_spin = QDoubleSpinBox()
        self.bf_k_spin.setRange(0.01, 10)
        self.bf_k_spin.setDecimals(4)
        self.bf_k_spin.setSuffix(" W/m·K")
        self.bf_k_spin.setToolTip("Thermal conductivity at reference temperature")
        editor_layout.addRow("Thermal Conductivity*:", self.bf_k_spin)
        
        self.bf_rho_spin = QDoubleSpinBox()
        self.bf_rho_spin.setRange(100, 20000)
        self.bf_rho_spin.setDecimals(0)
        self.bf_rho_spin.setSuffix(" kg/m³")
        self.bf_rho_spin.setToolTip("Density at reference temperature")
        editor_layout.addRow("Density*:", self.bf_rho_spin)
        
        self.bf_cp_spin = QDoubleSpinBox()
        self.bf_cp_spin.setRange(500, 10000)
        self.bf_cp_spin.setDecimals(0)
        self.bf_cp_spin.setSuffix(" J/kg·K")
        self.bf_cp_spin.setToolTip("Specific heat at reference temperature")
        editor_layout.addRow("Specific Heat*:", self.bf_cp_spin)
        
        self.bf_mu_spin = QDoubleSpinBox()
        self.bf_mu_spin.setRange(0.1, 1000)
        self.bf_mu_spin.setDecimals(4)
        self.bf_mu_spin.setSuffix(" mPa·s")
        self.bf_mu_spin.setToolTip("Dynamic viscosity at reference temperature")
        editor_layout.addRow("Viscosity*:", self.bf_mu_spin)
        
        self.bf_tref_spin = QDoubleSpinBox()
        self.bf_tref_spin.setRange(200, 500)
        self.bf_tref_spin.setDecimals(1)
        self.bf_tref_spin.setSuffix(" K")
        self.bf_tref_spin.setValue(300)
        self.bf_tref_spin.setToolTip("Reference temperature for properties")
        editor_layout.addRow("T_reference*:", self.bf_tref_spin)
        
        self.bf_desc_edit = QTextEdit()
        self.bf_desc_edit.setMaximumHeight(60)
        self.bf_desc_edit.setPlaceholderText("Optional description...")
        editor_layout.addRow("Description:", self.bf_desc_edit)
        
        editor_group.setLayout(editor_layout)
        right_layout.addWidget(editor_group)
        
        # Validation info
        validation_label = QLabel(
            "<b>Validation Ranges:</b><br>"
            "• k: 0.01 - 10 W/m·K<br>"
            "• ρ: 100 - 20,000 kg/m³<br>"
            "• cp: 500 - 10,000 J/kg·K<br>"
            "• μ: 0.1 - 1,000 mPa·s<br>"
            "• T_ref: 200 - 500 K"
        )
        validation_label.setStyleSheet("QLabel { background-color: #f0f8ff; padding: 10px; border-radius: 5px; }")
        right_layout.addWidget(validation_label)
        
        # Save/Update button
        save_bf_btn = QPushButton("💾 Save Changes")
        save_bf_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        save_bf_btn.clicked.connect(self._save_base_fluid)
        right_layout.addWidget(save_bf_btn)
        
        right_layout.addStretch()
        layout.addLayout(right_layout, stretch=1)
        
        return widget
    
    def load_data(self):
        """Load data into tables"""
        self._load_nanoparticles()
        self._load_base_fluids()
    
    def _load_nanoparticles(self):
        """Load nanoparticles into table"""
        self.np_table.setRowCount(0)
        
        for i, (name, mat) in enumerate(sorted(self.database.nanoparticles.items())):
            self.np_table.insertRow(i)
            self.np_table.setItem(i, 0, QTableWidgetItem(mat.name))
            self.np_table.setItem(i, 1, QTableWidgetItem(f"{mat.thermal_conductivity:.2f}"))
            self.np_table.setItem(i, 2, QTableWidgetItem(f"{mat.density:.0f}"))
            self.np_table.setItem(i, 3, QTableWidgetItem(f"{mat.specific_heat:.0f}"))
            self.np_table.setItem(i, 4, QTableWidgetItem(mat.description))
    
    def _load_base_fluids(self):
        """Load base fluids into table"""
        self.bf_table.setRowCount(0)
        
        for i, (name, fluid) in enumerate(sorted(self.database.base_fluids.items())):
            self.bf_table.insertRow(i)
            self.bf_table.setItem(i, 0, QTableWidgetItem(fluid.name))
            self.bf_table.setItem(i, 1, QTableWidgetItem(f"{fluid.thermal_conductivity:.4f}"))
            self.bf_table.setItem(i, 2, QTableWidgetItem(f"{fluid.density:.0f}"))
            self.bf_table.setItem(i, 3, QTableWidgetItem(f"{fluid.specific_heat:.0f}"))
            self.bf_table.setItem(i, 4, QTableWidgetItem(f"{fluid.viscosity*1000:.4f}"))
            self.bf_table.setItem(i, 5, QTableWidgetItem(f"{fluid.reference_temperature:.1f}"))
    
    # ===== NANOPARTICLE OPERATIONS =====
    
    def _on_np_selection_changed(self):
        """Handle nanoparticle selection"""
        selected = self.np_table.selectedItems()
        if not selected:
            return
        
        row = selected[0].row()
        name = self.np_table.item(row, 0).text()
        mat = self.database.get_nanoparticle(name)
        
        if mat:
            self.np_name_edit.setText(mat.name)
            self.np_k_spin.setValue(mat.thermal_conductivity)
            self.np_rho_spin.setValue(mat.density)
            self.np_cp_spin.setValue(mat.specific_heat)
            self.np_desc_edit.setPlainText(mat.description)
    
    def _add_nanoparticle(self):
        """Clear editor for new nanoparticle"""
        self.np_table.clearSelection()
        self.np_name_edit.clear()
        self.np_k_spin.setValue(40.0)
        self.np_rho_spin.setValue(3970)
        self.np_cp_spin.setValue(765)
        self.np_desc_edit.clear()
        self.np_name_edit.setFocus()
    
    def _save_nanoparticle(self):
        """Save nanoparticle (add or update)"""
        name = self.np_name_edit.text().strip()
        
        if not name:
            QMessageBox.warning(self, "Validation Error", "Material name is required")
            return
        
        mat = NanoparticleMaterial(
            name=name,
            thermal_conductivity=self.np_k_spin.value(),
            density=self.np_rho_spin.value(),
            specific_heat=self.np_cp_spin.value(),
            description=self.np_desc_edit.toPlainText().strip()
        )
        
        # Determine if adding or updating
        is_update = name in self.database.nanoparticles
        
        if is_update:
            success, message = self.database.update_nanoparticle(mat)
        else:
            success, message = self.database.add_nanoparticle(mat)
        
        if success:
            QMessageBox.information(self, "Success", message)
            self._load_nanoparticles()
        else:
            QMessageBox.warning(self, "Error", message)
    
    def _delete_nanoparticle(self):
        """Delete selected nanoparticle"""
        selected = self.np_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select a material to delete")
            return
        
        row = selected[0].row()
        name = self.np_table.item(row, 0).text()
        
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete material '{name}'?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            success, message = self.database.delete_nanoparticle(name)
            if success:
                QMessageBox.information(self, "Success", message)
                self._load_nanoparticles()
                self._add_nanoparticle()  # Clear editor
            else:
                QMessageBox.warning(self, "Error", message)
    
    # ===== BASE FLUID OPERATIONS =====
    
    def _on_bf_selection_changed(self):
        """Handle base fluid selection"""
        selected = self.bf_table.selectedItems()
        if not selected:
            return
        
        row = selected[0].row()
        name = self.bf_table.item(row, 0).text()
        fluid = self.database.get_base_fluid(name)
        
        if fluid:
            self.bf_name_edit.setText(fluid.name)
            self.bf_k_spin.setValue(fluid.thermal_conductivity)
            self.bf_rho_spin.setValue(fluid.density)
            self.bf_cp_spin.setValue(fluid.specific_heat)
            self.bf_mu_spin.setValue(fluid.viscosity * 1000)  # Convert to mPa·s
            self.bf_tref_spin.setValue(fluid.reference_temperature)
            self.bf_desc_edit.setPlainText(fluid.description)
    
    def _add_base_fluid(self):
        """Clear editor for new base fluid"""
        self.bf_table.clearSelection()
        self.bf_name_edit.clear()
        self.bf_k_spin.setValue(0.6)
        self.bf_rho_spin.setValue(1000)
        self.bf_cp_spin.setValue(4000)
        self.bf_mu_spin.setValue(1.0)
        self.bf_tref_spin.setValue(300)
        self.bf_desc_edit.clear()
        self.bf_name_edit.setFocus()
    
    def _save_base_fluid(self):
        """Save base fluid (add or update)"""
        name = self.bf_name_edit.text().strip()
        
        if not name:
            QMessageBox.warning(self, "Validation Error", "Fluid name is required")
            return
        
        fluid = BaseFluidMaterial(
            name=name,
            thermal_conductivity=self.bf_k_spin.value(),
            density=self.bf_rho_spin.value(),
            specific_heat=self.bf_cp_spin.value(),
            viscosity=self.bf_mu_spin.value() / 1000,  # Convert from mPa·s to Pa·s
            reference_temperature=self.bf_tref_spin.value(),
            description=self.bf_desc_edit.toPlainText().strip()
        )
        
        # Determine if adding or updating
        is_update = name in self.database.base_fluids
        
        if is_update:
            success, message = self.database.update_base_fluid(fluid)
        else:
            success, message = self.database.add_base_fluid(fluid)
        
        if success:
            QMessageBox.information(self, "Success", message)
            self._load_base_fluids()
        else:
            QMessageBox.warning(self, "Error", message)
    
    def _delete_base_fluid(self):
        """Delete selected base fluid"""
        selected = self.bf_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select a fluid to delete")
            return
        
        row = selected[0].row()
        name = self.bf_table.item(row, 0).text()
        
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete fluid '{name}'?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            success, message = self.database.delete_base_fluid(name)
            if success:
                QMessageBox.information(self, "Success", message)
                self._load_base_fluids()
                self._add_base_fluid()  # Clear editor
            else:
                QMessageBox.warning(self, "Error", message)
    
    # ===== DATABASE OPERATIONS =====
    
    def _export_database(self):
        """Export entire database"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Database", str(Path.home()), "JSON Files (*.json)"
        )
        
        if filepath:
            success, message = self.database.export_database(Path(filepath))
            if success:
                QMessageBox.information(self, "Export Successful", message)
            else:
                QMessageBox.warning(self, "Export Failed", message)
    
    def _import_database(self):
        """Import database"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Database", str(Path.home()), "JSON Files (*.json)"
        )
        
        if filepath:
            reply = QMessageBox.question(
                self, "Import Mode",
                "Merge with existing database?\n\n"
                "Yes = Merge (keep existing materials)\n"
                "No = Replace (delete existing materials)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Cancel:
                return
            
            merge = reply == QMessageBox.StandardButton.Yes
            success, message = self.database.import_database(Path(filepath), merge=merge)
            
            if success:
                QMessageBox.information(self, "Import Successful", message)
                self.load_data()
            else:
                QMessageBox.warning(self, "Import Failed", message)
    
    def _show_help(self):
        """Show help dialog"""
        help_text = """
<h3>Material Database Editor Help</h3>

<h4>Creating Custom Materials:</h4>
<ul>
  <li>Click <b>New</b> to create a new material</li>
  <li>Enter material properties in the editor</li>
  <li>Click <b>Save Changes</b> to add to database</li>
  <li>Materials are stored in: <code>~/Documents/BKPS_NFL/</code></li>
</ul>

<h4>Editing Materials:</h4>
<ul>
  <li>Select a material from the table</li>
  <li>Modify properties in the editor</li>
  <li>Click <b>Save Changes</b> to update</li>
</ul>

<h4>Validation Ranges:</h4>
<p>All properties must be within physically realistic ranges:</p>
<ul>
  <li><b>Nanoparticles:</b> k (0.1-10,000 W/m·K), ρ (100-50,000 kg/m³)</li>
  <li><b>Base Fluids:</b> k (0.01-10 W/m·K), μ (0.1-1,000 mPa·s)</li>
</ul>

<h4>Database Management:</h4>
<ul>
  <li><b>Export:</b> Save entire database to JSON file</li>
  <li><b>Import:</b> Load materials from JSON (merge or replace)</li>
</ul>

<p><i>Created materials are immediately available in the main GUI.</i></p>
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Help — Material Database Editor")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(help_text)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()
