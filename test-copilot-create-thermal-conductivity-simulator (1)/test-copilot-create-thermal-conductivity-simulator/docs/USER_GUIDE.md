# Nanofluid Simulator - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [GUI User Guide](#gui-user-guide)
4. [Python API Guide](#python-api-guide)
5. [Models Reference](#models-reference)
6. [Material Database](#material-database)
7. [Export and Reporting](#export-and-reporting)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## Introduction

The Nanofluid Thermal Conductivity Simulator is a research-grade tool for predicting thermophysical properties of nanofluids. It implements multiple validated models from peer-reviewed literature and provides both a professional GUI and a Python API.

### What are Nanofluids?

Nanofluids are engineered colloidal suspensions of nanoparticles (1-100 nm) dispersed in conventional base fluids. They exhibit enhanced thermal properties compared to base fluids, making them valuable for:

- Heat exchangers
- Cooling systems
- Solar thermal applications
- Electronics cooling
- Automotive applications

---

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone or Download the Repository**
   ```bash
   git clone https://github.com/msaurav625-lgtm/test.git
   cd test
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the Package**
   ```bash
   pip install -e .
   ```

4. **Verify Installation**
   ```bash
   python -c "import nanofluid_simulator; print(nanofluid_simulator.__version__)"
   ```

### Optional Dependencies

For full functionality, install optional packages:

```bash
# For 3D visualization
pip install vtk pyvista

# For interactive plots
pip install plotly

# For performance optimization
pip install numba
```

---

## GUI User Guide

### Launching the GUI

```bash
python run_gui.py
```

Or:

```bash
python -m nanofluid_simulator.gui.main_window
```

### GUI Overview

The interface consists of two main panels:

#### Left Panel: Controls
- **Base Fluid Selection**: Choose from predefined fluids
- **Temperature Control**: Set operating temperature (-50°C to 200°C)
- **Nanoparticle Components**: Add/remove nanoparticles
- **Calculation Buttons**: Run simulations

#### Right Panel: Visualization
- **Results Tab**: Tabular display of calculated properties
- **Visualization Tab**: Interactive plots
- **Configuration Tab**: Current simulation setup
- **Log Tab**: Activity log

### Step-by-Step Workflow

#### 1. Select Base Fluid

Click the **Base Fluid** dropdown and select from:
- water
- ethylene_glycol
- propylene_glycol
- water_eg_50_50 (hybrid mixture)
- engine_oil, mineral_oil, etc.

#### 2. Set Temperature

Use the **Temperature** spinner to set operating temperature:
- Range: -50°C to 200°C
- Default: 25°C (298.15 K)

#### 3. Add Nanoparticles

Click **➕ Add Component** to open the material selector dialog:

1. Select nanoparticle from dropdown (Cu, Al2O3, CNT, etc.)
2. Set volume fraction (0.001 to 0.5)
3. Set particle size (1 to 1000 nm)
4. Set sphericity (0.1 to 1.0)
5. Click OK

**For Hybrid Nanofluids**: Add multiple nanoparticle components

#### 4. Run Calculations

Choose calculation type:

- **🔬 Calculate Current State**: Run all applicable models for current configuration
- **📈 Temperature Sweep**: Vary temperature from 0°C to 100°C
- **📊 Concentration Sweep**: Vary volume fraction from 0.1% to 10%

#### 5. View Results

Switch between tabs:

- **Results Tab**: View calculated properties in table format
  - Thermal conductivity
  - Enhancement percentage
  - Viscosity
  - Density
  - Prandtl number

- **Visualization Tab**: View interactive plots
  - Model comparison bar chart
  - k vs Temperature
  - k vs Volume Fraction

#### 6. Export Results

**Menu → File → Export Results**

Choose format:
- CSV for raw data
- Excel for formatted spreadsheets
- PDF for comprehensive reports

### GUI Tips

- **Dark Mode**: Menu → View → Toggle Dark Mode (Ctrl+T)
- **Save Configuration**: Menu → File → Save Configuration (Ctrl+S)
- **Load Configuration**: Menu → File → Open Configuration (Ctrl+O)
- **Help**: Menu → Help → Documentation

---

## Python API Guide

### Basic Usage

```python
from nanofluid_simulator import EnhancedNanofluidSimulator

# Create simulator
sim = EnhancedNanofluidSimulator()

# Configure
sim.set_base_fluid("water")
sim.set_temperature_celsius(25)
sim.add_nanoparticle("Cu", volume_fraction=0.01, particle_size=25)

# Calculate
results = sim.calculate_all_applicable_models()

# Process results
for result in results:
    print(f"{result.model_name}: {result.k_effective:.6f} W/m·K")
```

### Advanced Features

#### Custom Materials

```python
from nanofluid_simulator import NanoparticleDatabase

# Add custom nanoparticle
NanoparticleDatabase.add_nanoparticle(
    formula="CustomNP",
    name="Custom Nanoparticle",
    thermal_conductivity=150.0,
    density=5000.0,
    specific_heat=600.0,
    particle_size_range=(10, 50)
)

# Add custom base fluid
NanoparticleDatabase.add_base_fluid(
    key="custom_fluid",
    name="Custom Fluid",
    thermal_conductivity=0.5,
    density=1000.0,
    specific_heat=3500.0,
    viscosity=0.002,
    reference_temperature=298.15
)
```

#### Temperature-Dependent Calculations

```python
# Set specific temperature
sim.set_temperature(323.15)  # 50°C in Kelvin

# Or use Celsius
sim.set_temperature_celsius(50)

# Temperature sweep
results_dict = sim.parametric_study_temperature(
    temperature_range=(273.15, 373.15),
    n_points=20
)
```

#### Hybrid Nanofluids

```python
sim = EnhancedNanofluidSimulator()
sim.set_base_fluid("water")
sim.set_temperature_celsius(40)

# Add multiple nanoparticles
sim.add_nanoparticle("Cu", volume_fraction=0.01, particle_size=30)
sim.add_nanoparticle("Al2O3", volume_fraction=0.01, particle_size=40)
sim.add_nanoparticle("CNT", volume_fraction=0.005, particle_size=20)

# Calculate (hybrid models will be used automatically)
results = sim.calculate_all_applicable_models()
```

---

## Models Reference

### Classical Models

#### 1. Maxwell Model (1881)

**Formula:**
```
k_eff/k_bf = (k_np + 2*k_bf + 2*φ*(k_np - k_bf)) / 
             (k_np + 2*k_bf - φ*(k_np - k_bf))
```

**Valid For:**
- Low volume fractions (φ < 0.05)
- Spherical particles
- No particle interaction

**Usage:**
```python
result = sim.calculate_maxwell()
```

#### 2. Hamilton-Crosser Model (1962)

**Formula:**
```
k_eff/k_bf = (k_np + (n-1)*k_bf - (n-1)*φ*(k_bf - k_np)) /
             (k_np + (n-1)*k_bf + φ*(k_bf - k_np))
where n = 3/sphericity
```

**Valid For:**
- Non-spherical particles
- Moderate volume fractions

**Usage:**
```python
sim.add_nanoparticle("CNT", volume_fraction=0.01, sphericity=0.5)
result = sim.calculate_hamilton_crosser()
```

#### 3. Yu-Choi Model (2003)

**Formula:**
Accounts for nanolayer effect around particles

**Valid For:**
- All volume fractions
- When nanolayer effects are significant

**Parameters:**
- layer_thickness: Nanolayer thickness (nm)
- particle_radius: Particle radius (nm)

### Advanced Models

#### 4. Patel Model (2003)

**Temperature-dependent enhancement**

**Formula:**
```
k_eff = k_maxwell * [1 + α*(T - T_ref)]
```

**Usage:**
```python
result = sim.calculate_patel()
```

#### 5. Koo-Kleinstreuer Model (2004)

**Includes Brownian motion effects**

**Formula:**
```
k_eff = k_static + k_Brownian
```

**Valid For:**
- All volume fractions
- Temperature-dependent behavior

**Usage:**
```python
result = sim.calculate_koo_kleinstreuer()
```

### Hybrid Models

#### 6. Hajjar Model (2014)

**For binary hybrid nanofluids**

**Requirements:**
- Exactly 2 nanoparticle types

**Usage:**
```python
sim.add_nanoparticle("Cu", volume_fraction=0.01)
sim.add_nanoparticle("Al2O3", volume_fraction=0.01)
result = sim.calculate_hajjar_hybrid()
```

#### 7. Sundar Model (2017)

**Comprehensive hybrid model with synergy**

**Features:**
- Particle size effects
- Temperature dependence
- Synergistic enhancement

**Usage:**
```python
result = sim.calculate_sundar_hybrid()
```

---

## Material Database

### Available Nanoparticles

| Material | Formula | k (W/m·K) | ρ (kg/m³) | cp (J/kg·K) |
|----------|---------|-----------|-----------|-------------|
| Copper | Cu | 401 | 8933 | 385 |
| Silver | Ag | 429 | 10490 | 235 |
| Gold | Au | 317 | 19300 | 129 |
| Aluminum | Al | 237 | 2700 | 897 |
| Alumina | Al₂O₃ | 40 | 3970 | 765 |
| Titania | TiO₂ | 8.9 | 4250 | 686 |
| Copper Oxide | CuO | 76.5 | 6310 | 535 |
| Silicon Carbide | SiC | 120 | 3210 | 750 |
| Aluminum Nitride | AlN | 285 | 3260 | 740 |
| Boron Nitride | BN | 300 | 2100 | 800 |
| CNT | CNT | 3000 | 2100 | 711 |
| Graphene | C | 5000 | 2200 | 709 |

### Available Base Fluids

| Fluid | k (W/m·K) | ρ (kg/m³) | cp (J/kg·K) | μ (Pa·s) |
|-------|-----------|-----------|-------------|----------|
| Water | 0.613 | 997 | 4182 | 0.001 |
| Ethylene Glycol | 0.252 | 1113 | 2415 | 0.0161 |
| Propylene Glycol | 0.200 | 1036 | 2500 | 0.042 |
| Water-EG (50:50) | 0.415 | 1055 | 3298 | 0.00405 |
| Engine Oil | 0.145 | 884 | 1909 | 0.486 |

---

## Export and Reporting

### Export to CSV

```python
from nanofluid_simulator.export import export_results

export_results(results, "output.csv", format='csv')
```

### Export to Excel

```python
from nanofluid_simulator.export import ResultExporter

exporter = ResultExporter()
exporter.to_excel(results, "output.xlsx", sheet_name="Results")
```

### Generate PDF Report

```python
from nanofluid_simulator.export import generate_pdf_report

generate_pdf_report(
    results=results,
    config=config,
    filename="report.pdf",
    plots=["fig1.png", "fig2.png"]
)
```

---

## Troubleshooting

### Common Issues

**Issue: GUI won't start**
```
Solution: Ensure PyQt6 is installed
pip install PyQt6
```

**Issue: Import errors**
```
Solution: Install package in development mode
pip install -e .
```

**Issue: Slow calculations**
```
Solution: Install numba for optimization
pip install numba
```

**Issue: Export fails**
```
Solution: Install export dependencies
pip install pandas openpyxl reportlab
```

---

## FAQ

**Q: What volume fractions are recommended?**
A: For practical applications, 0.1% to 5% (0.001 to 0.05). Higher concentrations may cause stability issues.

**Q: Can I add my own materials?**
A: Yes! Use `NanoparticleDatabase.add_nanoparticle()` or `add_base_fluid()`

**Q: Which model is most accurate?**
A: It depends on your system. Compare multiple models and validate against experimental data.

**Q: How do I cite this simulator?**
A: See the Citation section in README_FULL.md

**Q: Can I use this for commercial applications?**
A: Yes, under MIT license terms.

---

*For more help, visit: [GitHub Issues](https://github.com/msaurav625-lgtm/test/issues)*
