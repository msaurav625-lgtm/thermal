# Flow Calculator GUI Guide - v7.1

## Overview

The **Advanced Flow-Dependent Property Calculator** is a new GUI tab added in v7.1 that provides full user control over flow-dependent thermal conductivity and viscosity calculations.

## Features

### ✅ What's New in v7.1 GUI

1. **Dedicated "Flow Calculator" Tab**
   - Located between "Configuration" and "Parametric" tabs
   - Comprehensive interface for advanced flow-dependent calculations
   - Real-time results display

2. **Zero, One, or Multiple Nanoparticles**
   - Add as many nanoparticles as needed
   - Enable/disable nanoparticles without removing them
   - Compare multiple materials simultaneously

3. **Parameter Ranges**
   - Volume fraction: Single value or range (min-max-steps)
   - Velocity: Range-based sweeps
   - Temperature: Single value or sweeps
   - Diameter: Single value or sweeps

4. **Multiple Model Support**
   - **Thermal Conductivity**: Static, Buongiorno, Kumar, Rea-Guzman
   - **Viscosity**: Einstein, Brinkman, Batchelor, Shear-dependent
   - Select multiple models for comparison

5. **Calculation Modes**
   - Single Calculation
   - Volume Fraction Sweep
   - Velocity Sweep
   - Temperature Sweep
   - Diameter Sweep
   - Multi-dimensional Sweep
   - Material Comparison

6. **Results Display & Export**
   - Interactive table with all results
   - Export to CSV for further analysis
   - Automatic enhancement calculation

## User Interface Tour

### 1. Base Fluid Section
- **Base Fluid Selector**: Choose from Water, EG (Ethylene Glycol), or EG-Water mixture

### 2. Nanoparticles Table
- **Columns**:
  - `Enabled`: Checkbox to enable/disable nanoparticle
  - `Material`: Dropdown with 11 materials (Al₂O₃, CuO, TiO₂, SiO₂, ZnO, Fe₃O₄, Cu, Ag, Au, CNT, Graphene)
  - `φ (min-max-steps)`: Volume fraction - enter single value (e.g., `0.02`) or range (e.g., `0.01-0.05-10`)
  - `d (nm)`: Particle diameter in nanometers (1-500 nm)
  - `Shape`: Sphere, Cylinder, or Platelet
  - Remove button (×)

- **Buttons**:
  - `+ Add Nanoparticle`: Add a new row
  - `- Remove Selected`: Remove the currently selected row

### 3. Flow Conditions
- **Temperature**: 273-400 K (default: 300 K)
- **Pressure**: 10⁴-10⁷ Pa (default: 101325 Pa)
- **Velocity Range**: 
  - Min: 0.001-10 m/s (default: 0.1)
  - Max: 0.001-10 m/s (default: 0.1)
  - Steps: 1-100 (default: 1)
- **Shear Rate** (optional): 0-100,000 1/s

### 4. Model Selection
- **Thermal Conductivity Models**: Check one or more
  - Static (no flow enhancement)
  - Buongiorno (convective enhancement)
  - Kumar (shear-rate dependent)
  - Rea-Guzman (empirical correlation)

- **Viscosity Models**: Check one or more
  - Einstein (dilute suspensions)
  - Brinkman (moderate concentrations)
  - Batchelor (Brownian effects)
  - Shear-dependent (non-Newtonian)

### 5. Calculation Options
- **Calculation Mode Dropdown**:
  - Single Calculation: One set of conditions
  - Volume Fraction Sweep: Vary φ
  - Velocity Sweep: Vary flow velocity
  - Temperature Sweep: Vary temperature
  - Diameter Sweep: Vary particle size
  - Multi-dimensional Sweep: Combine parameters
  - Material Comparison: Compare multiple nanoparticles

### 6. Results Section
- **Results Table**: Displays calculated properties
  - Material name
  - Volume fraction (%)
  - Velocity (m/s)
  - Thermal conductivity (W/m·K)
  - Viscosity (mPa·s)
  - Enhancement (%)

- **Export Button**: Save results to CSV file

## Usage Examples

### Example 1: Base Fluid Only (No Nanoparticles)

1. Select base fluid: **Water**
2. Leave nanoparticle table empty (or disable all)
3. Set temperature: **300 K**
4. Set velocity: **0.1 m/s**
5. Click **Calculate**

**Result**: Base fluid properties at specified conditions

### Example 2: Single Nanoparticle

1. Select base fluid: **Water**
2. Click **+ Add Nanoparticle**
3. Configure row:
   - Enabled: ✓ (checked)
   - Material: **Al₂O₃**
   - φ: **0.02** (2%)
   - d: **30 nm**
   - Shape: **Sphere**
4. Set velocity: **0.1 m/s**
5. Select models: **Buongiorno** (k), **Brinkman** (μ)
6. Click **Calculate**

**Result**: Enhanced properties with ~5-7% thermal conductivity improvement

### Example 3: Multiple Nanoparticles Comparison

1. Add three nanoparticles:
   - Row 1: Al₂O₃, φ=0.02, d=30nm
   - Row 2: CuO, φ=0.02, d=30nm
   - Row 3: Cu, φ=0.02, d=30nm
2. All enabled ✓
3. Calculation mode: **Material Comparison**
4. Click **Calculate**

**Result**: Side-by-side comparison showing Cu > CuO > Al₂O₃

### Example 4: Volume Fraction Sweep

1. Add nanoparticle: Al₂O₃
2. Set φ: **0.005-0.05-10** (0.5% to 5%, 10 steps)
3. Calculation mode: **Volume Fraction Sweep**
4. Click **Calculate**

**Result**: 10 data points showing enhancement vs. concentration

### Example 5: Velocity Sweep

1. Add nanoparticle: CuO, φ=0.03
2. Velocity range:
   - Min: **0.01 m/s**
   - Max: **0.2 m/s**
   - Steps: **10**
3. Calculation mode: **Velocity Sweep**
4. Select: **Buongiorno** (flow-dependent model)
5. Click **Calculate**

**Result**: Enhancement increases with velocity

### Example 6: Enable/Disable Nanoparticles

1. Add three nanoparticles
2. Run calculation with all enabled
3. Uncheck one nanoparticle's "Enabled" box
4. Re-run calculation

**Result**: Calculation uses only enabled nanoparticles (no need to delete rows)

### Example 7: Multi-Model Comparison

1. Add nanoparticle: Al₂O₃, φ=0.02
2. Check **all** conductivity models:
   - Static ✓
   - Buongiorno ✓
   - Kumar ✓
   - Rea-Guzman ✓
3. Click **Calculate**

**Result**: Compare predictions from different models

## Integration with Existing Tabs

The Flow Calculator tab complements existing functionality:

- **Configuration Tab**: Quick setup for unified engine (Static/Flow/CFD/Hybrid modes)
- **Flow Calculator Tab**: Advanced flow-dependent analysis with full control
- **Parametric Tab**: Legacy parametric sweeps
- **Results Tab**: Standard results display
- **Visualization Tab**: Charts and plots
- **Validation Tab**: Research validation data
- **Advanced Tab**: Expert settings

## Technical Details

### Backend Integration

The GUI tab uses the v7.1 `AdvancedFlowCalculator` class:

```python
from nanofluid_simulator import (
    AdvancedFlowCalculator, 
    FlowDependentConfig, 
    NanoparticleSpec, 
    FlowConditions,
    calculate_flow_properties
)
```

### Data Flow

1. **User Input → Config**: GUI widgets → FlowDependentConfig dataclass
2. **Config → Calculator**: AdvancedFlowCalculator initialization
3. **Calculate**: calculator.calculate_flow_properties()
4. **Results → Display**: Dict → QTableWidget
5. **Export**: QTableWidget → CSV file

### Configuration Validation

The calculator validates:
- Volume fraction: 0 ≤ φ ≤ 0.2 (0-20%)
- Diameter: d > 0
- Temperature: T > 0 K
- At least one model selected
- Enabled nanoparticles only

## Tips & Best Practices

### 1. Start Simple
- Begin with single nanoparticle, single value
- Verify results match expectations
- Then add complexity (ranges, multiple materials)

### 2. Reasonable Ranges
- φ: 0.5% to 10% (typical experimental range)
- d: 10-100 nm (most common)
- V: 0.01-1 m/s (avoid extreme values)

### 3. Model Selection
- **Static**: Baseline (no flow effects)
- **Buongiorno**: Best for convective flows
- **Kumar**: When shear effects are important
- **Rea-Guzman**: Empirical, good for validation

### 4. Performance
- Large sweeps (>50 points) may take time
- Use progress bar to track completion
- Export results for offline analysis

### 5. Troubleshooting
- If calculation fails, check:
  - At least one nanoparticle enabled
  - Valid volume fraction format
  - At least one model selected
  - No negative values

## Comparison with Old Flow Mode

| Feature | Old "Flow" Mode | New "Flow Calculator" Tab |
|---------|----------------|---------------------------|
| Nanoparticles | 1 only | 0, 1, or multiple |
| Parameters | Single values | Ranges (min-max-steps) |
| Models | Fixed | User-selectable |
| Enable/Disable | Delete only | Toggle checkbox |
| Sweeps | Limited | Multi-dimensional |
| Export | JSON only | CSV + JSON |
| Comparison | No | Yes (multiple materials) |

## Keyboard Shortcuts

- **Ctrl+N**: New project (clears all)
- **Ctrl+S**: Save project
- **Ctrl+O**: Open project
- **Add NP button**: Click or Enter to add row
- **Table navigation**: Arrow keys, Tab

## Future Enhancements (Planned)

- 📊 Real-time plotting of sweep results
- 🔬 Direct comparison charts (k vs φ, k vs V)
- 💾 Save/load nanoparticle presets
- 🎨 Enhanced visualization integration
- 📈 Optimization mode (find optimal φ, d, V)

## Support

For questions or issues:
1. Check `FLOW_DEPENDENT_GUIDE.md` for API details
2. Run examples: `python examples/example_flow_dependent_advanced.py`
3. Check validation: `python validate_against_research.py`

## Changelog

### v7.1 - Flow Calculator GUI Release
- ✅ Added dedicated Flow Calculator tab
- ✅ Full integration with AdvancedFlowCalculator
- ✅ Multiple nanoparticle support
- ✅ Parameter range inputs
- ✅ Enable/disable toggles
- ✅ Model selection interface
- ✅ Results table with enhancement
- ✅ CSV export functionality
- ✅ No deprecated code - clean implementation

---

**Dedicated to: Brijesh Kumar Pandey**  
**BKPS NFL Thermal Pro v7.1** - Research-Grade Nanofluid Analysis
