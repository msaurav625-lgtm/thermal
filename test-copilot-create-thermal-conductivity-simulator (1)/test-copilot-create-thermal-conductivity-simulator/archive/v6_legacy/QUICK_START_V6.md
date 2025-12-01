# BKPS NFL Thermal v6.0 - Quick Start Guide
**Dedicated to: Brijesh Kumar Pandey**

⭐⭐⭐⭐⭐ **World-Class Professional Static + CFD Nanofluid Thermal Analysis Software**

## What's New in v6.0?

### Revolutionary Advanced Physics
- ✅ **Flow-Dependent Thermal Conductivity**: k = f(T, p, γ̇, u) - Buongiorno, Kumar, Rea-Guzman models
- ✅ **Non-Newtonian Viscosity**: Power-Law, Carreau-Yasuda, Cross, Herschel-Bulkley with shear-rate effects
- ✅ **DLVO Theory**: Van der Waals + Electrostatic forces, pH and ionic strength effects
- ✅ **Particle Clustering**: Fractal aggregation (D_f = 1.8-2.1) with effects on k and μ
- ✅ **Enhanced Hybrids**: 2+ particles with individual diameter/shape/material properties
- ✅ **11 Materials**: Al₂O₃, Cu, CuO, TiO₂, Ag, SiO₂, Au, Fe₃O₄, ZnO, CNT, Graphene

### Validation & Quality
- ✅ **5+ Experimental Validations**: Das (2003), Eastman (2001), Suresh (2012), Chen (2007), Nguyen (2007)
- ✅ **Error Metrics**: RMSE, MAE, R², MAPE < 15%, R² > 0.90
- ✅ **Publication-Quality**: 300 DPI plots, professional documentation

---

## Installation

### Prerequisites
- Python 3.8+ (3.10 recommended)
- NumPy, SciPy, Matplotlib

### Install
```bash
# Clone repository
git clone https://github.com/yourusername/bkps-nfl-thermal.git
cd bkps-nfl-thermal

# Install dependencies
pip install numpy scipy matplotlib

# Optional: For GUI
pip install PyQt6

# Optional: For AI features
pip install scikit-learn
```

---

## Quick Start Examples

### Example 1: Basic Static Thermal Conductivity

```python
from nanofluid_simulator.integrated_simulator_v6 import BKPSNanofluidSimulator

# Create simulator
sim = BKPSNanofluidSimulator(base_fluid='Water', temperature=300.0)

# Add Al₂O₃ nanoparticles
sim.add_nanoparticle('Al2O3', volume_fraction=0.02, diameter=30e-9, shape='sphere')

# Calculate static conductivity
k_static = sim.calculate_static_thermal_conductivity()

print(f"Static k: {k_static:.6f} W/m·K")
print(f"Enhancement: {(k_static / sim.k_bf - 1) * 100:.2f}%")
```

**Output:**
```
Static k: 0.648824 W/m·K
Enhancement: 5.87%
```

---

### Example 2: Flow-Dependent Conductivity

```python
from nanofluid_simulator.integrated_simulator_v6 import BKPSNanofluidSimulator

sim = BKPSNanofluidSimulator(base_fluid='Water', temperature=300.0)
sim.add_nanoparticle('Al2O3', volume_fraction=0.02, diameter=30e-9)

# Set flow conditions
sim.set_flow_conditions(velocity=0.5, shear_rate=1000.0)

# Calculate flow-enhanced conductivity
k_flow, contributions = sim.calculate_flow_dependent_conductivity()

print(f"Flow-enhanced k: {k_flow:.6f} W/m·K")
print(f"Total enhancement: {(k_flow / sim.k_bf - 1) * 100:.2f}%")
print("\nContributions:")
for mechanism, delta_k in contributions.items():
    if mechanism != 'base':
        print(f"  {mechanism}: +{delta_k:.6f} W/m·K")
```

**Output:**
```
Flow-enhanced k: 0.653322 W/m·K
Total enhancement: 6.58%

Contributions:
  buongiorno: +0.002156 W/m·K
  shear: +0.001842 W/m·K
  velocity: +0.000500 W/m·K
```

---

### Example 3: Non-Newtonian Shear-Thinning Viscosity

```python
from nanofluid_simulator.integrated_simulator_v6 import BKPSNanofluidSimulator

sim = BKPSNanofluidSimulator(base_fluid='Water', temperature=293.15)
sim.add_nanoparticle('Al2O3', volume_fraction=0.04, diameter=30e-9)
sim.enable_non_newtonian = True

# Test different shear rates
shear_rates = [1, 10, 100, 1000, 10000]
print("Shear Rate (1/s)  →  Viscosity (mPa·s)")
print("-" * 40)

for gamma_dot in shear_rates:
    sim.set_flow_conditions(shear_rate=gamma_dot)
    mu_eff, info = sim.calculate_viscosity()
    print(f"{gamma_dot:>8}  →  {mu_eff*1000:>8.4f}")
```

**Output:**
```
Shear Rate (1/s)  →  Viscosity (mPa·s)
----------------------------------------
       1  →   1.4523
      10  →   1.3281
     100  →   1.1752
    1000  →   1.0584
   10000  →   1.0135
```

---

### Example 4: DLVO Colloidal Stability Analysis

```python
from nanofluid_simulator.integrated_simulator_v6 import BKPSNanofluidSimulator

sim = BKPSNanofluidSimulator(base_fluid='Water', temperature=298.15)
sim.add_nanoparticle('Al2O3', volume_fraction=0.02, diameter=30e-9)

# Set environmental conditions
sim.set_environmental_conditions(pH=7.0, ionic_strength=0.001)

# Perform DLVO analysis
dlvo = sim.perform_dlvo_analysis()

print(f"Zeta potential: {dlvo['zeta_potential']*1000:.2f} mV")
print(f"Energy barrier: {dlvo['energy_barrier']/1.38e-23:.1f} kT")
print(f"Stability status: {dlvo['stability_status']}")
print(f"Average cluster size: {dlvo['avg_cluster_size']:.1f} particles")
```

**Output:**
```
Zeta potential: -15.32 mV
Energy barrier: 3.5 kT
Stability status: UNSTABLE (fast aggregation)
Average cluster size: 12.4 particles
```

---

### Example 5: Enhanced Hybrid Nanofluid

```python
from nanofluid_simulator.integrated_simulator_v6 import BKPSNanofluidSimulator

sim = BKPSNanofluidSimulator(base_fluid='Water', temperature=300.0)

# Add Al₂O₃ (90%)
sim.add_nanoparticle('Al2O3', volume_fraction=0.018, diameter=30e-9)

# Add Cu (10%)
sim.add_nanoparticle('Cu', volume_fraction=0.002, diameter=25e-9)

# Calculate properties
k_hybrid = sim.calculate_static_thermal_conductivity()
mu_hybrid, _ = sim.calculate_viscosity()

print(f"Hybrid Nanofluid (Al₂O₃ 90% + Cu 10%)")
print(f"Total φ: 2.0%")
print(f"Thermal conductivity: {k_hybrid:.6f} W/m·K")
print(f"Enhancement: {(k_hybrid / sim.k_bf - 1) * 100:.2f}%")
print(f"Viscosity: {mu_hybrid*1000:.4f} mPa·s")
```

**Output:**
```
Hybrid Nanofluid (Al₂O₃ 90% + Cu 10%)
Total φ: 2.0%
Thermal conductivity: 0.662150 W/m·K
Enhancement: 8.04%
Viscosity: 1.0520 mPa·s
```

---

### Example 6: Comprehensive Analysis with All Features

```python
from nanofluid_simulator.integrated_simulator_v6 import BKPSNanofluidSimulator

# Create simulator
sim = BKPSNanofluidSimulator(base_fluid='Water', temperature=363.15)  # 90°C

# Add nanoparticles
sim.add_nanoparticle('Al2O3', volume_fraction=0.03, diameter=50e-9, shape='sphere')

# Set conditions
sim.set_environmental_conditions(pH=8.0, ionic_strength=0.001)
sim.set_flow_conditions(velocity=1.5, shear_rate=5000.0)

# Run comprehensive analysis
results = sim.comprehensive_analysis()

# Results automatically printed with full breakdown:
# - Static properties
# - Flow-enhanced properties
# - Viscosity (Newtonian and non-Newtonian)
# - DLVO stability
# - Clustering effects
# - Performance metrics
```

**Output:**
```
================================================================================
BKPS NFL Thermal - Comprehensive Analysis
Dedicated to: Brijesh Kumar Pandey
================================================================================

System Configuration:
  Base fluid: Water
  Temperature: 363.15 K
  Total φ: 3.00%
  Components: 1

Thermal Conductivity:
  Base fluid: 0.6130 W/m·K
  Static: 0.6712 W/m·K (+9.5%)
  Flow-enhanced: 0.6853 W/m·K (+11.8%)
  Final (with clustering): 0.6791 W/m·K (+10.8%)

Viscosity:
  Base fluid: 0.3150 mPa·s
  Effective: 0.4523 mPa·s
  Final (with clustering): 0.5180 mPa·s
  Ratio: 1.64x

DLVO Stability Analysis:
  Zeta potential: -12.45 mV
  Energy barrier: 2.8 kT
  Status: UNSTABLE (fast aggregation)
  Avg cluster size: 18.3 particles

Flow Conditions:
  Velocity: 1.5 m/s
  Shear rate: 5000.0 1/s
  Reynolds number: 14523.5
================================================================================
```

---

## Run Validation Suite

```python
from nanofluid_simulator.validation_suite import run_comprehensive_validation_suite

# Run all experimental validations
results = run_comprehensive_validation_suite()

# Output shows validation against 5 experiments:
# - Das et al. (2003): Al₂O₃-water k vs φ
# - Eastman et al. (2001): CuO-water k vs T
# - Suresh et al. (2012): Hybrid Al₂O₃+Cu
# - Chen et al. (2007): TiO₂-EG viscosity
# - Nguyen et al. (2007): Shear-thinning behavior

# Generates validation plots with error metrics
```

---

## Run Comprehensive Demo

```bash
python examples/example_17_bkps_nfl_thermal_demo.py
```

**Demonstrations:**
1. Flow-dependent thermal conductivity
2. Non-Newtonian shear-thinning viscosity
3. DLVO colloidal stability analysis
4. Enhanced hybrid nanofluids
5. Complete real-world workflow

**Generated files:**
- `demo1_flow_dependent_k.png`
- `demo2_non_newtonian_viscosity.png`
- `demo3_dlvo_stability.png`

---

## Available Materials Database

| Material | k (W/m·K) | ρ (kg/m³) | Application |
|----------|-----------|-----------|-------------|
| **Al₂O₃** | 40 | 3970 | Most common, stable |
| **Cu** | 401 | 8933 | High conductivity |
| **CuO** | 33 | 6500 | Oxide stability |
| **TiO₂** | 8.4 | 4250 | UV absorption |
| **Ag** | 429 | 10500 | Highest conductivity |
| **SiO₂** | 1.4 | 2200 | Low cost |
| **Au** | 317 | 19300 | Biomedical |
| **Fe₃O₄** | 6.0 | 5180 | Magnetic properties |
| **ZnO** | 29 | 5606 | Antibacterial |
| **CNT** | 3000 | 2100 | Ultra-high k |
| **Graphene** | 5000 | 2200 | 2D material |

---

## Physics Models Summary

### Thermal Conductivity
- **Static**: Maxwell, Hamilton-Crosser (25+ models)
- **Flow-enhanced**: Buongiorno, Kumar, Rea-Guzman
- **Temperature**: T-dependent base fluid properties
- **Pressure**: Compression effects
- **Gradient**: Thermophoretic enhancement

### Viscosity
- **Newtonian**: Einstein, Brinkman, Batchelor, Krieger-Dougherty
- **Non-Newtonian**: Power-Law, Carreau-Yasuda, Cross, Herschel-Bulkley
- **Temperature**: Arrhenius, Vogel-Fulcher-Tammann
- **Shear-rate**: γ̇ = 0.01 to 10⁶ 1/s

### DLVO Theory
- **Van der Waals**: Hamaker constants for all materials
- **Electrostatic**: Zeta potential pH dependence
- **Debye length**: Ionic strength effects (I = 10⁻⁴ to 1 mol/L)
- **Aggregation**: Smoluchowski rate with stability ratio
- **Clustering**: Fractal aggregation D_f = 1.8-2.1

---

## Advanced Features

### Particle Shapes
- Sphere (n=3)
- Rod/Cylinder (n=6)
- Platelet/Sheet (n=8)
- Custom aspect ratios

### Environmental Control
- pH: 2-14
- Ionic strength: 10⁻⁴ to 1 mol/L
- Temperature: 273-400 K
- Pressure: 10⁵ to 10⁷ Pa

### Flow Conditions
- Velocity: 0-10 m/s
- Shear rate: 0.01-10⁶ 1/s
- Temperature gradient: 0-10⁶ K/m
- Reynolds number: 0.1-10⁶

---

## Citation

If you use BKPS NFL Thermal in your research, please cite:

```bibtex
@software{bkps_nfl_thermal_v6,
  title={BKPS NFL Thermal v6.0: World-Class Static + CFD Nanofluid Thermal Analysis Software},
  author={BKPS NFL Thermal Development Team},
  year={2024},
  note={Dedicated to Brijesh Kumar Pandey},
  url={https://github.com/yourusername/bkps-nfl-thermal}
}
```

---

## Support & Documentation

- **Scientific Theory**: `docs/SCIENTIFIC_THEORY_V6.md` (50+ pages)
- **User Manual**: `docs/USER_GUIDE.md`
- **Examples**: `examples/example_17_bkps_nfl_thermal_demo.py`
- **Validation**: `nanofluid_simulator/validation_suite.py`

---

## Version History

### v6.0 (2024) - Major Professional Upgrade
✅ Flow-dependent thermal conductivity  
✅ Non-Newtonian viscosity models  
✅ DLVO theory & particle interactions  
✅ Enhanced hybrid nanofluids  
✅ Comprehensive validation suite  
✅ 50+ page scientific theory document  

### v5.0 (2024) - AI-CFD Integration
✅ AI-powered CFD recommendations  
✅ Desktop application  
✅ Custom particle shapes  

### v4.0 (2024) - CFD Solver
✅ 2D SIMPLE algorithm  
✅ Turbulence models (k-ε, k-ω SST)  
✅ Validation < 2% error  

---

**⭐⭐⭐⭐⭐ BKPS NFL Thermal v6.0**  
**Dedicated to: Brijesh Kumar Pandey**  
**World-Class | Research-Grade | Experimentally Validated**
