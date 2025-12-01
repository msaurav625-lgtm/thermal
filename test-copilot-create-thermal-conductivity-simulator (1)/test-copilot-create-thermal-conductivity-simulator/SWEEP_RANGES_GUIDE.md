# Parameter Sweep Ranges Guide

**BKPS NFL Thermal Pro v7.1**  
**Quick Reference for UI Parameter Sweeps**

---

## 🎯 Available Parameter Sweeps

The system supports 4 main parameter sweep types. Here are the **exact ranges** you can use:

---

## 1. Temperature Sweep

### Range Details
- **Minimum:** 280 K (7°C)
- **Maximum:** 360 K (87°C)
- **Recommended Default:** 280-360 K
- **Typical Range:** 280-360 K (room to moderate heating)
- **Number of Points:** 10-50 (recommend 20)

### What Gets Calculated
- Base fluid thermal conductivity vs T
- Nanofluid thermal conductivity vs T
- Enhancement percentage vs T
- Viscosity vs T (both base and nanofluid)
- Density vs T
- Specific heat vs T

### UI Control Ranges (Current)
- **Temperature Control:** 273-373 K (0-100°C)
- **Sweep Range:** 280-360 K (default in code)

### Example Use Cases
- Study Brownian motion effects (increases with T)
- Thermal stability analysis
- Temperature-dependent property characterization
- Seasonal variation studies

---

## 2. Volume Fraction Sweep

### Range Details
- **Minimum:** 0.001 (0.1%)
- **Maximum:** 0.10 (10%)
- **Recommended Default:** 0.001-0.05 (0.1%-5%)
- **Practical Range:** 0.01-0.04 (1%-4%) most common
- **Number of Points:** 15-30 (recommend 20)

### What Gets Calculated
- Thermal conductivity vs φ
- Viscosity vs φ
- Enhancement percentage vs φ
- Viscosity ratio vs φ
- Density vs φ
- Specific heat vs φ

### UI Control Ranges (Current)
- **Volume Fraction Control:** 0.001-0.10 (0.1%-10%)
- **Sweep Range:** 0.001-0.05 (default in code)

### Example Use Cases
- Find optimal volume fraction for heat transfer
- Balance thermal enhancement vs viscosity increase
- Cost-benefit analysis (more particles = higher cost)
- Stability region identification

### Important Notes
- ⚠️ Above 5% (φ > 0.05): Stability issues common
- ⚠️ Above 10% (φ > 0.10): Not recommended (aggregation/settling)
- ✅ Sweet spot: 1-3% for most applications

---

## 3. Reynolds Number Sweep

### Range Details
- **Minimum:** 100 (laminar flow)
- **Maximum:** 10,000 (turbulent flow)
- **Recommended Default:** 100-10,000
- **Laminar Range:** 100-2,300 (Re < 2300)
- **Turbulent Range:** 2,300-10,000 (Re > 2300)
- **Number of Points:** 15-30 (recommend 20)

### What Gets Calculated
- Static thermal conductivity (baseline)
- Flow-enhanced thermal conductivity
- Nusselt number (Nu)
- Heat transfer coefficient (h) in W/m²·K
- Pressure drop in kPa
- Velocity in m/s

### UI Control Ranges (Current)
- **Reynolds Control:** 1-1,000,000 (1e6)
- **Sweep Range:** 100-10,000 (default in code)

### Example Use Cases
- Flow regime characterization (laminar to turbulent)
- Nusselt number vs Reynolds correlations
- Pressure drop penalty assessment
- Optimal flow rate determination
- Heat exchanger design optimization

### Important Notes
- Re < 2,300: Laminar flow (f = 64/Re)
- Re > 2,300: Turbulent flow (Blasius equation)
- Dittus-Boelter correlation used: Nu = 0.023·Re^0.8·Pr^0.4

---

## 4. Particle Diameter Sweep

### Range Details
- **Minimum:** 10 nm (10e-9 m)
- **Maximum:** 100 nm (100e-9 m)
- **Recommended Default:** 10-100 nm
- **Ultra-fine:** 10-30 nm (strong Brownian motion)
- **Fine:** 30-50 nm (balanced)
- **Coarse:** 50-100 nm (reduced Brownian effect)
- **Number of Points:** 15-25 (recommend 20)

### What Gets Calculated
- Thermal conductivity vs diameter
- Brownian velocity vs diameter (decreases with size)
- Peclet number vs diameter
- Diffusion coefficient vs diameter
- Thermal resistance vs diameter

### UI Control Ranges (Current)
- **Diameter Control:** 1-200 nm
- **Sweep Range:** 10-100 nm (default in code)

### Example Use Cases
- Brownian motion effect quantification
- Optimal particle size selection
- Stability vs enhancement tradeoff
- Manufacturing feasibility (smaller = more expensive)

### Important Notes
- Smaller particles: Higher enhancement, better stability, more expensive
- Larger particles: Lower enhancement, settling issues, cheaper
- Brownian velocity ∝ 1/diameter
- Diffusion coefficient ∝ 1/diameter

---

## 📊 Recommended Sweep Configurations

### Configuration 1: Basic Thermal Characterization
```
Temperature: 280-350 K (20 points)
Volume Fraction: 0.01-0.04 (20 points)
Nanoparticle: Al₂O₃
Base Fluid: Water
```

### Configuration 2: Flow System Optimization
```
Reynolds: 1000-8000 (25 points)
Volume Fraction: 0.02 (fixed)
Temperature: 300 K (fixed)
Nanoparticle: CuO
Channel Diameter: 0.01 m
```

### Configuration 3: Particle Size Study
```
Diameter: 10-80 nm (20 points)
Volume Fraction: 0.02 (fixed)
Temperature: 300 K (fixed)
Nanoparticle: Al₂O₃
```

### Configuration 4: Comprehensive Analysis
```
Temperature: 290-340 K (15 points)
Volume Fraction: 0.01-0.03 (15 points)
Reynolds: 500-5000 (15 points)
Diameter: 20-60 nm (15 points)
→ Total: 60 sweep runs
```

---

## 🎨 UI Integration Recommendations

### For Parameter Sweep Tab (v7.2)

**Temperature Sweep Controls:**
```python
min_temp_spin.setRange(250, 400)  # K
max_temp_spin.setRange(250, 400)  # K
min_temp_spin.setValue(280)  # Default min
max_temp_spin.setValue(360)  # Default max
points_spin.setRange(5, 100)
points_spin.setValue(20)
```

**Volume Fraction Sweep Controls:**
```python
min_phi_spin.setRange(0.001, 0.10)
max_phi_spin.setRange(0.001, 0.10)
min_phi_spin.setValue(0.001)  # 0.1%
max_phi_spin.setValue(0.05)   # 5%
points_spin.setValue(20)
```

**Reynolds Sweep Controls:**
```python
min_Re_spin.setRange(10, 100000)
max_Re_spin.setRange(10, 100000)
min_Re_spin.setValue(100)
max_Re_spin.setValue(10000)
points_spin.setValue(20)
```

**Diameter Sweep Controls:**
```python
min_d_spin.setRange(1, 200)  # nm
max_d_spin.setRange(1, 200)  # nm
min_d_spin.setValue(10)
max_d_spin.setValue(100)
points_spin.setValue(20)
```

---

## ⚙️ Current GUI Control Ranges (v7.0)

From `bkps_professional_gui_v7.py`:

| Control | Current Range | Units | Line |
|---------|--------------|-------|------|
| **Temperature** | 273-373 | K | 398 |
| **Pressure** | 50-500 | kPa | 404 |
| **Volume Fraction** | 0.001-0.10 | - | 420 |
| **Diameter** | 1-200 | nm | 427 |
| **Velocity** | 0.001-100 | m/s | 514 |
| **Reynolds** | 1-1,000,000 | - | 521 |
| **Mesh NX** | 10-500 | cells | 465 |
| **Mesh NY** | 10-500 | cells | 471 |
| **Max Iterations** | 10-10,000 | - | 485 |
| **Convergence Tol** | 1e-10 to 1e-3 | - | 491 |
| **Relaxation** | 0.1-1.0 | - | 499 |
| **Length** | 0.001-10 | m | 553 |
| **Height** | 0.001-1 | m | 561 |

---

## 🔬 Physical Constraints & Validation

### Temperature Constraints
- **Absolute Min:** 273 K (freezing point of water)
- **Absolute Max:** 373 K (boiling point of water at 1 atm)
- **Practical Range:** 280-360 K
- **Warning:** T < 273 K → Ice formation
- **Warning:** T > 373 K → Boiling (need pressure consideration)

### Volume Fraction Constraints
- **Physical Min:** 0.001 (0.1%) - Below this, effects negligible
- **Physical Max:** 0.10 (10%) - Above this, non-Newtonian behavior dominates
- **Stability Limit:** ~0.05 (5%) for most nanofluids
- **Warning:** φ > 0.06 → Aggregation likely
- **Warning:** φ > 0.10 → Settling, clogging issues

### Reynolds Number Constraints
- **Laminar Limit:** Re < 2,300
- **Turbulent:** Re > 2,300
- **Practical Max:** 10,000 (higher requires advanced turbulence models)
- **Warning:** Re > 10,000 → Need k-ε or k-ω SST turbulence model

### Particle Diameter Constraints
- **Practical Min:** 10 nm (manufacturing limit for many materials)
- **Practical Max:** 100 nm (larger → settling issues)
- **Brownian Motion Dominant:** < 50 nm
- **Warning:** d > 100 nm → Settling/aggregation issues
- **Warning:** d < 10 nm → Quantum effects may appear

---

## 📈 Execution Time Estimates

### Per Sweep Run
- **Temperature Sweep (20 points):** ~0.5 seconds
- **Volume Fraction Sweep (20 points):** ~0.6 seconds
- **Reynolds Sweep (20 points):** ~0.4 seconds
- **Diameter Sweep (20 points):** ~0.5 seconds

### Combined Sweeps
- **2D Parametric (20×20):** ~10-15 seconds
- **3D Parametric (15×15×15):** ~2-3 minutes
- **Full 4D Analysis:** ~10-15 minutes

---

## 💡 Usage Tips

### 1. Start with Default Ranges
Use the recommended default ranges first to get familiar with system behavior.

### 2. Adjust Based on Application
- **Electronics cooling:** T = 300-350 K, φ = 0.01-0.03
- **Solar collectors:** T = 320-370 K, φ = 0.02-0.05
- **Engine cooling:** T = 350-373 K, φ = 0.01-0.02

### 3. Number of Points
- **Quick check:** 10 points
- **Standard analysis:** 20 points
- **Publication quality:** 30-50 points

### 4. Avoid These Ranges
- ❌ T < 273 K (freezing)
- ❌ T > 373 K without pressure adjustment
- ❌ φ > 0.10 (instability)
- ❌ Re > 50,000 without turbulence model
- ❌ d < 5 nm (quantum effects)
- ❌ d > 200 nm (not nanofluids)

---

## 🚀 Quick Start Examples

### Example 1: Quick Temperature Study
```python
from nanofluid_simulator import ParameterSweepEngine

engine = ParameterSweepEngine()
result = engine.sweep_temperature(
    T_range=(280, 360),
    n_points=20,
    base_fluid="Water",
    nanoparticle="Al2O3",
    volume_fraction=0.02
)
```

### Example 2: Optimal Volume Fraction
```python
result = engine.sweep_volume_fraction(
    phi_range=(0.01, 0.04),
    n_points=25,
    base_fluid="Water",
    nanoparticle="CuO",
    temperature=300
)
```

### Example 3: Flow Regime Analysis
```python
result = engine.sweep_reynolds_number(
    Re_range=(500, 8000),
    n_points=30,
    base_fluid="Water",
    nanoparticle="Al2O3",
    volume_fraction=0.02
)
```

---

## 📋 Summary Table

| Parameter | Min | Max | Default Range | Units | Typical Points |
|-----------|-----|-----|---------------|-------|----------------|
| **Temperature** | 280 | 360 | 280-360 | K | 20 |
| **Volume Fraction** | 0.001 | 0.10 | 0.001-0.05 | - | 20 |
| **Reynolds Number** | 100 | 10,000 | 100-10,000 | - | 20 |
| **Diameter** | 10 | 100 | 10-100 | nm | 20 |

---

**Dedicated to: Brijesh Kumar Pandey**  
**BKPS NFL Thermal Pro v7.1**  
**Date: December 1, 2025**
