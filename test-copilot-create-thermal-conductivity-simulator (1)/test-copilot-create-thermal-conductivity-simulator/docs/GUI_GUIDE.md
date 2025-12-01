# CFD GUI User Guide

**Nanofluid Simulator v4.0**  
Interactive Graphical Interface for CFD Simulations

## Table of Contents
1. [Getting Started](#getting-started)
2. [Main Interface Overview](#main-interface-overview)
3. [Geometry & Mesh Setup](#geometry--mesh-setup)
4. [Nanofluid Properties](#nanofluid-properties)
5. [Boundary Conditions](#boundary-conditions)
6. [Solver Settings](#solver-settings)
7. [Running Simulations](#running-simulations)
8. [Viewing Results](#viewing-results)
9. [Troubleshooting](#troubleshooting)
10. [Example Workflows](#example-workflows)

---

## Getting Started

### Prerequisites

The GUI requires PyQt6. Install it using:
```bash
pip install PyQt6
```

### Launching the GUI

**Method 1: From command line**
```bash
python run_cfd_gui.py
```

**Method 2: From main GUI (v3)**
```bash
python run_gui_v3.py
```
Then click "CFD Simulation" button.

**Method 3: Programmatically**
```python
from nanofluid_simulator.gui.cfd_window import launch_cfd_gui
launch_cfd_gui()
```

---

## Main Interface Overview

The CFD GUI consists of two main panels:

### Left Panel - Configuration
- **4 tabs** for setup:
  1. Geometry & Mesh
  2. Nanofluid Properties
  3. Boundary Conditions
  4. Solver Settings
- **Control buttons**: Run, Stop
- **Progress bar**: Real-time iteration progress
- **Status log**: Detailed simulation messages

### Right Panel - Visualization
- **Mesh preview**: Visual representation of computational domain
- **Convergence plots**: Real-time residual monitoring
- **Results display**: Post-simulation analysis

---

## Geometry & Mesh Setup

### Domain Geometry

Configure the computational domain:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Length (m) | 0.001 - 10.0 | 1.0 | Domain length (x-direction) |
| Height (m) | 0.001 - 10.0 | 0.1 | Domain height (y-direction) |

**Tips:**
- For channel flow: Length >> Height (aspect ratio ~10:1)
- For cavity flow: Length ≈ Height (square cavity)
- For heat exchangers: Length = 0.5-1.0 m, Height = 0.01-0.05 m

### Mesh Resolution

Control the number of cells:

| Parameter | Range | Default | Recommendations |
|-----------|-------|---------|-----------------|
| Cells in X | 5 - 500 | 50 | Higher for longer domains |
| Cells in Y | 5 - 500 | 30 | Higher for complex flow |

**Mesh Guidelines:**
- **Coarse** (20×10): Quick testing, ~1 second
- **Medium** (50×30): Good balance, ~30 seconds
- **Fine** (100×60): High accuracy, ~5 minutes
- **Very Fine** (200×100): Research-grade, ~30 minutes

**Total cells** = nx × ny (displayed in preview)

**Memory estimate:**
- 50×30 (1,500 cells): ~10 MB
- 100×60 (6,000 cells): ~40 MB
- 200×100 (20,000 cells): ~160 MB

---

## Nanofluid Properties

### Nanoparticle Selection

Choose from 7 nanoparticle types:

| Nanoparticle | Thermal Conductivity | Best For |
|--------------|---------------------|----------|
| Al₂O₃ (Alumina) | 40 W/m·K | General purpose, cost-effective |
| CuO (Copper Oxide) | 76.5 W/m·K | High heat transfer |
| Cu (Copper) | 401 W/m·K | Maximum conductivity |
| TiO₂ (Titania) | 8.4 W/m·K | Stability, low cost |
| SiO₂ (Silica) | 1.4 W/m·K | Low viscosity increase |
| Fe₃O₄ (Magnetite) | 6 W/m·K | Magnetic applications |
| Carbon Nanotubes | 3000 W/m·K | Maximum enhancement |

### Volume Fraction (φ)

Range: 0% - 10%

**Recommendations:**
- **1-2%**: Good enhancement, minimal viscosity increase
- **3-5%**: Balanced performance
- **>5%**: Maximum heat transfer, higher pressure drop

**Trade-offs:**
- Higher φ → Better heat transfer ✓
- Higher φ → Higher viscosity (pumping power) ✗
- Higher φ → Higher cost ✗

### Particle Diameter

Range: 10 - 100 nm

**Effects:**
- Smaller particles → Better stability
- Smaller particles → Higher effective conductivity
- Larger particles → Lower viscosity increase
- Typical: 20-50 nm

### Temperature

Range: 273 - 373 K (0-100°C)

**Notes:**
- Properties calculated at this temperature
- Affects thermal conductivity and viscosity
- Water base fluid limited to 273-373 K

### Base Fluid

Currently supported: **Water** (H₂O)

Properties at 293 K:
- Density: 998 kg/m³
- Viscosity: 0.001 Pa·s
- Thermal conductivity: 0.6 W/m·K
- Specific heat: 4182 J/kg·K

---

## Boundary Conditions

### Available Boundary Types

#### 1. Velocity Inlet
Set velocity components at inlet:
- **u (m/s)**: Horizontal velocity
- **v (m/s)**: Vertical velocity (usually 0)
- **Temperature (K)**: Inlet fluid temperature

**Typical values:**
- Laminar flow: 0.01 - 0.5 m/s
- Turbulent flow: 1.0 - 10.0 m/s

#### 2. Pressure Outlet
Set pressure at outlet:
- **Pressure (Pa)**: Usually 0 (gauge pressure)
- Zero-gradient for velocity and temperature

#### 3. Wall
- **No-slip**: u = v = 0 (default for walls)
- **Temperature**: Fixed wall temperature
- **Heat flux**: Fixed heat input (W/m²)

**Wall types:**
- Adiabatic: No heat transfer
- Isothermal: Fixed temperature
- Heat flux: Constant q (e.g., 100 kW/m² for electronics)

#### 4. Symmetry
- Zero-gradient for all variables
- Use to reduce domain size

### Applying Boundary Conditions

1. Select boundary type from dropdown
2. Enter parameters
3. Click **"Add BC"**
4. BC appears in table
5. Repeat for all boundaries

**Required BCs:**
- At least one inlet OR wall with velocity
- At least one outlet OR pressure BC
- Wall BCs for top/bottom (if applicable)

**Example Setup (Channel Flow):**
- Inlet (left): Velocity inlet, u=0.1 m/s, T=293K
- Outlet (right): Pressure outlet, p=0 Pa
- Bottom wall: No-slip, T=320K (heated)
- Top wall: No-slip, adiabatic

---

## Solver Settings

### Turbulence Model

| Model | Use When | Description |
|-------|----------|-------------|
| Laminar | Re < 2300 | No turbulence modeling |
| k-ε | Re > 10,000 | Standard turbulence model |
| k-ω SST | Complex flows | Better near-wall treatment |

**Reynolds number:**  
Re = ρ u L / μ

Calculate before choosing:
- Re < 2300: Laminar
- 2300 < Re < 4000: Transitional (use laminar + safety)
- Re > 4000: Turbulent

### Maximum Iterations

Range: 10 - 10,000  
Default: 200

**Guidelines:**
- Quick test: 50-100 iterations
- Production: 200-500 iterations
- Research: 500-1000 iterations

**Note:** Solver stops early if converged

### Convergence Tolerance

Range: 1e-2 to 1e-8  
Default: 1e-4

**Recommendations:**
| Tolerance | Accuracy | Use Case |
|-----------|----------|----------|
| 1e-3 | ~1-2% | Quick testing |
| 1e-4 | ~0.1-0.5% | Production |
| 1e-5 | ~0.01-0.1% | Research |
| 1e-6 | ~0.001% | Validation |

### Linear Solver

| Solver | Best For | Speed |
|--------|----------|-------|
| Direct | <2,000 cells | Baseline |
| Gauss-Seidel | Simple problems | Slow |
| BiCGSTAB | >2,000 cells | 1.5× faster |
| CG | Symmetric systems | Fast |
| GMRES | Difficult problems | Robust |

**Recommendation:** Use BiCGSTAB for most cases

### Under-Relaxation Factors

Critical for stability!

| Variable | Range | Default | Notes |
|----------|-------|---------|-------|
| u, v | 0.3-0.7 | 0.7 | Lower = more stable |
| p | 0.1-0.3 | 0.3 | Pressure very sensitive |
| T | 0.5-0.9 | 0.8 | Temperature less critical |

**If diverging:**
- Reduce all factors by 0.1
- Especially reduce pressure factor

**If too slow:**
- Increase factors by 0.1
- Monitor for instability

---

## Running Simulations

### Pre-Run Checklist

Before clicking Run:
1. ✓ Geometry configured
2. ✓ Mesh generated (see preview)
3. ✓ Nanofluid properties set
4. ✓ Boundary conditions defined (minimum: inlet + outlet)
5. ✓ Solver settings configured

### During Simulation

**Progress indicators:**
- **Progress bar**: Shows iteration count
- **Status log**: Real-time messages
- **Convergence plot**: Residual history

**What to watch:**
- Residuals should **decrease** over time
- All residuals should drop below tolerance
- Typical: Exponential decay in first 50 iterations

**Red flags:**
- Residuals increasing → **Diverging!** Stop and reduce under-relaxation
- Residuals flat-lined above tolerance → Adjust solver settings
- Very slow progress → Consider coarser mesh for testing

### Stopping Simulation

Click **"Stop"** button to:
- Terminate simulation gracefully
- Save current state
- Review partial results

---

## Viewing Results

### Convergence Tab

**Residual plots** show convergence history:
- **U-velocity** (blue)
- **V-velocity** (green)
- **Pressure** (red)
- **Temperature** (orange)

**Good convergence:**
- All curves trend downward
- Reach tolerance line
- Smooth decay (no oscillations)

**Poor convergence:**
- Curves oscillate wildly → Reduce under-relaxation
- Curves increase → Diverging, stop immediately
- Curves plateau → Increase iterations or adjust BC

### Results Tab

After successful convergence:

**Field visualizations:**
- Temperature contours
- Velocity vectors
- Pressure distribution
- Streamlines

**Performance metrics:**
- Maximum velocity
- Average temperature
- Heat transfer rate
- Pressure drop

**Export options:**
- Save plots as PNG
- Export data to CSV
- Generate PDF report

---

## Troubleshooting

### Common Issues

#### "PyQt6 not available"
**Solution:** Install PyQt6
```bash
pip install PyQt6
```

#### "No boundary conditions defined"
**Problem:** Need at least inlet and outlet  
**Solution:** Add minimum required BCs before running

#### Simulation diverges
**Symptoms:** Residuals increase, NaN errors  
**Solutions:**
1. Reduce under-relaxation factors (especially pressure: 0.2)
2. Use finer mesh
3. Reduce inlet velocity
4. Start with laminar model

#### Very slow convergence
**Solutions:**
1. Use coarser mesh for initial tests
2. Increase under-relaxation (carefully)
3. Switch to BiCGSTAB solver
4. Improve initial guess

#### Out of memory
**Problem:** Mesh too fine  
**Solution:** Reduce nx and ny

**Memory estimates:**
- 50×30: ~10 MB
- 100×60: ~40 MB
- 200×100: ~160 MB

#### Mesh not displaying
**Check:**
- Valid geometry (Length and Height > 0)
- nx, ny >= 5
- Click "Update Mesh" button

---

## Example Workflows

### Example 1: Simple Channel Flow

**Goal:** Simulate water flow through heated channel

1. **Geometry & Mesh**
   - Length: 1.0 m
   - Height: 0.1 m
   - nx: 50, ny: 30

2. **Nanofluid**
   - Nanoparticle: None (pure water)
   - φ: 0%
   - Temperature: 293 K

3. **Boundary Conditions**
   - Inlet: Velocity u=0.1 m/s, T=293K
   - Outlet: Pressure p=0 Pa
   - Bottom: Wall, T=320K
   - Top: Wall, adiabatic

4. **Solver**
   - Turbulence: Laminar
   - Max iterations: 200
   - Tolerance: 1e-4
   - Linear solver: BiCGSTAB

5. **Run** and view temperature distribution

**Expected time:** ~30 seconds

---

### Example 2: Nanofluid Heat Transfer Enhancement

**Goal:** Compare heat transfer with/without nanoparticles

**Run 1: Baseline (water)**
- φ = 0%
- Note: Heat transfer rate, pressure drop

**Run 2: Nanofluid (3% Al₂O₃)**
- Change φ to 3%
- Keep all other settings same
- Note: Improved heat transfer (~15-20%)

**Comparison:**
- Temperature outlet: Lower with nanofluid ✓
- Pressure drop: Slightly higher (~10%) ✗
- Overall benefit: ~15% enhancement

---

### Example 3: Microchannel Cooling

**Goal:** Electronics cooling with high heat flux

1. **Geometry**
   - Length: 0.01 m (10 mm)
   - Height: 0.0002 m (200 μm)
   - nx: 50, ny: 15

2. **Nanofluid**
   - Nanoparticle: CuO
   - φ: 2%
   - d: 30 nm
   - Temperature: 293 K

3. **Boundary Conditions**
   - Inlet: u=0.5 m/s, T=293K
   - Outlet: Pressure
   - Bottom: Heat flux = 100,000 W/m²
   - Top: Adiabatic

4. **Solver**
   - Laminar (small Re in microchannel)
   - Tolerance: 1e-4

5. **Check:** Maximum temperature < 85°C (CPU safe limit)

**Expected time:** ~2-3 minutes

---

### Example 4: Natural Convection

**Goal:** Passive cooling in cavity

1. **Geometry**
   - Length: 0.1 m (square cavity)
   - Height: 0.1 m
   - nx: 50, ny: 50

2. **Nanofluid**
   - Nanoparticle: Al₂O₃
   - φ: 4%
   - Temperature: 300 K

3. **Boundary Conditions**
   - Left wall: T=310K (hot)
   - Right wall: T=290K (cold)
   - Top/Bottom: Adiabatic
   - All walls: u=v=0 (no-slip)

4. **Solver**
   - Laminar
   - Max iterations: 500 (natural convection needs more)
   - Tolerance: 1e-5

**Expected time:** ~5 minutes

**Note:** Full buoyancy effects require Boussinesq approximation (future feature)

---

## Tips for Success

### Start Simple
1. Begin with coarse mesh (20×10)
2. Use baseline fluid (water, φ=0%)
3. Simple geometry (channel)
4. Verify results before refining

### Gradual Refinement
1. Get coarse solution working
2. Increase mesh resolution
3. Add nanoparticles
4. Optimize under-relaxation

### Monitor Convergence
- Watch residual plots in real-time
- Expect exponential decay
- Stop if diverging

### Save Configurations
- Export successful setups
- Document working parameters
- Build library of test cases

### Use Performance Tools
Run `example_14_performance_benchmark.py` to:
- Determine optimal mesh size
- Compare solver performance
- Estimate run times

---

## Advanced Features

### Custom Boundary Conditions

For complex BCs not in GUI:
1. Export configuration
2. Edit BC definitions in code
3. Run via command line

### Batch Simulations

For parametric studies:
```python
from nanofluid_simulator.gui.cfd_window import CFDWindow

# Loop over parameters
for phi in [0.01, 0.03, 0.05]:
    # Setup and run simulation
    # Save results
    pass
```

### Integration with Post-Processing

After GUI simulation:
```python
from nanofluid_simulator.cfd_postprocess import FlowPostProcessor

post = FlowPostProcessor(mesh, results)
post.plot_streamlines()
post.calculate_heat_transfer()
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Ctrl+R | Run simulation |
| Ctrl+S | Stop simulation |
| Ctrl+E | Export results |
| Ctrl+Q | Quit |

---

## Additional Resources

- **Examples:** See `examples/example_8_cfd_demo.py` for command-line equivalent
- **Documentation:** `docs/CFD_GUIDE.md` for CFD theory
- **Performance:** `PERFORMANCE_OPTIMIZATION.md` for speed tips
- **Troubleshooting:** `TROUBLESHOOTING.md` for common issues

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review example workflows
3. Run validation examples to verify installation
4. Check GitHub issues

---

**Version:** 4.0  
**Last Updated:** November 30, 2025  
**License:** See LICENSE.txt
