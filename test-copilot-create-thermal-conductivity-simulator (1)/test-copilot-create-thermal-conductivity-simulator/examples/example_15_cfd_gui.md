# Example 15: Using the CFD GUI

This example demonstrates the interactive CFD GUI interface.

## Quick Start

```bash
python run_cfd_gui.py
```

## Features

### 1. **Geometry & Mesh Tab**
- Domain dimensions (length × height)
- Mesh resolution (nx × ny cells)
- Real-time mesh preview
- Cell count display

### 2. **Nanofluid Tab**
- Base fluid selection (water, ethylene glycol)
- Nanoparticle type (Al2O3, CuO, TiO2, SiO2, Fe3O4)
- Volume fraction (0-10%)
- Temperature (273-400 K)
- Property calculator with enhancement display

### 3. **Boundary Conditions Tab**
- **Inlet (Left)**: Velocity and temperature
- **Outlet (Right)**: Pressure
- **Walls (Top & Bottom)**: No-slip, heated, or adiabatic
- Color-coded visualization on mesh

### 4. **Solver Settings Tab**
- Max iterations (10-2000)
- Convergence tolerance (1e-3 to 1e-6)
- Linear solver selection (Direct, BiCGSTAB, Gauss-Seidel)
- Turbulence model (Laminar, k-epsilon)
- Under-relaxation factors (u/v, p, T)

### 5. **Real-Time Monitoring**
- Progress bar during simulation
- Status log with solver messages
- Convergence history display
- Residual tracking

### 6. **Results Display**
- Velocity field statistics
- Pressure field (max, min, drop)
- Temperature distribution
- Convergence plots
- Export to text file

## Usage Workflow

1. **Configure Geometry**
   - Set domain size (e.g., 1.0 m × 0.1 m)
   - Choose mesh resolution (e.g., 50×20 cells)
   - View mesh preview

2. **Select Nanofluid**
   - Choose nanoparticle (e.g., Al2O3)
   - Set volume fraction (e.g., 3%)
   - Calculate properties

3. **Set Boundary Conditions**
   - Inlet: u=0.1 m/s, T=300 K
   - Outlet: p=0 Pa
   - Walls: Heated at 320 K or adiabatic

4. **Configure Solver**
   - Max iterations: 200
   - Tolerance: 1e-4 (standard)
   - Linear solver: BiCGSTAB
   - Under-relaxation: u=0.7, p=0.3, T=0.8

5. **Run Simulation**
   - Click "▶ Run Simulation"
   - Monitor progress and convergence
   - View results when complete

6. **Analyze Results**
   - Check velocity and temperature fields
   - Review convergence history
   - Export results for documentation

## Example Scenarios

### Scenario 1: Channel Flow
```
Geometry: 1.0 m × 0.1 m
Mesh: 60×30 cells
Nanofluid: 3% Al2O3 in water at 300 K
Inlet: u=0.1 m/s, T=300 K
Outlet: p=0 Pa
Walls: No-slip, adiabatic
Expected: ~1-2 minutes
```

### Scenario 2: Heat Exchanger
```
Geometry: 0.5 m × 0.02 m
Mesh: 100×20 cells
Nanofluid: 5% CuO in water at 293 K
Inlet: u=0.1 m/s, T=320 K
Outlet: p=0 Pa
Walls: Bottom heated at 293 K, top adiabatic
Expected: ~2-3 minutes
```

### Scenario 3: Quick Test
```
Geometry: 0.5 m × 0.1 m
Mesh: 20×10 cells (coarse)
Nanofluid: Water (0%) at 300 K
Inlet: u=0.1 m/s, T=300 K
Tolerance: 1e-3 (fast)
Expected: <30 seconds
```

## Tips

1. **Start Coarse**: Use 20×10 mesh for quick tests
2. **Refine Gradually**: Increase to 50×30 → 100×60 as needed
3. **Monitor Convergence**: Stop if residuals plateau
4. **Adjust Relaxation**: Lower factors if diverging
5. **Check Results**: Verify physical behavior

## Keyboard Shortcuts

- **Ctrl+Q**: Quit application
- **F5**: Refresh mesh preview

## Troubleshooting

**GUI doesn't start:**
```bash
pip install PyQt6
```

**Slow performance:**
- Reduce mesh resolution
- Increase tolerance (1e-3)
- Use Direct solver for <2000 cells

**Divergence issues:**
- Lower under-relaxation factors
- Increase max iterations
- Check boundary conditions

**Memory errors:**
- Reduce mesh size
- Close other applications
- See PERFORMANCE_OPTIMIZATION.md

## Integration with Examples

The GUI provides the same capabilities as command-line examples:

| GUI Setup | Equivalent Example |
|-----------|-------------------|
| Default channel | Example 8 (CFD demo) |
| With heated wall | Example 11 (Heat exchanger) |
| Fine mesh validation | Example 10 (Validation) |

## Advanced Usage

**Batch Processing**: Use GUI to configure, then export settings and run via Python script for multiple cases.

**Custom Analysis**: Export results and post-process with Example 9 (post-processing tools).

**Research Workflow**:
1. GUI for initial exploration
2. Python scripts for parametric studies
3. GUI for visualization/presentation

## Requirements

- Python 3.8+
- PyQt6
- NumPy, SciPy, matplotlib
- All nanofluid_simulator modules

## Notes

- GUI runs solver in separate thread (non-blocking)
- All settings validated before simulation
- Results auto-saved in memory
- Export feature for documentation
- Color-coded boundary regions for clarity

---

**Pro Tip**: Use GUI for learning and quick tests, then transition to Python scripts for production research workflows with better automation and repeatability.
