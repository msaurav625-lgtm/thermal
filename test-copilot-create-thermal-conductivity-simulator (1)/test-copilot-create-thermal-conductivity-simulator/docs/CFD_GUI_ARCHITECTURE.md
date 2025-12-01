# CFD GUI Interface Documentation

## Overview

The CFD GUI provides an interactive visual interface for setting up and running computational fluid dynamics simulations with nanofluids.

## Architecture

```
CFD GUI Components:
├── cfd_window.py (950 lines)
│   ├── CFDWindow - Main window controller
│   ├── MeshCanvas - Visual mesh display widget
│   └── SolverThread - Background solver execution
│
├── Tabs:
│   ├── Geometry & Mesh - Domain and discretization
│   ├── Nanofluid - Properties and selection
│   ├── Boundary Conditions - BC setup
│   ├── Solver Settings - Algorithm parameters
│   ├── Convergence - Real-time monitoring
│   └── Results - Post-simulation analysis
│
└── Features:
    ├── Visual mesh preview with BC coloring
    ├── Interactive parameter adjustment
    ├── Progress tracking
    ├── Result export
    └── Non-blocking execution
```

## Key Classes

### CFDWindow
Main application window managing the entire simulation workflow.

**Responsibilities:**
- UI layout and tab management
- User input validation
- Solver configuration
- Result display
- File I/O operations

**Key Methods:**
- `update_mesh()`: Regenerates mesh from current settings
- `calculate_nanofluid_properties()`: Computes thermophysical properties
- `run_simulation()`: Launches solver in background thread
- `apply_boundary_conditions()`: Sets up BCs on mesh
- `display_results()`: Shows simulation output

### MeshCanvas
Custom Qt widget for visualizing computational mesh and boundary regions.

**Features:**
- Structured grid rendering
- Color-coded boundary regions
- Automatic scaling to widget size
- Coordinate transformation (mesh → screen)

**Rendering:**
```python
# Mesh coordinates → Screen coordinates
sx = margin + (x - x_min) * scale
sy = height - margin - (y - y_min) * scale
```

### SolverThread
Background thread for non-blocking CFD solver execution.

**Signals:**
- `progress_update`: Emits iteration count and messages
- `finished_signal`: Completion status and results

**Benefits:**
- Responsive GUI during long simulations
- User can monitor/stop execution
- Progress updates in real-time

## User Workflow

```
1. Configure Geometry
   ├─ Set domain dimensions (L × H)
   ├─ Choose mesh resolution (nx × ny)
   └─ Preview mesh

2. Select Nanofluid
   ├─ Base fluid (water, EG)
   ├─ Nanoparticle type (Al2O3, CuO, etc.)
   ├─ Volume fraction (φ)
   ├─ Temperature (T)
   └─ Calculate properties

3. Set Boundary Conditions
   ├─ Inlet: velocity, temperature
   ├─ Outlet: pressure
   └─ Walls: type and temperature

4. Configure Solver
   ├─ Max iterations
   ├─ Convergence tolerance
   ├─ Linear solver
   ├─ Turbulence model
   └─ Under-relaxation factors

5. Run Simulation
   ├─ Click "Run"
   ├─ Monitor progress
   └─ Wait for convergence

6. Analyze Results
   ├─ View statistics
   ├─ Check convergence
   └─ Export data
```

## Technical Details

### Mesh Generation
```python
mesh = StructuredMesh2D(
    x_range=(0.0, length),
    y_range=(0.0, height),
    nx=nx,
    ny=ny
)
```

### Boundary Region Identification
```python
# Inlet cells (left boundary)
inlet = [i for i in range(n_cells) 
         if cell_centers[i, 0] < tolerance]

# Outlet cells (right boundary)
outlet = [i for i in range(n_cells)
          if abs(cell_centers[i, 0] - x_max) < tolerance]

# Wall cells (top/bottom)
walls = [i for i in range(n_cells)
         if cell_centers[i, 1] < tolerance or 
            abs(cell_centers[i, 1] - y_max) < tolerance]
```

### Solver Configuration
```python
settings = SolverSettings(
    max_iterations=max_iter,
    convergence_tol=tolerance,
    linear_solver=solver_type,
    turbulence_model=turb_model,
    under_relaxation_u=alpha_u,
    under_relaxation_v=alpha_v,
    under_relaxation_p=alpha_p,
    under_relaxation_T=alpha_T
)
```

### Thread-Safe Execution
```python
# Create solver thread
thread = SolverThread(solver, max_iterations)
thread.progress_update.connect(on_progress)
thread.finished_signal.connect(on_finished)
thread.start()

# Signals ensure GUI updates happen on main thread
```

## Design Patterns

### Model-View-Controller (MVC)
- **Model**: CFD solver, mesh, nanofluid simulator
- **View**: Qt widgets, mesh canvas, tabs
- **Controller**: CFDWindow coordinating interactions

### Observer Pattern
- Solver thread emits signals
- GUI slots receive updates
- Decoupled communication

### Strategy Pattern
- Interchangeable linear solvers
- Different boundary condition types
- Flexible turbulence models

## Performance Considerations

### Responsive GUI
- Long computations in QThread
- Progress updates every N iterations
- Non-blocking UI during solve

### Memory Management
- Mesh regenerated only when needed
- Results cleared before new simulation
- Qt parent-child ownership for widgets

### Validation
- Input ranges enforced by spin boxes
- Mesh size limits prevent crashes
- Property calculation error handling

## Color Coding

Boundary regions visualized with distinct colors:

| Region | Color | RGB |
|--------|-------|-----|
| Inlet | Blue | (0, 0, 255) |
| Outlet | Red | (255, 0, 0) |
| Wall | Gray | (128, 128, 128) |
| Heated Wall | Orange | (255, 165, 0) |
| Symmetry | Green | (0, 255, 0) |

## Error Handling

```python
try:
    # Simulation logic
    mesh = create_mesh()
    solver = setup_solver()
    solver.solve()
except ValueError as e:
    QMessageBox.warning(self, "Input Error", str(e))
except RuntimeError as e:
    QMessageBox.critical(self, "Solver Error", str(e))
except Exception as e:
    log(f"Unexpected error: {str(e)}")
    QMessageBox.critical(self, "Error", "Simulation failed")
```

## Integration Points

### With Core Simulator
```python
from nanofluid_simulator.cfd_mesh import StructuredMesh2D
from nanofluid_simulator.cfd_solver import NavierStokesSolver
from nanofluid_simulator.simulator import NanofluidSimulator
```

### With Post-Processing
```python
from nanofluid_simulator.cfd_postprocess import FlowPostProcessor

processor = FlowPostProcessor(mesh, field)
vorticity = processor.calculate_vorticity()
```

### With Export
```python
# Export to text file
with open(filename, 'w') as f:
    f.write(results_text)
    f.write(convergence_history)
```

## Extension Points

### Adding New Boundary Conditions
1. Add combo box option in boundary tab
2. Implement BC application in `apply_boundary_conditions()`
3. Add color coding in `MeshCanvas.paintEvent()`
4. Update documentation

### Adding Visualization
1. Create new tab in results section
2. Use matplotlib FigureCanvas widget
3. Plot field variables (u, v, p, T)
4. Add export option

### Custom Solvers
1. Extend `SolverThread` with new solver type
2. Add selection in solver settings tab
3. Update `run_simulation()` method
4. Maintain signal compatibility

## Best Practices

1. **Always validate inputs** before starting simulation
2. **Use QThread** for long computations
3. **Emit signals** for cross-thread communication
4. **Handle exceptions** at every level
5. **Provide user feedback** via status log
6. **Test with small meshes** first
7. **Document assumptions** in tooltips
8. **Export results** for reproducibility

## Future Enhancements

- [ ] 3D visualization of results
- [ ] Interactive field plotting
- [ ] Animation of time-dependent solutions
- [ ] Mesh refinement controls
- [ ] Batch simulation mode
- [ ] Parameter sensitivity analysis
- [ ] Optimization integration
- [ ] Cloud computation support

---

**Note**: This GUI is designed for educational and research use. For production applications, consider additional validation, error recovery, and workflow management features.
