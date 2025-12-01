# BKPS NFL Thermal Pro - AI Coding Agent Instructions

## Project Overview

**BKPS NFL Thermal Pro 7.0** is a research-grade nanofluid thermal analysis platform combining static property calculations, flow-dependent physics, and full CFD simulation. The project is dedicated to Brijesh Kumar Pandey and validated against 6 experimental datasets (72.7% accuracy within ±20%, MAE=14.93%).

**Core Path**: `test-copilot-create-thermal-conductivity-simulator (1)/test-copilot-create-thermal-conductivity-simulator/`

## Architecture & Entry Points

### Three-Tier Simulator Architecture

1. **Unified Engine (v7.0)** - `nanofluid_simulator/unified_engine.py`
   - `BKPSNanofluidEngine` - Single entry point for all modes
   - `UnifiedConfig` dataclass system with validation
   - Quick-start factory: `BKPSNanofluidEngine.quick_start(mode="static", nanoparticle="Al2O3", ...)`
   - Lazy initialization - only loads required physics engines

2. **Legacy Simulators** - Still actively used
   - `BKPSNanofluidSimulator` (`integrated_simulator_v6.py`) - Advanced physics with DLVO theory, non-Newtonian rheology
   - `FlowNanofluidSimulator` (`flow_simulator.py`) - Flow-dependent properties
   - `EnhancedNanofluidSimulator` (`enhanced_simulator.py`) - Base class with common functionality

3. **CFD Solvers** - `nanofluid_simulator/cfd_solver.py`
   - `NavierStokesSolver` - SIMPLE algorithm implementation
   - Coupled with `StructuredMesh2D` from `cfd_mesh.py`

### Entry Point: `main.py`

```python
# CLI mode
python main.py --cli --mode static --nanoparticle Al2O3 --volume-fraction 0.02

# GUI mode (default)
python main.py  # launches bkps_professional_gui_v7.py
```

## Critical Developer Workflows

### Running Simulations

**v7.0 Unified API (Preferred)**:
```python
from nanofluid_simulator import BKPSNanofluidEngine

engine = BKPSNanofluidEngine.quick_start(
    mode="static",  # or "flow", "cfd", "hybrid"
    nanoparticle="Al2O3",
    volume_fraction=0.02,
    temperature=300
)
results = engine.run()  # Returns dict with all properties
engine.export_results("output.json")
```

**Configuration-based workflow**:
```python
from nanofluid_simulator.unified_engine import UnifiedConfig, create_static_config

config = create_static_config(base_fluid="Water", nanoparticle="CuO", ...)
config.save("project.json")  # Save for later
loaded_config = UnifiedConfig.load("project.json")
engine = BKPSNanofluidEngine(loaded_config)
```

### Testing

```bash
# Run end-to-end tests (no pytest setup required)
python test_end_to_end.py

# Unit tests exist in tests/ but use direct Python execution
python tests/test_simulator.py

# Validation against research data
python validate_against_research.py
```

### GUI Development

The GUI (`bkps_professional_gui_v7.py`) uses PyQt6 with:
- Threaded computation via `ComputationThread` (avoid UI freezing)
- Dark theme support with `QPalette` customization
- Dockable panels for flexible layout
- Auto-save functionality via `QSettings`

## Project-Specific Conventions

### Physics Model Selection

**Thermal Conductivity Models** (25+ available in `models.py` and `advanced_models.py`):
- Maxwell - Classical baseline
- Hamilton-Crosser - Shape factor support
- BKPS-enhanced - Project's validated model
- Access via: `simulator.calculate_thermal_conductivity(model='Maxwell')`

### Nanoparticle Materials Database

11 materials in `nanofluid_simulator/nanoparticles.py`:
- Oxides: Al₂O₃, CuO, TiO₂, SiO₂, ZnO, Fe₃O₄
- Metals: Cu, Ag, Au
- Carbon: CNT, Graphene

Access pattern:
```python
from nanofluid_simulator import NanoparticleDatabase
db = NanoparticleDatabase()
props = db.get_properties('Al2O3')  # Returns dict with k, rho, cp
```

### Configuration Validation Pattern

All config classes implement `.validate()` returning `(bool, List[str])`:
```python
valid, errors = config.validate()
if not valid:
    for error in errors:
        logger.error(error)
```

### Result Dictionary Structure

Unified engine returns nested dicts by mode:
```python
results = {
    'static': {
        'k_base': float,
        'k_static': float,
        'enhancement_k': float,  # percentage
        'mu_base': float,
        'mu_nf': float,
        'viscosity_ratio': float
    },
    'metadata': {...}
}
```

## Integration Points & Data Flow

### Physics Engine Integration

```
BKPSNanofluidEngine
├── Mode = STATIC → BKPSNanofluidSimulator
├── Mode = FLOW → FlowNanofluidSimulator
├── Mode = CFD → NavierStokesSolver + BKPSNanofluidSimulator
└── Mode = HYBRID → All engines
```

Engines are lazy-loaded on first `engine.run()` call.

### GUI ↔ Engine Communication

```
User Input (GUI) 
  → UnifiedConfig dataclass
  → ComputationThread (background)
  → BKPSNanofluidEngine.run()
  → pyqtSignal(results)
  → GUI update
```

**Critical**: Always use `ComputationThread` for long-running simulations to prevent UI freezing.

### Export System

Three export formats via `nanofluid_simulator/export.py`:
- JSON: `engine.export_results("file.json", format='json')`
- CSV: Tabular data for spreadsheets
- HDF5: Large CFD datasets

PDF reports: `PDFReportGenerator` from `pdf_report.py` generates publication-quality outputs.

## Key Files & Their Purpose

### Physics Core
- `models.py` - Classical thermal conductivity models (Maxwell, Hamilton-Crosser, etc.)
- `advanced_models.py` - Research models (Koo-Kleinstreuer, hybrid correlations)
- `dlvo_theory.py` - Particle interaction physics (van der Waals, electrostatic)
- `flow_dependent_conductivity.py` - Buongiorno, Kumar shear-enhanced models
- `non_newtonian_viscosity.py` - 7 viscosity models (Power-Law, Carreau-Yasuda, etc.)

### Solvers
- `cfd_solver.py` - SIMPLE algorithm, ~1200 lines
- `cfd_mesh.py` - Structured mesh generation
- `cfd_postprocess.py` - Flow visualization, vorticity analysis
- `cfd_turbulence.py` - k-ε, k-ω SST models

### AI Features
- `ai_recommender.py` - ML-based optimization (scikit-learn optional)
- `ai_cfd_integration.py` - Flow regime classification, convergence prediction

### Validation
- `validation_suite.py` - Framework for experimental comparison
- `validation_center.py` - v7.0 validation reporting
- `validate_against_research.py` - Runnable validation script

## Build & Deployment

### Dependencies
```bash
# Core (required)
pip install numpy scipy matplotlib

# GUI (optional but common)
pip install PyQt6

# Acceleration (optional)
pip install numba  # JIT compilation
```

No complex build system - pure Python execution.

### Windows Executable
```bash
# Build standalone .exe (uses build_v7_installer.py)
python build_v7_installer.py
```

Creates ~150-200MB self-contained executable with PyInstaller.

## Common Pitfalls & Solutions

### 1. Import Path Issues
The nested folder structure requires careful path management:
```python
# Always use from project root
from nanofluid_simulator import BKPSNanofluidEngine  # ✓ Correct

# Not:
from test-copilot-create-thermal-conductivity-simulator.nanofluid_simulator import ...  # ✗ Wrong
```

### 2. Simulator Initialization Order
Legacy simulators require explicit particle addition:
```python
sim = BKPSNanofluidSimulator(base_fluid="Water", temperature=300)
sim.add_nanoparticle(material="Al2O3", ...)  # Must call before calculations
```

### 3. CFD Convergence
Default solver settings are conservative. For faster iteration:
```python
solver_config = SolverConfig(
    max_iterations=500,  # Default: 1000
    convergence_tolerance=1e-5,  # Default: 1e-6
    relaxation_factor=0.8  # Default: 0.7 (higher = faster but less stable)
)
```

### 4. GUI Threading
Never call `engine.run()` directly in GUI event handlers:
```python
# ✓ Correct
thread = ComputationThread(engine)
thread.finished.connect(self.on_results)
thread.start()

# ✗ Wrong - freezes UI
results = engine.run()
```

### 5. Property Units
Be explicit about units (inconsistencies exist):
- Temperature: Kelvin (K)
- Diameter: meters (m) - convert nm via `diameter_nm * 1e-9`
- Pressure: Pascal (Pa)
- Thermal conductivity: W/m·K
- Viscosity: Pa·s (convert to mPa·s via `* 1000`)

## Examples Location

`examples/` contains 18 working scripts:
- `example_1_basic.py` - Quick start
- `example_7_quick_demo.py` - v7.0 unified API demo
- `example_8_cfd_demo.py` - CFD simulation (~2 min runtime)
- `example_16_ai_cfd_integration.py` - AI features

Run from project root: `python examples/example_1_basic.py`

## Documentation Files

- `CHANGELOG_V7.md` - v7.0 release notes, migration guide
- `V7_DELIVERY_SUMMARY.md` - Complete v7.0 feature documentation
- `RESEARCH_VALIDATION_SUMMARY.md` - Scientific validation details
- `VALIDATION_QUICK_REF.txt` - Expected accuracy by system type
- `docs/USER_GUIDE.md` - User-facing documentation
- `examples/README.md` - Example selection guide

## Version History Context

- **v7.0** (Current) - Unified API, validation center, project save/load
- **v6.0** - Flow-dependent conductivity, DLVO theory, non-Newtonian viscosity
- **v4.0-v5.0** - CFD solver, AI integration, GUI enhancements
- **v3.0** - Enhanced simulator with 25+ models

When extending features, maintain backward compatibility with v6.0 `BKPSNanofluidSimulator` API.

## Performance Considerations

- **Static calculations**: <0.1s per config
- **Flow simulations**: ~0.5s
- **CFD (50x50 mesh)**: 5-30s depending on convergence
- **Validation suite**: ~2-5 min (6 datasets)

Acceleration options (see `performance.py`):
- Numba JIT: 2-5x speedup
- Vectorized ops: Use `optimized_ops.py` for NumPy best practices
- Multi-threading: Plan for v7.1

## Contributing Patterns

When adding new physics models:
1. Add to appropriate module (`models.py` or `advanced_models.py`)
2. Update `__init__.py` exports
3. Add validation data if available (`validation_suite.py`)
4. Create example script in `examples/`
5. Update `CHANGELOG_V7.md`

Follow existing function signatures:
```python
def new_model(k_base, k_particle, volume_fraction, **kwargs) -> float:
    """
    Brief description with reference.
    
    Args:
        k_base: Base fluid conductivity (W/m·K)
        k_particle: Particle conductivity (W/m·K)
        volume_fraction: Volume fraction (0-1)
    
    Returns:
        Effective thermal conductivity (W/m·K)
        
    Reference:
        Author et al. (Year), Journal, DOI
    """
```

---

**For questions**: Check `TROUBLESHOOTING.md` or run `python diagnose.py` for system diagnostics.
