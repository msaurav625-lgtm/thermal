# Examples Directory

Comprehensive collection of nanofluid simulation examples covering various applications and use cases.

## Quick Start

Run any example:
```bash
python examples/example_1_basic.py
```

## Example Categories

### 🎯 Basic Usage (Examples 1-3)
**Purpose**: Learn fundamental simulator capabilities

- **example_1_basic.py** - Basic thermal conductivity calculation
  - Single nanoparticle type
  - Property calculation
  - Model comparison

- **example_2_hybrid.py** - Hybrid nanofluids
  - Multiple nanoparticle types
  - Synergistic effects
  - Optimization

- **example_3_parametric.py** - Parametric studies
  - Volume fraction sweep
  - Temperature effects
  - Diameter effects

### 🔬 Advanced Models (Examples 4-7)
**Purpose**: Explore advanced physics and models

- **example_4_interfacial.py** - Interfacial layer effects
  - Nanolayer modeling
  - Enhanced conductivity
  - Interface resistance

- **example_5_flow_physics.py** - Flow regime analysis
  - Laminar vs turbulent
  - Flow patterns
  - Performance metrics

- **example_6_solver_modes.py** - Different solution methods
  - Direct methods
  - Iterative methods
  - Convergence analysis

- **example_7_ai_recommendations.py** - AI-powered optimization
  - Intelligent recommendations
  - Performance prediction
  - Design guidance

### 💻 CFD Simulations (Examples 8-10)
**Purpose**: Computational fluid dynamics applications

- **example_8_cfd_demo.py** - Channel flow simulation
  - 2D finite volume method
  - SIMPLE algorithm
  - Nanofluid integration
  - ~2 minutes runtime

- **example_9_postprocessing.py** - Advanced visualization
  - Lid-driven cavity flow
  - Vorticity analysis
  - Publication-quality plots
  - ~3 minutes runtime

- **example_10_validation.py** - Solver validation
  - Analytical solutions (Poiseuille flow)
  - Benchmark data (Ghia et al. 1982)
  - Error analysis
  - ~5 minutes runtime

### 🏭 Engineering Applications (Examples 11-13)
**Purpose**: Real-world engineering problems

- **example_11_heat_exchanger.py** - Heat exchanger design
  - Parallel flow configuration
  - Effectiveness analysis
  - Performance comparison
  - Optimization guidance

- **example_12_microchannel_cooling.py** - Electronics cooling
  - High heat flux (100 kW/m²)
  - Microchannel geometry
  - Thermal resistance
  - Figure of merit (FOM)

- **example_13_natural_convection.py** - Buoyancy-driven flows
  - Differentially heated cavity
  - Rayleigh number effects
  - Passive cooling applications
  - Building thermal management

## Usage Guide

### Running Examples

**Single example:**
```bash
python examples/example_1_basic.py
```

**With output redirection:**
```bash
python examples/example_10_validation.py > validation_results.txt
```

**Time-intensive examples:**
```bash
# CFD examples may take 2-5 minutes
python examples/example_9_postprocessing.py
```

### Example Output

Most examples generate:
- **Console output**: Results, analysis, recommendations
- **Plots**: PNG images (if matplotlib available)
- **Reports**: Markdown files (validation, analysis)

### Customization

Each example is self-contained and can be modified:

```python
# Change nanoparticle type
nanoparticle=Nanoparticle.CUO  # instead of AL2O3

# Adjust volume fraction
volume_fraction=0.05  # 5% instead of default

# Modify temperature
temperature=350.0  # K

# Change mesh resolution (CFD examples)
nx, ny = 80, 60  # finer mesh
```

## Example Selection Guide

| Your Goal | Recommended Examples |
|-----------|---------------------|
| Learn basics | 1 → 2 → 3 |
| Understand models | 4 → 5 → 6 |
| Use CFD | 8 → 9 → 10 |
| Design heat exchanger | 11 |
| Cool electronics | 12 |
| Passive cooling | 13 |
| Validate solver | 10 |
| Optimize nanofluid | 7 → 3 |

## Performance Notes

### Fast Examples (<1 second)
- Examples 1-7: Property calculations

### Medium Examples (1-2 minutes)
- Example 8: Basic CFD

### Slow Examples (3-5 minutes)
- Example 9: Lid-driven cavity
- Example 10: Validation suite
- Examples 11-13: Engineering applications

## Output Files

Examples may generate:

- **Images**: `*.png` (visualization plots)
- **Reports**: `VALIDATION_REPORT.md`
- **Data**: CSV files (some examples)
- **Logs**: Convergence data

## Dependencies

**Minimum (Examples 1-7)**:
```
numpy
scipy
matplotlib (optional, for plots)
```

**Full (Examples 8-13 - CFD)**:
```
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.5
```

## Troubleshooting

**"Module not found":**
```bash
# Ensure you're in project root
cd /path/to/nanofluid_simulator
python examples/example_1_basic.py
```

**"Simulation not converging":**
- Increase `max_iterations` in SolverSettings
- Reduce under-relaxation factors
- Refine mesh (increase nx, ny)
- Check boundary conditions

**"Out of memory":**
- Reduce mesh resolution
- Use coarser grid for initial testing
- Close other applications

**Slow performance:**
- Normal for CFD examples (2-5 minutes)
- Check Task Manager/Activity Monitor
- Use smaller domains for testing

## Advanced Usage

### Batch Processing

Run multiple examples:
```bash
for i in {1..7}; do
    python examples/example_${i}_*.py
done
```

### Parallel Execution

CFD examples are CPU-bound. Run different cases in parallel:
```bash
# Terminal 1
python examples/example_11_heat_exchanger.py &

# Terminal 2
python examples/example_12_microchannel_cooling.py &
```

### Custom Analysis

Modify examples for your research:

1. Copy example file
2. Adjust parameters
3. Add custom analysis
4. Save results

Example:
```python
# my_custom_simulation.py
from examples.example_8_cfd_demo import *

# Modify parameters
phi_values = [0.01, 0.03, 0.05, 0.07, 0.10]

# Run custom analysis
for phi in phi_values:
    # ... simulation code ...
    save_results(phi, results)
```

## Research Applications

### Publications

For research papers, use:
- **Example 10**: Validation (establishes accuracy)
- **Examples 8-9**: CFD methodology
- **Examples 11-13**: Application-specific results

### Thesis Work

Comprehensive study:
1. Run Examples 1-3 (characterization)
2. Run Example 10 (validation)
3. Run Examples 11-13 (applications)
4. Use generated plots and data

### Teaching

Classroom demonstrations:
- Examples 1-3: Lecture on nanofluids
- Example 8: CFD introduction
- Example 9: Flow visualization
- Examples 11-13: Engineering applications

## Contributing

Add new examples following this structure:

```python
"""
Example XX: Title

Brief description.
Application: ...

Demonstrates:
- Feature 1
- Feature 2
"""

import numpy as np
# ... imports ...

print("="*80)
print("EXAMPLE TITLE")
print("="*80)

# Configuration
# ...

# Main simulation
# ...

# Results
print("\n📊 Results:")
# ...

# Conclusions
print("\n✅ Key Findings:")
# ...
```

## Support

- **Documentation**: See `docs/USER_GUIDE.md`
- **Issues**: Report bugs on GitHub
- **Questions**: Check `TROUBLESHOOTING.md`

## License

MIT License - See LICENSE.txt

## Citation

If you use these examples in research, please cite:
```
@software{nanofluid_simulator_2025,
  author = {Your Name},
  title = {Nanofluid Thermal Conductivity Simulator},
  year = {2025},
  version = {4.0},
  url = {https://github.com/yourusername/nanofluid-simulator}
}
```

---

**Happy Simulating! 🚀**

Questions? See `docs/USER_GUIDE.md` or `TROUBLESHOOTING.md`
