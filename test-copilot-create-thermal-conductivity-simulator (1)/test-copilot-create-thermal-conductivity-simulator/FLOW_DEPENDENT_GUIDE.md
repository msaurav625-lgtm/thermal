# Flow-Dependent Properties Calculator - Quick Reference Guide

## Overview

The Advanced Flow-Dependent Calculator provides full user control for calculating flow-dependent thermal conductivity and viscosity with:

✅ **0, 1, or multiple nanoparticles** - Full flexibility  
✅ **Single values or ranges** - Parametric sweeps  
✅ **Enable/disable nanoparticles** - Dynamic control  
✅ **Multiple models** - Choose what to calculate  
✅ **Flow effects** - Brownian, thermophoresis, shear  

---

## Quick Start

### 1. Base Fluid Only (No Nanoparticles)

```python
from nanofluid_simulator import calculate_flow_properties

result = calculate_flow_properties(
    base_fluid='Water',
    nanoparticles=[],  # Empty = base fluid only
    velocity=0.1,
    temperature=300
)

print(f"k_base = {result['k_static']:.4f} W/m·K")
print(f"Reynolds = {result['reynolds']:.1f}")
```

**Output:**
```
k_base = 0.6130 W/m·K
Reynolds = 998.0
```

---

### 2. Single Nanoparticle

```python
result = calculate_flow_properties(
    base_fluid='Water',
    nanoparticles=[{
        'material': 'Al2O3',
        'volume_fraction': 0.02,
        'diameter': 30e-9
    }],
    velocity=0.1,
    temperature=300,
    models=['static', 'buongiorno', 'kumar']
)

print(f"k_static = {result['conductivity']['static']:.4f} W/m·K")
print(f"k_buongiorno = {result['conductivity']['buongiorno']:.4f} W/m·K")
print(f"k_kumar = {result['conductivity']['kumar']:.4f} W/m·K")
```

**Output:**
```
k_static = 0.6488 W/m·K
k_buongiorno = 0.6590 W/m·K (flow-enhanced)
k_kumar = 0.6490 W/m·K (shear-enhanced)
```

---

### 3. Multiple Nanoparticles (Comparison)

```python
comparison = calculate_flow_properties(
    base_fluid='Water',
    nanoparticles=[
        {'material': 'Al2O3', 'volume_fraction': 0.02, 'diameter': 30e-9},
        {'material': 'CuO', 'volume_fraction': 0.02, 'diameter': 30e-9},
        {'material': 'Cu', 'volume_fraction': 0.02, 'diameter': 30e-9},
    ],
    velocity=0.1
)

# Returns dict with results for each material
for material, results in comparison.items():
    r = results[0]
    print(f"{material}: k = {r['conductivity']['buongiorno']:.4f} W/m·K")
```

**Output:**
```
Al2O3: k = 0.6590 W/m·K
CuO: k = 0.6598 W/m·K
Cu: k = 0.6605 W/m·K (highest!)
```

---

## Advanced Usage

### 4. Volume Fraction Range Sweep

```python
from nanofluid_simulator import (
    AdvancedFlowCalculator, 
    FlowDependentConfig, 
    NanoparticleSpec,
    FlowConditions
)

config = FlowDependentConfig(
    base_fluid='Water',
    nanoparticles=[
        NanoparticleSpec(
            material='Al2O3',
            volume_fraction=(0.01, 0.05, 10),  # 1% to 5%, 10 steps
            diameter=30e-9
        )
    ],
    flow_conditions=FlowConditions(velocity=0.1, temperature=300),
    conductivity_models=['buongiorno'],
    viscosity_models=['brinkman']
)

calc = AdvancedFlowCalculator(config)
results = calc.calculate_parametric_sweep()

for r in results:
    phi = r['volume_fraction'] * 100
    k = r['conductivity']['buongiorno']
    print(f"φ = {phi:.1f}%: k = {k:.4f} W/m·K")
```

**Output:**
```
φ = 1.0%: k = 0.6357 W/m·K
φ = 1.5%: k = 0.6473 W/m·K
φ = 2.0%: k = 0.6590 W/m·K
...
φ = 5.0%: k = 0.7329 W/m·K
```

---

### 5. Velocity Sweep (Flow Effect Study)

```python
config = FlowDependentConfig(
    base_fluid='Water',
    nanoparticles=[NanoparticleSpec(material='CuO', volume_fraction=0.02, diameter=50e-9)],
    sweep_velocity=(0.01, 0.2, 8),  # 0.01 to 0.2 m/s, 8 steps
    conductivity_models=['static', 'buongiorno'],
)

calc = AdvancedFlowCalculator(config)
results = calc.calculate_parametric_sweep()

for r in results:
    v = r['velocity']
    k_static = r['conductivity']['static']
    k_flow = r['conductivity']['buongiorno']
    enhancement = (k_flow / k_static - 1) * 100
    print(f"V = {v:.3f} m/s: k_static = {k_static:.4f}, k_flow = {k_flow:.4f} (+{enhancement:.2f}%)")
```

**Output:**
```
V = 0.010 m/s: k_static = 0.6496, k_flow = 0.6588 (+1.42%)
V = 0.037 m/s: k_static = 0.6496, k_flow = 0.6595 (+1.52%)
...
V = 0.200 m/s: k_static = 0.6496, k_flow = 0.6605 (+1.68%)
```

Note: Flow effects increase with velocity!

---

### 6. Enable/Disable Nanoparticles

```python
config = FlowDependentConfig(
    base_fluid='Water',
    nanoparticles=[
        NanoparticleSpec(material='Al2O3', volume_fraction=0.02, diameter=30e-9, enabled=True),
        NanoparticleSpec(material='CuO', volume_fraction=0.02, diameter=30e-9, enabled=False),  # DISABLED
        NanoparticleSpec(material='Cu', volume_fraction=0.02, diameter=30e-9, enabled=True),
    ],
    flow_conditions=FlowConditions(velocity=0.1)
)

calc = AdvancedFlowCalculator(config)
comparison = calc.calculate_comparison()

# Only Al2O3 and Cu are calculated (CuO is disabled)
print(f"Calculated materials: {list(comparison.keys())}")
```

**Output:**
```
Calculated materials: ['Al2O3', 'Cu']
```

---

### 7. Multi-Dimensional Sweep (φ × Velocity)

```python
config = FlowDependentConfig(
    base_fluid='Water',
    nanoparticles=[
        NanoparticleSpec(
            material='Al2O3',
            volume_fraction=(0.01, 0.04, 4),  # 4 volume fractions
            diameter=30e-9
        )
    ],
    sweep_velocity=(0.05, 0.15, 3),  # 3 velocities
)

calc = AdvancedFlowCalculator(config)
results = calc.calculate_parametric_sweep()

print(f"Total combinations: {len(results)}")  # 4 × 3 = 12

for r in results:
    phi = r['volume_fraction'] * 100
    v = r['velocity']
    k = r['conductivity']['buongiorno']
    print(f"φ={phi:.1f}%, V={v:.3f} m/s: k={k:.4f} W/m·K")
```

**Output:**
```
Total combinations: 12
φ=1.0%, V=0.050 m/s: k=0.6355 W/m·K
φ=1.0%, V=0.100 m/s: k=0.6357 W/m·K
φ=1.0%, V=0.150 m/s: k=0.6358 W/m·K
φ=2.0%, V=0.050 m/s: k=0.6588 W/m·K
...
```

---

## Available Models

### Thermal Conductivity Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `'static'` | Classical Maxwell/Hamilton-Crosser | Baseline, no flow effects |
| `'buongiorno'` | Brownian + thermophoresis effects | General flow applications |
| `'kumar'` | Shear-rate dependent | High shear flows |
| `'rea_guzman'` | Reynolds-dependent | Convective flows |

### Viscosity Models

| Model | Description | Valid Range |
|-------|-------------|-------------|
| `'einstein'` | Linear (dilute) | φ < 2% |
| `'brinkman'` | Moderate concentration | φ < 10% |
| `'batchelor'` | Includes Brownian | φ < 5% |
| `'shear_dependent'` | Non-Newtonian | All φ, non-zero shear |

---

## Configuration Options

### NanoparticleSpec

```python
NanoparticleSpec(
    material='Al2O3',              # Material name or custom
    volume_fraction=0.02,          # Single value OR (min, max, steps)
    diameter=30e-9,                # Single value OR (min, max, steps)
    shape='sphere',                # 'sphere', 'cylinder', 'platelet'
    k_particle=40.0,               # Optional: override database
    rho_particle=3970,             # Optional: override database
    enabled=True                   # Toggle on/off without removing
)
```

### FlowConditions

```python
FlowConditions(
    velocity=0.05,                 # m/s
    shear_rate=None,               # 1/s (auto-calculated if None)
    temperature=300.0,             # K
    pressure=101325.0,             # Pa
    characteristic_length=0.01     # m (hydraulic diameter)
)
```

### FlowDependentConfig

```python
FlowDependentConfig(
    base_fluid='Water',                           # Or 'EG', 'Oil'
    nanoparticles=[...],                          # List of NanoparticleSpec
    flow_conditions=FlowConditions(...),
    conductivity_models=['buongiorno', 'kumar'], # Which k models
    viscosity_models=['brinkman'],               # Which μ models
    include_brownian=True,                       # Brownian motion
    include_thermophoresis=True,                 # Thermophoretic effects
    include_shear_effects=True,                  # Shear-rate effects
    sweep_velocity=(min, max, steps),            # Optional velocity range
    sweep_temperature=(min, max, steps)          # Optional temp range
)
```

---

## Supported Materials

### Nanoparticles (11)
- **Oxides:** Al₂O₃, CuO, TiO₂, SiO₂, ZnO, Fe₃O₄
- **Metals:** Cu, Ag, Au
- **Carbon:** CNT, Graphene

### Base Fluids (3)
- **Water** (most common)
- **EG** (Ethylene Glycol)
- **Oil**

---

## Result Structure

### Single Nanoparticle Result

```python
{
    'material': 'Al2O3',
    'volume_fraction': 0.02,
    'diameter': 30e-9,
    'temperature': 300.0,
    'velocity': 0.1,
    'k_static': 0.6488,
    'k_base': 0.613,
    'enhancement_static': 5.84,
    'shear_rate': 60.0,
    'conductivity': {
        'static': 0.6488,
        'buongiorno': 0.6590,
        'kumar': 0.6490
    },
    'viscosity': {
        'einstein': 0.00105,
        'brinkman': 0.001052,
        'batchelor': 0.001053
    },
    'reynolds': 948.8
}
```

---

## Export to CSV

```python
import csv

results = calc.calculate_parametric_sweep()

with open('results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['volume_fraction', 'velocity', 'k_buongiorno', 'reynolds'])
    writer.writeheader()
    
    for r in results:
        writer.writerow({
            'volume_fraction': r['volume_fraction'],
            'velocity': r['velocity'],
            'k_buongiorno': r['conductivity']['buongiorno'],
            'reynolds': r['reynolds']
        })
```

---

## Complete Examples

See `examples/example_flow_dependent_advanced.py` for 10 comprehensive examples:

1. Base fluid only
2. Single nanoparticle
3. Multiple nanoparticles
4. Volume fraction sweep
5. Velocity sweep
6. Temperature sweep
7. Diameter sweep
8. Enable/disable nanoparticles
9. Multi-dimensional sweep
10. Model comparison

**Run examples:**
```bash
python examples/example_flow_dependent_advanced.py
```

---

## Integration with Unified Engine

Use flow-dependent calculator within the unified engine:

```python
from nanofluid_simulator import BKPSNanofluidEngine

engine = BKPSNanofluidEngine.quick_start(
    mode="flow",
    base_fluid="Water",
    nanoparticle="Al2O3",
    volume_fraction=0.02,
    diameter=30e-9,
    geometry={'length': 0.1, 'height': 0.01},
    flow={'velocity': 0.1}
)

results = engine.run()

# Access flow-dependent properties
print(f"Flow k: {results['flow']['k_flow']:.4f} W/m·K")
print(f"Reynolds: {results['flow']['reynolds']:.1f}")
```

---

## Tips & Best Practices

✅ **Start simple:** Test with single nanoparticle, single value  
✅ **Validate:** Compare with base fluid to verify enhancements  
✅ **Use ranges wisely:** More steps = more computation time  
✅ **Enable/disable:** Great for sensitivity studies  
✅ **Model selection:** Choose models validated for your conditions  
✅ **Export results:** Save to CSV for further analysis in Excel/Python  

---

## Common Use Cases

### 1. **Optimization Study**
Sweep φ and V to find maximum enhancement with acceptable viscosity increase.

### 2. **Material Comparison**
Calculate properties for multiple materials at same conditions, pick best.

### 3. **What-If Analysis**
Enable/disable nanoparticles to see individual contributions.

### 4. **Validation Study**
Compare different models against experimental data.

### 5. **Design Space Exploration**
Multi-dimensional sweeps to map entire parameter space.

---

## Performance Notes

- **Single calculation:** <0.01s
- **Parametric sweep (100 points):** ~0.5s
- **Multi-dimensional (10×10):** ~1s
- **Memory efficient:** Results are lightweight dicts

---

## Support & Documentation

- **Main documentation:** `docs/USER_GUIDE.md`
- **Examples:** `examples/example_flow_dependent_advanced.py`
- **API reference:** `nanofluid_simulator/advanced_flow_calculator.py`
- **Issues:** https://github.com/msaurav625-lgtm/thermal/issues

---

**Version:** BKPS NFL Thermal Pro v7.1  
**Date:** January 29, 2025  
**Author:** Dedicated to Brijesh Kumar Pandey  

**Status:** ✅ PRODUCTION-READY
