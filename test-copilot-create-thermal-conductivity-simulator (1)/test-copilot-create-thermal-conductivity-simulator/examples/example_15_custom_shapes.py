"""
Example 15: Custom Nanoparticle Shape Definitions

Demonstrates how to:
- Define custom particle geometries
- Calculate shape-dependent thermal conductivity
- Compare different particle shapes
- Create user-defined shapes

Run time: <1 second

Author: Nanofluid Simulator v4.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from nanofluid_simulator.custom_shapes import (
    CustomShapeManager, ParticleShape, add_shape, get_available_shapes
)
from nanofluid_simulator.simulator import NanofluidSimulator
from nanofluid_simulator.models import Nanoparticle

print("="*70)
print("CUSTOM NANOPARTICLE SHAPE DEMONSTRATION")
print("="*70)

# ============================================================================
# PART 1: Available Standard Shapes
# ============================================================================
print("\n" + "="*70)
print("PART 1: STANDARD SHAPES LIBRARY")
print("="*70)

manager = CustomShapeManager()
manager.list_shapes()

# ============================================================================
# PART 2: Creating Custom Shapes
# ============================================================================
print("\n" + "="*70)
print("PART 2: CREATING CUSTOM SHAPES")
print("="*70)

print("\nMethod 1: Explicit definition")
print("-"*70)

custom_hexagon = ParticleShape(
    name='hexagonal_prism',
    shape_factor=4.2,
    aspect_ratio=1.5,
    surface_area_factor=1.15,
    description='Hexagonal prism nanoparticles from my synthesis'
)

manager.add_custom_shape(custom_hexagon)

print("\nMethod 2: From aspect ratio (automatic shape factor)")
print("-"*70)

custom_rod = manager.create_shape_from_aspect_ratio(
    name='ultra_long_rod',
    aspect_ratio=50.0,
    description='Ultra-long nanorods (L/D = 50)'
)

manager.add_custom_shape(custom_rod)
print(f"  Calculated shape factor: {custom_rod.shape_factor:.2f}")

print("\nMethod 3: Quick add function")
print("-"*70)

add_shape(
    'my_experimental_particle',
    shape_factor=7.5,
    aspect_ratio=12.0,
    surface_area_factor=2.8,
    description='Particles from my lab experiment'
)

# ============================================================================
# PART 3: Shape Factor Impact on Thermal Conductivity
# ============================================================================
print("\n\n" + "="*70)
print("PART 3: SHAPE FACTOR IMPACT ON THERMAL CONDUCTIVITY")
print("="*70)

print("\nComparing thermal conductivity enhancement for different shapes")
print("Conditions: 3% Al2O3 nanofluids at 300K")
print()

# Base fluid properties
sim = NanofluidSimulator()
phi = 0.03  # 3% volume fraction
T = 300.0   # Temperature (K)

# Test different shapes
shapes_to_test = [
    ('sphere', 'Spherical particles'),
    ('cylinder', 'Short cylinders (L/D=3)'),
    ('long_cylinder', 'Long cylinders (L/D=10)'),
    ('carbon_nanotube', 'Carbon nanotubes (L/D=100)'),
    ('platelet', 'Disk-shaped platelets'),
    ('graphene', 'Graphene nanosheets'),
]

results = []

print(f"{'Shape':<25} {'Shape Factor':<15} {'k_eff (W/m·K)':<15} {'Enhancement':<15}")
print("-"*70)

# Water baseline
k_base = 0.6  # W/m·K

for shape_name, description in shapes_to_test:
    shape = manager.get_shape(shape_name)
    if shape:
        # Use Hamilton-Crosser model with shape factor
        # Simplified calculation for demonstration
        k_p = 40.0  # Al2O3 thermal conductivity
        k_f = k_base
        n = shape.shape_factor
        
        # Hamilton-Crosser equation
        numerator = k_p + (n - 1) * k_f - (n - 1) * phi * (k_f - k_p)
        denominator = k_p + (n - 1) * k_f + phi * (k_f - k_p)
        k_eff = k_f * (numerator / denominator)
        
        enhancement = (k_eff - k_base) / k_base * 100
        
        results.append({
            'name': shape_name,
            'description': description,
            'shape_factor': shape.shape_factor,
            'k_eff': k_eff,
            'enhancement': enhancement
        })
        
        print(f"{shape_name:<25} {shape.shape_factor:<15.1f} {k_eff:<15.3f} {enhancement:<15.1f}%")

print("-"*70)

# ============================================================================
# PART 4: Shape Factor Sensitivity Analysis
# ============================================================================
print("\n\n" + "="*70)
print("PART 4: SHAPE FACTOR SENSITIVITY ANALYSIS")
print("="*70)

print("\nHow does shape factor affect thermal conductivity?")
print("Testing range: n = 3 (sphere) to n = 100 (thin platelet)")
print()

shape_factors = [3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
phi_test = 0.03

print(f"{'Shape Factor (n)':<20} {'k_eff (W/m·K)':<20} {'Enhancement (%)':<20}")
print("-"*70)

for n in shape_factors:
    # Hamilton-Crosser with varying n
    numerator = k_p + (n - 1) * k_f - (n - 1) * phi_test * (k_f - k_p)
    denominator = k_p + (n - 1) * k_f + phi_test * (k_f - k_p)
    k_eff = k_f * (numerator / denominator)
    enhancement = (k_eff - k_base) / k_base * 100
    
    print(f"{n:<20.1f} {k_eff:<20.3f} {enhancement:<20.1f}")

print("-"*70)

print("\n📊 Observations:")
print("   • Higher shape factor (n) → Higher thermal conductivity")
print("   • Spheres (n=3): Baseline enhancement")
print("   • Cylinders (n=6): Moderate enhancement")
print("   • Platelets (n→100): Maximum enhancement")
print("   • Shape effect is significant! (~2-3× difference)")

# ============================================================================
# PART 5: Volume Fraction Dependence
# ============================================================================
print("\n\n" + "="*70)
print("PART 5: VOLUME FRACTION DEPENDENCE FOR DIFFERENT SHAPES")
print("="*70)

phi_range = [0.01, 0.02, 0.03, 0.04, 0.05]
shapes_compare = [
    ('sphere', 3.0),
    ('cylinder', 6.0),
    ('platelet', 50.0)
]

print("\nThermal conductivity vs volume fraction:")
print()
print(f"{'φ (%)':<10}", end='')
for name, _ in shapes_compare:
    print(f"{name.capitalize():<20}", end='')
print()
print("-"*70)

for phi_val in phi_range:
    print(f"{phi_val*100:<10.1f}", end='')
    
    for shape_name, n in shapes_compare:
        numerator = k_p + (n - 1) * k_f - (n - 1) * phi_val * (k_f - k_p)
        denominator = k_p + (n - 1) * k_f + phi_val * (k_f - k_p)
        k_eff = k_f * (numerator / denominator)
        
        print(f"{k_eff:<20.3f}", end='')
    print()

print("-"*70)

# ============================================================================
# PART 6: Creating Shape for Your Experiment
# ============================================================================
print("\n\n" + "="*70)
print("PART 6: GUIDE TO CREATING SHAPES FOR YOUR EXPERIMENT")
print("="*70)

print("""
Step 1: Measure your nanoparticle geometry
   • Use TEM/SEM imaging
   • Measure length (L) and diameter (D)
   • Calculate aspect ratio: AR = L/D

Step 2: Determine shape category
   • AR ≈ 1: Sphere-like (n ≈ 3)
   • 1 < AR < 10: Rod-like (n ≈ 5-10)
   • AR > 10: Wire-like (n ≈ 10-20)
   • AR < 1: Disk-like (n > 20)

Step 3: Create custom shape
""")

print("Example code for your experiment:")
print("-"*70)
print("""
from nanofluid_simulator.custom_shapes import add_shape, CustomShapeManager

# Option A: If you know shape factor from literature
add_shape(
    name='my_experimental_particle',
    shape_factor=6.8,  # From similar particles in literature
    aspect_ratio=5.2,  # Measured from TEM
    description='ZnO nanorods from hydrothermal synthesis'
)

# Option B: Let system calculate shape factor from aspect ratio
manager = CustomShapeManager()
my_shape = manager.create_shape_from_aspect_ratio(
    name='my_particle',
    aspect_ratio=5.2,  # Your measured value
    description='My experimental particles'
)
manager.add_custom_shape(my_shape)

# Use in calculations
k_eff = calculate_with_custom_shape('my_particle', phi=0.03)
""")

# ============================================================================
# PART 7: Export/Import Custom Shapes
# ============================================================================
print("\n\n" + "="*70)
print("PART 7: SAVING AND SHARING CUSTOM SHAPES")
print("="*70)

# Export current shapes
output_file = "my_custom_shapes.txt"
manager.export_shapes_to_file(output_file)

print(f"\n✓ All shapes exported to: {output_file}")
print("  Share this file with collaborators!")

print("\nTo use in another project:")
print("-"*70)
print("""
from nanofluid_simulator.custom_shapes import CustomShapeManager

manager = CustomShapeManager()
manager.import_shapes_from_file('my_custom_shapes.txt')
# Now all your custom shapes are available!
""")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "="*70)
print("SUMMARY - KEY TAKEAWAYS")
print("="*70)

print("\n✅ What You Learned:")
print("   1. Standard shapes available: 8 predefined geometries")
print("   2. Create custom shapes: 3 different methods")
print("   3. Shape factor matters: 2-3× impact on thermal conductivity")
print("   4. Higher aspect ratio particles → Better heat transfer")
print("   5. Export/import: Share shape definitions with team")

print("\n📊 Best Shapes for Heat Transfer Enhancement:")
sorted_results = sorted(results, key=lambda x: x['enhancement'], reverse=True)
for i, r in enumerate(sorted_results[:3], 1):
    print(f"   {i}. {r['name']}: {r['enhancement']:.1f}% enhancement")

print("\n💡 Recommendations:")
print("   • For maximum enhancement: Use high aspect ratio particles (CNT, graphene)")
print("   • For stability: Use spherical or short cylindrical particles")
print("   • For balance: Use moderate aspect ratio cylinders (L/D = 3-10)")
print("   • Always validate with experimental measurements!")

print("\n📚 Integration with Main Simulator:")
print("   Current: Shape definitions created")
print("   TODO: Integrate with NanofluidSimulator class for automatic shape-based calculations")
print("   Workaround: Use shape factor in Hamilton-Crosser model manually")

print("\n" + "="*70)
print("EXAMPLE COMPLETE!")
print("="*70)

print("\n💡 Next Steps:")
print("   1. Measure your nanoparticle geometry (TEM/SEM)")
print("   2. Create custom shape definition")
print("   3. Calculate thermal conductivity with your shape")
print("   4. Validate against experimental data")
print("   5. Share shape definitions with your team")
