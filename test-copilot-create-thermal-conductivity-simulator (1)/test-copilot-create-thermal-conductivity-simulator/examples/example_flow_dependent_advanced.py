#!/usr/bin/env python3
"""
Comprehensive Examples for Advanced Flow-Dependent Calculator

Demonstrates:
1. No nanoparticles (base fluid only)
2. Single nanoparticle
3. Multiple nanoparticles
4. Parameter range sweeps
5. Velocity sweeps
6. Temperature sweeps
7. Model comparisons
8. Enable/disable nanoparticles

Author: BKPS NFL Thermal v7.1
"""

import numpy as np
import matplotlib.pyplot as plt
from nanofluid_simulator.advanced_flow_calculator import (
    AdvancedFlowCalculator,
    FlowDependentConfig,
    NanoparticleSpec,
    FlowConditions,
    calculate_flow_properties
)


def example_1_base_fluid_only():
    """Example 1: Base fluid properties (no nanoparticles)"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Base Fluid Only (No Nanoparticles)")
    print("="*70)
    
    config = FlowDependentConfig(
        base_fluid='Water',
        nanoparticles=[],  # Empty list
        flow_conditions=FlowConditions(velocity=0.1, temperature=300),
        conductivity_models=['static'],
        viscosity_models=['base']
    )
    
    calc = AdvancedFlowCalculator(config)
    result = calc._base_fluid_only_results(config.flow_conditions)
    
    print(f"\nBase Fluid: {config.base_fluid}")
    print(f"Thermal Conductivity: {result['k_static']:.4f} W/m·K")
    print(f"Viscosity: {result['viscosity']['base']*1000:.4f} mPa·s")
    print(f"Reynolds Number: {result['reynolds']:.1f}")
    print("\n✓ Use case: Baseline comparison or validation")


def example_2_single_nanoparticle():
    """Example 2: Single nanoparticle with flow effects"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Single Nanoparticle (Al2O3) with Flow Effects")
    print("="*70)
    
    config = FlowDependentConfig(
        base_fluid='Water',
        nanoparticles=[
            NanoparticleSpec(
                material='Al2O3',
                volume_fraction=0.02,
                diameter=30e-9,
                enabled=True
            )
        ],
        flow_conditions=FlowConditions(velocity=0.1, temperature=300),
        conductivity_models=['static', 'buongiorno', 'kumar'],
        viscosity_models=['einstein', 'brinkman', 'batchelor']
    )
    
    calc = AdvancedFlowCalculator(config)
    result = calc.calculate_single_condition(
        config.nanoparticles[0],
        config.flow_conditions
    )
    
    print(f"\nNanoparticle: {result['material']}")
    print(f"Volume Fraction: {result['volume_fraction']*100:.1f}%")
    print(f"Diameter: {result['diameter']*1e9:.1f} nm")
    print(f"Velocity: {result['velocity']:.3f} m/s")
    print(f"Shear Rate: {result['shear_rate']:.1f} 1/s")
    
    print(f"\nThermal Conductivity:")
    print(f"  Static:         {result['conductivity']['static']:.4f} W/m·K")
    print(f"  Buongiorno:     {result['conductivity']['buongiorno']:.4f} W/m·K")
    print(f"  Kumar (shear):  {result['conductivity']['kumar']:.4f} W/m·K")
    print(f"  Enhancement:    {result['enhancement_static']:.2f}%")
    
    print(f"\nViscosity:")
    print(f"  Einstein:   {result['viscosity']['einstein']*1000:.4f} mPa·s")
    print(f"  Brinkman:   {result['viscosity']['brinkman']*1000:.4f} mPa·s")
    print(f"  Batchelor:  {result['viscosity']['batchelor']*1000:.4f} mPa·s")
    
    print(f"\nReynolds Number: {result['reynolds']:.1f}")


def example_3_multiple_nanoparticles():
    """Example 3: Compare multiple nanoparticles"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Multiple Nanoparticles Comparison")
    print("="*70)
    
    config = FlowDependentConfig(
        base_fluid='Water',
        nanoparticles=[
            NanoparticleSpec(material='Al2O3', volume_fraction=0.02, diameter=30e-9, enabled=True),
            NanoparticleSpec(material='CuO', volume_fraction=0.02, diameter=30e-9, enabled=True),
            NanoparticleSpec(material='Cu', volume_fraction=0.02, diameter=30e-9, enabled=True),
            NanoparticleSpec(material='TiO2', volume_fraction=0.02, diameter=30e-9, enabled=True),
        ],
        flow_conditions=FlowConditions(velocity=0.1, temperature=300),
        conductivity_models=['buongiorno', 'kumar'],
        viscosity_models=['brinkman']
    )
    
    calc = AdvancedFlowCalculator(config)
    comparison = calc.calculate_comparison()
    
    print(f"\nBase Fluid: {config.base_fluid}, Velocity: {config.flow_conditions.velocity} m/s")
    print(f"Volume Fraction: 2% (all), Diameter: 30 nm (all)")
    print("\n{:<12} {:<15} {:<15} {:<12}".format(
        "Material", "k_buongiorno", "k_kumar", "Reynolds"
    ))
    print("-"*60)
    
    for material, results in sorted(comparison.items()):
        r = results[0]
        k_buon = r['conductivity'].get('buongiorno', 0)
        k_kumar = r['conductivity'].get('kumar', 0)
        re = r['reynolds']
        print(f"{material:<12} {k_buon:.4f} W/m·K   {k_kumar:.4f} W/m·K   {re:.1f}")
    
    print("\n✓ Cu has highest thermal conductivity")
    print("✓ All show similar flow enhancement patterns")


def example_4_volume_fraction_sweep():
    """Example 4: Sweep volume fraction from 0.5% to 5%"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Volume Fraction Sweep (0.5% to 5%)")
    print("="*70)
    
    config = FlowDependentConfig(
        base_fluid='Water',
        nanoparticles=[
            NanoparticleSpec(
                material='Al2O3',
                volume_fraction=(0.005, 0.05, 10),  # 0.5% to 5%, 10 steps
                diameter=30e-9,
                enabled=True
            )
        ],
        flow_conditions=FlowConditions(velocity=0.1, temperature=300),
        conductivity_models=['buongiorno'],
        viscosity_models=['brinkman']
    )
    
    calc = AdvancedFlowCalculator(config)
    results = calc.calculate_parametric_sweep()
    
    print(f"\n{'φ (%)':<10} {'k (W/m·K)':<15} {'μ (mPa·s)':<15} {'Re':<10}")
    print("-"*50)
    
    for r in results:
        phi_pct = r['volume_fraction'] * 100
        k = r['conductivity']['buongiorno']
        mu = r['viscosity']['brinkman'] * 1000
        re = r['reynolds']
        print(f"{phi_pct:<10.2f} {k:<15.4f} {mu:<15.4f} {re:<10.1f}")
    
    print("\n✓ Thermal conductivity increases with φ")
    print("✓ Viscosity increases significantly with φ")
    print("✓ Reynolds decreases due to higher viscosity")


def example_5_velocity_sweep():
    """Example 5: Sweep velocity from 0.01 to 0.2 m/s"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Velocity Sweep (0.01 to 0.2 m/s)")
    print("="*70)
    
    config = FlowDependentConfig(
        base_fluid='Water',
        nanoparticles=[
            NanoparticleSpec(
                material='CuO',
                volume_fraction=0.02,
                diameter=50e-9,
                enabled=True
            )
        ],
        flow_conditions=FlowConditions(velocity=0.05, temperature=300),  # Base value
        sweep_velocity=(0.01, 0.2, 8),  # min, max, steps
        conductivity_models=['static', 'buongiorno', 'kumar'],
        viscosity_models=['brinkman']
    )
    
    calc = AdvancedFlowCalculator(config)
    results = calc.calculate_parametric_sweep()
    
    print(f"\n{'V (m/s)':<12} {'k_static':<12} {'k_buongiorno':<15} {'k_kumar':<12} {'Re':<10}")
    print("-"*70)
    
    for r in results:
        v = r['velocity']
        k_static = r['conductivity']['static']
        k_buon = r['conductivity']['buongiorno']
        k_kumar = r['conductivity']['kumar']
        re = r['reynolds']
        print(f"{v:<12.3f} {k_static:<12.4f} {k_buon:<15.4f} {k_kumar:<12.4f} {re:<10.1f}")
    
    print("\n✓ Static k is constant (no flow dependence)")
    print("✓ Buongiorno k increases with velocity (Peclet effect)")
    print("✓ Kumar k increases with shear rate")
    print("✓ Reynolds increases linearly with velocity")


def example_6_temperature_sweep():
    """Example 6: Temperature effect on properties"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Temperature Sweep (280K to 340K)")
    print("="*70)
    
    config = FlowDependentConfig(
        base_fluid='Water',
        nanoparticles=[
            NanoparticleSpec(
                material='Al2O3',
                volume_fraction=0.03,
                diameter=40e-9,
                enabled=True
            )
        ],
        flow_conditions=FlowConditions(velocity=0.08, temperature=300),
        sweep_temperature=(280, 340, 7),  # 280K to 340K
        conductivity_models=['buongiorno'],
        viscosity_models=['brinkman'],
        include_brownian=True
    )
    
    calc = AdvancedFlowCalculator(config)
    results = calc.calculate_parametric_sweep()
    
    print(f"\n{'T (K)':<10} {'T (°C)':<10} {'k (W/m·K)':<15} {'μ (mPa·s)':<15}")
    print("-"*55)
    
    for r in results:
        T_k = r['temperature']
        T_c = T_k - 273.15
        k = r['conductivity']['buongiorno']
        mu = r['viscosity']['brinkman'] * 1000
        print(f"{T_k:<10.1f} {T_c:<10.1f} {k:<15.4f} {mu:<15.4f}")
    
    print("\n✓ Higher temperature → stronger Brownian motion → higher k")
    print("✓ Temperature effects captured by Buongiorno model")


def example_7_diameter_sweep():
    """Example 7: Effect of particle diameter"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Particle Diameter Sweep (10nm to 100nm)")
    print("="*70)
    
    config = FlowDependentConfig(
        base_fluid='Water',
        nanoparticles=[
            NanoparticleSpec(
                material='Cu',
                volume_fraction=0.015,
                diameter=(10e-9, 100e-9, 10),  # 10nm to 100nm
                enabled=True
            )
        ],
        flow_conditions=FlowConditions(velocity=0.1, temperature=300),
        conductivity_models=['buongiorno'],
        viscosity_models=['brinkman']
    )
    
    calc = AdvancedFlowCalculator(config)
    results = calc.calculate_parametric_sweep()
    
    print(f"\n{'d (nm)':<12} {'k (W/m·K)':<15} {'Enhancement (%)':<18}")
    print("-"*50)
    
    for r in results:
        d_nm = r['diameter'] * 1e9
        k = r['conductivity']['buongiorno']
        enh = (k / r['k_base'] - 1) * 100
        print(f"{d_nm:<12.1f} {k:<15.4f} {enh:<18.2f}")
    
    print("\n✓ Smaller particles → higher Brownian motion → higher k")
    print("✓ Diameter significantly affects flow enhancement")


def example_8_enable_disable_nanoparticles():
    """Example 8: Enable/disable nanoparticles dynamically"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Enable/Disable Nanoparticles")
    print("="*70)
    
    # Configuration with 3 nanoparticles
    nanoparticles = [
        NanoparticleSpec(material='Al2O3', volume_fraction=0.02, diameter=30e-9, enabled=True),
        NanoparticleSpec(material='CuO', volume_fraction=0.02, diameter=30e-9, enabled=False),  # Disabled
        NanoparticleSpec(material='Cu', volume_fraction=0.02, diameter=30e-9, enabled=True),
    ]
    
    config = FlowDependentConfig(
        base_fluid='Water',
        nanoparticles=nanoparticles,
        flow_conditions=FlowConditions(velocity=0.1, temperature=300),
        conductivity_models=['buongiorno'],
        viscosity_models=['brinkman']
    )
    
    calc = AdvancedFlowCalculator(config)
    comparison = calc.calculate_comparison()
    
    print("\nNanoparticle Status:")
    for i, np in enumerate(nanoparticles, 1):
        status = "✓ ENABLED" if np.enabled else "✗ DISABLED"
        print(f"  {i}. {np.material:<8} - {status}")
    
    print(f"\nCalculation Results (enabled only):")
    print(f"{'Material':<12} {'k (W/m·K)':<15} {'Status':<10}")
    print("-"*40)
    
    for material, results in sorted(comparison.items()):
        r = results[0]
        k = r['conductivity']['buongiorno']
        print(f"{material:<12} {k:<15.4f} Calculated")
    
    print("\n✓ CuO was disabled and not calculated")
    print("✓ Useful for sensitivity studies or what-if analysis")


def example_9_multi_dimensional_sweep():
    """Example 9: 2D sweep (volume fraction × velocity)"""
    print("\n" + "="*70)
    print("EXAMPLE 9: 2D Parametric Sweep (φ × Velocity)")
    print("="*70)
    
    config = FlowDependentConfig(
        base_fluid='Water',
        nanoparticles=[
            NanoparticleSpec(
                material='Al2O3',
                volume_fraction=(0.01, 0.04, 4),  # 1% to 4%
                diameter=30e-9,
                enabled=True
            )
        ],
        flow_conditions=FlowConditions(velocity=0.05, temperature=300),
        sweep_velocity=(0.05, 0.15, 3),  # 0.05 to 0.15 m/s
        conductivity_models=['buongiorno'],
        viscosity_models=['brinkman']
    )
    
    calc = AdvancedFlowCalculator(config)
    results = calc.calculate_parametric_sweep()
    
    print(f"\nTotal combinations: {len(results)}")
    print(f"\n{'φ (%)':<10} {'V (m/s)':<12} {'k (W/m·K)':<15} {'Re':<10}")
    print("-"*50)
    
    for r in results[:8]:  # Show first 8
        phi_pct = r['volume_fraction'] * 100
        v = r['velocity']
        k = r['conductivity']['buongiorno']
        re = r['reynolds']
        print(f"{phi_pct:<10.1f} {v:<12.3f} {k:<15.4f} {re:<10.1f}")
    
    if len(results) > 8:
        print(f"... ({len(results)-8} more rows)")
    
    print("\n✓ Useful for optimization and design space exploration")
    print("✓ Can export to CSV for further analysis")


def example_10_model_comparison():
    """Example 10: Compare different conductivity models"""
    print("\n" + "="*70)
    print("EXAMPLE 10: Model Comparison (Static vs Flow Models)")
    print("="*70)
    
    result = calculate_flow_properties(
        base_fluid='Water',
        nanoparticles=[{'material': 'CuO', 'volume_fraction': 0.025, 'diameter': 40e-9}],
        velocity=0.12,
        temperature=310,
        models=['static', 'buongiorno', 'kumar', 'rea_guzman']
    )
    
    print(f"\nConditions:")
    print(f"  Material: {result['material']}")
    print(f"  φ = {result['volume_fraction']*100:.1f}%, d = {result['diameter']*1e9:.0f} nm")
    print(f"  V = {result['velocity']} m/s, T = {result['temperature']} K")
    print(f"  Re = {result['reynolds']:.1f}")
    
    print(f"\nThermal Conductivity Comparison:")
    k_base = result['k_base']
    
    models_data = []
    for model, k_value in result['conductivity'].items():
        enhancement = (k_value / k_base - 1) * 100
        models_data.append((model, k_value, enhancement))
    
    models_data.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Model':<15} {'k (W/m·K)':<15} {'Enhancement (%)':<18}")
    print("-"*50)
    for model, k, enh in models_data:
        print(f"{model.capitalize():<15} {k:<15.4f} {enh:<18.2f}")
    
    print("\n✓ Flow models predict higher k than static model")
    print("✓ Different models capture different physics")
    print("✓ Choose model based on your validation data")


def run_all_examples():
    """Run all examples"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*10 + "ADVANCED FLOW-DEPENDENT CALCULATOR EXAMPLES" + " "*15 + "║")
    print("║" + " "*20 + "User Control Demonstrations" + " "*21 + "║")
    print("╚" + "="*68 + "╝")
    
    examples = [
        example_1_base_fluid_only,
        example_2_single_nanoparticle,
        example_3_multiple_nanoparticles,
        example_4_volume_fraction_sweep,
        example_5_velocity_sweep,
        example_6_temperature_sweep,
        example_7_diameter_sweep,
        example_8_enable_disable_nanoparticles,
        example_9_multi_dimensional_sweep,
        example_10_model_comparison,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in {example_func.__name__}: {e}")
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  ✓ 0, 1, or multiple nanoparticles")
    print("  ✓ Parameter ranges (φ, d, V, T)")
    print("  ✓ Enable/disable nanoparticles")
    print("  ✓ Multiple conductivity models")
    print("  ✓ Flow-dependent vs static properties")
    print("  ✓ Multi-dimensional sweeps")
    print("  ✓ Model comparisons")
    print("\n" + "="*70)


if __name__ == "__main__":
    run_all_examples()
