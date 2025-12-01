"""
BKPS NFL Thermal v6.0 - Comprehensive Demonstration
Dedicated to: Brijesh Kumar Pandey

This example showcases all advanced features:
- Flow-dependent thermal conductivity
- Non-Newtonian viscosity (shear-rate effects)
- DLVO theory & colloidal stability
- Particle clustering & aggregation
- Enhanced hybrid nanofluids
- Comprehensive validation

Author: BKPS NFL Thermal v6.0
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import BKPS NFL Thermal modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanofluid_simulator.integrated_simulator_v6 import BKPSNanofluidSimulator
from nanofluid_simulator.validation_suite import run_comprehensive_validation_suite


def demo_flow_dependent_conductivity():
    """
    Demonstrate flow-dependent thermal conductivity enhancement.
    """
    print("\n" + "="*90)
    print("DEMO 1: Flow-Dependent Thermal Conductivity")
    print("="*90)
    
    # Create simulator
    sim = BKPSNanofluidSimulator(base_fluid='Water', temperature=300.0)
    sim.add_nanoparticle('Al2O3', volume_fraction=0.02, diameter=30e-9, shape='sphere')
    
    # Calculate conductivity at different flow velocities
    velocities = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 2.0])
    k_values = []
    
    print("\nThermal conductivity vs. Flow velocity:")
    print("-" * 70)
    print(f"{'Velocity (m/s)':<20} {'k_eff (W/m·K)':<20} {'Enhancement (%)':<20}")
    print("-" * 70)
    
    k_static = sim.calculate_static_thermal_conductivity()
    
    for v in velocities:
        sim.set_flow_conditions(velocity=v)
        k_eff, _ = sim.calculate_flow_dependent_conductivity()
        k_values.append(k_eff)
        
        enhancement = (k_eff / sim.k_bf - 1) * 100
        print(f"{v:<20.2f} {k_eff:<20.6f} {enhancement:<20.2f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(velocities, k_values, 'o-', linewidth=2, markersize=8, label='BKPS NFL Thermal v6.0')
    ax.axhline(y=k_static, color='r', linestyle='--', label=f'Static k = {k_static:.4f} W/m·K')
    ax.axhline(y=sim.k_bf, color='gray', linestyle=':', label=f'Base fluid k = {sim.k_bf:.4f} W/m·K')
    
    ax.set_xlabel('Flow Velocity (m/s)', fontsize=12)
    ax.set_ylabel('Thermal Conductivity (W/m·K)', fontsize=12)
    ax.set_title('Flow-Dependent Thermal Conductivity (Al₂O₃-water, φ=2%)', fontsize=14, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo1_flow_dependent_k.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: demo1_flow_dependent_k.png")
    plt.close()


def demo_non_newtonian_viscosity():
    """
    Demonstrate non-Newtonian shear-thinning behavior.
    """
    print("\n" + "="*90)
    print("DEMO 2: Non-Newtonian Shear-Thinning Viscosity")
    print("="*90)
    
    # Create simulator with higher concentration
    sim = BKPSNanofluidSimulator(base_fluid='Water', temperature=293.15)
    sim.add_nanoparticle('Al2O3', volume_fraction=0.04, diameter=30e-9)
    sim.enable_non_newtonian = True
    
    # Shear rate range
    shear_rates = np.logspace(-1, 4, 50)
    mu_values = []
    
    print("\nViscosity vs. Shear Rate:")
    print("-" * 70)
    
    for gamma_dot in [0.1, 1, 10, 100, 1000, 10000]:
        sim.set_flow_conditions(shear_rate=gamma_dot)
        mu_eff, info = sim.calculate_viscosity()
        mu_values.append(mu_eff)
        
        print(f"γ̇ = {gamma_dot:>8.1f} 1/s  →  μ = {mu_eff*1000:>8.4f} mPa·s")
    
    # Calculate full curve
    mu_curve = []
    for gamma_dot in shear_rates:
        sim.set_flow_conditions(shear_rate=gamma_dot)
        mu_eff, _ = sim.calculate_viscosity()
        mu_curve.append(mu_eff * 1000)  # Convert to mPa·s
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.loglog(shear_rates, mu_curve, linewidth=2, label='Non-Newtonian (Carreau-Yasuda)')
    ax.axhline(y=sim.mu_bf*1000, color='gray', linestyle=':', label=f'Base fluid μ = {sim.mu_bf*1000:.2f} mPa·s')
    
    ax.set_xlabel('Shear Rate, γ̇ (1/s)', fontsize=12)
    ax.set_ylabel('Apparent Viscosity, μ (mPa·s)', fontsize=12)
    ax.set_title('Shear-Thinning Behavior (Al₂O₃-water, φ=4%)', fontsize=14, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add shear-thinning annotation
    ax.annotate('Shear-thinning\nregion', xy=(50, 2.5), xytext=(200, 4),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', weight='bold')
    
    plt.tight_layout()
    plt.savefig('demo2_non_newtonian_viscosity.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: demo2_non_newtonian_viscosity.png")
    plt.close()


def demo_dlvo_stability():
    """
    Demonstrate DLVO stability analysis under different conditions.
    """
    print("\n" + "="*90)
    print("DEMO 3: DLVO Colloidal Stability Analysis")
    print("="*90)
    
    # Test conditions
    conditions = [
        (7.0, 0.001, "Neutral pH, DI water"),
        (7.0, 0.1, "Neutral pH, high salt"),
        (4.0, 0.001, "Acidic pH, DI water"),
        (10.0, 0.001, "Basic pH, DI water")
    ]
    
    results_summary = []
    
    for pH, ionic_strength, description in conditions:
        sim = BKPSNanofluidSimulator(base_fluid='Water', temperature=298.15)
        sim.add_nanoparticle('Al2O3', volume_fraction=0.02, diameter=30e-9)
        sim.set_environmental_conditions(pH=pH, ionic_strength=ionic_strength)
        
        dlvo = sim.perform_dlvo_analysis()
        
        results_summary.append({
            'condition': description,
            'pH': pH,
            'ionic_strength': ionic_strength,
            'zeta_potential': dlvo['zeta_potential'] * 1000,  # mV
            'energy_barrier': dlvo['energy_barrier'] / 1.38e-23,  # kT
            'stability_ratio': dlvo['stability_ratio'],
            'status': dlvo['stability_status'],
            'cluster_size': dlvo['avg_cluster_size']
        })
    
    # Print results table
    print("\nDLVO Stability Analysis Results:")
    print("-" * 90)
    print(f"{'Condition':<25} {'ζ (mV)':<12} {'Barrier (kT)':<15} {'Status':<25}")
    print("-" * 90)
    
    for r in results_summary:
        print(f"{r['condition']:<25} {r['zeta_potential']:>10.1f}  {r['energy_barrier']:>13.1f}  {r['status']:<25}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # Plot 1: Zeta potential
    conditions_labels = [r['condition'].split(',')[0] for r in results_summary]
    zeta_values = [r['zeta_potential'] for r in results_summary]
    colors = ['green' if abs(z) > 30 else 'orange' if abs(z) > 15 else 'red' for z in zeta_values]
    
    ax1.bar(range(len(conditions_labels)), zeta_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Stable (|ζ| > 30 mV)')
    ax1.axhline(y=-30, color='g', linestyle='--', alpha=0.5)
    ax1.axhline(y=15, color='orange', linestyle='--', alpha=0.5, label='Metastable (15-30 mV)')
    ax1.axhline(y=-15, color='orange', linestyle='--', alpha=0.5)
    
    ax1.set_ylabel('Zeta Potential (mV)', fontsize=12)
    ax1.set_title('Zeta Potential vs. Conditions', fontsize=13, weight='bold')
    ax1.set_xticks(range(len(conditions_labels)))
    ax1.set_xticklabels(conditions_labels, rotation=15, ha='right')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Energy barrier
    barrier_values = [r['energy_barrier'] for r in results_summary]
    colors2 = ['green' if b > 15 else 'orange' if b > 5 else 'red' for b in barrier_values]
    
    ax2.bar(range(len(conditions_labels)), barrier_values, color=colors2, alpha=0.7, edgecolor='black')
    ax2.axhline(y=15, color='g', linestyle='--', alpha=0.5, label='Stable (>15 kT)')
    ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Metastable (5-15 kT)')
    
    ax2.set_ylabel('Energy Barrier (kT)', fontsize=12)
    ax2.set_title('DLVO Energy Barrier', fontsize=13, weight='bold')
    ax2.set_xticks(range(len(conditions_labels)))
    ax2.set_xticklabels(conditions_labels, rotation=15, ha='right')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('demo3_dlvo_stability.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: demo3_dlvo_stability.png")
    plt.close()


def demo_hybrid_nanofluid():
    """
    Demonstrate enhanced hybrid nanofluid with 2 components.
    """
    print("\n" + "="*90)
    print("DEMO 4: Enhanced Hybrid Nanofluid (Al₂O₃ + Cu)")
    print("="*90)
    
    # Create hybrid nanofluid simulator
    sim = BKPSNanofluidSimulator(base_fluid='Water', temperature=300.0)
    
    # Add Al2O3 (90%)
    sim.add_nanoparticle('Al2O3', volume_fraction=0.018, diameter=30e-9, shape='sphere')
    
    # Add Cu (10%)
    sim.add_nanoparticle('Cu', volume_fraction=0.002, diameter=25e-9, shape='sphere')
    
    print("\nHybrid Composition:")
    print("  Al₂O₃: 90% (φ = 0.018)")
    print("  Cu:    10% (φ = 0.002)")
    print("  Total: φ = 0.020 (2.0%)")
    
    # Perform comprehensive analysis
    results = sim.comprehensive_analysis()
    
    # Compare with mono-nanofluids
    print("\n\nComparison with Mono-Nanofluids:")
    print("-" * 70)
    
    # Pure Al2O3
    sim_al2o3 = BKPSNanofluidSimulator(base_fluid='Water', temperature=300.0)
    sim_al2o3.add_nanoparticle('Al2O3', volume_fraction=0.02, diameter=30e-9)
    k_al2o3 = sim_al2o3.calculate_static_thermal_conductivity()
    
    # Pure Cu
    sim_cu = BKPSNanofluidSimulator(base_fluid='Water', temperature=300.0)
    sim_cu.add_nanoparticle('Cu', volume_fraction=0.02, diameter=25e-9)
    k_cu = sim_cu.calculate_static_thermal_conductivity()
    
    # Hybrid
    k_hybrid = results['k_static']
    
    print(f"Pure Al₂O₃ (φ=2%):     k = {k_al2o3:.5f} W/m·K  (+{(k_al2o3/sim.k_bf-1)*100:.2f}%)")
    print(f"Pure Cu (φ=2%):        k = {k_cu:.5f} W/m·K  (+{(k_cu/sim.k_bf-1)*100:.2f}%)")
    print(f"Hybrid Al₂O₃+Cu (2%):  k = {k_hybrid:.5f} W/m·K  (+{(k_hybrid/sim.k_bf-1)*100:.2f}%)")
    
    print(f"\n✓ Hybrid enhancement: {(k_hybrid/k_al2o3-1)*100:+.2f}% over pure Al₂O₃")


def demo_comprehensive_validation():
    """
    Run comprehensive validation suite against published experiments.
    """
    print("\n" + "="*90)
    print("DEMO 5: Comprehensive Validation Against Published Experiments")
    print("="*90)
    
    # Run validation suite
    validation_results = run_comprehensive_validation_suite()
    
    print("\n✓ Validation complete! Check plots and report.")


def demo_full_workflow():
    """
    Demonstrate complete workflow with all features.
    """
    print("\n" + "="*90)
    print("DEMO 6: Complete Workflow - Real-World Application")
    print("="*90)
    print("Scenario: Automotive cooling system with Al₂O₃-water nanofluid")
    print("="*90)
    
    # System specifications
    print("\nSystem Specifications:")
    print("  Application: Automotive radiator")
    print("  Operating temperature: 90°C (363 K)")
    print("  Coolant: Water-based")
    print("  Nanofluid: Al₂O₃, d=50nm, φ=3%")
    print("  Flow conditions: v=1.5 m/s, turbulent")
    print("  Environment: pH=8, low ionic strength")
    
    # Create simulator
    sim = BKPSNanofluidSimulator(base_fluid='Water', temperature=363.15)
    sim.add_nanoparticle('Al2O3', volume_fraction=0.03, diameter=50e-9, shape='sphere')
    sim.set_environmental_conditions(pH=8.0, ionic_strength=0.001)
    sim.set_flow_conditions(velocity=1.5, shear_rate=5000.0)
    
    # Comprehensive analysis
    results = sim.comprehensive_analysis()
    
    # Performance metrics
    print("\n\nPerformance Metrics:")
    print("-" * 70)
    
    k_enhancement = results['k_enhancement_total']
    heat_transfer_enhancement = k_enhancement * 1.2  # Approximate correlation
    
    print(f"Thermal conductivity enhancement: +{k_enhancement:.1f}%")
    print(f"Heat transfer coefficient enhancement: +{heat_transfer_enhancement:.1f}% (estimated)")
    print(f"Viscosity increase: {results['mu_ratio']:.2f}x")
    print(f"Pressure drop increase: ~{(results['mu_ratio']**0.8 - 1)*100:.1f}% (estimated)")
    
    # Stability assessment
    if results['dlvo_analysis']:
        dlvo = results['dlvo_analysis']
        print(f"\nColloidal Stability: {dlvo['stability_status']}")
        print(f"  Energy barrier: {dlvo['energy_barrier']/1.38e-23:.1f} kT")
        print(f"  Average cluster size: {dlvo['avg_cluster_size']:.1f} particles")
    
    # Recommendations
    print("\n\nRecommendations:")
    print("-" * 70)
    
    if k_enhancement > 10:
        print("✓ Excellent thermal performance - significant cooling improvement expected")
    
    if results['mu_ratio'] < 1.5:
        print("✓ Acceptable pressure drop - pumping power increase is manageable")
    
    if results['dlvo_analysis'] and 'STABLE' in dlvo['stability_status']:
        print("✓ Good colloidal stability - long-term performance assured")
    
    print("\n✓ BKPS NFL Thermal v6.0 analysis complete!")


def main():
    """
    Run all demonstrations.
    """
    print("="*90)
    print("BKPS NFL THERMAL v6.0 - COMPREHENSIVE DEMONSTRATION")
    print("Dedicated to: Brijesh Kumar Pandey")
    print("="*90)
    print("\nWorld-Class Static + CFD Nanofluid Thermal Analysis Software")
    print("⭐⭐⭐⭐⭐ Research-Grade | Experimentally Validated | Publication-Quality")
    print()
    
    # Run demonstrations
    demo_flow_dependent_conductivity()
    demo_non_newtonian_viscosity()
    demo_dlvo_stability()
    demo_hybrid_nanofluid()
    demo_full_workflow()
    
    # Optional: Run validation (takes longer)
    print("\n\nRun comprehensive validation suite? (This will take several minutes)")
    print("Note: Validation requires matplotlib for plot generation")
    # Uncomment to run: demo_comprehensive_validation()
    
    print("\n" + "="*90)
    print("ALL DEMONSTRATIONS COMPLETE!")
    print("="*90)
    print("\nGenerated files:")
    print("  - demo1_flow_dependent_k.png")
    print("  - demo2_non_newtonian_viscosity.png")
    print("  - demo3_dlvo_stability.png")
    print("\nFor validation plots, run demo_comprehensive_validation()")
    print("\n⭐ BKPS NFL Thermal v6.0 - World-Class Professional Research Tool")


if __name__ == "__main__":
    main()
