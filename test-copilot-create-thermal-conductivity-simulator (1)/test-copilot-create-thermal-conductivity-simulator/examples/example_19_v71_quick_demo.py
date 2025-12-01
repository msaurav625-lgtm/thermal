#!/usr/bin/env python3
"""
BKPS NFL Thermal Pro v7.1 — Quick Demonstration
===============================================

Demonstrates all new v7.1 visualization capabilities:
1. Parameter sweeps (T, φ, Re)
2. DLVO particle interaction visualization
3. Enhanced CFD visualization

Dedicated to: Brijesh Kumar Pandey
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import v7.1 modules
try:
    from nanofluid_simulator import (
        ParameterSweepEngine,
        ParticleInteractionVisualizer,
        RealTimeCFDVisualizer,
        __version__
    )
    print(f"✓ Successfully loaded BKPS NFL Thermal Pro {__version__}")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def demo_temperature_sweep():
    """Demonstrate temperature sweep with real physics."""
    print("\n" + "="*60)
    print("DEMO 1: Temperature Sweep (k_eff vs T)")
    print("="*60)
    
    try:
        engine = ParameterSweepEngine()
        
        result = engine.sweep_temperature(
            T_range=(280, 360),
            n_points=15,
            base_fluid="Water",
            nanoparticle="Al2O3",
            volume_fraction=0.02,
            particle_diameter=30e-9
        )
        
        print(f"✓ Temperature sweep complete")
        print(f"  T range: {result.parameter_values[0]:.1f} - {result.parameter_values[-1]:.1f} K")
        print(f"  k_base: {result.output_values['k_base'][0]:.4f} - {result.output_values['k_base'][-1]:.4f} W/m·K")
        print(f"  k_nf: {result.output_values['k_nf'][0]:.4f} - {result.output_values['k_nf'][-1]:.4f} W/m·K")
        print(f"  Enhancement: {result.output_values['enhancement_k'][0]:.2f}% - {result.output_values['enhancement_k'][-1]:.2f}%")
        
        # Create plot
        figure = engine.create_sweep_plot(
            result,
            output_keys=['k_base', 'k_nf', 'enhancement_k'],
            title='Thermal Conductivity vs Temperature (Al₂O₃/Water, φ=2%)'
        )
        figure.savefig('v71_demo_temperature_sweep.png', dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: v71_demo_temperature_sweep.png")
        
        return True
        
    except Exception as e:
        print(f"✗ Temperature sweep failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_volume_fraction_sweep():
    """Demonstrate volume fraction sweep."""
    print("\n" + "="*60)
    print("DEMO 2: Volume Fraction Sweep (k_eff vs φ)")
    print("="*60)
    
    try:
        engine = ParameterSweepEngine()
        
        result = engine.sweep_volume_fraction(
            phi_range=(0.005, 0.04),
            n_points=15,
            base_fluid="Water",
            nanoparticle="CuO",
            temperature=300.0,
            particle_diameter=30e-9
        )
        
        print(f"✓ Volume fraction sweep complete")
        print(f"  φ range: {result.parameter_values[0]:.4f} - {result.parameter_values[-1]:.4f}")
        print(f"  k_nf: {result.output_values['k_nf'][0]:.4f} - {result.output_values['k_nf'][-1]:.4f} W/m·K")
        print(f"  μ_nf: {result.output_values['mu_nf'][0]:.6f} - {result.output_values['mu_nf'][-1]:.6f} Pa·s")
        print(f"  Enhancement: {result.output_values['enhancement_k'][0]:.2f}% - {result.output_values['enhancement_k'][-1]:.2f}%")
        
        # Create plot
        figure = engine.create_sweep_plot(
            result,
            output_keys=['k_nf', 'mu_nf', 'enhancement_k'],
            title='Properties vs Volume Fraction (CuO/Water, T=300K)'
        )
        figure.savefig('v71_demo_volume_fraction_sweep.png', dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: v71_demo_volume_fraction_sweep.png")
        
        return True
        
    except Exception as e:
        print(f"✗ Volume fraction sweep failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_dlvo_potential():
    """Demonstrate DLVO potential visualization."""
    print("\n" + "="*60)
    print("DEMO 3: DLVO Potential Energy Curve")
    print("="*60)
    
    try:
        visualizer = ParticleInteractionVisualizer()
        
        fig = visualizer.plot_dlvo_potential(
            particle_diameter=30e-9,
            temperature=300.0,
            ionic_strength=0.001,
            hamaker=3.7e-20,
            zeta_potential=-0.030
        )
        
        print(f"✓ DLVO potential calculated")
        print(f"  Particle diameter: 30 nm")
        print(f"  Temperature: 300 K")
        print(f"  Ionic strength: 0.001 M")
        print(f"  Hamaker constant: 3.7e-20 J")
        print(f"  Zeta potential: -30 mV")
        
        fig.savefig('v71_demo_dlvo_potential.png', dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: v71_demo_dlvo_potential.png")
        
        return True
        
    except Exception as e:
        print(f"✗ DLVO visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_aggregation_kinetics():
    """Demonstrate aggregation kinetics visualization."""
    print("\n" + "="*60)
    print("DEMO 4: Aggregation Kinetics (Cluster Growth)")
    print("="*60)
    
    try:
        visualizer = ParticleInteractionVisualizer()
        
        fig = visualizer.plot_aggregation_kinetics(
            particle_diameter=30e-9,
            volume_fraction=0.02,
            temperature=300.0,
            stability_ratio=10.0,
            time_hours=24.0
        )
        
        print(f"✓ Aggregation kinetics calculated")
        print(f"  Initial diameter: 30 nm")
        print(f"  Volume fraction: 2%")
        print(f"  Stability ratio: 10")
        print(f"  Time span: 24 hours")
        
        fig.savefig('v71_demo_aggregation_kinetics.png', dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: v71_demo_aggregation_kinetics.png")
        
        return True
        
    except Exception as e:
        print(f"✗ Aggregation visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_brownian_motion():
    """Demonstrate Brownian motion effects."""
    print("\n" + "="*60)
    print("DEMO 5: Brownian Motion Effects")
    print("="*60)
    
    try:
        visualizer = ParticleInteractionVisualizer()
        
        fig = visualizer.plot_brownian_motion_effect(
            particle_diameter=30e-9,
            temperature=300.0,
            base_fluid="Water"
        )
        
        print(f"✓ Brownian motion effects calculated")
        print(f"  Particle diameter: 30 nm")
        print(f"  Temperature: 300 K")
        print(f"  Base fluid: Water")
        
        fig.savefig('v71_demo_brownian_motion.png', dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: v71_demo_brownian_motion.png")
        
        return True
        
    except Exception as e:
        print(f"✗ Brownian motion visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_cfd_visualization():
    """Demonstrate CFD visualization (with synthetic data for demo)."""
    print("\n" + "="*60)
    print("DEMO 6: CFD Visualization")
    print("="*60)
    
    try:
        from nanofluid_simulator.cfd_mesh import StructuredMesh2D
        
        # Create simple mesh
        nx, ny = 50, 30
        length, height = 0.1, 0.01
        mesh = StructuredMesh2D(length, height, nx, ny)
        
        # Create synthetic fields for demonstration
        x = np.linspace(0, length, nx)
        y = np.linspace(0, height, ny)
        X, Y = np.meshgrid(x, y)
        
        # Temperature field (hot inlet, cooled walls)
        temperature = 300 + 50 * (1 - X/length) * (1 - (2*Y/height - 1)**2)
        
        # Velocity field (parabolic profile)
        u_max = 0.1
        u = u_max * (1 - (2*Y/height - 1)**2) * np.ones_like(X)
        v = np.zeros_like(Y)
        
        # Pressure field (linear drop)
        pressure = 101325 + 1000 * (1 - X/length)
        
        visualizer = RealTimeCFDVisualizer()
        
        # Temperature contours
        fig_temp = visualizer.plot_temperature_contours(
            mesh, temperature.T,
            levels=20,
            cmap='hot',
            show_mesh=False
        )
        fig_temp.savefig('v71_demo_cfd_temperature.png', dpi=300, bbox_inches='tight')
        print(f"✓ Temperature contours saved: v71_demo_cfd_temperature.png")
        
        # Velocity field
        fig_vel = visualizer.plot_velocity_field(
            mesh, u.T, v.T,
            show_vectors=True,
            show_streamlines=True,
            vector_spacing=5
        )
        fig_vel.savefig('v71_demo_cfd_velocity.png', dpi=300, bbox_inches='tight')
        print(f"✓ Velocity field saved: v71_demo_cfd_velocity.png")
        
        # Pressure field
        fig_pres = visualizer.plot_pressure_field(
            mesh, pressure.T,
            levels=15,
            cmap='viridis'
        )
        fig_pres.savefig('v71_demo_cfd_pressure.png', dpi=300, bbox_inches='tight')
        print(f"✓ Pressure field saved: v71_demo_cfd_pressure.png")
        
        print(f"✓ CFD visualization complete")
        print(f"  Mesh: {nx}×{ny} cells")
        print(f"  Domain: {length}m × {height}m")
        print(f"  T range: {temperature.min():.1f} - {temperature.max():.1f} K")
        print(f"  U_max: {u_max:.3f} m/s")
        
        return True
        
    except Exception as e:
        print(f"✗ CFD visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print(" BKPS NFL Thermal Pro v7.1 — Quick Demonstration")
    print(" Real Physics Visualization Engine")
    print(" Dedicated to: Brijesh Kumar Pandey")
    print("=" * 70)
    
    results = []
    
    # Run demonstrations
    results.append(("Temperature Sweep", demo_temperature_sweep()))
    results.append(("Volume Fraction Sweep", demo_volume_fraction_sweep()))
    results.append(("DLVO Potential", demo_dlvo_potential()))
    results.append(("Aggregation Kinetics", demo_aggregation_kinetics()))
    results.append(("Brownian Motion", demo_brownian_motion()))
    results.append(("CFD Visualization", demo_cfd_visualization()))
    
    # Summary
    print("\n" + "=" * 70)
    print(" DEMONSTRATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} {name}")
    
    print(f"\nOverall: {passed}/{total} demonstrations successful")
    
    if passed == total:
        print("\n🎉 All v7.1 features demonstrated successfully!")
        print("📊 Generated plots:")
        print("   - v71_demo_temperature_sweep.png")
        print("   - v71_demo_volume_fraction_sweep.png")
        print("   - v71_demo_dlvo_potential.png")
        print("   - v71_demo_aggregation_kinetics.png")
        print("   - v71_demo_brownian_motion.png")
        print("   - v71_demo_cfd_temperature.png")
        print("   - v71_demo_cfd_velocity.png")
        print("   - v71_demo_cfd_pressure.png")
        print("\n✅ BKPS NFL Thermal Pro v7.1 is fully operational!")
    else:
        print(f"\n⚠️  {total - passed} demonstration(s) failed - see errors above")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
