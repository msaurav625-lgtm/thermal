#!/usr/bin/env python3
"""
Example 1: Basic Mono Nanofluid Simulation

This example demonstrates how to:
- Create a simple nanofluid simulation
- Calculate thermal conductivity using multiple models
- Display and compare results
"""

from nanofluid_simulator import EnhancedNanofluidSimulator

def main():
    print("=" * 70)
    print("EXAMPLE 1: Basic Mono Nanofluid Simulation")
    print("=" * 70)
    print()
    
    # Create simulator
    sim = EnhancedNanofluidSimulator()
    
    # Configure: Water + Copper nanoparticles
    print("Configuring nanofluid...")
    sim.set_base_fluid("water")
    sim.set_temperature_celsius(25)
    sim.add_nanoparticle("Cu", volume_fraction=0.01, particle_size=25)
    
    # Display configuration
    print(sim.get_configuration_summary())
    print()
    
    # Calculate using all models
    print("Calculating thermal conductivity...")
    results = sim.calculate_all_applicable_models()
    print(f"Computed {len(results)} models successfully!\n")
    
    # Display results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Model':<30} {'k_eff (W/m·K)':<15} {'Enhancement (%)'}")
    print("-" * 70)
    
    for result in results:
        print(f"{result.model_name:<30} {result.k_effective:<15.6f} "
              f"{result.enhancement_k:>10.2f}%")
    
    print("=" * 70)
    print()
    
    # Show detailed properties for first model
    print("Detailed Properties (Maxwell Model):")
    print("-" * 70)
    first_result = results[0]
    print(f"Thermal Conductivity: {first_result.k_effective:.6f} W/m·K")
    print(f"Dynamic Viscosity: {first_result.mu_effective:.6f} Pa·s")
    print(f"Density: {first_result.rho_effective:.2f} kg/m³")
    print(f"Specific Heat: {first_result.cp_effective:.2f} J/kg·K")
    print(f"Thermal Diffusivity: {first_result.alpha_effective*1e7:.4f} × 10⁻⁷ m²/s")
    print(f"Prandtl Number: {first_result.pr_effective:.4f}")
    print()
    
    print("✓ Example completed successfully!")

if __name__ == "__main__":
    main()
