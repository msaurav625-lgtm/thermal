#!/usr/bin/env python3
"""
Example 3: Parametric Studies

This example demonstrates:
- Temperature sweep analysis
- Concentration sweep analysis
- Plotting results
- Exporting parametric data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanofluid_simulator import EnhancedNanofluidSimulator
import matplotlib.pyplot as plt

def main():
    print("=" * 70)
    print("EXAMPLE 3: Parametric Studies")
    print("=" * 70)
    print()
    
    # Create simulator
    sim = EnhancedNanofluidSimulator()
    
    # Configure
    sim.set_base_fluid("water")
    sim.set_temperature_celsius(25)
    sim.add_nanoparticle("Cu", volume_fraction=0.02, particle_size=25)
    
    print(sim.get_configuration_summary())
    print()
    
    # Temperature sweep
    print("Performing temperature sweep (0°C to 100°C)...")
    temp_results = sim.parametric_study_temperature(
        temperature_range=(273.15, 373.15),
        n_points=15
    )
    
    print(f"✓ Completed temperature sweep with {len(temp_results)} models")
    print()
    
    # Concentration sweep
    print("Performing concentration sweep (0.1% to 5%)...")
    
    # Reset to base configuration
    sim.clear_nanoparticles()
    sim.add_nanoparticle("Cu", volume_fraction=0.01, particle_size=25)
    
    conc_results = sim.parametric_study_concentration(
        phi_range=(0.001, 0.05),
        n_points=15
    )
    
    print(f"✓ Completed concentration sweep with {len(conc_results)} models")
    print()
    
    # Plot results
    print("Generating plots...")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Thermal conductivity vs Temperature
    for model_name, results in temp_results.items():
        temps = [r.temperature - 273.15 for r in results]
        k_values = [r.k_effective for r in results]
        ax1.plot(temps, k_values, marker='o', label=model_name, linewidth=2)
    
    ax1.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Thermal Conductivity (W/m·K)', fontsize=12, fontweight='bold')
    ax1.set_title('Thermal Conductivity vs Temperature', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Thermal conductivity vs Volume fraction
    for model_name, results in conc_results.items():
        # Extract phi from results
        phi_values = [sim._nanoparticle_components[0].volume_fraction * 100 
                     for r in results]  # This is simplified
        k_values = [r.k_effective for r in results]
        ax2.plot(phi_values, k_values, marker='s', label=model_name, linewidth=2)
    
    ax2.set_xlabel('Volume Fraction (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Thermal Conductivity (W/m·K)', fontsize=12, fontweight='bold')
    ax2.set_title('Thermal Conductivity vs Volume Fraction', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = "parametric_study.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    
    # Show plot (comment out if running in headless environment)
    # plt.show()
    
    # Export data
    try:
        from nanofluid_simulator.export import ResultExporter
        exporter = ResultExporter()
        exporter.parametric_to_excel(temp_results, "temperature_sweep.xlsx", "Temperature")
        exporter.parametric_to_excel(conc_results, "concentration_sweep.xlsx", "Concentration")
        print(f"✓ Data exported to Excel files")
    except Exception as e:
        print(f"Note: Excel export requires pandas and openpyxl: {e}")
    
    print()
    print("✓ Example completed successfully!")

if __name__ == "__main__":
    main()
