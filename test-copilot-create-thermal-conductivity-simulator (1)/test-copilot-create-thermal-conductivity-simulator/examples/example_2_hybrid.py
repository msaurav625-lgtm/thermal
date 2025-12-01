#!/usr/bin/env python3
"""
Example 2: Hybrid Nanofluid Simulation

This example demonstrates:
- Creating hybrid nanofluids with multiple nanoparticle types
- Using mixed base fluids
- Comparing hybrid models
- Exporting results
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanofluid_simulator import EnhancedNanofluidSimulator
from nanofluid_simulator.export import export_results
import json

def main():
    print("=" * 70)
    print("EXAMPLE 2: Hybrid Nanofluid Simulation")
    print("=" * 70)
    print()
    
    # Create simulator
    sim = EnhancedNanofluidSimulator()
    
    # Configure hybrid nanofluid: Water-EG mixture + Cu + Al2O3
    print("Configuring hybrid nanofluid...")
    sim.set_base_fluid("water_eg_50_50")  # 50:50 Water-Ethylene Glycol
    sim.set_temperature_celsius(40)
    
    # Add two different nanoparticles
    sim.add_nanoparticle("Cu", volume_fraction=0.01, particle_size=30, sphericity=1.0)
    sim.add_nanoparticle("Al2O3", volume_fraction=0.01, particle_size=40, sphericity=0.9)
    
    # Display configuration
    print(sim.get_configuration_summary())
    print()
    
    # Calculate
    print("Calculating thermal conductivity for hybrid nanofluid...")
    results = sim.calculate_all_applicable_models()
    print(f"Computed {len(results)} models successfully!\n")
    
    # Display results
    print("=" * 70)
    print("RESULTS - HYBRID NANOFLUID")
    print("=" * 70)
    print(f"{'Model':<35} {'k_eff (W/m·K)':<15} {'Enhancement (%)'}")
    print("-" * 70)
    
    for result in results:
        print(f"{result.model_name:<35} {result.k_effective:<15.6f} "
              f"{result.enhancement_k:>10.2f}%")
    
    print("=" * 70)
    print()
    
    # Export results to JSON
    output_file = "hybrid_nanofluid_results.json"
    json_data = sim.export_to_json(results)
    
    with open(output_file, 'w') as f:
        f.write(json_data)
    
    print(f"✓ Results exported to: {output_file}")
    
    # Export to CSV
    try:
        export_results(results, "hybrid_nanofluid_results.csv", format='csv')
        print(f"✓ Results exported to: hybrid_nanofluid_results.csv")
    except Exception as e:
        print(f"Note: CSV export requires pandas: {e}")
    
    print()
    print("✓ Example completed successfully!")

if __name__ == "__main__":
    main()
