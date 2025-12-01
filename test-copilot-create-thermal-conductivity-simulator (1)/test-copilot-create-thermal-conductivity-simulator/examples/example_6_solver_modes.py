"""
Example 6: Solver Mode Demonstration - Static vs Flow-Enhanced

This example demonstrates the difference between Static Property Solver
and Flow-Enhanced Solver modes in the Nanofluid Simulator v2.1.0.

Features demonstrated:
- Static Mode: Temperature + concentration effects only
- Flow Mode: Full flow physics integration
- Performance comparison between modes
- Visualization of key differences
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from nanofluid_simulator import (
    EnhancedNanofluidSimulator,
    FlowNanofluidSimulator,
    SolverMode,
    SolverModeConfig
)


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def demonstrate_static_mode():
    """Demonstrate Static Property Solver."""
    print_section_header("STATIC PROPERTY SOLVER MODE 🔬")
    
    mode_info = SolverModeConfig.get_mode_description(SolverMode.STATIC)
    print(f"{mode_info['icon']} {mode_info['name']}")
    print(f"Description: {mode_info['description']}\n")
    print(f"Best for: {mode_info['best_for']}\n")
    
    # Create static simulator
    sim = EnhancedNanofluidSimulator()
    
    # Configuration
    print("Configuration:")
    print("  Base fluid: Water")
    print("  Temperature: 40°C")
    print("  Nanoparticle: 2% Cu (30 nm)")
    print()
    
    sim.set_base_fluid("water")
    sim.set_temperature_celsius(40)
    sim.add_nanoparticle("Cu", volume_fraction=0.02, particle_size=30)
    
    # Calculate with classical models
    print("Calculating with classical models...")
    results = sim.calculate_all_applicable_models()
    
    print(f"\nResults ({len(results)} models calculated):\n")
    print(f"{'Model':<30} {'k_eff (W/m·K)':<15} {'Enhancement (%)':<15}")
    print("-" * 60)
    
    for r in results[:10]:  # Show first 10 models
        print(f"{r.model_name:<30} {r.k_effective:<15.6f} {r.enhancement_k:<15.2f}")
    
    if len(results) > 10:
        print(f"... and {len(results)-10} more models\n")
    
    # Get average enhancement
    avg_enhancement = np.mean([r.enhancement_k for r in results])
    print(f"\n📊 Average Enhancement: {avg_enhancement:.2f}%")
    print(f"📊 Range: {min(r.enhancement_k for r in results):.2f}% to {max(r.enhancement_k for r in results):.2f}%")
    
    print("\n✓ Static mode complete - no flow effects included")
    
    return results


def demonstrate_flow_mode():
    """Demonstrate Flow-Enhanced Solver."""
    print_section_header("FLOW-ENHANCED SOLVER MODE (DYNAMIC) 🌊")
    
    mode_info = SolverModeConfig.get_mode_description(SolverMode.FLOW)
    print(f"{mode_info['icon']} {mode_info['name']}")
    print(f"Description: {mode_info['description']}\n")
    print(f"Best for: {mode_info['best_for']}\n")
    
    # Create flow simulator
    sim = FlowNanofluidSimulator()
    
    # Configuration
    print("Configuration:")
    print("  Base fluid: Water")
    print("  Temperature: 40°C")
    print("  Nanoparticle: 2% Cu (30 nm)")
    print("  Flow velocity: 1.0 m/s")
    print("  Channel: D=10 mm, L=1 m")
    print()
    
    sim.set_base_fluid("water")
    sim.set_temperature_celsius(40)
    sim.add_nanoparticle("Cu", volume_fraction=0.02, particle_size=30)
    sim.set_flow_velocity(1.0)
    sim.set_channel_geometry(diameter=0.01, length=1.0)
    
    # Calculate flow physics
    print("Calculating with flow-enhanced models...")
    results = sim.calculate_all_flow_models()
    
    print(f"\nResults ({len(results)} flow models calculated):\n")
    print(f"{'Model':<25} {'k_eff':<10} {'Enh%':<8} {'Re':<8} {'Nu':<8} {'h (W/m²·K)':<12} {'PI':<8}")
    print("-" * 85)
    
    for r in results:
        print(f"{r.model_name:<25} {r.k_effective:<10.4f} {r.enhancement_k:<8.2f} "
              f"{r.Reynolds:<8.0f} {r.Nusselt:<8.2f} {r.h_convective:<12.1f} {r.performance_index:<8.3f}")
    
    print(f"\n🌊 Flow Regime: {results[0].flow_regime}")
    print(f"📊 Reynolds Number: {results[0].Reynolds:.0f}")
    print(f"📊 Pressure Drop: {results[0].pressure_drop_Pa/1000:.3f} kPa")
    print(f"📊 Pumping Power: {results[0].pumping_power_W:.3f} W")
    print(f"\n⚡ Performance Index: {min(r.performance_index for r in results):.3f} to {max(r.performance_index for r in results):.3f}")
    print(f"   (PI > 1.0 means nanofluid is beneficial despite pumping cost)")
    
    print("\n✓ Flow mode complete - full flow physics integrated")
    
    return results


def compare_modes():
    """Compare Static vs Flow modes."""
    print_section_header("MODE COMPARISON: STATIC vs FLOW")
    
    # Static mode
    sim_static = EnhancedNanofluidSimulator()
    sim_static.set_base_fluid("water")
    sim_static.set_temperature_celsius(40)
    sim_static.add_nanoparticle("Cu", volume_fraction=0.02, particle_size=30)
    
    results_static = sim_static.calculate_all_applicable_models()
    k_static_avg = np.mean([r.k_effective for r in results_static])
    k_static_range = (min(r.k_effective for r in results_static),
                      max(r.k_effective for r in results_static))
    
    # Flow mode
    sim_flow = FlowNanofluidSimulator()
    sim_flow.set_base_fluid("water")
    sim_flow.set_temperature_celsius(40)
    sim_flow.add_nanoparticle("Cu", volume_fraction=0.02, particle_size=30)
    sim_flow.set_flow_velocity(1.0)
    sim_flow.set_channel_geometry(diameter=0.01, length=1.0)
    
    results_flow = sim_flow.calculate_all_flow_models()
    k_flow_avg = np.mean([r.k_effective for r in results_flow])
    k_flow_range = (min(r.k_effective for r in results_flow),
                    max(r.k_effective for r in results_flow))
    
    print("┌─────────────────────────────────────────────────────────┐")
    print("│  Feature Comparison                                     │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│  Models Available        │ Static: {len(results_static):<3} │ Flow: {len(results_flow):<3}       │")
    print(f"│  k_eff Average (W/m·K)   │ {k_static_avg:<8.4f}    │ {k_flow_avg:<8.4f}    │")
    print(f"│  k_eff Range (W/m·K)     │ {k_static_range[0]:.4f}-{k_static_range[1]:.4f} │ {k_flow_range[0]:.4f}-{k_flow_range[1]:.4f} │")
    print("│  Reynolds Number         │ N/A         │ {:>8.0f}      │".format(results_flow[0].Reynolds))
    print("│  Nusselt Number          │ N/A         │ {:>8.2f}      │".format(results_flow[0].Nusselt))
    print("│  Heat Transfer (W/m²·K)  │ N/A         │ {:>8.1f}      │".format(results_flow[0].h_convective))
    print("│  Pressure Drop (kPa)     │ N/A         │ {:>8.3f}      │".format(results_flow[0].pressure_drop_Pa/1000))
    print("│  Pumping Power (W)       │ N/A         │ {:>8.3f}      │".format(results_flow[0].pumping_power_W))
    print("│  Performance Index       │ N/A         │ {:>8.3f}      │".format(results_flow[0].performance_index))
    print("└─────────────────────────────────────────────────────────┘")
    
    print("\n🎯 Key Insights:")
    print("  1. Static mode provides property estimation only")
    print("  2. Flow mode adds thermal-hydraulic performance analysis")
    print("  3. Flow effects can significantly enhance thermal conductivity")
    print(f"  4. Flow enhancement: {((k_flow_avg - k_static_avg)/k_static_avg*100):.2f}% on average")
    print("  5. Performance Index shows cost-benefit of using nanofluids")


def demonstrate_velocity_effect():
    """Demonstrate effect of velocity in Flow mode."""
    print_section_header("VELOCITY EFFECT IN FLOW MODE")
    
    sim = FlowNanofluidSimulator()
    sim.set_base_fluid("water")
    sim.set_temperature_celsius(40)
    sim.add_nanoparticle("Cu", volume_fraction=0.02, particle_size=30)
    sim.set_channel_geometry(diameter=0.01, length=1.0)
    
    velocities = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("Velocity sweep analysis:")
    print(f"\n{'Velocity (m/s)':<15} {'Re':<10} {'k_eff (W/m·K)':<15} {'h (W/m²·K)':<15} {'P_pump (W)':<12} {'PI':<8}")
    print("-" * 85)
    
    for v in velocities:
        sim.set_flow_velocity(v)
        results = sim.calculate_all_flow_models()
        
        # Use Buongiorno model as representative
        r = results[0]
        
        print(f"{v:<15.1f} {r.Reynolds:<10.0f} {r.k_effective:<15.6f} {r.h_convective:<15.1f} "
              f"{r.pumping_power_W:<12.3f} {r.performance_index:<8.3f}")
    
    print("\n📈 Observations:")
    print("  • Heat transfer coefficient (h) increases with velocity")
    print("  • Pumping power increases with velocity³")
    print("  • Performance Index helps optimize velocity selection")
    print("  • Higher velocities = better heat transfer but higher cost")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("  NANOFLUID SIMULATOR v2.1.0 - SOLVER MODE DEMONSTRATION")
    print("  World's Most Advanced Flow-Integrated Nanofluid Modeling")
    print("="*80)
    
    print("\nThis example demonstrates the two solver modes:")
    print("  🔬 Static Property Solver - Classical property estimation")
    print("  🌊 Flow-Enhanced Solver - Complete flow physics integration")
    
    # Demonstrate each mode
    print("\n" + "─"*80)
    input("Press Enter to demonstrate Static Mode...")
    results_static = demonstrate_static_mode()
    
    print("\n" + "─"*80)
    input("Press Enter to demonstrate Flow Mode...")
    results_flow = demonstrate_flow_mode()
    
    print("\n" + "─"*80)
    input("Press Enter to compare modes...")
    compare_modes()
    
    print("\n" + "─"*80)
    input("Press Enter to demonstrate velocity effects...")
    demonstrate_velocity_effect()
    
    print_section_header("DEMONSTRATION COMPLETE ✓")
    print("Key Takeaways:")
    print("  1. Static Mode: Fast, simple property calculations")
    print("  2. Flow Mode: Complete thermal-hydraulic analysis")
    print("  3. Flow physics can significantly enhance performance")
    print("  4. Performance Index provides optimization guidance")
    print("  5. Choose mode based on application requirements")
    print("\n🚀 Both modes fully operational and integrated!")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
