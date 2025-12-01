"""
Example 7: AI Recommendation System Demonstration

This example demonstrates the AI-powered recommendation engine that
suggests optimal nanofluid configurations based on application requirements.

Features demonstrated:
- Intelligent nanoparticle selection
- Optimal concentration recommendations
- Flow velocity optimization
- Multi-objective optimization
- Stability-aware recommendations
- Cost-benefit analysis
"""

from nanofluid_simulator.ai_recommender import (
    AIRecommendationEngine,
    ApplicationType,
    OptimizationObjective,
    RecommendationConstraints
)


def print_section_header(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def print_recommendation(rec, rank=None):
    """Print a recommendation in formatted style."""
    prefix = f"#{rank} " if rank else ""
    print(f"{prefix}🎯 {rec.recommendation_text}")
    print(f"   Overall Score: {rec.overall_score:.3f}")
    print(f"   ├─ Thermal: {rec.thermal_score:.3f}")
    print(f"   ├─ Cost: {rec.cost_score:.3f}")
    print(f"   ├─ Stability: {rec.stability_score:.3f}")
    print(f"   └─ Efficiency: {rec.efficiency_score:.3f}")
    
    if rec.warnings:
        print(f"\n   ⚠️  Warnings:")
        for warning in rec.warnings:
            print(f"       • {warning}")
    
    if rec.alternatives:
        print(f"\n   🔄 Alternatives: {', '.join(rec.alternatives[:2])}")
    print()


def demo_heat_exchanger():
    """Demonstrate AI recommendations for heat exchanger application."""
    print_section_header("DEMO 1: Heat Exchanger Optimization")
    
    print("Application: Industrial heat exchanger")
    print("Requirements:")
    print("  • Operating temperature: 60°C")
    print("  • Maximize thermal performance")
    print("  • Budget-conscious")
    print("  • Stable operation required")
    print()
    
    # Create AI engine
    ai = AIRecommendationEngine(base_fluid="water")
    
    # Define constraints
    constraints = RecommendationConstraints(
        max_volume_fraction=0.05,
        max_viscosity_ratio=2.0,
        min_performance_index=1.0,
        require_stability=True
    )
    
    # Get recommendations (static mode first)
    print("Analyzing static properties...")
    recommendations = ai.recommend_configuration(
        application=ApplicationType.HEAT_EXCHANGER,
        temperature_celsius=60,
        objective=OptimizationObjective.BALANCE,
        constraints=constraints,
        top_n=3
    )
    
    print(f"\nTop 3 Recommendations:\n")
    for i, rec in enumerate(recommendations, 1):
        print_recommendation(rec, rank=i)


def demo_cpu_cooling():
    """Demonstrate AI recommendations for CPU cooling with flow."""
    print_section_header("DEMO 2: CPU Cooling with Flow Optimization")
    
    print("Application: High-performance CPU cooling")
    print("Requirements:")
    print("  • Operating temperature: 40°C")
    print("  • Flow velocity: 0.5 m/s")
    print("  • Channel: 3mm diameter, 0.5m length")
    print("  • Maximize heat transfer")
    print("  • Low pumping power")
    print()
    
    ai = AIRecommendationEngine(base_fluid="water")
    
    constraints = RecommendationConstraints(
        max_volume_fraction=0.03,
        max_viscosity_ratio=1.8,
        min_performance_index=1.05,
        max_pumping_power=1.0,  # 1 Watt max
        require_stability=True
    )
    
    flow_conditions = {
        'velocity': 0.5,
        'diameter': 0.003,  # 3mm
        'length': 0.5       # 0.5m
    }
    
    print("Analyzing flow physics and thermal performance...")
    recommendations = ai.recommend_configuration(
        application=ApplicationType.CPU_COOLING,
        temperature_celsius=40,
        objective=OptimizationObjective.MAXIMIZE_HEAT_TRANSFER,
        constraints=constraints,
        flow_conditions=flow_conditions,
        top_n=3
    )
    
    print(f"\nTop 3 Recommendations:\n")
    for i, rec in enumerate(recommendations, 1):
        print_recommendation(rec, rank=i)


def demo_concentration_optimization():
    """Demonstrate concentration optimization for specific nanoparticle."""
    print_section_header("DEMO 3: Concentration Optimization")
    
    print("Task: Find optimal concentration of Al2O3 in water")
    print("Goal: Maximize efficiency while maintaining stability")
    print()
    
    ai = AIRecommendationEngine(base_fluid="water")
    
    constraints = RecommendationConstraints(
        max_volume_fraction=0.06,
        max_viscosity_ratio=2.0,
        require_stability=True
    )
    
    print("Optimizing Al2O3 concentration...")
    rec = ai.optimize_concentration(
        nanoparticle="Al2O3",
        temperature_celsius=50,
        objective=OptimizationObjective.MAXIMIZE_EFFICIENCY,
        constraints=constraints
    )
    
    if rec:
        print("\n✅ Optimal Configuration Found:\n")
        print_recommendation(rec)
    else:
        print("\n❌ No optimal configuration found within constraints")


def demo_velocity_optimization():
    """Demonstrate flow velocity optimization."""
    print_section_header("DEMO 4: Flow Velocity Optimization")
    
    print("Task: Find optimal flow velocity for 2% Cu-water nanofluid")
    print("Channel: 10mm diameter, 1m length")
    print("Temperature: 40°C")
    print("Goal: Maximize Performance Index")
    print()
    
    ai = AIRecommendationEngine(base_fluid="water")
    
    constraints = RecommendationConstraints(
        max_pressure_drop=5000,  # 5 kPa max
        max_pumping_power=2.0,   # 2 W max
        min_performance_index=1.0
    )
    
    print("Scanning velocity range 0.1 - 5.0 m/s...")
    optimal_velocity, rec = ai.optimize_velocity(
        nanoparticle="Cu",
        volume_fraction=0.02,
        temperature_celsius=40,
        channel_diameter=0.01,
        channel_length=1.0,
        constraints=constraints
    )
    
    if rec:
        print(f"\n✅ Optimal Velocity Found: {optimal_velocity:.2f} m/s\n")
        print_recommendation(rec)
        
        print("Velocity Analysis:")
        print(f"  • Reynolds: {rec.reynolds:.0f} ({rec.flow_regime if hasattr(rec, 'flow_regime') else 'Turbulent'})")
        print(f"  • Nusselt: {rec.nusselt:.2f}")
        print(f"  • Heat Transfer: {rec.heat_transfer_coeff:.1f} W/m²·K")
        print(f"  • Pressure Drop: {rec.pressure_drop/1000:.2f} kPa")
        print(f"  • Pumping Power: {rec.pumping_power:.3f} W")
        print(f"  • Performance Index: {rec.performance_index:.3f}")
    else:
        print("\n❌ No optimal velocity found within constraints")


def demo_comparison():
    """Compare recommendations across different objectives."""
    print_section_header("DEMO 5: Multi-Objective Comparison")
    
    print("Application: General cooling system")
    print("Temperature: 50°C")
    print("Comparing different optimization objectives...")
    print()
    
    ai = AIRecommendationEngine(base_fluid="water")
    
    objectives = [
        (OptimizationObjective.MAXIMIZE_HEAT_TRANSFER, "Max Heat Transfer"),
        (OptimizationObjective.MINIMIZE_COST, "Min Cost"),
        (OptimizationObjective.MAXIMIZE_EFFICIENCY, "Max Efficiency"),
        (OptimizationObjective.BALANCE, "Balanced")
    ]
    
    for obj, name in objectives:
        print(f"\n{'─'*80}")
        print(f"Objective: {name}")
        print('─'*80)
        
        recommendations = ai.recommend_configuration(
            application=ApplicationType.GENERAL,
            temperature_celsius=50,
            objective=obj,
            top_n=1
        )
        
        if recommendations:
            rec = recommendations[0]
            print(f"\n🏆 Winner: {rec.nanoparticle} at {rec.volume_fraction*100:.1f}%")
            print(f"   Enhancement: {rec.enhancement:.2f}%")
            print(f"   Overall Score: {rec.overall_score:.3f}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("  AI RECOMMENDATION SYSTEM DEMONSTRATION")
    print("  Nanofluid Simulator v2.1.0 - Intelligent Configuration Optimizer")
    print("="*80)
    
    print("\nThe AI Recommendation Engine uses multi-objective optimization")
    print("to suggest the best nanofluid configurations for your application.")
    print("\nFeatures:")
    print("  🎯 Intelligent nanoparticle selection")
    print("  📊 Optimal concentration recommendations")
    print("  🌊 Flow velocity optimization")
    print("  ⚖️  Multi-objective optimization (thermal/cost/stability/efficiency)")
    print("  ⚠️  Stability prediction and warnings")
    print("  💰 Cost-benefit analysis")
    
    # Run demonstrations
    input("\n\nPress Enter to see Heat Exchanger recommendations...")
    demo_heat_exchanger()
    
    input("\nPress Enter to see CPU Cooling with Flow optimization...")
    demo_cpu_cooling()
    
    input("\nPress Enter to see Concentration optimization...")
    demo_concentration_optimization()
    
    input("\nPress Enter to see Velocity optimization...")
    demo_velocity_optimization()
    
    input("\nPress Enter to see Multi-Objective comparison...")
    demo_comparison()
    
    print_section_header("DEMONSTRATION COMPLETE ✓")
    print("🎉 The AI Recommendation System is fully operational!")
    print("\nKey Benefits:")
    print("  ✅ Saves experimental time by predicting optimal configurations")
    print("  ✅ Considers multiple objectives simultaneously")
    print("  ✅ Provides stability warnings to prevent failures")
    print("  ✅ Cost-conscious recommendations")
    print("  ✅ Application-specific optimization")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
