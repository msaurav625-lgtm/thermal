"""
Quick AI Recommendation Demo - Nanofluid Simulator v2.1.0

Fast demonstration of the AI recommendation system capabilities.
"""

from nanofluid_simulator import (
    AIRecommendationEngine,
    ApplicationType,
    OptimizationObjective,
    RecommendationConstraints
)


def main():
    """Quick AI recommendation demonstration."""
    
    print("=" * 70)
    print(" AI RECOMMENDATION SYSTEM - QUICK DEMO")
    print(" Nanofluid Simulator v2.1.0")
    print("=" * 70)
    print()
    
    engine = AIRecommendationEngine()
    
    # DEMO 1: Heat Exchanger with Balance Objective
    print("📊 DEMO 1: Heat Exchanger - Balanced Optimization")
    print("-" * 70)
    
    recs = engine.recommend_configuration(
        application=ApplicationType.HEAT_EXCHANGER,
        temperature_celsius=60,
        objective=OptimizationObjective.BALANCE,
        top_n=5
    )
    
    if recs:
        print(f"\n✅ Top {len(recs)} Recommendations:\n")
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec.nanoparticle} @ {rec.volume_fraction:.1%} (d={rec.particle_size:.0f} nm)")
            print(f"   Thermal: k_eff = {rec.thermal_conductivity:.3f} W/m·K "
                  f"(+{rec.enhancement:.1f}%)")
            print(f"   Viscosity: μ = {rec.viscosity:.6f} Pa·s "
                  f"({rec.viscosity_ratio:.2f}x base)")
            print(f"   Stability: {rec.stability_status}")
            print(f"   Overall Score: {rec.overall_score:.3f}")
            if rec.recommendation_text:
                print(f"   💡 {rec.recommendation_text}")
            print()
    else:
        print("❌ No recommendations found\n")
    
    # DEMO 2: CPU Cooling - Maximize Heat Transfer
    print("\n" + "=" * 70)
    print("🔥 DEMO 2: CPU Cooling - Maximum Heat Transfer")
    print("-" * 70)
    
    # Relax viscosity constraint for high performance
    constraints = RecommendationConstraints(
        max_viscosity_ratio=2.5,  # Allow higher viscosity for better thermal
        require_stability=True
    )
    
    recs = engine.recommend_configuration(
        application=ApplicationType.CPU_COOLING,
        temperature_celsius=70,
        objective=OptimizationObjective.MAXIMIZE_HEAT_TRANSFER,
        constraints=constraints,
        top_n=3
    )
    
    if recs:
        print(f"\n✅ Top {len(recs)} High-Performance Recommendations:\n")
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec.nanoparticle} @ {rec.volume_fraction:.1%}")
            print(f"   Enhancement: +{rec.enhancement:.1f}%")
            print(f"   Viscosity Ratio: {rec.viscosity_ratio:.2f}x")
            print(f"   Score: {rec.overall_score:.3f}")
            print()
    else:
        print("❌ No recommendations found\n")
    
    # DEMO 3: Cost-Optimized Solution
    print("\n" + "=" * 70)
    print("💰 DEMO 3: Cost-Optimized Configuration")
    print("-" * 70)
    
    recs = engine.recommend_configuration(
        application=ApplicationType.HEAT_EXCHANGER,
        temperature_celsius=50,
        objective=OptimizationObjective.MINIMIZE_COST,
        top_n=3
    )
    
    if recs:
        print(f"\n✅ Top {len(recs)} Budget-Friendly Recommendations:\n")
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec.nanoparticle} @ {rec.volume_fraction:.1%}")
            print(f"   Enhancement: +{rec.enhancement:.1f}%")
            print(f"   Cost Score: {rec.cost_score:.3f} (lower is better)")
            print(f"   Overall Score: {rec.overall_score:.3f}")
            print()
    else:
        print("❌ No recommendations found\n")
    
    print("=" * 70)
    print("✨ AI Recommendation Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
