"""
Test New Interfacial Layer Models and Features
"""

from nanofluid_simulator import EnhancedNanofluidSimulator

# Create simulator
sim = EnhancedNanofluidSimulator()
sim.set_base_fluid("water")
sim.set_temperature_celsius(25)
sim.add_nanoparticle("Cu", volume_fraction=0.02, particle_size=30)

print("="*70)
print(" Nanofluid Simulator - Testing New Features")
print("="*70)
print(f"\nConfiguration:")
print(f"  Base Fluid: Water")
print(f"  Nanoparticle: Cu")
print(f"  Volume Fraction: 2.0%")
print(f"  Particle Size: 30 nm")
print(f"  Temperature: 25°C")
print("\n" + "="*70)
print(" INTERFACIAL LAYER MODELS (Surface Interaction Effects)")
print("="*70)

# Test interfacial models
print("\n1. Xue Interfacial Layer Model (β=0.1):")
result = sim.calculate_xue_interfacial(beta=0.1)
print(f"   k_eff = {result.k_effective:.6f} W/m·K")
print(f"   Enhancement: {result.enhancement_k:.2f}%")
print(f"   μ_eff = {result.mu_effective*1000:.4f} mPa·s")

print("\n2. Leong-Yang Interfacial Model (h=2nm, k_ratio=2.0):")
result = sim.calculate_leong_yang_interfacial(h=2.0, k_layer_ratio=2.0)
print(f"   k_eff = {result.k_effective:.6f} W/m·K")
print(f"   Enhancement: {result.enhancement_k:.2f}%")
print(f"   μ_eff = {result.mu_effective*1000:.4f} mPa·s")

print("\n3. Yu-Choi Interfacial Model (β=0.1):")
result = sim.calculate_yu_choi_interfacial(beta=0.1)
print(f"   k_eff = {result.k_effective:.6f} W/m·K")
print(f"   Enhancement: {result.enhancement_k:.2f}%")
print(f"   μ_eff = {result.mu_effective*1000:.4f} mPa·s")

# Compare with classical models
print("\n" + "="*70)
print(" COMPARISON WITH CLASSICAL MODELS")
print("="*70)

all_results = sim.calculate_all_applicable_models()
print(f"\nTotal models calculated: {len(all_results)}")
print("\n{:<35} {:>15} {:>12}".format("Model", "k_eff (W/m·K)", "Enhancement"))
print("-"*70)
for r in all_results:
    print("{:<35} {:>15.6f} {:>11.2f}%".format(
        r.model_name, r.k_effective, r.enhancement_k
    ))

# Test viscosity vs temperature
print("\n" + "="*70)
print(" VISCOSITY VS TEMPERATURE ANALYSIS")
print("="*70)

results_dict = sim.parametric_study_temperature(
    temperature_range=(293.15, 353.15),  # 20°C to 80°C
    n_points=7
)

print("\nTemperature effects on viscosity (Maxwell model):")
print("\n{:>10} {:>15} {:>15}".format("T (°C)", "μ (mPa·s)", "μ/μ_bf"))
print("-"*45)

maxwell_results = results_dict.get("Maxwell", [])
for r in maxwell_results:
    T_celsius = r.temperature - 273.15
    mu_ratio = r.mu_effective / r.to_dict().get('mu_bf', 0.001)
    print("{:>10.1f} {:>15.4f} {:>15.2f}".format(
        T_celsius, r.mu_effective*1000, mu_ratio
    ))

print("\n" + "="*70)
print(" All Properties at Different Temperatures")
print("="*70)

print("\n{:>8} {:>12} {:>12} {:>10} {:>10}".format(
    "T (°C)", "k (W/m·K)", "μ (mPa·s)", "ρ (kg/m³)", "Pr"
))
print("-"*70)

for r in maxwell_results:
    T_celsius = r.temperature - 273.15
    print("{:>8.1f} {:>12.6f} {:>12.4f} {:>10.1f} {:>10.3f}".format(
        T_celsius, r.k_effective, r.mu_effective*1000, 
        r.rho_effective, r.pr_effective
    ))

print("\n" + "="*70)
print(" KEY OBSERVATIONS")
print("="*70)
print("""
1. INTERFACIAL LAYER EFFECTS:
   - Xue, Leong-Yang, and Yu-Choi models account for nanolayer
   - Surface interaction increases thermal conductivity enhancement
   - Critical for understanding nanofluid behavior

2. VISCOSITY-TEMPERATURE RELATIONSHIP:
   - Viscosity decreases with temperature (expected behavior)
   - Nanoparticles increase viscosity slightly at all temperatures
   - Important for pumping power calculations

3. PRANDTL NUMBER:
   - Changes with temperature due to varying properties
   - Critical for heat transfer correlations
   - Used in convective heat transfer design

4. GUI NEW FEATURES:
   - "💧 Viscosity vs T" button: Plot viscosity variation
   - "🔬 All Properties vs T" button: 2x2 subplot with all properties
   - Enhanced analysis capabilities for research
""")

print("\n" + "="*70)
print(" Test Complete! ✓")
print("="*70)
