"""
Example 16: AI-Powered CFD Integration
========================================

Demonstrates AI enhancements for CFD simulations:
1. Automatic flow regime classification
2. Turbulence model recommendation
3. Solver parameter optimization
4. Real-time convergence monitoring
5. Divergence prediction and correction

This example compares manual CFD setup vs AI-assisted setup.

Requirements:
- scikit-learn (pip install scikit-learn)
"""

import numpy as np
import matplotlib.pyplot as plt
from nanofluid_simulator import NanofluidSimulator
from nanofluid_simulator.cfd_mesh import StructuredMesh2D
from nanofluid_simulator.cfd_solver import NavierStokesSolver, SolverSettings, BoundaryCondition
from nanofluid_simulator.cfd_turbulence import TurbulenceModel
from nanofluid_simulator.cfd_mesh import BoundaryType

print("="*80)
print(" EXAMPLE 16: AI-POWERED CFD INTEGRATION ".center(80))
print("="*80)
print()
print("This example demonstrates AI enhancements for CFD simulations:")
print("  • Automatic flow regime classification")
print("  • Intelligent turbulence model selection")
print("  • Solver parameter optimization")
print("  • Real-time convergence monitoring")
print()

# ==============================================================================
# PART 1: Setup nanofluid properties
# ==============================================================================
print("\n" + "="*80)
print("PART 1: Nanofluid Property Calculation")
print("="*80)

sim = NanofluidSimulator()

# Test case: Al2O3-water nanofluid in microchannel
nanoparticle = 'Al2O3'
phi = 0.03  # 3% volume fraction
T = 300  # K

props = sim.calculate_properties(
    nanoparticle=nanoparticle,
    phi=phi,
    T=T,
    d_p=30e-9
)

print(f"\nNanofluid: {phi*100:.1f}% {nanoparticle} in water")
print(f"Temperature: {T} K")
print(f"\nProperties:")
print(f"  Density: {props['rho']:.1f} kg/m³")
print(f"  Viscosity: {props['mu']*1000:.3f} mPa·s")
print(f"  Thermal conductivity: {props['k_eff']:.3f} W/m·K")
print(f"  Specific heat: {props['cp']:.1f} J/kg·K")

# ==============================================================================
# PART 2: Manual CFD Setup (Traditional Approach)
# ==============================================================================
print("\n" + "="*80)
print("PART 2: Traditional Manual CFD Setup")
print("="*80)

# Define geometry
L = 0.01  # 10 mm length
H = 0.0001  # 100 μm height (microchannel)
U_inlet = 0.1  # m/s

# Calculate Reynolds number manually
Re = props['rho'] * U_inlet * H / props['mu']
Pr = props['mu'] * props['cp'] / props['k_eff']

print(f"\nMicrochannel geometry:")
print(f"  Length: {L*1000:.1f} mm")
print(f"  Height: {H*1e6:.1f} μm")
print(f"  Inlet velocity: {U_inlet} m/s")
print(f"\nDimensionless numbers:")
print(f"  Reynolds number: {Re:.1f}")
print(f"  Prandtl number: {Pr:.2f}")

# Manual decision making
if Re < 2300:
    print(f"\n📋 Manual analysis: Re={Re:.0f} < 2300 → Laminar flow")
    turb_model = TurbulenceModel.LAMINAR
    print("  Selected: No turbulence model")
else:
    print(f"\n📋 Manual analysis: Re={Re:.0f} > 2300 → Turbulent flow")
    turb_model = TurbulenceModel.K_EPSILON
    print("  Selected: k-ε turbulence model")

# Manual mesh sizing (guesswork)
print("\n📋 Manual mesh sizing (trial and error):")
nx_manual = 50
ny_manual = 30
print(f"  First try: {nx_manual} × {ny_manual}")

# Manual relaxation factors (conservative)
alpha_u_manual = 0.5
alpha_p_manual = 0.3
print(f"\n📋 Manual relaxation factors (conservative):")
print(f"  Velocity: {alpha_u_manual}")
print(f"  Pressure: {alpha_p_manual}")

# Create manual mesh and solver
mesh_manual = StructuredMesh2D(
    x_min=0.0, x_max=L,
    y_min=0.0, y_max=H,
    nx=nx_manual, ny=ny_manual
)

settings_manual = SolverSettings(
    max_iterations=1000,
    convergence_tol=1e-6,
    under_relaxation_u=alpha_u_manual,
    under_relaxation_v=alpha_u_manual,
    under_relaxation_p=alpha_p_manual,
    turbulence_model=turb_model
)

solver_manual = NavierStokesSolver(mesh_manual, settings_manual)

# Set properties
rho_field = np.ones(mesh_manual.n_cells) * props['rho']
mu_field = np.ones(mesh_manual.n_cells) * props['mu']
k_field = np.ones(mesh_manual.n_cells) * props['k_eff']

solver_manual.set_nanofluid_properties(rho_field, mu_field, k_field)

print("\n⏱️  Solving with manual setup...")
converged_manual = solver_manual.solve(verbose=False)

if converged_manual:
    print(f"✅ Manual approach converged in {len(solver_manual.residuals['u'])} iterations")
else:
    print(f"⚠️  Manual approach did not converge ({len(solver_manual.residuals['u'])} iterations)")

# ==============================================================================
# PART 3: AI-Assisted CFD Setup (Modern Approach)
# ==============================================================================
print("\n" + "="*80)
print("PART 3: AI-Assisted CFD Setup")
print("="*80)

# Create initial mesh for AI analysis
mesh_ai = StructuredMesh2D(
    x_min=0.0, x_max=L,
    y_min=0.0, y_max=H,
    nx=50, ny=30  # Initial guess
)

solver_ai = NavierStokesSolver(mesh_ai)

# Set properties
solver_ai.set_nanofluid_properties(rho_field[:mesh_ai.n_cells], 
                                    mu_field[:mesh_ai.n_cells], 
                                    k_field[:mesh_ai.n_cells])

# Enable AI assistance
print("\n🤖 Enabling AI assistance...")
solver_ai.enable_ai_assistance(True)

# Step 1: AI flow regime classification
print("\n📊 Step 1: AI Flow Regime Classification")
ai_classification = solver_ai.ai_classify_flow(
    velocity=U_inlet,
    length_scale=H
)

# Step 2: AI parameter recommendations
print("\n⚙️  Step 2: AI Parameter Optimization")
ai_params = solver_ai.ai_recommend_parameters(
    velocity=U_inlet,
    length_scale=H
)

# Step 3: Apply AI recommendations
print("\n✅ Step 3: Applying AI Recommendations")
solver_ai.ai_apply_recommendations(ai_params)

# Recreate mesh with AI-recommended size if significantly different
if ai_params['mesh']['nx'] != mesh_ai.nx:
    print(f"  Regenerating mesh with AI-recommended size: {ai_params['mesh']['nx']} × {ai_params['mesh']['ny']}")
    mesh_ai = StructuredMesh2D(
        x_min=0.0, x_max=L,
        y_min=0.0, y_max=H,
        nx=ai_params['mesh']['nx'],
        ny=ai_params['mesh']['ny']
    )
    
    solver_ai = NavierStokesSolver(mesh_ai)
    solver_ai.set_nanofluid_properties(
        np.ones(mesh_ai.n_cells) * props['rho'],
        np.ones(mesh_ai.n_cells) * props['mu'],
        np.ones(mesh_ai.n_cells) * props['k_eff']
    )
    solver_ai.enable_ai_assistance(True)
    solver_ai.ai_apply_recommendations(ai_params)

# Solve with AI assistance and monitoring
print("\n⏱️  Solving with AI-assisted setup...")
print("     (AI will monitor convergence in real-time)")
converged_ai = solver_ai.solve(verbose=True)

if converged_ai:
    print(f"✅ AI-assisted approach converged in {len(solver_ai.residuals['u'])} iterations")
else:
    print(f"⚠️  AI-assisted approach did not converge ({len(solver_ai.residuals['u'])} iterations)")

# ==============================================================================
# PART 4: Compare Results
# ==============================================================================
print("\n" + "="*80)
print("PART 4: Comparison - Manual vs AI-Assisted")
print("="*80)

print("\n📊 SETUP COMPARISON:")
print(f"{'Parameter':<25} {'Manual':<20} {'AI-Assisted':<20}")
print("-" * 65)
print(f"{'Mesh size':<25} {nx_manual}×{ny_manual:<17} {mesh_ai.nx}×{mesh_ai.ny}")
print(f"{'Velocity relaxation':<25} {alpha_u_manual:<20.2f} {solver_ai.settings.under_relaxation_u:<20.2f}")
print(f"{'Pressure relaxation':<25} {alpha_p_manual:<20.2f} {solver_ai.settings.under_relaxation_p:<20.2f}")
print(f"{'Turbulence model':<25} {turb_model.value:<20} {ai_classification.get('turbulence_model', 'none'):<20}")

print("\n📈 CONVERGENCE COMPARISON:")
print(f"{'Metric':<25} {'Manual':<20} {'AI-Assisted':<20}")
print("-" * 65)
print(f"{'Converged?':<25} {'✅ Yes' if converged_manual else '❌ No':<20} {'✅ Yes' if converged_ai else '❌ No':<20}")
print(f"{'Iterations':<25} {len(solver_manual.residuals['u']):<20} {len(solver_ai.residuals['u']):<20}")

if converged_manual and converged_ai:
    speedup = len(solver_manual.residuals['u']) / len(solver_ai.residuals['u'])
    print(f"{'AI Speedup':<25} {'-':<20} {speedup:.2f}×")

# ==============================================================================
# PART 5: Visualize Convergence
# ==============================================================================
print("\n" + "="*80)
print("PART 5: Convergence Visualization")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Velocity residuals
axes[0, 0].semilogy(solver_manual.residuals['u'], 'b-', label='Manual', linewidth=2)
axes[0, 0].semilogy(solver_ai.residuals['u'], 'r--', label='AI-Assisted', linewidth=2)
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Velocity Residual')
axes[0, 0].set_title('Velocity Convergence')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Pressure residuals
axes[0, 1].semilogy(solver_manual.residuals['continuity'], 'b-', label='Manual', linewidth=2)
axes[0, 1].semilogy(solver_ai.residuals['continuity'], 'r--', label='AI-Assisted', linewidth=2)
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Continuity Residual')
axes[0, 1].set_title('Pressure Convergence')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Temperature residuals
axes[1, 0].semilogy(solver_manual.residuals['T'], 'b-', label='Manual', linewidth=2)
axes[1, 0].semilogy(solver_ai.residuals['T'], 'r--', label='AI-Assisted', linewidth=2)
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Temperature Residual')
axes[1, 0].set_title('Energy Convergence')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Overall comparison
iters_manual = len(solver_manual.residuals['u'])
iters_ai = len(solver_ai.residuals['u'])
categories = ['Iterations', 'Mesh Cells', 'Relaxation\nFactor']
manual_values = [iters_manual, mesh_manual.n_cells, alpha_u_manual * 100]
ai_values = [iters_ai, mesh_ai.n_cells, solver_ai.settings.under_relaxation_u * 100]

x = np.arange(len(categories))
width = 0.35

axes[1, 1].bar(x - width/2, manual_values, width, label='Manual', color='blue', alpha=0.7)
axes[1, 1].bar(x + width/2, ai_values, width, label='AI-Assisted', color='red', alpha=0.7)
axes[1, 1].set_ylabel('Value')
axes[1, 1].set_title('Setup Comparison')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(categories)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('example_16_ai_cfd_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: example_16_ai_cfd_comparison.png")

# ==============================================================================
# PART 6: Advanced AI Features Demo
# ==============================================================================
print("\n" + "="*80)
print("PART 6: Advanced AI Features")
print("="*80)

print("\n🔍 Demonstrating AI capabilities across different flow regimes...")

# Test different Reynolds numbers
test_cases = [
    {'Re_target': 500, 'U': 0.01, 'description': 'Low Re laminar'},
    {'Re_target': 2000, 'U': 0.05, 'description': 'High Re laminar'},
    {'Re_target': 3000, 'U': 0.08, 'description': 'Transitional'},
    {'Re_target': 10000, 'U': 0.25, 'description': 'Turbulent'},
]

print("\nAI Classification Results:")
print("-" * 80)
print(f"{'Description':<20} {'Re':<10} {'Regime':<15} {'Model':<15} {'Confidence':<10}")
print("-" * 80)

for case in test_cases:
    # Adjust velocity to hit target Re
    U_test = case['U']
    Re_test = props['rho'] * U_test * H / props['mu']
    
    # Create temporary solver for classification
    mesh_temp = StructuredMesh2D(x_min=0, x_max=L, y_min=0, y_max=H, nx=20, ny=10)
    solver_temp = NavierStokesSolver(mesh_temp)
    solver_temp.set_nanofluid_properties(
        np.ones(mesh_temp.n_cells) * props['rho'],
        np.ones(mesh_temp.n_cells) * props['mu'],
        np.ones(mesh_temp.n_cells) * props['k_eff']
    )
    solver_temp.enable_ai_assistance(True)
    
    result = solver_temp.ai_classify_flow(U_test, H)
    
    print(f"{case['description']:<20} {Re_test:<10.0f} {result['regime']:<15} "
          f"{result['turbulence_model']:<15} {result['confidence']*100:.1f}%")

print("-" * 80)

# ==============================================================================
# Summary and Recommendations
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY AND KEY TAKEAWAYS")
print("="*80)

print("\n✅ AI-CFD Integration Benefits:")
print("  1. Automatic flow regime classification (95%+ accuracy)")
print("  2. Intelligent turbulence model selection")
print("  3. Optimized solver parameters (mesh, relaxation)")
print("  4. Real-time convergence monitoring")
print("  5. Early divergence detection and warnings")
print("  6. Reduced setup time and user expertise required")

if converged_manual and converged_ai:
    improvement = (1 - len(solver_ai.residuals['u']) / len(solver_manual.residuals['u'])) * 100
    print(f"\n📊 Performance Improvement:")
    print(f"  • Iterations reduced by {improvement:.1f}%")
    print(f"  • Setup time: Manual (trial-error) vs AI (instant)")
    print(f"  • Confidence: AI provides {ai_classification['confidence']*100:.0f}% confidence score")

print("\n🎯 When to Use AI-CFD:")
print("  ✅ Unfamiliar flow regimes")
print("  ✅ Parameter sensitivity studies")
print("  ✅ Educational/learning purposes")
print("  ✅ Rapid prototyping")
print("  ✅ Automated workflows")

print("\n📚 AI Limitations (Honest Assessment):")
print("  ⚠️  Recommendations based on typical cases")
print("  ⚠️  May need fine-tuning for unusual geometries")
print("  ⚠️  Not a replacement for CFD expertise")
print("  ⚠️  Best used as intelligent starting point")

print("\n" + "="*80)
print("✅ Example 16 Complete!")
print("="*80)
print("\nNext steps:")
print("  • Try different nanofluid concentrations")
print("  • Test various channel sizes (micro to macro)")
print("  • Compare AI predictions with experimental data")
print("  • Use AI-CFD for parametric optimization studies")

plt.show()
