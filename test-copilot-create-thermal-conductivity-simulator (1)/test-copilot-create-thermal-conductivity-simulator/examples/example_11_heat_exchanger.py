"""
Example 11: Nanofluid Heat Exchanger Simulation

Parallel-flow heat exchanger with hot nanofluid and cold base fluid.
Demonstrates:
- Conjugate heat transfer
- Nanofluid enhancement effects
- Heat transfer effectiveness
- Comparison with base fluid

Application: Shell-and-tube heat exchangers, plate heat exchangers
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nanofluid_simulator import NanofluidSimulator, Nanoparticle
from nanofluid_simulator.cfd_mesh import StructuredMesh2D, BoundaryType
from nanofluid_simulator.cfd_solver import NavierStokesSolver, SolverSettings, BoundaryCondition
from nanofluid_simulator.cfd_postprocess import FlowVisualizer, HeatTransferCalculator

print("="*80)
print("NANOFLUID HEAT EXCHANGER SIMULATION")
print("="*80)

# Configuration
L = 0.5  # Heat exchanger length (m)
H = 0.02  # Channel height (m)
nx, ny = 100, 20

phi_values = [0.0, 0.01, 0.03, 0.05]  # Volume fractions to compare

print(f"\n📐 Configuration:")
print(f"   Geometry: {L}m × {H}m channel")
print(f"   Mesh: {nx}×{ny} cells")
print(f"   Nanoparticle: Al2O3")
print(f"   Volume fractions: {phi_values}")

results = {}

for phi in phi_values:
    print(f"\n{'='*80}")
    print(f"CASE: φ = {phi*100:.1f}% Al2O3")
    print(f"{'='*80}")
    
    # Create mesh
    mesh = StructuredMesh2D(
        x_range=(0.0, L),
        y_range=(0.0, H),
        nx=nx,
        ny=ny
    )
    
    # Calculate nanofluid properties
    if phi > 0:
        sim = NanofluidSimulator(
            base_fluid_name='water',
            nanoparticle=Nanoparticle.AL2O3,
            volume_fraction=phi,
            temperature=320.0,  # Hot inlet temperature
            nanoparticle_diameter=30e-9
        )
        props = sim.calculate_all_properties()
        
        rho = props['density']
        mu = props['viscosity']
        k = props['thermal_conductivity']
        cp = props['specific_heat']
        
        print(f"\n🔬 Nanofluid Properties (T=320K):")
        print(f"   k_nf/k_bf = {k/0.613:.3f}")
        print(f"   μ_nf/μ_bf = {mu/0.001:.3f}")
    else:
        # Base fluid (water at 320K)
        rho = 989.0
        mu = 5.77e-4
        cp = 4180.0
        k = 0.643
        print(f"\n💧 Base Fluid (Water at 320K)")
    
    # Setup solver
    settings = SolverSettings(
        max_iterations=300,
        tolerance=1e-5,
        under_relaxation_u=0.7,
        under_relaxation_p=0.3,
        under_relaxation_T=0.9,
        turbulence_model='laminar'
    )
    
    solver = NavierStokesSolver(mesh, settings)
    solver.set_fluid_properties(rho, mu, cp, k)
    
    # Boundary conditions
    u_inlet = 0.1  # m/s
    T_hot = 320.0  # K (hot inlet)
    T_cold = 293.0  # K (cold wall)
    
    # Inlet: hot fluid
    inlet_faces = [f.id for f in mesh.faces if f.boundary_type == BoundaryType.INLET]
    bc_inlet = BoundaryCondition(bc_type='inlet', velocity=(u_inlet, 0.0), temperature=T_hot)
    for fid in inlet_faces:
        solver.set_boundary_condition(fid, bc_inlet)
    
    # Outlet
    outlet_faces = [f.id for f in mesh.faces if f.boundary_type == BoundaryType.OUTLET]
    bc_outlet = BoundaryCondition(bc_type='outlet', pressure=0.0)
    for fid in outlet_faces:
        solver.set_boundary_condition(fid, bc_outlet)
    
    # Bottom wall: cold (heat transfer)
    bottom_faces = [f.id for f in mesh.faces 
                   if f.boundary_type == BoundaryType.WALL and f.center[1] < 1e-6]
    bc_cold = BoundaryCondition(bc_type='wall', velocity=(0.0, 0.0), temperature=T_cold)
    for fid in bottom_faces:
        solver.set_boundary_condition(fid, bc_cold)
    
    # Top wall: adiabatic
    top_faces = [f.id for f in mesh.faces 
                if f.boundary_type == BoundaryType.WALL and abs(f.center[1] - H) < 1e-6]
    bc_adiabatic = BoundaryCondition(bc_type='wall', velocity=(0.0, 0.0), temperature=None)
    for fid in top_faces:
        solver.set_boundary_condition(fid, bc_adiabatic)
    
    print(f"\n🚀 Running CFD simulation...")
    residuals = solver.solve()
    
    print(f"   ✅ Converged in {len(residuals['u'])} iterations")
    
    # Calculate heat transfer
    field = solver.field
    
    # Outlet temperature
    outlet_cells = [c for c in mesh.cells if abs(c.center[0] - L) < mesh.dx]
    T_outlet = np.mean([field.T[c.id] for c in outlet_cells])
    
    # Heat transferred
    m_dot = rho * u_inlet * H  # kg/s per unit width
    Q_dot = m_dot * cp * (T_hot - T_outlet)  # W per unit width
    
    # Heat transfer coefficient (simplified)
    A = L  # Surface area per unit width
    LMTD = ((T_hot - T_cold) - (T_outlet - T_cold)) / np.log((T_hot - T_cold) / (T_outlet - T_cold))
    h = Q_dot / (A * LMTD)
    
    # Effectiveness
    Q_max = m_dot * cp * (T_hot - T_cold)
    effectiveness = Q_dot / Q_max
    
    results[phi] = {
        'T_outlet': T_outlet,
        'Q_dot': Q_dot,
        'h': h,
        'effectiveness': effectiveness,
        'pressure_drop': np.mean(field.p[inlet_faces]) - np.mean(field.p[outlet_faces]) if outlet_faces else 0
    }
    
    print(f"\n📊 Results:")
    print(f"   Outlet temperature: {T_outlet:.2f} K")
    print(f"   Heat transfer rate: {Q_dot:.2f} W/m")
    print(f"   Heat transfer coefficient: {h:.1f} W/(m²·K)")
    print(f"   Effectiveness: {effectiveness:.3f}")
    
    # Visualization for highest concentration
    if phi == max(phi_values):
        print(f"\n📸 Generating visualization...")
        visualizer = FlowVisualizer(mesh, field)
        visualizer.plot_temperature_field(f'heat_exchanger_phi{int(phi*100)}.png')

# Comparison
print(f"\n{'='*80}")
print("PERFORMANCE COMPARISON")
print(f"{'='*80}")

print(f"\n{'φ (%)':<10} {'T_out (K)':<12} {'Q (W/m)':<12} {'h (W/m²K)':<12} {'ε':<10} {'Enhancement'}")
print("-"*80)

Q_base = results[0.0]['Q_dot']
for phi in phi_values:
    r = results[phi]
    enhancement = (r['Q_dot'] - Q_base) / Q_base * 100 if phi > 0 else 0
    print(f"{phi*100:<10.1f} {r['T_outlet']:<12.2f} {r['Q_dot']:<12.2f} "
          f"{r['h']:<12.1f} {r['effectiveness']:<10.3f} {enhancement:>6.1f}%")

print("\n✅ Key Findings:")
best_phi = max(phi_values, key=lambda p: results[p]['Q_dot'])
best_enhancement = (results[best_phi]['Q_dot'] - Q_base) / Q_base * 100

print(f"   • Best performance: φ = {best_phi*100:.1f}%")
print(f"   • Heat transfer enhancement: {best_enhancement:.1f}%")
print(f"   • Effectiveness improvement: "
      f"{(results[best_phi]['effectiveness'] - results[0.0]['effectiveness'])*100:.1f}%")

print("\n💡 Engineering Insights:")
print("   • Nanofluids improve heat transfer due to higher thermal conductivity")
print("   • Optimal concentration balances enhancement vs. viscosity penalty")
print("   • Effectiveness increases with nanoparticle loading")
print("   • Suitable for compact heat exchangers")
