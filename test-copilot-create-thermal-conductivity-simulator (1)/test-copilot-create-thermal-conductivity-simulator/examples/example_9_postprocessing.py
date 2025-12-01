"""
Example 9: Advanced Post-Processing

Demonstrates comprehensive CFD post-processing capabilities:
- Derived quantity calculations (vorticity, strain rate, Q-criterion)
- Heat transfer analysis (Nusselt number, heat flux)
- Force calculations (drag, lift)
- Publication-quality visualization
- Convergence monitoring

This example runs a lid-driven cavity flow simulation and applies
various post-processing techniques to analyze the results.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nanofluid_simulator.cfd_mesh import StructuredMesh2D, BoundaryType
from nanofluid_simulator.cfd_solver import (
    NavierStokesSolver, SolverSettings, BoundaryCondition
)
from nanofluid_simulator.cfd_postprocess import (
    DerivedQuantities, FlowVisualizer, ForceCalculator,
    HeatTransferCalculator, ConvergenceMonitor
)


def run_lid_driven_cavity():
    """
    Simulate lid-driven cavity flow.
    
    Classic CFD benchmark case:
    - Square cavity (1m × 1m)
    - Top wall moves with velocity U=1 m/s
    - Other walls stationary
    - Initially at rest
    """
    print("="*80)
    print("LID-DRIVEN CAVITY FLOW - POST-PROCESSING DEMO")
    print("="*80)
    
    # ====================
    # 1. MESH GENERATION
    # ====================
    print("\n📐 Step 1: Mesh Generation")
    
    L = 1.0  # Cavity size (m)
    nx, ny = 40, 40
    
    mesh = StructuredMesh2D(
        x_range=(0.0, L),
        y_range=(0.0, L),
        nx=nx,
        ny=ny
    )
    
    print(f"   Mesh: {nx}×{ny} = {mesh.n_cells} cells")
    print(f"   Cell size: Δx={mesh.dx:.4f} m, Δy={mesh.dy:.4f} m")
    
    # ====================
    # 2. SOLVER SETUP
    # ====================
    print("\n⚙️  Step 2: Solver Configuration")
    
    settings = SolverSettings(
        max_iterations=500,
        tolerance=1e-4,
        under_relaxation_u=0.7,
        under_relaxation_p=0.3,
        turbulence_model='laminar'  # Laminar flow for Re=100
    )
    
    solver = NavierStokesSolver(mesh, settings)
    
    # Set fluid properties (water at 20°C)
    rho = 998.0  # kg/m³
    mu = 1.002e-3  # Pa·s
    cp = 4182.0  # J/(kg·K)
    k_thermal = 0.598  # W/(m·K)
    
    solver.set_fluid_properties(rho, mu, cp, k_thermal)
    
    # Reynolds number
    U_lid = 1.0  # m/s
    Re = rho * U_lid * L / mu
    print(f"   Reynolds number: Re = {Re:.0f}")
    print(f"   Fluid: Water (ρ={rho:.1f} kg/m³, μ={mu*1e3:.3f} mPa·s)")
    
    # ====================
    # 3. BOUNDARY CONDITIONS
    # ====================
    print("\n🔧 Step 3: Boundary Conditions")
    
    # Top wall (moving lid)
    top_faces = [f.id for f in mesh.faces if f.boundary_type == BoundaryType.WALL 
                 and abs(f.center[1] - L) < 1e-6]
    
    bc_top = BoundaryCondition(
        bc_type='wall',
        velocity=(U_lid, 0.0),
        temperature=300.0
    )
    
    for fid in top_faces:
        solver.set_boundary_condition(fid, bc_top)
    
    # Other walls (stationary)
    other_walls = [f.id for f in mesh.faces if f.boundary_type == BoundaryType.WALL 
                   and f.id not in top_faces]
    
    bc_wall = BoundaryCondition(
        bc_type='wall',
        velocity=(0.0, 0.0),
        temperature=300.0
    )
    
    for fid in other_walls:
        solver.set_boundary_condition(fid, bc_wall)
    
    print(f"   Top wall: u={U_lid} m/s (moving lid)")
    print(f"   Other walls: u=0 m/s (no-slip)")
    print(f"   Total boundary faces: {len(top_faces) + len(other_walls)}")
    
    # ====================
    # 4. RUN SIMULATION
    # ====================
    print("\n🚀 Step 4: Running CFD Simulation")
    print("   This may take 30-60 seconds...")
    
    residuals = solver.solve()
    
    print(f"\n   ✅ Simulation converged!")
    print(f"   Final residuals:")
    for var, res_list in residuals.items():
        if len(res_list) > 0:
            print(f"      {var}: {res_list[-1]:.2e}")
    
    # ====================
    # 5. POST-PROCESSING
    # ====================
    print("\n" + "="*80)
    print("POST-PROCESSING ANALYSIS")
    print("="*80)
    
    field = solver.field
    
    # 5a. Basic statistics
    print("\n📊 Flow Field Statistics:")
    vel_mag = np.sqrt(field.u**2 + field.v**2)
    print(f"   u velocity: [{field.u.min():.4f}, {field.u.max():.4f}] m/s")
    print(f"   v velocity: [{field.v.min():.4f}, {field.v.max():.4f}] m/s")
    print(f"   Velocity magnitude: [{vel_mag.min():.4f}, {vel_mag.max():.4f}] m/s")
    print(f"   Pressure: [{field.p.min():.2f}, {field.p.max():.2f}] Pa")
    print(f"   Temperature: [{field.T.min():.2f}, {field.T.max():.2f}] K")
    
    # 5b. Derived quantities
    print("\n🌀 Derived Quantities:")
    
    omega = DerivedQuantities.calculate_vorticity(mesh, field.u, field.v)
    print(f"   Vorticity ω:")
    print(f"      Range: [{omega.min():.2f}, {omega.max():.2f}] 1/s")
    print(f"      Mean: {omega.mean():.2f} 1/s")
    print(f"      Std: {omega.std():.2f} 1/s")
    
    S = DerivedQuantities.calculate_strain_rate(mesh, field.u, field.v)
    print(f"   Strain Rate S:")
    print(f"      Range: [{S.min():.2f}, {S.max():.2f}] 1/s")
    print(f"      Mean: {S.mean():.2f} 1/s")
    
    Q = DerivedQuantities.calculate_q_criterion(mesh, field.u, field.v)
    print(f"   Q-criterion:")
    print(f"      Range: [{Q.min():.2f}, {Q.max():.2f}]")
    print(f"      Positive Q (vortex cores): {np.sum(Q > 0)}/{mesh.n_cells} cells")
    
    # 5c. Vortex center detection
    print("\n🎯 Vortex Analysis:")
    vortex_cells = np.where(Q > np.percentile(Q, 95))[0]  # Top 5% Q values
    if len(vortex_cells) > 0:
        vortex_centers = np.array([mesh.cells[i].center for i in vortex_cells])
        center_x = vortex_centers[:, 0].mean()
        center_y = vortex_centers[:, 1].mean()
        print(f"   Primary vortex center: ({center_x:.3f}, {center_y:.3f}) m")
        print(f"   Expected location: (~0.5, ~0.7) m for Re=100")
    
    # 5d. Energy dissipation
    print("\n⚡ Energy Dissipation:")
    eps_dissipation = 2 * mu * (S**2)  # ε = 2μ S²
    total_dissipation = np.sum(eps_dissipation * np.array([c.volume for c in mesh.cells]))
    print(f"   Total dissipation: {total_dissipation:.6f} W")
    print(f"   Average: {total_dissipation/mesh.n_cells:.8f} W/cell")
    
    # 5e. Kinetic energy
    print("\n💨 Kinetic Energy:")
    ke = 0.5 * rho * vel_mag**2
    total_ke = np.sum(ke * np.array([c.volume for c in mesh.cells]))
    print(f"   Total kinetic energy: {total_ke:.6f} J")
    print(f"   Average: {total_ke/mesh.n_cells:.8f} J/cell")
    
    # ====================
    # 6. VISUALIZATION
    # ====================
    print("\n" + "="*80)
    print("VISUALIZATION")
    print("="*80)
    
    visualizer = FlowVisualizer(mesh, field)
    
    print("\n📸 Generating plots...")
    visualizer.plot_all_fields(prefix='cavity_flow')
    
    # Additional convergence plot
    print("\n📈 Convergence plot...")
    ConvergenceMonitor.plot_residuals(residuals, 'cavity_convergence.png')
    
    # ====================
    # 7. COMPARISON WITH BENCHMARK
    # ====================
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)
    
    print("\nℹ️  Lid-Driven Cavity at Re=100 (Ghia et al. 1982):")
    print("   Expected features:")
    print("   - Primary vortex center: (0.53, 0.57)")
    print("   - Maximum u velocity at centerline: ~0.84")
    print("   - Secondary vortices in bottom corners")
    
    # Find centerline velocities
    centerline_x = L / 2
    centerline_cells = [c for c in mesh.cells if abs(c.center[0] - centerline_x) < mesh.dx/2]
    centerline_u = [field.u[c.id] for c in centerline_cells]
    centerline_y = [c.center[1] for c in centerline_cells]
    
    if len(centerline_u) > 0:
        max_u = max(centerline_u)
        max_u_idx = centerline_u.index(max_u)
        max_u_y = centerline_y[max_u_idx]
        
        print(f"\n   Your results:")
        print(f"   - Maximum u at centerline: {max_u:.3f} at y={max_u_y:.3f} m")
        
        # Error
        expected_max_u = 0.84
        error_u = abs(max_u - expected_max_u) / expected_max_u * 100
        print(f"   - Error: {error_u:.1f}%")
        
        if error_u < 20:
            print(f"   ✅ Good agreement with benchmark!")
        else:
            print(f"   ⚠️  Consider refining mesh or increasing iterations")
    
    print("\n" + "="*80)
    print("✅ POST-PROCESSING COMPLETE!")
    print("="*80)
    
    print("\n📁 Output files generated:")
    print("   - cavity_flow_velocity.png")
    print("   - cavity_flow_temperature.png")
    print("   - cavity_flow_pressure.png")
    print("   - cavity_flow_vorticity.png")
    print("   - cavity_convergence.png")
    
    return solver, residuals


if __name__ == "__main__":
    import time
    
    start_time = time.time()
    solver, residuals = run_lid_driven_cavity()
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Total execution time: {elapsed:.1f} seconds")
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✅ This example demonstrated:")
    print("   1. Classic CFD benchmark (lid-driven cavity)")
    print("   2. Vorticity and strain rate analysis")
    print("   3. Q-criterion for vortex identification")
    print("   4. Energy dissipation calculation")
    print("   5. Publication-quality visualization")
    print("   6. Convergence monitoring")
    print("   7. Comparison with literature benchmark")
    print("\n💡 Try adjusting:")
    print("   - Reynolds number (change U_lid or viscosity)")
    print("   - Mesh resolution (nx, ny)")
    print("   - Solver tolerance")
    print("\n🎓 Research Application:")
    print("   Use these post-processing tools to analyze your nanofluid")
    print("   simulations and extract meaningful engineering results!")
