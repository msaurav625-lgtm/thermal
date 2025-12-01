"""
Example: CFD Simulation with Nanofluid

Demonstrates 2D flow simulation in a pipe/channel with nanofluid properties.
Shows how to couple thermal conductivity models with CFD solver.

Author: Nanofluid Simulator v4.0
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanofluid_simulator.cfd_mesh import StructuredMesh2D, BoundaryType
from nanofluid_simulator.cfd_solver import (
    NavierStokesSolver, 
    SolverSettings, 
    BoundaryCondition,
    InterpolationScheme
)
from nanofluid_simulator.simulator import NanofluidSimulator
from nanofluid_simulator.nanoparticles import get_nanoparticle_properties


def calculate_nanofluid_properties(volume_fraction: float, 
                                   temperature: float = 300.0):
    """
    Calculate nanofluid thermophysical properties for CFD.
    
    Parameters
    ----------
    volume_fraction : float
        Nanoparticle volume fraction (0-0.05)
    temperature : float
        Temperature in Kelvin
        
    Returns
    -------
    dict
        Properties: rho, mu, k, cp
    """
    # Create nanofluid simulator
    sim = NanofluidSimulator(
        base_fluid='water',
        nanoparticle='Al2O3',
        volume_fraction=volume_fraction,
        temperature=temperature
    )
    
    # Calculate thermal conductivity
    results = sim.calculate_all_models()
    k_nf = results['Maxwell']  # Use Maxwell model
    
    # Calculate density (simple mixture rule)
    np_props = get_nanoparticle_properties('Al2O3')
    rho_bf = 997.0  # Water density at 300K
    rho_nf = (1 - volume_fraction) * rho_bf + volume_fraction * np_props['density']
    
    # Calculate viscosity (Einstein model for dilute suspensions)
    mu_bf = 0.001  # Water viscosity at 300K
    mu_nf = mu_bf * (1 + 2.5 * volume_fraction)
    
    # Heat capacity (mixture rule)
    cp_bf = 4180.0  # Water
    cp_np = 880.0   # Al2O3
    cp_nf = ((1 - volume_fraction) * rho_bf * cp_bf + 
             volume_fraction * np_props['density'] * cp_np) / rho_nf
    
    return {
        'rho': rho_nf,
        'mu': mu_nf,
        'k': k_nf,
        'cp': cp_nf
    }


def simulate_channel_flow(volume_fraction: float = 0.02,
                          Re: float = 100.0):
    """
    Simulate 2D channel flow with nanofluid.
    
    Parameters
    ----------
    volume_fraction : float
        Nanoparticle volume fraction
    Re : float
        Reynolds number
    """
    print("\n" + "="*70)
    print(f"  CFD SIMULATION: Channel Flow with Nanofluid")
    print("="*70)
    print(f"  Nanofluid: Water + Al2O3")
    print(f"  Volume fraction: {volume_fraction*100:.1f}%")
    print(f"  Reynolds number: {Re:.1f}")
    print()
    
    # ===========================
    # 1. MESH GENERATION
    # ===========================
    print("📐 Step 1: Generating mesh...")
    
    # Channel dimensions
    length = 0.1  # 10 cm
    height = 0.02  # 2 cm
    
    # Create mesh (finer than demo for better accuracy)
    mesh = StructuredMesh2D(
        x_range=(0.0, length),
        y_range=(0.0, height),
        nx=50,
        ny=20,
        boundary_types={
            'left': BoundaryType.INLET,
            'right': BoundaryType.OUTLET,
            'top': BoundaryType.WALL,
            'bottom': BoundaryType.WALL
        }
    )
    
    quality = mesh.get_mesh_quality()
    print(f"   ✅ Mesh created: {quality.n_cells} cells, {quality.n_faces} faces")
    
    # ===========================
    # 2. CALCULATE NANOFLUID PROPERTIES
    # ===========================
    print("\n🧪 Step 2: Calculating nanofluid properties...")
    
    props = calculate_nanofluid_properties(volume_fraction)
    
    print(f"   Density:              ρ = {props['rho']:.2f} kg/m³")
    print(f"   Dynamic viscosity:    μ = {props['mu']:.6f} Pa·s")
    print(f"   Thermal conductivity: k = {props['k']:.4f} W/m·K")
    print(f"   Heat capacity:       cp = {props['cp']:.1f} J/kg·K")
    
    # Calculate inlet velocity from Reynolds number
    D_h = 2 * height  # Hydraulic diameter for channel
    u_inlet = Re * props['mu'] / (props['rho'] * D_h)
    print(f"\n   Inlet velocity: u = {u_inlet:.4f} m/s")
    
    # ===========================
    # 3. SETUP SOLVER
    # ===========================
    print("\n⚙️  Step 3: Setting up CFD solver...")
    
    settings = SolverSettings(
        max_iterations=50,  # Limited for demo
        convergence_tol=1e-4,
        under_relaxation_u=0.7,
        under_relaxation_v=0.7,
        under_relaxation_p=0.3,
        time_step=0.001,
        interpolation_scheme=InterpolationScheme.CENTRAL
    )
    
    solver = NavierStokesSolver(mesh, settings)
    
    # Set nanofluid properties (uniform for now)
    rho_field = np.ones(mesh.n_cells) * props['rho']
    mu_field = np.ones(mesh.n_cells) * props['mu']
    k_field = np.ones(mesh.n_cells) * props['k']
    
    solver.set_nanofluid_properties(rho_field, mu_field, k_field)
    
    # ===========================
    # 4. BOUNDARY CONDITIONS
    # ===========================
    print("🔧 Step 4: Applying boundary conditions...")
    
    # Inlet: specified velocity and temperature
    bc_inlet = BoundaryCondition(
        bc_type=BoundaryType.INLET,
        velocity=(u_inlet, 0.0),
        temperature=300.0
    )
    solver.set_boundary_condition(BoundaryType.INLET, bc_inlet)
    
    # Outlet: zero gradient (pressure outlet)
    bc_outlet = BoundaryCondition(
        bc_type=BoundaryType.OUTLET,
        pressure=0.0
    )
    solver.set_boundary_condition(BoundaryType.OUTLET, bc_outlet)
    
    # Walls: no-slip, constant temperature
    bc_wall = BoundaryCondition(
        bc_type=BoundaryType.WALL,
        velocity=(0.0, 0.0),
        temperature=320.0  # Heated wall
    )
    solver.set_boundary_condition(BoundaryType.WALL, bc_wall)
    
    print("   ✅ Inlet:  u = {:.4f} m/s, T = 300 K".format(u_inlet))
    print("   ✅ Outlet: p = 0 Pa (reference)")
    print("   ✅ Walls:  no-slip, T = 320 K (heated)")
    
    # ===========================
    # 5. INITIALIZE FLOW FIELD
    # ===========================
    print("\n🌊 Step 5: Initializing flow field...")
    
    solver.initialize_field(
        u0=u_inlet * 0.5,  # Initial guess
        v0=0.0,
        p0=0.0,
        T0=300.0
    )
    
    print("   ✅ Initial conditions set")
    
    # ===========================
    # 6. SOLVE
    # ===========================
    print("\n🚀 Step 6: Running CFD solver...")
    print("   (This will take a moment...)")
    
    converged = solver.solve(max_iterations=50)
    
    # ===========================
    # 7. POST-PROCESS
    # ===========================
    print("\n📊 Step 7: Post-processing results...")
    
    field = solver.get_results()
    
    # Calculate derived quantities
    u_max = np.max(field.u)
    u_mean = np.mean(field.u)
    T_max = np.max(field.T)
    T_min = np.min(field.T)
    p_max = np.max(field.p)
    
    print(f"\n   Flow field statistics:")
    print(f"   ├─ Max velocity:     {u_max:.4f} m/s")
    print(f"   ├─ Mean velocity:    {u_mean:.4f} m/s")
    print(f"   ├─ Temperature range: {T_min:.2f} - {T_max:.2f} K")
    print(f"   └─ Max pressure:     {p_max:.2f} Pa")
    
    # ===========================
    # 8. VISUALIZATION
    # ===========================
    print("\n🎨 Step 8: Creating visualizations...")
    
    visualize_results(mesh, field, volume_fraction)
    
    print("\n✅ CFD simulation complete!")
    
    return mesh, solver, field


def visualize_results(mesh, field, volume_fraction):
    """Create visualization of CFD results"""
    
    # Extract coordinates and reshape fields
    nx, ny = mesh.nx, mesh.ny
    x = np.array([cell.center[0] for cell in mesh.cells]).reshape(ny, nx)
    y = np.array([cell.center[1] for cell in mesh.cells]).reshape(ny, nx)
    
    u = field.u.reshape(ny, nx)
    v = field.v.reshape(ny, nx)
    T = field.T.reshape(ny, nx)
    p = field.p.reshape(ny, nx)
    
    # Velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Velocity magnitude contour
    ax = axes[0, 0]
    contour1 = ax.contourf(x, y, vel_mag, levels=20, cmap='viridis')
    ax.streamplot(x, y, u, v, color='white', linewidth=0.5, density=1.5, arrowsize=1)
    plt.colorbar(contour1, ax=ax, label='Velocity magnitude (m/s)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Velocity Field with Streamlines')
    ax.set_aspect('equal')
    
    # 2. Temperature field
    ax = axes[0, 1]
    contour2 = ax.contourf(x, y, T, levels=20, cmap='hot')
    plt.colorbar(contour2, ax=ax, label='Temperature (K)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Temperature Distribution')
    ax.set_aspect('equal')
    
    # 3. Pressure field
    ax = axes[1, 0]
    contour3 = ax.contourf(x, y, p, levels=20, cmap='coolwarm')
    plt.colorbar(contour3, ax=ax, label='Pressure (Pa)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Pressure Field')
    ax.set_aspect('equal')
    
    # 4. Velocity profile at outlet
    ax = axes[1, 1]
    u_outlet = u[:, -1]
    y_outlet = y[:, -1]
    ax.plot(u_outlet, y_outlet, 'b-', linewidth=2, label='Computed')
    
    # Analytical parabolic profile for comparison (fully developed laminar flow)
    u_max_theory = 1.5 * np.mean(u_outlet)
    height = np.max(y_outlet) - np.min(y_outlet)
    y_theory = np.linspace(0, height, 100)
    u_theory = u_max_theory * (1 - (2*y_theory/height - 1)**2)
    ax.plot(u_theory, y_theory, 'r--', linewidth=2, label='Parabolic (theory)')
    
    ax.set_xlabel('u-velocity (m/s)')
    ax.set_ylabel('y (m)')
    ax.set_title('Velocity Profile at Outlet')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle(f'CFD Simulation: Water + Al₂O₃ Nanofluid (φ = {volume_fraction*100:.1f}%)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'cfd_nanofluid_phi{int(volume_fraction*100)}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"   ✅ Results saved to: {filename}")


if __name__ == "__main__":
    print("="*70)
    print("  NANOFLUID SIMULATOR v4.0 - CFD MODULE")
    print("  Example: 2D Channel Flow with Heat Transfer")
    print("="*70)
    
    # Run simulation with default parameters
    mesh, solver, field = simulate_channel_flow(
        volume_fraction=0.02,  # 2% nanoparticles
        Re=100.0               # Laminar flow
    )
    
    # Compare with pure base fluid
    print("\n" + "="*70)
    print("  COMPARISON: Pure Water vs Nanofluid")
    print("="*70)
    
    print("\n🧪 Running simulation with pure water...")
    mesh_bf, solver_bf, field_bf = simulate_channel_flow(
        volume_fraction=0.0,  # Pure water
        Re=100.0
    )
    
    # Calculate enhancement
    k_enhancement = (np.mean(field.k) - np.mean(field_bf.k)) / np.mean(field_bf.k) * 100
    
    print("\n📈 Performance Enhancement:")
    print(f"   Thermal conductivity: +{k_enhancement:.2f}%")
    print(f"   Base fluid: k = {np.mean(field_bf.k):.4f} W/m·K")
    print(f"   Nanofluid:  k = {np.mean(field.k):.4f} W/m·K")
    
    print("\n" + "="*70)
    print("  🎉 CFD SIMULATION COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   - cfd_nanofluid_phi2.png (nanofluid results)")
    print("   - cfd_nanofluid_phi0.png (base fluid results)")
    print("\n💡 This demonstrates:")
    print("   ✓ CFD mesh generation")
    print("   ✓ Nanofluid property integration")
    print("   ✓ Navier-Stokes solution")
    print("   ✓ Heat transfer simulation")
    print("   ✓ Post-processing & visualization")
