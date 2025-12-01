"""
Lid-Driven Cavity CFD Benchmark for Code Verification

Implements the classic 2D lid-driven cavity flow test case for CFD solver validation.
Compares velocity profiles with Ghia et al. (1982) reference data.

References:
- Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. J. Comput. Phys., 48(3), 387-411.
"""

import numpy as np
import matplotlib.pyplot as plt
from nanofluid_simulator.cfd_mesh import StructuredMesh2D
from nanofluid_simulator.cfd_solver import NavierStokesSolver, SolverSettings
from nanofluid_simulator.cfd_turbulence import TurbulenceModel

# Reference velocity profiles from Ghia et al. (Re=100)
GHIA_U = np.array([
    0.0000, 0.0923, 0.1836, 0.2706, 0.3468, 0.4085, 0.4579, 0.4897, 0.4987, 0.4819, 0.4451, 0.3866, 0.3127, 0.2297, 0.1461, 0.0626, 0.0000
])
GHIA_Y = np.linspace(1, 0, len(GHIA_U))

GHIA_V = np.array([
    0.0000, -0.0598, -0.1232, -0.1937, -0.2746, -0.3573, -0.4291, -0.4778, -0.4954, -0.4819, -0.4451, -0.3866, -0.3127, -0.2297, -0.1461, -0.0626, 0.0000
])
GHIA_X = np.linspace(0, 1, len(GHIA_V))


def run_lid_driven_cavity(nx=50, ny=50, Re=100, plot=True):
    """
    Run lid-driven cavity CFD benchmark and compare with Ghia et al. data.
    Args:
        nx, ny: Mesh resolution
        Re: Reynolds number
        plot: If True, plot velocity profiles
    Returns:
        results: dict with velocity fields and error metrics
    """
    # Domain: 1x1 square
    mesh = StructuredMesh2D(x_range=(0, 1), y_range=(0, 1), nx=nx, ny=ny)
    
    # Fluid properties for Re=100 (u_lid=1, nu=0.01)
    u_lid = 1.0
    nu = u_lid / Re
    
    settings = SolverSettings(
        max_iterations=500,
        convergence_tol=1e-5,
        under_relaxation_u=0.7,
        under_relaxation_v=0.7,
        under_relaxation_p=0.3,
        under_relaxation_T=0.8,
        time_step=0.001,
        turbulence_model=TurbulenceModel.LAMINAR
    )
    
    solver = NavierStokesSolver(mesh, settings)
    
    # Set uniform properties
    rho = np.ones_like(mesh.x) * 1.0
    mu = np.ones_like(mesh.x) * nu
    k = np.ones_like(mesh.x) * 1.0
    
    solver.set_nanofluid_properties(rho=rho, mu=mu, k=k)
    
    # Set boundary conditions
    solver.set_boundary_conditions({
        'top': {'velocity': (u_lid, 0)},
        'bottom': {'velocity': (0, 0)},
        'left': {'velocity': (0, 0)},
        'right': {'velocity': (0, 0)}
    })
    
    # Solve
    result = solver.solve()
    u = result['u']
    v = result['v']
    X = mesh.X
    Y = mesh.Y
    
    # Extract centerline profiles
    mid_x = nx // 2
    mid_y = ny // 2
    u_center = u[:, mid_x]
    v_center = v[mid_y, :]
    y = Y[:, mid_x]
    x = X[mid_y, :]
    
    # Interpolate to Ghia points
    u_interp = np.interp(GHIA_Y, y[::-1], u_center[::-1])
    v_interp = np.interp(GHIA_X, x, v_center)
    
    # Compute errors
    u_error = np.abs(u_interp - GHIA_U)
    v_error = np.abs(v_interp - GHIA_V)
    mae_u = np.mean(u_error)
    mae_v = np.mean(v_error)
    
    if plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(u_center, y, label='Simulated')
        plt.plot(GHIA_U, GHIA_Y, 'o-', label='Ghia et al.')
        plt.xlabel('u (centerline)')
        plt.ylabel('y')
        plt.title('U velocity profile')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, v_center, label='Simulated')
        plt.plot(GHIA_X, GHIA_V, 'o-', label='Ghia et al.')
        plt.xlabel('x')
        plt.ylabel('v (centerline)')
        plt.title('V velocity profile')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return {
        'u_center': u_center,
        'v_center': v_center,
        'y': y,
        'x': x,
        'ghia_u': GHIA_U,
        'ghia_y': GHIA_Y,
        'ghia_v': GHIA_V,
        'ghia_x': GHIA_X,
        'mae_u': mae_u,
        'mae_v': mae_v
    }

if __name__ == '__main__':
    results = run_lid_driven_cavity()
    print(f"U profile MAE: {results['mae_u']:.4f}")
    print(f"V profile MAE: {results['mae_v']:.4f}")
