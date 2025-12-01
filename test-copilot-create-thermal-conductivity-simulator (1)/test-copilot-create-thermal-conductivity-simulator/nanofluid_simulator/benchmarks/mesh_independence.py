"""
Mesh Independence Study Tool

Automates grid refinement studies to verify spatial convergence
and establish mesh-independent solutions for CFD simulations.

Usage:
    from nanofluid_simulator.benchmarks.mesh_independence import mesh_independence_study
    
    resolutions = [(20, 20), (40, 40), (80, 80), (160, 160)]
    results = mesh_independence_study(solver_func, resolutions, metric='centerline_u')
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MeshIndependenceResult:
    """Results from mesh independence study"""
    resolutions: List[Tuple[int, int]]
    grid_sizes: List[float]  # characteristic grid spacing
    metrics: List[float]  # observed metric value
    relative_errors: List[float]  # relative change from previous mesh
    gci: List[float]  # Grid Convergence Index
    converged: bool
    convergence_order: float


def mesh_independence_study(
    solver_func: Callable[[int, int], Dict[str, Any]],
    resolutions: List[Tuple[int, int]],
    metric_extractor: Callable[[Dict], float],
    domain_length: float = 1.0,
    convergence_threshold: float = 0.01,
    plot: bool = True,
    safety_factor: float = 1.25
) -> MeshIndependenceResult:
    """
    Perform mesh independence study using Richardson extrapolation.
    
    Args:
        solver_func: Function that takes (nx, ny) and returns solver results dict
        resolutions: List of (nx, ny) grid resolutions to test
        metric_extractor: Function to extract scalar metric from solver results
        domain_length: Physical domain length for characteristic spacing
        convergence_threshold: Relative error threshold for convergence (default 1%)
        plot: If True, plot convergence curves
        safety_factor: Safety factor for GCI calculation (default 1.25)
    
    Returns:
        MeshIndependenceResult with convergence metrics
    
    Example:
        def solve_cavity(nx, ny):
            mesh = StructuredMesh2D((0, 1), (0, 1), nx, ny)
            solver = NavierStokesSolver(mesh)
            return solver.solve()
        
        def get_max_u(results):
            return np.max(results['u'])
        
        study = mesh_independence_study(solve_cavity, [(20,20), (40,40), (80,80)], get_max_u)
    """
    
    if len(resolutions) < 3:
        raise ValueError("Need at least 3 mesh resolutions for convergence analysis")
    
    # Sort by grid size
    resolutions = sorted(resolutions, key=lambda r: r[0])
    
    # Run simulations
    print("Running mesh independence study...")
    metrics = []
    for i, (nx, ny) in enumerate(resolutions):
        print(f"  Resolution {i+1}/{len(resolutions)}: {nx}×{ny}")
        result = solver_func(nx, ny)
        metric_value = metric_extractor(result)
        metrics.append(metric_value)
        print(f"    Metric value: {metric_value:.6f}")
    
    # Calculate grid sizes (characteristic spacing)
    grid_sizes = [domain_length / nx for nx, ny in resolutions]
    
    # Calculate relative errors
    relative_errors = [0.0]  # First mesh has no reference
    for i in range(1, len(metrics)):
        if metrics[i-1] != 0:
            rel_error = abs((metrics[i] - metrics[i-1]) / metrics[i-1])
        else:
            rel_error = abs(metrics[i] - metrics[i-1])
        relative_errors.append(rel_error)
    
    # Estimate convergence order using last three meshes
    if len(metrics) >= 3:
        r21 = grid_sizes[-2] / grid_sizes[-1]
        r32 = grid_sizes[-3] / grid_sizes[-2]
        epsilon32 = metrics[-2] - metrics[-3]
        epsilon21 = metrics[-1] - metrics[-2]
        
        if epsilon21 != 0 and epsilon32 != 0:
            # Richardson extrapolation order
            p = abs(np.log(abs(epsilon32 / epsilon21)) / np.log(r21))
            p = np.clip(p, 0.5, 5.0)  # Clamp to reasonable range
        else:
            p = 2.0  # Assume second order if division by zero
    else:
        p = 2.0
    
    # Calculate Grid Convergence Index (GCI) using Roache's method
    gci = [0.0]  # First mesh has no GCI
    for i in range(1, len(metrics)):
        r = grid_sizes[i-1] / grid_sizes[i]
        rel_err = relative_errors[i]
        gci_value = (safety_factor * rel_err) / (r**p - 1)
        gci.append(gci_value)
    
    # Check convergence
    converged = relative_errors[-1] < convergence_threshold and gci[-1] < convergence_threshold
    
    result = MeshIndependenceResult(
        resolutions=resolutions,
        grid_sizes=grid_sizes,
        metrics=metrics,
        relative_errors=relative_errors,
        gci=gci,
        converged=converged,
        convergence_order=p
    )
    
    if plot:
        _plot_mesh_independence(result, convergence_threshold)
    
    # Print summary
    print("\n" + "="*60)
    print("MESH INDEPENDENCE STUDY SUMMARY")
    print("="*60)
    print(f"Convergence Order (p): {p:.3f}")
    print(f"Final Relative Error: {relative_errors[-1]:.4%}")
    print(f"Final GCI: {gci[-1]:.4%}")
    print(f"Converged: {'✓ YES' if converged else '✗ NO - Refine further'}")
    print("="*60)
    print("\nResolution Details:")
    for i, ((nx, ny), h, metric, err, gci_val) in enumerate(
        zip(resolutions, grid_sizes, metrics, relative_errors, gci)
    ):
        print(f"  {nx:3d}×{ny:3d}  h={h:.4f}  metric={metric:.6f}  "
              f"rel_err={err:.4%}  GCI={gci_val:.4%}")
    print("="*60 + "\n")
    
    return result


def _plot_mesh_independence(result: MeshIndependenceResult, threshold: float):
    """Plot mesh independence convergence curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Metric vs grid size
    ax1.plot(result.grid_sizes, result.metrics, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Grid Size h', fontsize=11)
    ax1.set_ylabel('Metric Value', fontsize=11)
    ax1.set_title('Mesh Convergence', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Mark finest mesh
    ax1.plot(result.grid_sizes[-1], result.metrics[-1], 'r*', markersize=15, 
             label='Finest Mesh')
    ax1.legend()
    
    # Plot 2: Relative error and GCI
    x_labels = [f"{nx}×{ny}" for nx, ny in result.resolutions]
    x_pos = np.arange(len(x_labels))
    
    ax2.semilogy(x_pos[1:], result.relative_errors[1:], 'o-', label='Relative Error', 
                 linewidth=2, markersize=8)
    ax2.semilogy(x_pos[1:], result.gci[1:], 's-', label='GCI', 
                 linewidth=2, markersize=8)
    ax2.axhline(threshold, color='r', linestyle='--', alpha=0.5, 
                label=f'{threshold:.1%} Threshold')
    
    ax2.set_xlabel('Mesh Resolution', fontsize=11)
    ax2.set_ylabel('Error Metric', fontsize=11)
    ax2.set_title('Convergence Metrics', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()


def quick_mesh_study(
    solver_func: Callable[[int, int], Dict],
    base_res: int = 20,
    num_levels: int = 4,
    metric_key: str = 'u',
    metric_func: Optional[Callable] = None
) -> MeshIndependenceResult:
    """
    Quick mesh independence study with automatic doubling.
    
    Args:
        solver_func: Solver function taking (nx, ny)
        base_res: Base resolution (will double for each level)
        num_levels: Number of refinement levels
        metric_key: Key in results dict to extract (if metric_func not provided)
        metric_func: Custom function to extract metric from results
    
    Returns:
        MeshIndependenceResult
    """
    resolutions = [(base_res * (2**i), base_res * (2**i)) for i in range(num_levels)]
    
    if metric_func is None:
        # Default: max absolute value of field
        def metric_func(res):
            return np.max(np.abs(res[metric_key]))
    
    return mesh_independence_study(solver_func, resolutions, metric_func)


if __name__ == '__main__':
    # Example: simple Poisson equation solver for demo
    def solve_poisson(nx, ny):
        """Dummy solver for demonstration"""
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        # Analytical solution to Laplace equation
        u = np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)
        return {'u': u, 'nx': nx, 'ny': ny}
    
    def get_max_u(results):
        return np.max(results['u'])
    
    # Run study
    resolutions = [(10, 10), (20, 20), (40, 40), (80, 80)]
    result = mesh_independence_study(solve_poisson, resolutions, get_max_u)
    
    print(f"\nExample completed. Converged: {result.converged}")
