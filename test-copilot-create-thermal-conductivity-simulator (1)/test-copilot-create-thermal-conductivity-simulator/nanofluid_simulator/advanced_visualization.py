#!/usr/bin/env python3
"""
BKPS NFL Thermal Pro 7.0 - Advanced Visualization Tools
Dedicated to: Brijesh Kumar Pandey

Features:
- 2D contour plots (temperature, velocity fields)
- Velocity vector fields
- Q-criterion vortex identification
- Streamlines
- Sensitivity analysis (Sobol indices, Morris screening)
- High-resolution export (600+ DPI, SVG, PDF)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from SALib.analyze import sobol, morris
    from SALib.sample import saltelli, morris as morris_sample
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False


class AdvancedVisualizer:
    """
    Advanced visualization tools for nanofluid simulations
    
    Provides scientific-grade plotting for:
    - CFD results (contours, vectors, streamlines)
    - Sensitivity analysis
    - Publication-ready figures
    """
    
    def __init__(self, dpi: int = 300):
        """
        Initialize visualizer
        
        Args:
            dpi: Resolution for exports (default: 300, recommend 600 for publication)
        """
        self.dpi = dpi
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.family'] = 'serif'
    
    def plot_2d_contour(self,
                       x: np.ndarray,
                       y: np.ndarray,
                       field: np.ndarray,
                       title: str = "Field Contour",
                       xlabel: str = "X",
                       ylabel: str = "Y",
                       clabel: str = "Field Value",
                       levels: int = 20,
                       cmap: str = 'jet',
                       figsize: Tuple[float, float] = (10, 8)) -> plt.Figure:
        """
        Create 2D contour plot
        
        Args:
            x, y: Coordinate meshgrids
            field: 2D array of field values
            title: Plot title
            xlabel, ylabel: Axis labels
            clabel: Colorbar label
            levels: Number of contour levels
            cmap: Colormap name
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Contour plot
        contour = ax.contourf(x, y, field, levels=levels, cmap=cmap)
        
        # Contour lines
        contour_lines = ax.contour(x, y, field, levels=levels, 
                                   colors='black', alpha=0.3, linewidths=0.5)
        
        # Colorbar
        cbar = plt.colorbar(contour, ax=ax, label=clabel)
        cbar.ax.tick_params(labelsize=9)
        
        # Labels
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Aspect ratio
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def plot_velocity_vectors(self,
                             x: np.ndarray,
                             y: np.ndarray,
                             u: np.ndarray,
                             v: np.ndarray,
                             magnitude: Optional[np.ndarray] = None,
                             title: str = "Velocity Field",
                             xlabel: str = "X (m)",
                             ylabel: str = "Y (m)",
                             skip: int = 2,
                             scale: float = None,
                             figsize: Tuple[float, float] = (10, 8)) -> plt.Figure:
        """
        Create velocity vector field plot
        
        Args:
            x, y: Coordinate meshgrids
            u, v: Velocity components
            magnitude: Velocity magnitude (computed if None)
            title: Plot title
            xlabel, ylabel: Axis labels
            skip: Skip every N vectors for clarity
            scale: Arrow scale (auto if None)
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate magnitude if not provided
        if magnitude is None:
            magnitude = np.sqrt(u**2 + v**2)
        
        # Background: velocity magnitude contour
        contour = ax.contourf(x, y, magnitude, levels=20, cmap='viridis', alpha=0.7)
        cbar = plt.colorbar(contour, ax=ax, label='Velocity Magnitude (m/s)')
        
        # Velocity vectors
        skip_slice = (slice(None, None, skip), slice(None, None, skip))
        ax.quiver(x[skip_slice], y[skip_slice], 
                 u[skip_slice], v[skip_slice],
                 magnitude[skip_slice],
                 scale=scale,
                 cmap='jet',
                 alpha=0.8,
                 width=0.003,
                 headwidth=4,
                 headlength=5)
        
        # Labels
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.set_aspect('equal')
        plt.tight_layout()
        return fig
    
    def plot_q_criterion(self,
                        x: np.ndarray,
                        y: np.ndarray,
                        u: np.ndarray,
                        v: np.ndarray,
                        threshold: float = 0.0,
                        title: str = "Q-Criterion (Vortex Identification)",
                        figsize: Tuple[float, float] = (10, 8)) -> plt.Figure:
        """
        Calculate and plot Q-criterion for vortex identification
        
        Q = 0.5 * (Ω² - S²)
        where Ω is vorticity magnitude and S is strain rate magnitude
        
        Args:
            x, y: Coordinate meshgrids
            u, v: Velocity components
            threshold: Q-criterion threshold for vortex regions
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        # Calculate velocity gradients
        du_dx = np.gradient(u, axis=1)
        du_dy = np.gradient(u, axis=0)
        dv_dx = np.gradient(v, axis=1)
        dv_dy = np.gradient(v, axis=0)
        
        # Vorticity (Ω)
        omega = dv_dx - du_dy
        omega_squared = omega**2
        
        # Strain rate (S)
        S11 = du_dx
        S12 = 0.5 * (du_dy + dv_dx)
        S22 = dv_dy
        S_squared = S11**2 + 2*S12**2 + S22**2
        
        # Q-criterion
        Q = 0.5 * (omega_squared - S_squared)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Q-criterion contour
        levels = np.linspace(max(threshold, Q.min()), Q.max(), 20)
        contour = ax.contourf(x, y, Q, levels=levels, cmap='RdBu_r', extend='both')
        
        # Zero contour (vortex boundaries)
        ax.contour(x, y, Q, levels=[threshold], colors='black', linewidths=2)
        
        # Colorbar
        cbar = plt.colorbar(contour, ax=ax, label='Q-Criterion (s⁻²)')
        
        # Labels
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.set_aspect('equal')
        plt.tight_layout()
        return fig
    
    def plot_streamlines(self,
                        x: np.ndarray,
                        y: np.ndarray,
                        u: np.ndarray,
                        v: np.ndarray,
                        density: float = 1.5,
                        title: str = "Streamlines",
                        figsize: Tuple[float, float] = (10, 8)) -> plt.Figure:
        """
        Create streamline plot
        
        Args:
            x, y: Coordinate meshgrids
            u, v: Velocity components
            density: Streamline density
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Velocity magnitude for background
        magnitude = np.sqrt(u**2 + v**2)
        
        # Background contour
        contour = ax.contourf(x, y, magnitude, levels=20, cmap='viridis', alpha=0.5)
        cbar = plt.colorbar(contour, ax=ax, label='Velocity Magnitude (m/s)')
        
        # Streamlines
        ax.streamplot(x, y, u, v, 
                     density=density,
                     color='white',
                     linewidth=1.5,
                     arrowsize=1.5,
                     arrowstyle='->')
        
        # Labels
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.set_aspect('equal')
        plt.tight_layout()
        return fig
    
    def sobol_sensitivity_analysis(self,
                                   problem: Dict,
                                   model_function: callable,
                                   n_samples: int = 1024) -> Dict[str, np.ndarray]:
        """
        Perform Sobol sensitivity analysis
        
        Args:
            problem: SALib problem definition with 'num_vars', 'names', 'bounds'
            model_function: Function to evaluate (takes array, returns scalar)
            n_samples: Number of samples
            
        Returns:
            Dictionary with S1 (first-order) and ST (total-order) indices
        """
        if not SALIB_AVAILABLE:
            raise ImportError("SALib not installed. Install with: pip install SALib")
        
        # Generate samples
        param_values = saltelli.sample(problem, n_samples)
        
        # Evaluate model
        Y = np.array([model_function(X) for X in param_values])
        
        # Perform Sobol analysis
        Si = sobol.analyze(problem, Y)
        
        return {
            'S1': Si['S1'],  # First-order indices
            'ST': Si['ST'],  # Total-order indices
            'S2': Si.get('S2', None),  # Second-order indices (if computed)
            'names': problem['names']
        }
    
    def morris_screening(self,
                        problem: Dict,
                        model_function: callable,
                        n_trajectories: int = 10) -> Dict[str, np.ndarray]:
        """
        Perform Morris screening (elementary effects)
        
        Args:
            problem: SALib problem definition
            model_function: Function to evaluate
            n_trajectories: Number of trajectories
            
        Returns:
            Dictionary with mu (mean), mu_star (absolute mean), sigma (std)
        """
        if not SALIB_AVAILABLE:
            raise ImportError("SALib not installed. Install with: pip install SALib")
        
        # Generate samples
        param_values = morris_sample.sample(problem, n_trajectories)
        
        # Evaluate model
        Y = np.array([model_function(X) for X in param_values])
        
        # Perform Morris analysis
        Si = morris.analyze(problem, param_values, Y)
        
        return {
            'mu': Si['mu'],  # Mean elementary effect
            'mu_star': Si['mu_star'],  # Absolute mean
            'sigma': Si['sigma'],  # Standard deviation
            'names': problem['names']
        }
    
    def plot_sobol_indices(self,
                          sobol_results: Dict,
                          title: str = "Sobol Sensitivity Indices",
                          figsize: Tuple[float, float] = (12, 6)) -> plt.Figure:
        """
        Plot Sobol sensitivity indices
        
        Args:
            sobol_results: Results from sobol_sensitivity_analysis
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        names = sobol_results['names']
        S1 = sobol_results['S1']
        ST = sobol_results['ST']
        
        x = np.arange(len(names))
        width = 0.35
        
        # First-order indices
        ax1.bar(x, S1, width, label='S1 (First-order)', color='#2E86AB')
        ax1.set_xlabel('Parameters', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Sensitivity Index', fontsize=12, fontweight='bold')
        ax1.set_title('First-Order Indices', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend()
        
        # Total-order indices
        ax2.bar(x, ST, width, label='ST (Total-order)', color='#06A77D')
        ax2.set_xlabel('Parameters', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Sensitivity Index', fontsize=12, fontweight='bold')
        ax2.set_title('Total-Order Indices', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_morris_screening(self,
                             morris_results: Dict,
                             title: str = "Morris Screening",
                             figsize: Tuple[float, float] = (10, 8)) -> plt.Figure:
        """
        Plot Morris screening results (mu* vs sigma)
        
        Args:
            morris_results: Results from morris_screening
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        names = morris_results['names']
        mu_star = morris_results['mu_star']
        sigma = morris_results['sigma']
        
        # Scatter plot
        scatter = ax.scatter(mu_star, sigma, s=200, c=range(len(names)),
                           cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
        
        # Annotate points
        for i, name in enumerate(names):
            ax.annotate(name, (mu_star[i], sigma[i]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
        
        # Labels
        ax.set_xlabel('μ* (Mean Absolute Effect)', fontsize=12, fontweight='bold')
        ax.set_ylabel('σ (Standard Deviation)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Parameter Index')
        
        plt.tight_layout()
        return fig
    
    def export_high_res(self,
                       fig: plt.Figure,
                       filepath: str,
                       dpi: int = 600,
                       formats: List[str] = ['png', 'pdf', 'svg']) -> List[str]:
        """
        Export figure in multiple high-resolution formats
        
        Args:
            fig: matplotlib Figure object
            filepath: Base filepath (without extension)
            dpi: Resolution (600+ recommended for publication)
            formats: List of formats to export
            
        Returns:
            List of created file paths
        """
        created_files = []
        
        for fmt in formats:
            output_path = f"{filepath}.{fmt}"
            
            if fmt == 'svg':
                fig.savefig(output_path, format='svg', bbox_inches='tight')
            else:
                fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches='tight')
            
            created_files.append(output_path)
        
        return created_files


def create_sample_cfd_field(nx: int = 50, ny: int = 50) -> Dict[str, np.ndarray]:
    """
    Create sample CFD field for demonstration
    
    Args:
        nx, ny: Grid dimensions
        
    Returns:
        Dictionary with x, y, u, v, T fields
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Sample velocity field (lid-driven cavity-like)
    U = np.sin(np.pi * X) * np.cos(np.pi * Y)
    V = -np.cos(np.pi * X) * np.sin(np.pi * Y)
    
    # Sample temperature field
    T = 300 + 20 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    
    return {'x': X, 'y': Y, 'u': U, 'v': V, 'T': T}


if __name__ == '__main__':
    # Demo: Create sample visualizations
    print("BKPS NFL Thermal Pro 7.0 - Advanced Visualization Demo")
    print("=" * 60)
    
    # Create visualizer
    viz = AdvancedVisualizer(dpi=300)
    
    # Generate sample CFD data
    data = create_sample_cfd_field(nx=40, ny=40)
    
    # 1. Temperature contour
    print("Creating temperature contour...")
    fig1 = viz.plot_2d_contour(
        data['x'], data['y'], data['T'],
        title="Temperature Field",
        clabel="Temperature (K)",
        cmap='hot'
    )
    plt.savefig('/tmp/temperature_contour.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: /tmp/temperature_contour.png")
    
    # 2. Velocity vectors
    print("Creating velocity vector field...")
    fig2 = viz.plot_velocity_vectors(
        data['x'], data['y'], data['u'], data['v'],
        skip=2
    )
    plt.savefig('/tmp/velocity_vectors.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: /tmp/velocity_vectors.png")
    
    # 3. Q-criterion
    print("Creating Q-criterion vortex plot...")
    fig3 = viz.plot_q_criterion(
        data['x'], data['y'], data['u'], data['v']
    )
    plt.savefig('/tmp/q_criterion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: /tmp/q_criterion.png")
    
    # 4. Streamlines
    print("Creating streamlines...")
    fig4 = viz.plot_streamlines(
        data['x'], data['y'], data['u'], data['v']
    )
    plt.savefig('/tmp/streamlines.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: /tmp/streamlines.png")
    
    print("\n✅ Advanced visualization demo complete!")
    print("Generated 4 high-quality visualization examples")
