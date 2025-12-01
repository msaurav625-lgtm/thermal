"""
Comprehensive Visualization Module for Nanofluid Simulator

Provides flow field visualization, thermal contours, streamlines, and vector plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
from typing import Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class FlowVisualizer:
    """Handles flow field visualization with thermal contours."""
    
    @staticmethod
    def create_flow_field(
        geometry: str = "pipe",
        nx: int = 50,
        ny: int = 50,
        reynolds: float = 1000,
        temperature_range: Tuple[float, float] = (293, 353)
    ) -> Dict[str, np.ndarray]:
        """
        Create a flow field with temperature distribution.
        
        Args:
            geometry: Flow geometry type ('pipe', 'channel', 'plate')
            nx, ny: Grid resolution
            reynolds: Reynolds number
            temperature_range: (T_inlet, T_wall) in Kelvin
            
        Returns:
            Dictionary with X, Y, U, V velocity components and Temperature
        """
        # Create meshgrid
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        T_inlet, T_wall = temperature_range
        
        if geometry == "pipe":
            # Fully developed laminar pipe flow (Hagen-Poiseuille)
            R = 0.5  # Pipe radius
            r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
            
            # Velocity profile: parabolic
            U_max = reynolds * 0.01
            U = U_max * np.maximum(0, 1 - (r / R)**2)
            V = np.zeros_like(U)
            
            # Temperature profile: radial conduction
            T = T_inlet + (T_wall - T_inlet) * (r / R)**2
            T = np.clip(T, T_inlet, T_wall)
            
        elif geometry == "channel":
            # Plane Poiseuille flow
            U_max = reynolds * 0.01
            U = U_max * 4 * Y * (1 - Y)
            V = np.zeros_like(U)
            
            # Temperature: linear in y-direction
            T = T_inlet + (T_wall - T_inlet) * Y
            
        elif geometry == "plate":
            # Boundary layer over flat plate
            delta = 5 / np.sqrt(reynolds)  # Boundary layer thickness
            eta = Y / np.maximum(delta * np.sqrt(X + 0.01), 1e-6)
            
            # Blasius solution approximation
            U = np.tanh(eta)
            V = 0.5 * eta * (1 - np.tanh(eta)**2) * delta / np.maximum(np.sqrt(X + 0.01), 1e-6)
            
            # Thermal boundary layer
            T = T_inlet + (T_wall - T_inlet) * (1 - np.exp(-eta))
            
        else:
            # Default: uniform flow
            U = np.ones_like(X) * reynolds * 0.01
            V = np.zeros_like(U)
            T = T_inlet + (T_wall - T_inlet) * Y
        
        return {
            'X': X,
            'Y': Y,
            'U': U,
            'V': V,
            'T': T,
            'reynolds': reynolds,
            'geometry': geometry
        }
    
    @staticmethod
    def plot_thermal_contours(
        flow_data: Dict[str, np.ndarray],
        fig: Optional[Figure] = None,
        ax = None,
        show_colorbar: bool = True
    ):
        """
        Plot thermal contours with flow overlay.
        
        Args:
            flow_data: Dictionary from create_flow_field
            fig: Matplotlib figure (optional)
            ax: Matplotlib axis (optional)
            show_colorbar: Whether to show colorbar
        """
        if ax is None:
            if fig is None:
                fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
        
        X, Y = flow_data['X'], flow_data['Y']
        T = flow_data['T']
        
        # Plot thermal contours
        levels = np.linspace(T.min(), T.max(), 20)
        contourf = ax.contourf(X, Y, T, levels=levels, cmap='jet', alpha=0.8)
        contour = ax.contour(X, Y, T, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        if show_colorbar and fig is not None:
            cbar = fig.colorbar(contourf, ax=ax, label='Temperature (K)')
            cbar.ax.tick_params(labelsize=8)
        
        ax.set_xlabel('x/L')
        ax.set_ylabel('y/L')
        ax.set_title(f'Thermal Contours - {flow_data["geometry"].title()} Flow (Re={flow_data["reynolds"]:.0f})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    @staticmethod
    def plot_velocity_field(
        flow_data: Dict[str, np.ndarray],
        fig: Optional[Figure] = None,
        ax = None,
        vector_density: int = 10,
        show_magnitude: bool = True
    ):
        """
        Plot velocity vector field with magnitude contours.
        
        Args:
            flow_data: Dictionary from create_flow_field
            fig: Matplotlib figure (optional)
            ax: Matplotlib axis (optional)
            vector_density: Spacing between vectors
            show_magnitude: Whether to show velocity magnitude contours
        """
        if ax is None:
            if fig is None:
                fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
        
        X, Y = flow_data['X'], flow_data['Y']
        U, V = flow_data['U'], flow_data['V']
        
        # Calculate velocity magnitude
        magnitude = np.sqrt(U**2 + V**2)
        
        # Plot magnitude contours
        if show_magnitude:
            levels = np.linspace(0, magnitude.max(), 15)
            contourf = ax.contourf(X, Y, magnitude, levels=levels, cmap='viridis', alpha=0.6)
            if fig is not None:
                cbar = fig.colorbar(contourf, ax=ax, label='Velocity Magnitude (m/s)')
                cbar.ax.tick_params(labelsize=8)
        
        # Plot velocity vectors
        skip = max(1, X.shape[0] // vector_density)
        ax.quiver(
            X[::skip, ::skip], Y[::skip, ::skip],
            U[::skip, ::skip], V[::skip, ::skip],
            magnitude[::skip, ::skip],
            cmap='plasma',
            scale=magnitude.max() * 20,
            width=0.003,
            headwidth=4,
            headlength=5,
            alpha=0.8
        )
        
        ax.set_xlabel('x/L')
        ax.set_ylabel('y/L')
        ax.set_title(f'Velocity Field - {flow_data["geometry"].title()} Flow (Re={flow_data["reynolds"]:.0f})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    @staticmethod
    def plot_streamlines(
        flow_data: Dict[str, np.ndarray],
        fig: Optional[Figure] = None,
        ax = None,
        density: float = 2.0,
        show_thermal: bool = True
    ):
        """
        Plot flow streamlines with thermal overlay.
        
        Args:
            flow_data: Dictionary from create_flow_field
            fig: Matplotlib figure (optional)
            ax: Matplotlib axis (optional)
            density: Streamline density
            show_thermal: Whether to overlay thermal contours
        """
        if ax is None:
            if fig is None:
                fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
        
        X, Y = flow_data['X'], flow_data['Y']
        U, V = flow_data['U'], flow_data['V']
        T = flow_data['T']
        
        # Plot thermal background
        if show_thermal:
            contourf = ax.contourf(X, Y, T, levels=20, cmap='coolwarm', alpha=0.4)
            if fig is not None:
                cbar = fig.colorbar(contourf, ax=ax, label='Temperature (K)')
                cbar.ax.tick_params(labelsize=8)
        
        # Plot streamlines
        magnitude = np.sqrt(U**2 + V**2)
        lw = 2 * magnitude / magnitude.max()  # Line width varies with velocity
        
        strm = ax.streamplot(
            X, Y, U, V,
            color=magnitude,
            linewidth=lw,
            cmap='viridis',
            density=density,
            arrowsize=1.5,
            arrowstyle='->',
            minlength=0.1
        )
        
        ax.set_xlabel('x/L')
        ax.set_ylabel('y/L')
        ax.set_title(f'Streamlines - {flow_data["geometry"].title()} Flow (Re={flow_data["reynolds"]:.0f})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    @staticmethod
    def plot_combined_analysis(
        flow_data: Dict[str, np.ndarray],
        fig: Optional[Figure] = None
    ):
        """
        Create a comprehensive 2x2 plot with all visualizations.
        
        Args:
            flow_data: Dictionary from create_flow_field
            fig: Matplotlib figure (optional)
        """
        if fig is None:
            fig = Figure(figsize=(14, 12))
        
        fig.suptitle(
            f'Complete Flow Analysis - {flow_data["geometry"].title()} '
            f'(Re = {flow_data["reynolds"]:.0f})',
            fontsize=14, fontweight='bold'
        )
        
        # Thermal contours
        ax1 = fig.add_subplot(2, 2, 1)
        FlowVisualizer.plot_thermal_contours(flow_data, fig, ax1)
        
        # Velocity field
        ax2 = fig.add_subplot(2, 2, 2)
        FlowVisualizer.plot_velocity_field(flow_data, fig, ax2, vector_density=12)
        
        # Streamlines
        ax3 = fig.add_subplot(2, 2, 3)
        FlowVisualizer.plot_streamlines(flow_data, fig, ax3, density=1.5)
        
        # Profiles
        ax4 = fig.add_subplot(2, 2, 4)
        FlowVisualizer._plot_profiles(flow_data, ax4)
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def _plot_profiles(flow_data: Dict[str, np.ndarray], ax):
        """Plot velocity and temperature profiles."""
        Y = flow_data['Y']
        U = flow_data['U']
        T = flow_data['T']
        
        # Extract centerline or mid-plane profiles
        mid_x = U.shape[1] // 2
        
        y_profile = Y[:, mid_x]
        u_profile = U[:, mid_x]
        t_profile = T[:, mid_x]
        
        # Normalize
        u_norm = u_profile / u_profile.max() if u_profile.max() > 0 else u_profile
        t_norm = (t_profile - t_profile.min()) / (t_profile.max() - t_profile.min() + 1e-10)
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(u_norm, y_profile, 'b-', linewidth=2, label='Velocity')
        line2 = ax2.plot(t_norm, y_profile, 'r-', linewidth=2, label='Temperature')
        
        ax.set_xlabel('Normalized Velocity')
        ax.set_ylabel('y/L', color='b')
        ax2.set_ylabel('Normalized Temperature', color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_title('Centerline Profiles')
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')
        
        return ax


class NanofluidPropertyVisualizer:
    """Visualize nanofluid thermal properties."""
    
    @staticmethod
    def plot_conductivity_enhancement(
        base_k: float,
        particle_k: float,
        volume_fractions: np.ndarray,
        temperature: float = 300,
        models: list = None,
        fig: Optional[Figure] = None,
        ax = None
    ):
        """
        Plot thermal conductivity enhancement vs volume fraction.
        
        Args:
            base_k: Base fluid thermal conductivity
            particle_k: Nanoparticle thermal conductivity
            volume_fractions: Array of volume fractions to plot
            temperature: Temperature in K
            models: List of model names to compare
            fig, ax: Matplotlib figure and axis
        """
        if ax is None:
            if fig is None:
                fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
        
        from ..models import MaxwellModel, HamiltonCrosserModel, YuChoiModel
        
        if models is None:
            models = ['Maxwell', 'Hamilton-Crosser', 'Yu-Choi']
        
        model_map = {
            'Maxwell': MaxwellModel(),
            'Hamilton-Crosser': HamiltonCrosserModel(sphericity=0.5),
            'Yu-Choi': YuChoiModel()
        }
        
        colors = ['b', 'r', 'g', 'm', 'c']
        
        for idx, model_name in enumerate(models):
            if model_name in model_map:
                model = model_map[model_name]
                k_eff = []
                
                for phi in volume_fractions:
                    k_nf = model.calculate_conductivity(
                        base_k=base_k,
                        particle_k=particle_k,
                        volume_fraction=phi,
                        temperature=temperature
                    )
                    k_eff.append(k_nf / base_k)  # Enhancement ratio
                
                ax.plot(
                    volume_fractions * 100,
                    k_eff,
                    color=colors[idx % len(colors)],
                    marker='o',
                    linewidth=2,
                    markersize=6,
                    label=model_name
                )
        
        ax.set_xlabel('Volume Fraction (%)', fontsize=12)
        ax.set_ylabel('k_nf / k_bf (Enhancement Ratio)', fontsize=12)
        ax.set_title(f'Thermal Conductivity Enhancement at T = {temperature} K', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        ax.set_xlim(left=0)
        
        return ax


def create_sample_visualization():
    """Create sample visualization for testing."""
    fig = Figure(figsize=(14, 10))
    
    # Create flow field
    flow_data = FlowVisualizer.create_flow_field(
        geometry="channel",
        nx=60,
        ny=60,
        reynolds=1000,
        temperature_range=(300, 350)
    )
    
    # Plot combined analysis
    FlowVisualizer.plot_combined_analysis(flow_data, fig)
    
    return fig


if __name__ == '__main__':
    # Test visualization
    fig = create_sample_visualization()
    plt.show()
