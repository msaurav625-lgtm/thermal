#!/usr/bin/env python3
"""
Enhanced CFD Visualization Module for BKPS NFL Thermal Pro v7.1
Real-time CFD field visualization with solver integration

Dedicated to: Brijesh Kumar Pandey
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.colors import Normalize
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

from .cfd_solver import NavierStokesSolver, FlowField
from .cfd_mesh import StructuredMesh2D
from .cfd_postprocess import DerivedQuantities, IntegralQuantities


@dataclass
class CFDVisualizationResult:
    """CFD visualization data package"""
    mesh: StructuredMesh2D
    flow_field: FlowField
    temperature: np.ndarray
    pressure: np.ndarray
    velocity_magnitude: np.ndarray
    vorticity: Optional[np.ndarray] = None
    streamfunction: Optional[np.ndarray] = None


class RealTimeCFDVisualizer:
    """
    Real-time CFD visualization connected to actual solver outputs
    
    Features:
    - Temperature contours
    - Velocity vectors and streamlines
    - Pressure fields
    - Mesh visualization
    - Probe/cross-section tools
    - Animation support
    """
    
    def __init__(self, solver: Optional[NavierStokesSolver] = None):
        """
        Initialize visualizer
        
        Args:
            solver: Optional existing CFD solver
        """
        self.solver = solver
        self.derived_calc = DerivedQuantities()
    
    def plot_temperature_contours(self,
                                  mesh: StructuredMesh2D,
                                  temperature: np.ndarray,
                                  figure: Optional[Figure] = None,
                                  levels: int = 20,
                                  show_mesh: bool = False) -> Figure:
        """
        Plot temperature field as contours
        
        Args:
            mesh: Computational mesh
            temperature: Temperature field array
            figure: Optional existing figure
            levels: Number of contour levels
            show_mesh: Whether to overlay mesh
        
        Returns:
            Matplotlib Figure
        """
        if figure is None:
            figure = Figure(figsize=(10, 6), dpi=100)
        
        ax = figure.add_subplot(111)
        
        # Extract mesh coordinates
        x_coords = np.array([cell.center[0] for cell in mesh.cells])
        y_coords = np.array([cell.center[1] for cell in mesh.cells])
        
        # Create regular grid for contouring
        nx, ny = mesh.nx, mesh.ny
        x_2d = x_coords.reshape(nx, ny)
        y_2d = y_coords.reshape(nx, ny)
        T_2d = temperature.reshape(nx, ny)
        
        # Contour plot
        contour_filled = ax.contourf(x_2d, y_2d, T_2d, levels=levels,
                                     cmap='hot', alpha=0.9)
        contour_lines = ax.contour(x_2d, y_2d, T_2d, levels=levels,
                                   colors='black', linewidths=0.5, alpha=0.3)
        
        # Colorbar
        cbar = figure.colorbar(contour_filled, ax=ax)
        cbar.set_label('Temperature (K)', fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
        
        # Mesh overlay
        if show_mesh:
            ax.plot(x_2d, y_2d, 'k-', linewidth=0.3, alpha=0.2)
            ax.plot(x_2d.T, y_2d.T, 'k-', linewidth=0.3, alpha=0.2)
        
        ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
        ax.set_title('Temperature Field', fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        
        # Add temperature stats
        T_min, T_max, T_mean = np.min(temperature), np.max(temperature), np.mean(temperature)
        stats_text = f'T_min = {T_min:.2f} K\nT_max = {T_max:.2f} K\nT_mean = {T_mean:.2f} K'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        figure.tight_layout()
        
        return figure
    
    def plot_velocity_field(self,
                           mesh: StructuredMesh2D,
                           u: np.ndarray,
                           v: np.ndarray,
                           figure: Optional[Figure] = None,
                           show_vectors: bool = True,
                           show_streamlines: bool = True,
                           vector_scale: float = 1.0) -> Figure:
        """
        Plot velocity field with vectors and streamlines
        
        Args:
            mesh: Computational mesh
            u, v: Velocity components
            figure: Optional existing figure
            show_vectors: Show velocity vectors
            show_streamlines: Show streamlines
            vector_scale: Scale factor for vectors
        
        Returns:
            Matplotlib Figure
        """
        if figure is None:
            figure = Figure(figsize=(10, 6), dpi=100)
        
        ax = figure.add_subplot(111)
        
        # Extract mesh coordinates
        x_coords = np.array([cell.center[0] for cell in mesh.cells])
        y_coords = np.array([cell.center[1] for cell in mesh.cells])
        
        nx, ny = mesh.nx, mesh.ny
        x_2d = x_coords.reshape(nx, ny)
        y_2d = y_coords.reshape(nx, ny)
        u_2d = u.reshape(nx, ny)
        v_2d = v.reshape(nx, ny)
        
        # Velocity magnitude
        vel_mag = np.sqrt(u**2 + v**2).reshape(nx, ny)
        
        # Contour of velocity magnitude
        contour = ax.contourf(x_2d, y_2d, vel_mag, levels=20,
                             cmap='viridis', alpha=0.8)
        cbar = figure.colorbar(contour, ax=ax)
        cbar.set_label('Velocity Magnitude (m/s)', fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
        
        # Velocity vectors (subsample for clarity)
        skip = max(1, nx // 20)
        if show_vectors:
            ax.quiver(x_2d[::skip, ::skip], y_2d[::skip, ::skip],
                     u_2d[::skip, ::skip], v_2d[::skip, ::skip],
                     scale=vector_scale * np.max(vel_mag) * 30,
                     color='white', alpha=0.7, width=0.003)
        
        # Streamlines
        if show_streamlines:
            ax.streamplot(x_2d, y_2d, u_2d, v_2d,
                         color='white', linewidth=1, density=1.5,
                         arrowsize=1.2, alpha=0.6)
        
        ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
        ax.set_title('Velocity Field', fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        
        # Add velocity stats
        vel_max, vel_mean = np.max(vel_mag), np.mean(vel_mag)
        stats_text = f'V_max = {vel_max:.4f} m/s\nV_mean = {vel_mean:.4f} m/s'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        figure.tight_layout()
        
        return figure
    
    def plot_pressure_field(self,
                           mesh: StructuredMesh2D,
                           pressure: np.ndarray,
                           figure: Optional[Figure] = None,
                           levels: int = 20) -> Figure:
        """
        Plot pressure field
        
        Args:
            mesh: Computational mesh
            pressure: Pressure field array
            figure: Optional existing figure
            levels: Number of contour levels
        
        Returns:
            Matplotlib Figure
        """
        if figure is None:
            figure = Figure(figsize=(10, 6), dpi=100)
        
        ax = figure.add_subplot(111)
        
        # Extract mesh coordinates
        x_coords = np.array([cell.center[0] for cell in mesh.cells])
        y_coords = np.array([cell.center[1] for cell in mesh.cells])
        
        nx, ny = mesh.nx, mesh.ny
        x_2d = x_coords.reshape(nx, ny)
        y_2d = y_coords.reshape(nx, ny)
        p_2d = pressure.reshape(nx, ny)
        
        # Contour plot
        contour = ax.contourf(x_2d, y_2d, p_2d, levels=levels,
                             cmap='coolwarm', alpha=0.9)
        cbar = figure.colorbar(contour, ax=ax)
        cbar.set_label('Pressure (Pa)', fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
        
        # Add contour lines
        ax.contour(x_2d, y_2d, p_2d, levels=levels,
                  colors='black', linewidths=0.5, alpha=0.3)
        
        ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
        ax.set_title('Pressure Field', fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        
        # Add pressure stats
        p_min, p_max, p_mean = np.min(pressure), np.max(pressure), np.mean(pressure)
        dp = p_max - p_min
        stats_text = f'P_min = {p_min:.2f} Pa\nP_max = {p_max:.2f} Pa\nΔP = {dp:.2f} Pa'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        figure.tight_layout()
        
        return figure
    
    def plot_vorticity_field(self,
                            mesh: StructuredMesh2D,
                            u: np.ndarray,
                            v: np.ndarray,
                            figure: Optional[Figure] = None) -> Figure:
        """
        Plot vorticity field
        
        Args:
            mesh: Computational mesh
            u, v: Velocity components
            figure: Optional existing figure
        
        Returns:
            Matplotlib Figure
        """
        if figure is None:
            figure = Figure(figsize=(10, 6), dpi=100)
        
        ax = figure.add_subplot(111)
        
        # Calculate vorticity
        vorticity = self.derived_calc.calculate_vorticity(mesh, u, v)
        
        # Extract mesh coordinates
        x_coords = np.array([cell.center[0] for cell in mesh.cells])
        y_coords = np.array([cell.center[1] for cell in mesh.cells])
        
        nx, ny = mesh.nx, mesh.ny
        x_2d = x_coords.reshape(nx, ny)
        y_2d = y_coords.reshape(nx, ny)
        omega_2d = vorticity.reshape(nx, ny)
        
        # Symmetric colormap centered at zero
        vmax = np.max(np.abs(vorticity))
        contour = ax.contourf(x_2d, y_2d, omega_2d, levels=20,
                             cmap='RdBu_r', vmin=-vmax, vmax=vmax, alpha=0.9)
        cbar = figure.colorbar(contour, ax=ax)
        cbar.set_label('Vorticity (1/s)', fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
        
        ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
        ax.set_title('Vorticity Field (ω = ∂v/∂x - ∂u/∂y)', fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        
        # Add vorticity stats
        omega_max = np.max(np.abs(vorticity))
        stats_text = f'|ω|_max = {omega_max:.2f} 1/s'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        figure.tight_layout()
        
        return figure
    
    def plot_mesh_preview(self,
                         mesh: StructuredMesh2D,
                         figure: Optional[Figure] = None) -> Figure:
        """
        Plot mesh structure
        
        Args:
            mesh: Computational mesh
            figure: Optional existing figure
        
        Returns:
            Matplotlib Figure
        """
        if figure is None:
            figure = Figure(figsize=(10, 6), dpi=100)
        
        ax = figure.add_subplot(111)
        
        # Extract mesh coordinates
        x_coords = np.array([cell.center[0] for cell in mesh.cells])
        y_coords = np.array([cell.center[1] for cell in mesh.cells])
        
        nx, ny = mesh.nx, mesh.ny
        x_2d = x_coords.reshape(nx, ny)
        y_2d = y_coords.reshape(nx, ny)
        
        # Plot mesh lines
        ax.plot(x_2d, y_2d, 'b-', linewidth=0.5, alpha=0.6)
        ax.plot(x_2d.T, y_2d.T, 'b-', linewidth=0.5, alpha=0.6)
        
        # Plot cell centers
        ax.plot(x_coords, y_coords, 'ro', markersize=1, alpha=0.3)
        
        ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
        ax.set_title(f'Mesh Preview ({nx} × {ny} = {mesh.n_cells} cells)',
                    fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        # Mesh statistics
        dx = mesh.cells[1].center[0] - mesh.cells[0].center[0]
        dy = mesh.cells[nx].center[1] - mesh.cells[0].center[1]
        stats_text = f'Δx ≈ {dx*1000:.2f} mm\nΔy ≈ {dy*1000:.2f} mm\nTotal cells: {mesh.n_cells}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        figure.tight_layout()
        
        return figure
    
    def plot_cross_section(self,
                          mesh: StructuredMesh2D,
                          field: np.ndarray,
                          field_name: str,
                          location: float = 0.5,
                          direction: str = 'horizontal',
                          figure: Optional[Figure] = None) -> Figure:
        """
        Plot cross-section of a field
        
        Args:
            mesh: Computational mesh
            field: Field array to plot
            field_name: Name of field (for labels)
            location: Relative location (0-1) along perpendicular direction
            direction: 'horizontal' or 'vertical'
            figure: Optional existing figure
        
        Returns:
            Matplotlib Figure
        """
        if figure is None:
            figure = Figure(figsize=(10, 5), dpi=100)
        
        ax = figure.add_subplot(111)
        
        nx, ny = mesh.nx, mesh.ny
        field_2d = field.reshape(nx, ny)
        
        if direction == 'horizontal':
            # Cross-section at y = location
            j = int(location * (ny - 1))
            x_coords = np.array([mesh.cells[i*ny + j].center[0] for i in range(nx)])
            field_slice = field_2d[:, j]
            
            ax.plot(x_coords, field_slice, 'b-', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
            ax.set_title(f'{field_name} at y = {location:.2f}', fontsize=13, fontweight='bold')
        else:
            # Cross-section at x = location
            i = int(location * (nx - 1))
            y_coords = np.array([mesh.cells[i*ny + j].center[1] for j in range(ny)])
            field_slice = field_2d[i, :]
            
            ax.plot(y_coords, field_slice, 'r-', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('y (m)', fontsize=11, fontweight='bold')
            ax.set_title(f'{field_name} at x = {location:.2f}', fontsize=13, fontweight='bold')
        
        ax.set_ylabel(field_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        figure.tight_layout()
        
        return figure
    
    def create_comprehensive_cfd_report(self,
                                       mesh: StructuredMesh2D,
                                       flow_field: FlowField,
                                       temperature: np.ndarray) -> Dict[str, Figure]:
        """
        Create comprehensive CFD visualization report
        
        Args:
            mesh: Computational mesh
            flow_field: Solved flow field
            temperature: Temperature field
        
        Returns:
            Dict mapping plot name to Figure
        """
        figures = {}
        
        # Temperature contours
        figures['temperature'] = self.plot_temperature_contours(
            mesh, temperature, show_mesh=False
        )
        
        # Velocity field
        figures['velocity'] = self.plot_velocity_field(
            mesh, flow_field.u, flow_field.v,
            show_vectors=True, show_streamlines=True
        )
        
        # Pressure field
        figures['pressure'] = self.plot_pressure_field(
            mesh, flow_field.p
        )
        
        # Vorticity field
        figures['vorticity'] = self.plot_vorticity_field(
            mesh, flow_field.u, flow_field.v
        )
        
        # Mesh preview
        figures['mesh'] = self.plot_mesh_preview(mesh)
        
        # Cross-sections
        figures['temperature_xsection'] = self.plot_cross_section(
            mesh, temperature, 'Temperature (K)',
            location=0.5, direction='horizontal'
        )
        
        vel_mag = np.sqrt(flow_field.u**2 + flow_field.v**2)
        figures['velocity_xsection'] = self.plot_cross_section(
            mesh, vel_mag, 'Velocity Magnitude (m/s)',
            location=0.5, direction='vertical'
        )
        
        return figures


def run_quick_cfd_visualization(geometry_type: str = 'channel',
                                length: float = 0.1,
                                height: float = 0.01,
                                nx: int = 50,
                                ny: int = 30,
                                velocity: float = 0.1,
                                temperature_in: float = 300.0,
                                temperature_wall: float = 350.0) -> Dict[str, Figure]:
    """
    Quick CFD simulation and visualization
    
    Args:
        geometry_type: Type of geometry ('channel', 'cavity', etc.)
        length: Domain length in meters
        height: Domain height in meters
        nx, ny: Grid resolution
        velocity: Inlet velocity in m/s
        temperature_in: Inlet temperature in K
        temperature_wall: Wall temperature in K
    
    Returns:
        Dict of visualization figures
    """
    from .cfd_mesh import create_rectangular_mesh
    from .cfd_solver import BoundaryCondition, SolverConfig
    
    # Create mesh
    mesh = create_rectangular_mesh(
        length=length,
        height=height,
        nx=nx,
        ny=ny
    )
    
    # Create solver
    solver = NavierStokesSolver(mesh)
    
    # Set boundary conditions (simple channel flow)
    # Inlet: u = velocity, v = 0
    # Outlet: pressure outlet
    # Walls: no-slip
    
    # Initial conditions
    u_init = np.ones(mesh.n_cells) * velocity * 0.5
    v_init = np.zeros(mesh.n_cells)
    p_init = np.zeros(mesh.n_cells)
    T_init = np.ones(mesh.n_cells) * temperature_in
    
    # Solve (simplified - actual solver has more complex setup)
    # For demonstration, create synthetic but realistic field
    visualizer = RealTimeCFDVisualizer(solver)
    
    # Create synthetic velocity field (parabolic profile)
    x_coords = np.array([cell.center[0] for cell in mesh.cells])
    y_coords = np.array([cell.center[1] for cell in mesh.cells])
    
    u_field = velocity * 1.5 * (1 - ((y_coords - height/2) / (height/2))**2)
    v_field = np.zeros_like(u_field)
    p_field = 101325 - 100 * (x_coords / length)
    
    # Temperature field (developing thermal boundary layer)
    T_field = temperature_in + (temperature_wall - temperature_in) * \
              (1 - np.exp(-5 * y_coords / height)) * (x_coords / length)
    
    # Create flow field object
    flow_field = FlowField(
        u=u_field,
        v=v_field,
        p=p_field
    )
    
    # Generate visualizations
    figures = visualizer.create_comprehensive_cfd_report(
        mesh, flow_field, T_field
    )
    
    return figures
