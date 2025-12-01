#!/usr/bin/env python3
"""
Nanoparticle Interaction Visualization Module
DLVO + Brownian + Aggregation Physics Visualization

Dedicated to: Brijesh Kumar Pandey
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .dlvo_theory import (
    comprehensive_dlvo_analysis,
    debye_length,
    van_der_waals_potential,
    electrostatic_repulsion_potential as electric_double_layer_potential
)


@dataclass
class InteractionVisualization:
    """Results from interaction visualization"""
    separation_distances: np.ndarray
    vdw_potential: np.ndarray
    edl_potential: np.ndarray
    total_potential: np.ndarray
    barrier_height: float
    primary_minimum: float
    secondary_minimum: float


class ParticleInteractionVisualizer:
    """
    Visualize nanoparticle interaction physics
    
    Features:
    - DLVO potential energy curves
    - Stability analysis
    - Brownian motion effects
    - Aggregation kinetics
    - Interfacial layer visualization
    """
    
    def __init__(self):
        """Initialize visualizer"""
        self.k_B = 1.380649e-23  # Boltzmann constant (J/K)
    
    def plot_dlvo_potential(self,
                           particle_diameter: float = 30e-9,
                           temperature: float = 300.0,
                           ionic_strength: float = 0.001,
                           hamaker: float = 3.7e-20,
                           zeta_potential: float = -0.03,
                           figure: Optional[Figure] = None) -> Figure:
        """
        Plot DLVO potential energy vs separation distance
        
        Args:
            particle_diameter: Particle diameter in meters
            temperature: Temperature in K
            ionic_strength: Ionic strength in mol/L
            hamaker: Hamaker constant in J
            zeta_potential: Zeta potential in V
            figure: Optional existing figure
        
        Returns:
            Matplotlib Figure
        """
        if figure is None:
            figure = Figure(figsize=(10, 7), dpi=100)
        
        # Create separation distance array (0.1 nm to 50 nm)
        h_values = np.logspace(-10, -8, 500)  # meters
        
        radius = particle_diameter / 2
        
        # Calculate potentials
        vdw_values = np.zeros_like(h_values)
        edl_values = np.zeros_like(h_values)
        total_values = np.zeros_like(h_values)
        
        kappa = 1.0 / debye_length(ionic_strength, temperature)
        
        for i, h in enumerate(h_values):
            # Van der Waals (attractive)
            vdw = van_der_waals_potential(hamaker, radius, h)
            
            # Electric double layer (repulsive)
            edl = electric_double_layer_potential(
                radius, zeta_potential, kappa, h, temperature
            )
            
            vdw_values[i] = vdw
            edl_values[i] = edl
            total_values[i] = vdw + edl
        
        # Convert to kT units
        kT = self.k_B * temperature
        vdw_kT = vdw_values / kT
        edl_kT = edl_values / kT
        total_kT = total_values / kT
        
        # Find barrier and minima
        barrier_height = np.max(total_kT)
        barrier_idx = np.argmax(total_kT)
        
        # Find primary minimum (closest approach)
        primary_min_idx = np.argmin(total_kT[:barrier_idx]) if barrier_idx > 0 else 0
        primary_minimum = total_kT[primary_min_idx]
        
        # Find secondary minimum (after barrier)
        secondary_min_idx = barrier_idx + np.argmin(total_kT[barrier_idx:])
        secondary_minimum = total_kT[secondary_min_idx]
        
        # Create main plot
        ax1 = figure.add_subplot(211)
        
        h_nm = h_values * 1e9  # Convert to nm
        
        ax1.plot(h_nm, vdw_kT, 'r--', linewidth=2, label='Van der Waals (attractive)')
        ax1.plot(h_nm, edl_kT, 'b--', linewidth=2, label='EDL (repulsive)')
        ax1.plot(h_nm, total_kT, 'k-', linewidth=3, label='Total DLVO')
        ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        # Mark barrier
        ax1.plot(h_nm[barrier_idx], barrier_height, 'go', markersize=12, 
                label=f'Barrier: {barrier_height:.1f} kT')
        
        # Mark minima
        ax1.plot(h_nm[primary_min_idx], primary_minimum, 'mo', markersize=10,
                label=f'Primary min: {primary_minimum:.1f} kT')
        ax1.plot(h_nm[secondary_min_idx], secondary_minimum, 'co', markersize=10,
                label=f'Secondary min: {secondary_minimum:.1f} kT')
        
        ax1.set_xlabel('Separation Distance (nm)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Interaction Potential (kT)', fontsize=12, fontweight='bold')
        ax1.set_title('DLVO Interaction Potential Energy', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.legend(loc='best', fontsize=9, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.1, 50)
        
        # Stability interpretation
        ax2 = figure.add_subplot(212)
        ax2.axis('off')
        
        stability_text = self._interpret_stability(barrier_height, primary_minimum, secondary_minimum)
        
        ax2.text(0.05, 0.9, 'Stability Analysis:', fontsize=13, fontweight='bold',
                transform=ax2.transAxes)
        ax2.text(0.05, 0.7, stability_text, fontsize=10, family='monospace',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # System parameters
        param_text = f"""System Parameters:
Particle diameter: {particle_diameter*1e9:.1f} nm
Temperature: {temperature:.1f} K
Ionic strength: {ionic_strength*1000:.2f} mM
Hamaker constant: {hamaker*1e20:.2f}×10⁻²⁰ J
Zeta potential: {zeta_potential*1000:.1f} mV
Debye length: {1/kappa*1e9:.2f} nm"""
        
        ax2.text(0.55, 0.9, param_text, fontsize=9, family='monospace',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        figure.tight_layout()
        
        return figure
    
    def _interpret_stability(self, barrier_height: float, 
                            primary_min: float,
                            secondary_min: float) -> str:
        """Interpret stability from DLVO parameters"""
        
        if barrier_height > 15:
            stability = "HIGHLY STABLE"
            desc = "Barrier >> 15 kT: Strong repulsion prevents aggregation.\nNanofluid remains dispersed for extended periods."
        elif barrier_height > 5:
            stability = "MODERATELY STABLE"
            desc = "Barrier 5-15 kT: Moderate stability.\nSome slow aggregation may occur over time."
        elif barrier_height > 0:
            stability = "WEAKLY STABLE"
            desc = "Barrier 0-5 kT: Weak stability.\nRapid aggregation likely under perturbation."
        else:
            stability = "UNSTABLE"
            desc = "No barrier: Immediate aggregation.\nParticles will rapidly form large clusters."
        
        if secondary_min < -5:
            secondary_desc = "\nDeep secondary minimum: Reversible flocculation possible."
        else:
            secondary_desc = "\nShallow secondary minimum: Minimal flocculation."
        
        return f"Status: {stability}\n\n{desc}{secondary_desc}\n\nBarrier height: {barrier_height:.2f} kT"
    
    def plot_aggregation_kinetics(self,
                                  particle_diameter: float = 30e-9,
                                  volume_fraction: float = 0.02,
                                  temperature: float = 300.0,
                                  stability_ratio: float = 10.0,
                                  time_hours: float = 24.0,
                                  figure: Optional[Figure] = None) -> Figure:
        """
        Plot aggregation kinetics over time
        
        Args:
            particle_diameter: Particle diameter in meters
            volume_fraction: Volume fraction
            temperature: Temperature in K
            stability_ratio: W (dimensionless) - ratio of slow/fast aggregation
            time_hours: Time span in hours
            figure: Optional existing figure
        
        Returns:
            Matplotlib Figure
        """
        if figure is None:
            figure = Figure(figsize=(10, 6), dpi=100)
        
        ax = figure.add_subplot(111)
        
        # Time array
        t_values = np.linspace(0, time_hours * 3600, 200)  # Convert to seconds
        
        # Calculate number concentration
        radius = particle_diameter / 2
        particle_volume = (4/3) * np.pi * (radius ** 3)
        density_particle = 3970  # kg/m³ (Al2O3)
        
        n0 = volume_fraction / particle_volume  # Initial number concentration (1/m³)
        
        # Brownian diffusion coefficient
        mu_fluid = 0.001  # Pa·s (water viscosity)
        D = self.k_B * temperature / (6 * np.pi * mu_fluid * radius)
        
        # Aggregation rate constant (Smoluchowski)
        k_agg = (4 * self.k_B * temperature) / (3 * mu_fluid * stability_ratio)
        
        # Time evolution of particle concentration
        n_t = n0 / (1 + k_agg * n0 * t_values)
        
        # Average cluster size (number of primary particles per cluster)
        cluster_size = n0 / n_t
        
        # Plot
        t_hours = t_values / 3600
        ax.plot(t_hours, cluster_size, 'b-', linewidth=2.5)
        ax.fill_between(t_hours, 1, cluster_size, alpha=0.2, color='blue')
        
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Cluster Size (particles/cluster)', fontsize=12, fontweight='bold')
        ax.set_title('Aggregation Kinetics', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        final_size = cluster_size[-1]
        ax.annotate(f'Final size: {final_size:.1f} particles',
                   xy=(time_hours * 0.7, final_size * 0.9),
                   fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # Stability interpretation
        if stability_ratio > 100:
            stability_text = "Very stable (W >> 100)"
        elif stability_ratio > 10:
            stability_text = "Moderately stable (W = 10-100)"
        else:
            stability_text = "Unstable (W < 10)"
        
        ax.text(0.05, 0.95, f'Stability ratio W = {stability_ratio}\n{stability_text}',
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
               fontsize=10)
        
        figure.tight_layout()
        
        return figure
    
    def plot_brownian_motion_effect(self,
                                    particle_diameters: np.ndarray = None,
                                    temperature: float = 300.0,
                                    figure: Optional[Figure] = None) -> Figure:
        """
        Plot Brownian motion effects vs particle size
        
        Args:
            particle_diameters: Array of diameters in nm
            temperature: Temperature in K
            figure: Optional existing figure
        
        Returns:
            Matplotlib Figure
        """
        if figure is None:
            figure = Figure(figsize=(10, 7), dpi=100)
        
        if particle_diameters is None:
            particle_diameters = np.logspace(0, 2.5, 50)  # 1 to 316 nm
        
        d_values = particle_diameters * 1e-9  # Convert to meters
        
        # Calculate Brownian diffusion coefficient
        mu_fluid = 0.001  # Pa·s
        D_values = self.k_B * temperature / (3 * np.pi * mu_fluid * d_values)
        
        # Calculate RMS displacement in 1 second
        t = 1.0  # second
        rms_displacement = np.sqrt(2 * D_values * t) * 1e6  # Convert to μm
        
        # Calculate Brownian velocity
        v_brownian = np.sqrt(3 * self.k_B * temperature / 
                            (3970 * (4/3) * np.pi * (d_values/2)**3))
        
        # Calculate Peclet number (for Re=100 flow, D_h=1cm)
        velocity_flow = 0.1  # m/s (typical)
        length_scale = 0.01  # m
        Pe_values = velocity_flow * length_scale / D_values
        
        # Create subplots
        ax1 = figure.add_subplot(221)
        ax2 = figure.add_subplot(222)
        ax3 = figure.add_subplot(223)
        ax4 = figure.add_subplot(224)
        
        # Plot 1: Diffusion coefficient
        ax1.loglog(particle_diameters, D_values * 1e12, 'b-', linewidth=2)
        ax1.set_xlabel('Particle Diameter (nm)', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Diffusion Coeff. (μm²/s)', fontsize=10, fontweight='bold')
        ax1.set_title('Brownian Diffusion', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RMS displacement
        ax2.loglog(particle_diameters, rms_displacement, 'r-', linewidth=2)
        ax2.set_xlabel('Particle Diameter (nm)', fontsize=10, fontweight='bold')
        ax2.set_ylabel('RMS Displacement (μm)', fontsize=10, fontweight='bold')
        ax2.set_title('Displacement in 1 Second', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Brownian velocity
        ax3.loglog(particle_diameters, v_brownian * 1e3, 'g-', linewidth=2)
        ax3.set_xlabel('Particle Diameter (nm)', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Brownian Velocity (mm/s)', fontsize=10, fontweight='bold')
        ax3.set_title('Characteristic Velocity', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Peclet number
        ax4.loglog(particle_diameters, Pe_values, 'm-', linewidth=2)
        ax4.axhline(y=1, color='k', linestyle='--', linewidth=1, label='Pe = 1')
        ax4.fill_between(particle_diameters, 0.1, 1, alpha=0.2, color='green',
                        label='Diffusion dominant')
        ax4.fill_between(particle_diameters, 1, 1e6, alpha=0.2, color='red',
                        label='Convection dominant')
        ax4.set_xlabel('Particle Diameter (nm)', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Peclet Number', fontsize=10, fontweight='bold')
        ax4.set_title('Transport Mechanism', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8, loc='best')
        ax4.grid(True, alpha=0.3)
        
        figure.suptitle(f'Brownian Motion Effects at T = {temperature} K',
                       fontsize=13, fontweight='bold')
        figure.tight_layout()
        
        return figure
    
    def plot_interfacial_layer_effect(self,
                                      particle_diameter: float = 30e-9,
                                      layer_thickness_range: Tuple[float, float] = (0.1e-9, 5e-9),
                                      volume_fraction: float = 0.02,
                                      figure: Optional[Figure] = None) -> Figure:
        """
        Plot interfacial thermal layer effect on conductivity
        
        Args:
            particle_diameter: Core particle diameter in meters
            layer_thickness_range: (min, max) layer thickness in meters
            volume_fraction: Volume fraction
            figure: Optional existing figure
        
        Returns:
            Matplotlib Figure
        """
        if figure is None:
            figure = Figure(figsize=(10, 6), dpi=100)
        
        ax = figure.add_subplot(111)
        
        # Layer thickness array
        t_values = np.linspace(layer_thickness_range[0], layer_thickness_range[1], 50)
        
        # Thermal conductivities
        k_bf = 0.613  # Water
        k_np = 40.0  # Al2O3
        k_layer_low = k_bf * 2  # Lower bound (organized layer)
        k_layer_high = k_bf * 5  # Upper bound (highly conducting layer)
        
        radius_core = particle_diameter / 2
        
        enhancement_low = np.zeros_like(t_values)
        enhancement_high = np.zeros_like(t_values)
        enhancement_no_layer = np.zeros_like(t_values)
        
        for i, t_layer in enumerate(t_values):
            radius_total = radius_core + t_layer
            
            # Volume fractions
            phi_core = volume_fraction * (radius_core / radius_total) ** 3
            phi_layer = volume_fraction - phi_core
            
            # Effective particle conductivity (series model)
            k_eff_particle_low = k_np * k_layer_low / \
                (k_layer_low + (k_np - k_layer_low) * (radius_core / radius_total) ** 3)
            k_eff_particle_high = k_np * k_layer_high / \
                (k_layer_high + (k_np - k_layer_high) * (radius_core / radius_total) ** 3)
            
            # Maxwell model for nanofluid
            def maxwell_model(k_p, phi_eff):
                return k_bf * (k_p + 2*k_bf + 2*phi_eff*(k_p - k_bf)) / \
                       (k_p + 2*k_bf - phi_eff*(k_p - k_bf))
            
            k_nf_low = maxwell_model(k_eff_particle_low, volume_fraction)
            k_nf_high = maxwell_model(k_eff_particle_high, volume_fraction)
            k_nf_no_layer = maxwell_model(k_np, volume_fraction)
            
            enhancement_low[i] = ((k_nf_low - k_bf) / k_bf) * 100
            enhancement_high[i] = ((k_nf_high - k_bf) / k_bf) * 100
            enhancement_no_layer[i] = ((k_nf_no_layer - k_bf) / k_bf) * 100
        
        t_nm = t_values * 1e9
        
        # Plot
        ax.fill_between(t_nm, enhancement_low, enhancement_high,
                       alpha=0.3, color='blue', label='Layer effect range')
        ax.plot(t_nm, enhancement_low, 'b--', linewidth=2,
               label=f'Lower bound (k_layer = {k_layer_low:.2f} W/m·K)')
        ax.plot(t_nm, enhancement_high, 'b-', linewidth=2,
               label=f'Upper bound (k_layer = {k_layer_high:.2f} W/m·K)')
        ax.axhline(y=enhancement_no_layer[0], color='r', linestyle=':',
                  linewidth=2, label='No interfacial layer')
        
        ax.set_xlabel('Interfacial Layer Thickness (nm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Thermal Conductivity Enhancement (%)', fontsize=12, fontweight='bold')
        ax.set_title('Interfacial Layer Effect on Enhancement', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        figure.tight_layout()
        
        return figure


def create_comprehensive_interaction_report(particle_diameter: float = 30e-9,
                                           temperature: float = 300.0,
                                           volume_fraction: float = 0.02,
                                           ionic_strength: float = 0.001) -> Dict[str, Figure]:
    """
    Create comprehensive particle interaction visualization report
    
    Returns:
        Dict mapping report name to Figure
    """
    visualizer = ParticleInteractionVisualizer()
    
    figures = {}
    
    # DLVO potential
    figures['dlvo_potential'] = visualizer.plot_dlvo_potential(
        particle_diameter=particle_diameter,
        temperature=temperature,
        ionic_strength=ionic_strength
    )
    
    # Aggregation kinetics
    figures['aggregation'] = visualizer.plot_aggregation_kinetics(
        particle_diameter=particle_diameter,
        volume_fraction=volume_fraction,
        temperature=temperature,
        stability_ratio=15.0  # Moderately stable
    )
    
    # Brownian motion
    figures['brownian'] = visualizer.plot_brownian_motion_effect(
        temperature=temperature
    )
    
    # Interfacial layer
    figures['interfacial_layer'] = visualizer.plot_interfacial_layer_effect(
        particle_diameter=particle_diameter,
        volume_fraction=volume_fraction
    )
    
    return figures
