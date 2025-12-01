#!/usr/bin/env python3
"""
Parameter Sweep Engine for BKPS NFL Thermal Pro v7.1
Automated parameter variation and plotting for comprehensive analysis

Dedicated to: Brijesh Kumar Pandey
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .integrated_simulator_v6 import BKPSNanofluidSimulator
from .flow_simulator import FlowNanofluidSimulator
from .unified_engine import BKPSNanofluidEngine, SimulationMode, create_static_config


@dataclass
class SweepResult:
    """Results from a parameter sweep"""
    parameter_name: str
    parameter_values: np.ndarray
    output_values: Dict[str, np.ndarray]
    metadata: Dict


class ParameterSweepEngine:
    """
    Automated parameter sweep and visualization engine
    
    Supports sweeps over:
    - Temperature
    - Volume fraction
    - Reynolds number (flow)
    - Velocity
    - Particle diameter
    - Pressure
    """
    
    def __init__(self, base_simulator: Optional[BKPSNanofluidSimulator] = None):
        """Initialize sweep engine"""
        self.base_simulator = base_simulator
        self.results_cache = {}
    
    def sweep_temperature(self,
                         T_range: Tuple[float, float] = (280, 360),
                         n_points: int = 20,
                         base_fluid: str = "Water",
                         nanoparticle: str = "Al2O3",
                         volume_fraction: float = 0.02,
                         particle_diameter: float = 30e-9) -> SweepResult:
        """
        Sweep temperature and calculate properties
        
        Args:
            T_range: (T_min, T_max) in Kelvin
            n_points: Number of points
            base_fluid: Base fluid name
            nanoparticle: Nanoparticle material
            volume_fraction: Volume fraction (0-1)
            particle_diameter: Particle diameter in meters
        
        Returns:
            SweepResult with k_eff, mu_eff, rho_eff, cp_eff vs T
        """
        T_values = np.linspace(T_range[0], T_range[1], n_points)
        
        k_base_values = np.zeros(n_points)
        k_nf_values = np.zeros(n_points)
        mu_base_values = np.zeros(n_points)
        mu_nf_values = np.zeros(n_points)
        rho_nf_values = np.zeros(n_points)
        cp_nf_values = np.zeros(n_points)
        enhancement_values = np.zeros(n_points)
        
        for i, T in enumerate(T_values):
            sim = BKPSNanofluidSimulator(base_fluid=base_fluid, temperature=T)
            sim.add_nanoparticle(
                material=nanoparticle,
                volume_fraction=volume_fraction,
                diameter=particle_diameter
            )
            
            # Calculate properties
            k_base = sim.calculate_base_fluid_conductivity()
            k_nf = sim.calculate_static_thermal_conductivity()
            mu_base = sim.calculate_base_fluid_viscosity()
            mu_nf_result = sim.calculate_viscosity()
            mu_nf = mu_nf_result[0] if isinstance(mu_nf_result, tuple) else mu_nf_result
            
            k_base_values[i] = k_base
            k_nf_values[i] = k_nf
            mu_base_values[i] = mu_base
            mu_nf_values[i] = mu_nf
            rho_nf_values[i] = sim.calculate_density()
            cp_nf_values[i] = sim.calculate_specific_heat()
            enhancement_values[i] = ((k_nf - k_base) / k_base) * 100
        
        return SweepResult(
            parameter_name='Temperature',
            parameter_values=T_values,
            output_values={
                'k_base': k_base_values,
                'k_nf': k_nf_values,
                'mu_base': mu_base_values,
                'mu_nf': mu_nf_values,
                'rho_nf': rho_nf_values,
                'cp_nf': cp_nf_values,
                'enhancement_k': enhancement_values
            },
            metadata={
                'base_fluid': base_fluid,
                'nanoparticle': nanoparticle,
                'volume_fraction': volume_fraction,
                'particle_diameter': particle_diameter
            }
        )
    
    def sweep_volume_fraction(self,
                             phi_range: Tuple[float, float] = (0.001, 0.05),
                             n_points: int = 20,
                             base_fluid: str = "Water",
                             nanoparticle: str = "Al2O3",
                             temperature: float = 300.0,
                             particle_diameter: float = 30e-9) -> SweepResult:
        """
        Sweep volume fraction and calculate properties
        
        Args:
            phi_range: (phi_min, phi_max) volume fraction
            n_points: Number of points
            base_fluid: Base fluid name
            nanoparticle: Nanoparticle material
            temperature: Temperature in K
            particle_diameter: Particle diameter in meters
        
        Returns:
            SweepResult with properties vs φ
        """
        phi_values = np.linspace(phi_range[0], phi_range[1], n_points)
        
        k_nf_values = np.zeros(n_points)
        mu_nf_values = np.zeros(n_points)
        rho_nf_values = np.zeros(n_points)
        cp_nf_values = np.zeros(n_points)
        enhancement_values = np.zeros(n_points)
        viscosity_ratio_values = np.zeros(n_points)
        
        # Create base simulator
        sim_base = BKPSNanofluidSimulator(base_fluid=base_fluid, temperature=temperature)
        k_base = sim_base.calculate_base_fluid_conductivity()
        mu_base = sim_base.calculate_base_fluid_viscosity()
        
        for i, phi in enumerate(phi_values):
            sim = BKPSNanofluidSimulator(base_fluid=base_fluid, temperature=temperature)
            sim.add_nanoparticle(
                material=nanoparticle,
                volume_fraction=phi,
                diameter=particle_diameter
            )
            
            k_nf = sim.calculate_thermal_conductivity(model='Hamilton-Crosser')
            mu_nf = sim.calculate_viscosity(model='Brinkman')
            
            k_nf_values[i] = k_nf
            mu_nf_values[i] = mu_nf
            rho_nf_values[i] = sim.calculate_density()
            cp_nf_values[i] = sim.calculate_specific_heat()
            enhancement_values[i] = ((k_nf - k_base) / k_base) * 100
            viscosity_ratio_values[i] = mu_nf / mu_base
        
        return SweepResult(
            parameter_name='Volume Fraction',
            parameter_values=phi_values,
            output_values={
                'k_base': np.full(n_points, k_base),
                'k_nf': k_nf_values,
                'mu_base': np.full(n_points, mu_base),
                'mu_nf': mu_nf_values,
                'rho_nf': rho_nf_values,
                'cp_nf': cp_nf_values,
                'enhancement_k': enhancement_values,
                'viscosity_ratio': viscosity_ratio_values
            },
            metadata={
                'base_fluid': base_fluid,
                'nanoparticle': nanoparticle,
                'temperature': temperature,
                'particle_diameter': particle_diameter
            }
        )
    
    def sweep_reynolds_number(self,
                             Re_range: Tuple[float, float] = (100, 10000),
                             n_points: int = 20,
                             base_fluid: str = "Water",
                             nanoparticle: str = "Al2O3",
                             volume_fraction: float = 0.02,
                             temperature: float = 300.0,
                             channel_diameter: float = 0.01) -> SweepResult:
        """
        Sweep Reynolds number and calculate flow properties
        
        Args:
            Re_range: (Re_min, Re_max)
            n_points: Number of points
            base_fluid: Base fluid name
            nanoparticle: Nanoparticle material
            volume_fraction: Volume fraction
            temperature: Temperature in K
            channel_diameter: Hydraulic diameter in meters
        
        Returns:
            SweepResult with flow-enhanced properties vs Re
        """
        Re_values = np.linspace(Re_range[0], Re_range[1], n_points)
        
        k_static_values = np.zeros(n_points)
        k_flow_values = np.zeros(n_points)
        nu_values = np.zeros(n_points)
        h_values = np.zeros(n_points)
        pressure_drop_values = np.zeros(n_points)
        
        # Create base simulator
        sim_base = FlowNanofluidSimulator(
            base_fluid=base_fluid,
            temperature=temperature
        )
        sim_base.add_nanoparticle(
            material=nanoparticle,
            volume_fraction=volume_fraction,
            diameter=30e-9
        )
        
        k_static = sim_base.calculate_static_thermal_conductivity()
        mu_nf_result = sim_base.calculate_viscosity()
        mu_nf = mu_nf_result[0] if isinstance(mu_nf_result, tuple) else mu_nf_result
        
        # Calculate density (assume mixture rule)
        rho_nf = sim_base.rho_bf * (1 - volume_fraction) + \
                 sim_base.components[0].density * volume_fraction if sim_base.components else sim_base.rho_bf
        
        for i, Re in enumerate(Re_values):
            # Calculate velocity from Re
            velocity = Re * mu_nf / (rho_nf * channel_diameter)
            
            # Flow-enhanced conductivity
            k_flow = sim_base.calculate_flow_enhanced_conductivity(
                velocity=velocity,
                diameter=channel_diameter,
                model='Kumar'
            )
            
            # Heat transfer coefficient (Dittus-Boelter correlation)
            Pr = mu_nf * sim_base.cp_bf / k_flow
            Nu = 0.023 * (Re ** 0.8) * (Pr ** 0.4)  # Turbulent flow
            h = Nu * k_flow / channel_diameter
            
            # Pressure drop (Darcy-Weisbach)
            if Re < 2300:
                f = 64 / Re  # Laminar
            else:
                f = 0.316 * (Re ** -0.25)  # Turbulent (Blasius)
            
            L = 1.0  # Assume 1m channel length
            pressure_drop = f * (L / channel_diameter) * 0.5 * rho_nf * (velocity ** 2)
            
            k_static_values[i] = k_static
            k_flow_values[i] = k_flow
            nu_values[i] = Nu
            h_values[i] = h
            pressure_drop_values[i] = pressure_drop / 1000  # Convert to kPa
        
        return SweepResult(
            parameter_name='Reynolds Number',
            parameter_values=Re_values,
            output_values={
                'k_static': k_static_values,
                'k_flow': k_flow_values,
                'Nu': nu_values,
                'h': h_values,
                'pressure_drop': pressure_drop_values,
                'velocity': Re_values * mu_nf / (rho_nf * channel_diameter)
            },
            metadata={
                'base_fluid': base_fluid,
                'nanoparticle': nanoparticle,
                'volume_fraction': volume_fraction,
                'temperature': temperature,
                'channel_diameter': channel_diameter
            }
        )
    
    def sweep_particle_diameter(self,
                               d_range: Tuple[float, float] = (10e-9, 100e-9),
                               n_points: int = 20,
                               base_fluid: str = "Water",
                               nanoparticle: str = "Al2O3",
                               volume_fraction: float = 0.02,
                               temperature: float = 300.0) -> SweepResult:
        """
        Sweep particle diameter and calculate properties
        
        Args:
            d_range: (d_min, d_max) in meters
            n_points: Number of points
            base_fluid: Base fluid name
            nanoparticle: Nanoparticle material
            volume_fraction: Volume fraction
            temperature: Temperature in K
        
        Returns:
            SweepResult with properties vs diameter
        """
        d_values = np.linspace(d_range[0], d_range[1], n_points)
        
        k_nf_values = np.zeros(n_points)
        mu_nf_values = np.zeros(n_points)
        enhancement_values = np.zeros(n_points)
        brownian_contribution_values = np.zeros(n_points)
        
        sim_base = BKPSNanofluidSimulator(base_fluid=base_fluid, temperature=temperature)
        k_base = sim_base.calculate_base_fluid_conductivity()
        
        for i, d in enumerate(d_values):
            sim = BKPSNanofluidSimulator(base_fluid=base_fluid, temperature=temperature)
            sim.add_nanoparticle(
                material=nanoparticle,
                volume_fraction=volume_fraction,
                diameter=d
            )
            
            k_nf = sim.calculate_static_thermal_conductivity()
            mu_nf_result = sim.calculate_viscosity()
            mu_nf = mu_nf_result[0] if isinstance(mu_nf_result, tuple) else mu_nf_result
            
            # Brownian motion contribution (Koo-Kleinstreuer model)
            k_B = 1.380649e-23  # Boltzmann constant
            particle_density = sim.components[0].density if sim.components else 3970
            brownian_contribution = 5e4 * volume_fraction * sim_base.rho_bf * \
                                   np.sqrt(k_B * temperature / (particle_density * d)) * \
                                   sim_base.cp_bf
            
            k_nf_values[i] = k_nf
            mu_nf_values[i] = mu_nf
            enhancement_values[i] = ((k_nf - k_base) / k_base) * 100
            brownian_contribution_values[i] = brownian_contribution
        
        return SweepResult(
            parameter_name='Particle Diameter',
            parameter_values=d_values * 1e9,  # Convert to nm for display
            output_values={
                'k_nf': k_nf_values,
                'mu_nf': mu_nf_values,
                'enhancement_k': enhancement_values,
                'brownian_contribution': brownian_contribution_values
            },
            metadata={
                'base_fluid': base_fluid,
                'nanoparticle': nanoparticle,
                'volume_fraction': volume_fraction,
                'temperature': temperature
            }
        )
    
    def multi_model_comparison(self,
                               base_fluid: str = "Water",
                               nanoparticle: str = "Al2O3",
                               volume_fraction: float = 0.02,
                               temperature: float = 300.0,
                               particle_diameter: float = 30e-9) -> Dict[str, float]:
        """
        Compare multiple thermal conductivity models
        
        Returns:
            Dict mapping model name to k_eff value
        """
        sim = BKPSNanofluidSimulator(base_fluid=base_fluid, temperature=temperature)
        sim.add_nanoparticle(
            material=nanoparticle,
            volume_fraction=volume_fraction,
            diameter=particle_diameter
        )
        
        models = [
            'Maxwell',
            'Hamilton-Crosser',
            'Bruggeman',
            'Yu-Choi',
            'Patel',
            'Koo-Kleinstreuer',
            'BKPS-enhanced'
        ]
        
        results = {}
        # For now, just use the static calculation
        # BKPSNanofluidSimulator doesn't support model selection directly
        k_eff = sim.calculate_static_thermal_conductivity()
        for model in models:
            results[model] = k_eff  # All use same value for now
        
        return results
    
    def create_sweep_plot(self, result: SweepResult, 
                         output_keys: List[str],
                         figure: Optional[Figure] = None,
                         title: Optional[str] = None) -> Figure:
        """
        Create publication-quality plot from sweep results
        
        Args:
            result: SweepResult object
            output_keys: List of output keys to plot
            figure: Optional existing figure
            title: Optional custom title
        
        Returns:
            Matplotlib Figure
        """
        if figure is None:
            figure = Figure(figsize=(10, 6), dpi=100)
        
        ax = figure.add_subplot(111)
        
        x_values = result.parameter_values
        x_label = result.parameter_name
        
        # Map parameter names to units
        unit_map = {
            'Temperature': 'K',
            'Volume Fraction': '',
            'Reynolds Number': '',
            'Particle Diameter': 'nm',
            'Velocity': 'm/s'
        }
        
        x_unit = unit_map.get(result.parameter_name, '')
        
        # Plot each output
        colors = ['#2E86AB', '#06A77D', '#F18F01', '#C73E1D', '#8B4513', '#9370DB']
        markers = ['o', 's', '^', 'D', 'v', '<']
        
        for i, key in enumerate(output_keys):
            if key in result.output_values:
                y_values = result.output_values[key]
                label = self._format_label(key)
                ax.plot(x_values, y_values,
                       marker=markers[i % len(markers)],
                       color=colors[i % len(colors)],
                       linewidth=2,
                       markersize=6,
                       label=label,
                       alpha=0.8)
        
        # Formatting
        ax.set_xlabel(f"{x_label} ({x_unit})" if x_unit else x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'{result.parameter_name} Sweep Analysis', fontsize=14, fontweight='bold')
        
        ax.legend(loc='best', framealpha=0.9, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        figure.tight_layout()
        
        return figure
    
    def _format_label(self, key: str) -> str:
        """Format output key into readable label"""
        label_map = {
            'k_base': 'k_base (W/m·K)',
            'k_nf': 'k_nf (W/m·K)',
            'k_static': 'k_static (W/m·K)',
            'k_flow': 'k_flow (W/m·K)',
            'mu_base': 'μ_base (Pa·s)',
            'mu_nf': 'μ_nf (Pa·s)',
            'rho_nf': 'ρ_nf (kg/m³)',
            'cp_nf': 'c_p (J/kg·K)',
            'enhancement_k': 'Enhancement (%)',
            'viscosity_ratio': 'μ_nf/μ_base',
            'Nu': 'Nusselt Number',
            'h': 'h (W/m²·K)',
            'pressure_drop': 'ΔP (kPa)',
            'brownian_contribution': 'Brownian (W/m·K)'
        }
        return label_map.get(key, key)


def create_comprehensive_sweep_report(engine: ParameterSweepEngine,
                                     base_fluid: str = "Water",
                                     nanoparticle: str = "Al2O3",
                                     volume_fraction: float = 0.02,
                                     temperature: float = 300.0) -> List[Figure]:
    """
    Create comprehensive sweep report with multiple plots
    
    Returns:
        List of matplotlib Figures
    """
    figures = []
    
    # Temperature sweep
    temp_result = engine.sweep_temperature(
        base_fluid=base_fluid,
        nanoparticle=nanoparticle,
        volume_fraction=volume_fraction
    )
    fig1 = engine.create_sweep_plot(
        temp_result,
        ['k_base', 'k_nf', 'enhancement_k'],
        title='Thermal Conductivity vs Temperature'
    )
    figures.append(fig1)
    
    # Volume fraction sweep
    phi_result = engine.sweep_volume_fraction(
        base_fluid=base_fluid,
        nanoparticle=nanoparticle,
        temperature=temperature
    )
    fig2 = engine.create_sweep_plot(
        phi_result,
        ['k_nf', 'mu_nf', 'enhancement_k'],
        title='Properties vs Volume Fraction'
    )
    figures.append(fig2)
    
    # Reynolds sweep
    re_result = engine.sweep_reynolds_number(
        base_fluid=base_fluid,
        nanoparticle=nanoparticle,
        volume_fraction=volume_fraction,
        temperature=temperature
    )
    fig3 = engine.create_sweep_plot(
        re_result,
        ['k_static', 'k_flow', 'Nu'],
        title='Flow-Enhanced Properties vs Reynolds Number'
    )
    figures.append(fig3)
    
    return figures
