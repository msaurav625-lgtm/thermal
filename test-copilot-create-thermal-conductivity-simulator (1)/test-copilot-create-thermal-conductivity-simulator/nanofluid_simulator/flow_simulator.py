"""
Comprehensive Flow-Integrated Nanofluid Simulator

This is the world-class simulator integrating:
- Static + Flow-dependent thermal conductivity
- Complete viscosity models (temperature + shear-rate dependent)
- Particle aggregation and DLVO stability
- Flow regime analysis and heat transfer performance
- Pumping power and thermal-hydraulic optimization

This simulator goes beyond traditional tools by modeling realistic
flow physics that affects nanofluid behavior.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .enhanced_simulator import EnhancedNanofluidSimulator, EnhancedSimulationResult
from .nanoparticles import NanoparticleDatabase
from .flow_models import (
    buongiorno_convective_model,
    corcione_model,
    rea_bonnet_convective_model,
    shear_enhanced_conductivity,
    velocity_dependent_conductivity
)
from .viscosity_models import (
    BaseFluidViscosity,
    NanofluidViscosityCalculator,
    carreau_model,
    aggregated_nanofluid_viscosity
)
from .aggregation_models import (
    ParticleInteractionAnalyzer,
    assess_colloidal_stability,
    interfacial_layer_thickness,
    interfacial_layer_conductivity
)
from .flow_regime_analysis import FlowRegimeAnalyzer


@dataclass
class FlowSimulationResult:
    """
    Complete simulation result including flow effects.
    
    Attributes:
        model_name: Name of thermal conductivity model used
        temperature: Temperature (K)
        
        # Thermal properties
        k_effective: Effective thermal conductivity (W/m·K)
        k_static: Static component of k_eff (W/m·K)
        k_flow_contribution: Flow enhancement to k_eff (W/m·K)
        
        # Viscosity properties
        mu_effective: Effective dynamic viscosity (Pa·s)
        mu_base_fluid: Base fluid viscosity at T (Pa·s)
        viscosity_behavior: "Newtonian", "Shear-thinning", or "Shear-thickening"
        
        # Other thermophysical
        rho_effective: Effective density (kg/m³)
        cp_effective: Effective specific heat (J/kg·K)
        alpha_effective: Thermal diffusivity (m²/s)
        
        # Flow regime
        Reynolds: Reynolds number
        Prandtl: Prandtl number
        Nusselt: Nusselt number
        flow_regime: "Laminar", "Transitional", or "Turbulent"
        
        # Heat transfer
        h_convective: Heat transfer coefficient (W/m²·K)
        
        # Pressure & power
        pressure_drop_Pa: Pressure drop (Pa)
        pumping_power_W: Required pumping power (W)
        friction_factor: Darcy friction factor
        
        # Aggregation
        aggregation_state: "Stable", "Moderate", or "Severe"
        energy_barrier_kT: DLVO energy barrier
        collision_efficiency: Probability of aggregation upon collision
        
        # Performance
        enhancement_k: Thermal conductivity enhancement (%)
        enhancement_h: Heat transfer enhancement vs base fluid (%)
        pumping_penalty: Pumping power increase vs base fluid (%)
        performance_index: Thermal-hydraulic performance index
    """
    model_name: str
    temperature: float
    
    # Thermal
    k_effective: float
    k_static: float
    k_flow_contribution: float
    
    # Viscosity
    mu_effective: float
    mu_base_fluid: float
    viscosity_behavior: str
    
    # Thermophysical
    rho_effective: float
    cp_effective: float
    alpha_effective: float
    
    # Flow
    Reynolds: float
    Prandtl: float
    Nusselt: float
    flow_regime: str
    
    # Heat transfer
    h_convective: float
    
    # Pressure & power
    pressure_drop_Pa: float
    pumping_power_W: float
    friction_factor: float
    
    # Aggregation
    aggregation_state: str
    energy_barrier_kT: float
    collision_efficiency: float
    
    # Performance
    enhancement_k: float
    enhancement_h: float
    pumping_penalty: float
    performance_index: float


class FlowNanofluidSimulator(EnhancedNanofluidSimulator):
    """
    World-class nanofluid simulator with complete flow physics.
    
    This simulator pioneers next-generation modeling by integrating:
    - Static + dynamic thermal conductivity
    - Temperature + shear-rate dependent viscosity
    - Particle aggregation and stability analysis
    - Flow regime effects on all properties
    - Realistic thermal-hydraulic performance prediction
    
    Use Cases:
    - Heat exchanger design and optimization
    - Cooling system performance prediction
    - Energy efficiency analysis
    - Research and development of novel nanofluids
    """
    
    def __init__(self):
        super().__init__()
        
        # Flow parameters
        self._velocity: float = 0.0  # m/s
        self._shear_rate: Optional[float] = None  # 1/s
        self._hydraulic_diameter: float = 0.01  # m (10 mm default)
        self._channel_length: float = 1.0  # m
        self._roughness: float = 0.0  # m (smooth pipe)
        
        # Aggregation parameters
        self._aggregation_state: str = "stable"  # "stable", "moderate", "severe"
        self._zeta_potential: float = 30.0  # mV
        self._ionic_strength: float = 0.001  # mol/L
        
        # Performance parameters
        self._pump_efficiency: float = 0.75
    
    # ========================================================================
    # SETTERS FOR FLOW PARAMETERS
    # ========================================================================
    
    def set_flow_velocity(self, velocity: float):
        """
        Set flow velocity (m/s).
        
        Typical values:
        - Laminar flow: 0.01 - 1 m/s
        - Turbulent flow: 1 - 10 m/s
        - High-speed cooling: > 10 m/s
        """
        if velocity < 0:
            raise ValueError("Velocity must be non-negative")
        self._velocity = velocity
    
    def set_shear_rate(self, shear_rate: float):
        """
        Set shear rate (1/s).
        
        If not set, will be estimated from velocity.
        Typical values: 10 - 10000 s⁻¹
        """
        if shear_rate < 0:
            raise ValueError("Shear rate must be non-negative")
        self._shear_rate = shear_rate
    
    def set_channel_geometry(self, diameter: float, length: float):
        """
        Set channel dimensions.
        
        Args:
            diameter: Hydraulic diameter (m)
            length: Channel length (m)
        """
        if diameter <= 0 or length <= 0:
            raise ValueError("Dimensions must be positive")
        self._hydraulic_diameter = diameter
        self._channel_length = length
    
    def set_surface_roughness(self, roughness: float):
        """Set surface roughness (m). 0 = smooth pipe."""
        self._roughness = max(0, roughness)
    
    def set_aggregation_state(self, state: str):
        """
        Set nanoparticle aggregation state.
        
        Args:
            state: "stable", "moderate", or "severe"
        """
        valid_states = ["stable", "moderate", "severe"]
        if state not in valid_states:
            raise ValueError(f"State must be one of {valid_states}")
        self._aggregation_state = state
    
    def set_colloidal_parameters(self, zeta_potential: float, ionic_strength: float):
        """
        Set parameters for DLVO stability analysis.
        
        Args:
            zeta_potential: Zeta potential (mV)
            ionic_strength: Ionic strength (mol/L)
        """
        self._zeta_potential = zeta_potential
        self._ionic_strength = ionic_strength
    
    # ========================================================================
    # FLOW-DEPENDENT THERMAL CONDUCTIVITY CALCULATIONS
    # ========================================================================
    
    def calculate_buongiorno_flow(self) -> FlowSimulationResult:
        """Calculate using Buongiorno two-phase model with flow effects."""
        self._validate_configuration()
        
        if len(self._nanoparticle_components) != 1:
            raise ValueError("Buongiorno model requires single nanoparticle type")
        
        c = self._nanoparticle_components[0]
        d_p = c.particle_size
        
        # Get accurate base fluid viscosity at temperature
        mu_bf = BaseFluidViscosity.get_base_fluid_viscosity(
            self._base_fluid_name, self._temperature
        )
        
        k_eff, diagnostics = buongiorno_convective_model(
            self._base_fluid.thermal_conductivity,
            c.material.thermal_conductivity,
            c.volume_fraction,
            self._temperature,
            d_p,
            self._base_fluid.density,
            c.material.density,
            mu_bf,
            velocity=self._velocity,
            shear_rate=self._shear_rate
        )
        
        return self._create_flow_result(
            "Buongiorno Flow",
            k_eff,
            diagnostics.get('k_static', k_eff),
            diagnostics.get('k_Brownian', 0),
            c.material.density,
            c.material.specific_heat,
            c.volume_fraction,
            mu_bf,
            diagnostics
        )
    
    def calculate_corcione_flow(self) -> FlowSimulationResult:
        """Calculate using Corcione empirical model."""
        self._validate_configuration()
        
        if len(self._nanoparticle_components) != 1:
            raise ValueError("Corcione model requires single nanoparticle type")
        
        c = self._nanoparticle_components[0]
        
        mu_bf = BaseFluidViscosity.get_base_fluid_viscosity(
            self._base_fluid_name, self._temperature
        )
        
        k_eff, diagnostics = corcione_model(
            self._base_fluid.thermal_conductivity,
            c.material.thermal_conductivity,
            c.volume_fraction,
            self._temperature,
            c.particle_size,
            self._base_fluid.density,
            c.material.density,
            mu_bf
        )
        
        # Add flow enhancement if velocity > 0
        if self._velocity > 0:
            k_static = k_eff
            k_flow_enh, _ = shear_enhanced_conductivity(
                k_eff,
                c.volume_fraction,
                self._shear_rate if self._shear_rate else 10 * self._velocity,
                c.particle_size,
                self._temperature,
                mu_bf
            )
            k_flow_contribution = k_flow_enh - k_static
            k_eff = k_flow_enh
        else:
            k_static = k_eff
            k_flow_contribution = 0
        
        return self._create_flow_result(
            "Corcione Flow",
            k_eff,
            k_static,
            k_flow_contribution,
            c.material.density,
            c.material.specific_heat,
            c.volume_fraction,
            mu_bf,
            diagnostics
        )
    
    def calculate_rea_bonnet_convective(self) -> FlowSimulationResult:
        """Calculate using Rea-Bonnet convective heat transfer model."""
        self._validate_configuration()
        
        if self._velocity == 0:
            raise ValueError("Rea-Bonnet model requires flow velocity > 0")
        
        if len(self._nanoparticle_components) != 1:
            raise ValueError("Rea-Bonnet model requires single nanoparticle type")
        
        c = self._nanoparticle_components[0]
        
        mu_bf = BaseFluidViscosity.get_base_fluid_viscosity(
            self._base_fluid_name, self._temperature
        )
        
        k_eff, diagnostics = rea_bonnet_convective_model(
            self._base_fluid.thermal_conductivity,
            c.material.thermal_conductivity,
            c.volume_fraction,
            self._temperature,
            self._velocity,
            self._hydraulic_diameter,
            self._channel_length,
            self._base_fluid.density,
            mu_bf,
            self._base_fluid.specific_heat
        )
        
        return self._create_flow_result(
            "Rea-Bonnet Convective",
            k_eff,
            diagnostics.get('k_static', k_eff * 0.9),
            k_eff - diagnostics.get('k_static', k_eff * 0.9),
            c.material.density,
            c.material.specific_heat,
            c.volume_fraction,
            mu_bf,
            diagnostics
        )
    
    def calculate_all_flow_models(self) -> List[FlowSimulationResult]:
        """Calculate using all applicable flow models."""
        results = []
        
        # Only single-component models for now
        if len(self._nanoparticle_components) == 1:
            try:
                results.append(self.calculate_buongiorno_flow())
            except Exception as e:
                print(f"Buongiorno failed: {e}")
            
            try:
                results.append(self.calculate_corcione_flow())
            except Exception as e:
                print(f"Corcione failed: {e}")
            
            if self._velocity > 0:
                try:
                    results.append(self.calculate_rea_bonnet_convective())
                except Exception as e:
                    print(f"Rea-Bonnet failed: {e}")
        
        return results
    
    # ========================================================================
    # HELPER METHOD TO CREATE COMPREHENSIVE RESULTS
    # ========================================================================
    
    def _create_flow_result(
        self,
        model_name: str,
        k_eff: float,
        k_static: float,
        k_flow_contrib: float,
        rho_np: float,
        cp_np: float,
        phi_total: float,
        mu_bf_at_T: float,
        diagnostics: dict
    ) -> FlowSimulationResult:
        """Create comprehensive flow simulation result."""
        
        # Calculate effective properties
        from .thermophysical_properties import nanofluid_density, nanofluid_specific_heat
        
        rho_eff = nanofluid_density(self._base_fluid.density, rho_np, phi_total)
        cp_eff = nanofluid_specific_heat(
            self._base_fluid.density, rho_np,
            self._base_fluid.specific_heat, cp_np,
            phi_total
        )
        
        # Calculate viscosity with all effects
        visc_results = NanofluidViscosityCalculator.calculate_complete(
            self._base_fluid_name,
            self._temperature,
            phi_total,
            d_p=self._nanoparticle_components[0].particle_size if self._nanoparticle_components else 50.0,
            shear_rate=self._shear_rate if self._shear_rate else (10 * self._velocity if self._velocity > 0 else 0),
            aggregation_state=self._aggregation_state
        )
        
        # Use Krieger-Dougherty as primary model
        mu_eff = visc_results.get('Krieger-Dougherty', mu_bf_at_T * (1 + 2.5 * phi_total))
        visc_behavior = visc_results.get('_behavior', 'Newtonian')
        
        # Thermal diffusivity
        alpha_eff = k_eff / (rho_eff * cp_eff)
        
        # Flow regime analysis
        if self._velocity > 0:
            flow_analysis = FlowRegimeAnalyzer.analyze_complete(
                rho_eff, mu_eff, k_eff, cp_eff,
                self._velocity, self._hydraulic_diameter, self._channel_length,
                self._roughness, self._pump_efficiency
            )
            
            Re = flow_analysis['Reynolds']
            Pr = flow_analysis['Prandtl']
            Nu = flow_analysis['Nusselt']
            flow_regime = flow_analysis['flow_regime']
            h_conv = flow_analysis['h_conv']
            dP = flow_analysis['pressure_drop_Pa']
            P_pump = flow_analysis['pumping_power_W']
            f = flow_analysis['friction_factor']
        else:
            Re = 0
            Pr = mu_eff * cp_eff / k_eff
            Nu = 0
            flow_regime = "Static"
            h_conv = 0
            dP = 0
            P_pump = 0
            f = 0
        
        # Aggregation analysis
        if len(self._nanoparticle_components) == 1:
            c = self._nanoparticle_components[0]
            stability = assess_colloidal_stability(
                c.particle_size,
                self._zeta_potential,
                self._ionic_strength,
                self._temperature
            )
            energy_barrier = stability['energy_barrier_kT']
            agg_state = stability['stability_status']
            
            # Collision efficiency
            from .aggregation_models import particle_collision_efficiency
            coll_eff = particle_collision_efficiency(
                c.particle_size, self._velocity, self._temperature,
                mu_eff, energy_barrier
            )
        else:
            energy_barrier = 10.0
            agg_state = "Stable"
            coll_eff = 0.0
        
        # Calculate enhancements vs base fluid
        enhancement_k = ((k_eff / self._base_fluid.thermal_conductivity) - 1) * 100
        
        # Heat transfer enhancement (if flow)
        if self._velocity > 0:
            # Base fluid analysis
            bf_analysis = FlowRegimeAnalyzer.analyze_complete(
                self._base_fluid.density, mu_bf_at_T,
                self._base_fluid.thermal_conductivity, self._base_fluid.specific_heat,
                self._velocity, self._hydraulic_diameter, self._channel_length,
                self._roughness, self._pump_efficiency
            )
            h_bf = bf_analysis['h_conv']
            P_bf = bf_analysis['pumping_power_W']
            
            enhancement_h = ((h_conv / h_bf) - 1) * 100 if h_bf > 0 else 0
            pumping_penalty = ((P_pump / P_bf) - 1) * 100 if P_bf > 0 else 0
            
            # Performance index
            from .flow_regime_analysis import performance_index
            perf_index = performance_index(h_conv, h_bf, P_pump, P_bf)
        else:
            enhancement_h = 0
            pumping_penalty = 0
            perf_index = 0
        
        return FlowSimulationResult(
            model_name=model_name,
            temperature=self._temperature,
            k_effective=k_eff,
            k_static=k_static,
            k_flow_contribution=k_flow_contrib,
            mu_effective=mu_eff,
            mu_base_fluid=mu_bf_at_T,
            viscosity_behavior=visc_behavior,
            rho_effective=rho_eff,
            cp_effective=cp_eff,
            alpha_effective=alpha_eff,
            Reynolds=Re,
            Prandtl=Pr,
            Nusselt=Nu,
            flow_regime=flow_regime,
            h_convective=h_conv,
            pressure_drop_Pa=dP,
            pumping_power_W=P_pump,
            friction_factor=f,
            aggregation_state=agg_state,
            energy_barrier_kT=energy_barrier,
            collision_efficiency=coll_eff,
            enhancement_k=enhancement_k,
            enhancement_h=enhancement_h,
            pumping_penalty=pumping_penalty,
            performance_index=perf_index
        )
