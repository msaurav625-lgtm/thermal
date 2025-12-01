"""
Enhanced Nanofluid Simulator with Advanced Features

This module provides an enhanced simulator with support for:
- Temperature-dependent calculations
- Hybrid nanofluids
- Multiple model comparisons
- Comprehensive thermophysical properties
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np

from .models import (
    maxwell_model,
    hamilton_crosser_model,
    bruggeman_model,
    yu_choi_model,
    wasp_model,
)
from .advanced_models import (
    patel_model,
    koo_kleinstreuer_model,
    hajjar_hybrid_model,
    esfe_hybrid_model,
    sundar_hybrid_model,
    takabi_salehi_model,
    xue_interfacial_layer_model,
    leong_yang_interfacial_model,
    yu_choi_interfacial_model,
)
from .thermophysical_properties import (
    ThermophysicalProperties,
    batchelor_viscosity,
    nanofluid_density,
    nanofluid_specific_heat,
)
from .nanoparticles import NanoparticleDatabase, NanoparticleMaterial, BaseFluid


@dataclass
class NanofluidComponent:
    """
    Represents a single nanoparticle component in a (hybrid) nanofluid.
    
    Attributes:
        material: NanoparticleMaterial object
        volume_fraction: Volume fraction of this component (0 to 1)
        particle_size: Average particle size in nm
        sphericity: Particle sphericity (0 to 1)
    """
    material: NanoparticleMaterial
    volume_fraction: float
    particle_size: float = 25.0
    sphericity: float = 1.0


@dataclass
class EnhancedSimulationResult:
    """
    Comprehensive simulation result with all properties.
    
    Attributes:
        model_name: Name of the model used
        temperature: Temperature in K
        k_effective: Effective thermal conductivity (W/m·K)
        mu_effective: Effective viscosity (Pa·s)
        rho_effective: Effective density (kg/m³)
        cp_effective: Effective specific heat (J/kg·K)
        alpha_effective: Thermal diffusivity (m²/s)
        pr_effective: Prandtl number
        enhancement_k: Thermal conductivity enhancement (%)
        enhancement_mu: Viscosity increase (%)
    """
    model_name: str
    temperature: float
    k_effective: float
    mu_effective: Optional[float] = None
    rho_effective: Optional[float] = None
    cp_effective: Optional[float] = None
    alpha_effective: Optional[float] = None
    pr_effective: Optional[float] = None
    enhancement_k: Optional[float] = None
    enhancement_mu: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model": self.model_name,
            "temperature_K": self.temperature,
            "temperature_C": self.temperature - 273.15,
            "thermal_conductivity": self.k_effective,
            "viscosity": self.mu_effective,
            "density": self.rho_effective,
            "specific_heat": self.cp_effective,
            "thermal_diffusivity": self.alpha_effective,
            "prandtl_number": self.pr_effective,
            "enhancement_k_percent": self.enhancement_k,
            "enhancement_mu_percent": self.enhancement_mu,
        }


class EnhancedNanofluidSimulator:
    """
    Advanced simulator for mono and hybrid nanofluids.
    
    Features:
    - Temperature-dependent calculations
    - Hybrid nanofluid support (multiple nanoparticles)
    - All thermophysical properties
    - Advanced models with Brownian motion
    - Parametric studies
    """
    
    def __init__(self):
        """Initialize the enhanced simulator."""
        self._base_fluid: Optional[BaseFluid] = None
        self._base_fluid_name: Optional[str] = None
        self._nanoparticle_components: List[NanofluidComponent] = []
        self._temperature: float = 298.15  # K
        self._layer_thickness: float = 1.0  # nm
        
    def set_base_fluid(
        self,
        name: str,
        temperature: Optional[float] = None
    ) -> "EnhancedNanofluidSimulator":
        """
        Set the base fluid from database.
        
        Args:
            name: Name of base fluid from database
            temperature: Temperature in K (optional)
            
        Returns:
            Self for method chaining
        """
        self._base_fluid = NanoparticleDatabase.get_base_fluid(name)
        self._base_fluid_name = name
        
        if temperature is not None:
            self._temperature = temperature
            
        return self
    
    def add_nanoparticle(
        self,
        formula: str,
        volume_fraction: float,
        particle_size: float = 25.0,
        sphericity: float = 1.0
    ) -> "EnhancedNanofluidSimulator":
        """
        Add a nanoparticle component (for hybrid nanofluids).
        
        Args:
            formula: Chemical formula from database
            volume_fraction: Volume fraction of this component
            particle_size: Average particle size in nm
            sphericity: Particle sphericity (0 to 1)
            
        Returns:
            Self for method chaining
        """
        material = NanoparticleDatabase.get_nanoparticle(formula)
        
        component = NanofluidComponent(
            material=material,
            volume_fraction=volume_fraction,
            particle_size=particle_size,
            sphericity=sphericity
        )
        
        self._nanoparticle_components.append(component)
        
        return self
    
    def clear_nanoparticles(self) -> "EnhancedNanofluidSimulator":
        """Clear all nanoparticle components."""
        self._nanoparticle_components = []
        return self
    
    def set_temperature(self, temperature: float) -> "EnhancedNanofluidSimulator":
        """
        Set simulation temperature.
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Self for method chaining
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self._temperature = temperature
        return self
    
    def set_temperature_celsius(self, temperature_c: float) -> "EnhancedNanofluidSimulator":
        """
        Set simulation temperature in Celsius.
        
        Args:
            temperature_c: Temperature in Celsius
            
        Returns:
            Self for method chaining
        """
        return self.set_temperature(temperature_c + 273.15)
    
    def _validate_configuration(self):
        """Validate simulator configuration."""
        if self._base_fluid is None:
            raise ValueError("Base fluid not set")
        if not self._nanoparticle_components:
            raise ValueError("No nanoparticle components added")
        
        total_phi = sum(c.volume_fraction for c in self._nanoparticle_components)
        if total_phi > 1:
            raise ValueError(f"Total volume fraction {total_phi} exceeds 1")
    
    def _get_total_volume_fraction(self) -> float:
        """Get total volume fraction of all nanoparticles."""
        return sum(c.volume_fraction for c in self._nanoparticle_components)
    
    def _get_effective_nanoparticle_properties(self) -> Tuple[float, float, float, float]:
        """
        Calculate effective properties for hybrid nanofluid.
        
        Returns:
            Tuple of (k_eff, rho_eff, cp_eff, d_eff)
        """
        phi_total = self._get_total_volume_fraction()
        
        if phi_total == 0:
            return 0, 0, 0, 0
        
        k_eff = sum(c.material.thermal_conductivity * c.volume_fraction 
                   for c in self._nanoparticle_components) / phi_total
        
        rho_eff = sum(c.material.density * c.volume_fraction 
                     for c in self._nanoparticle_components) / phi_total
        
        cp_eff = sum(c.material.specific_heat * c.volume_fraction 
                    for c in self._nanoparticle_components) / phi_total
        
        d_eff = sum(c.particle_size * c.volume_fraction 
                   for c in self._nanoparticle_components) / phi_total
        
        return k_eff, rho_eff, cp_eff, d_eff
    
    def is_hybrid(self) -> bool:
        """Check if this is a hybrid nanofluid (multiple components)."""
        return len(self._nanoparticle_components) > 1
    
    def calculate_maxwell(self) -> EnhancedSimulationResult:
        """Calculate using Maxwell model."""
        self._validate_configuration()
        
        k_np, rho_np, cp_np, _ = self._get_effective_nanoparticle_properties()
        phi_total = self._get_total_volume_fraction()
        
        k_eff = maxwell_model(
            self._base_fluid.thermal_conductivity,
            k_np,
            phi_total
        )
        
        return self._create_result("Maxwell", k_eff, rho_np, cp_np, phi_total)
    
    def calculate_patel(self) -> EnhancedSimulationResult:
        """Calculate using temperature-dependent Patel model."""
        self._validate_configuration()
        
        k_np, rho_np, cp_np, _ = self._get_effective_nanoparticle_properties()
        phi_total = self._get_total_volume_fraction()
        
        k_eff = patel_model(
            self._base_fluid.thermal_conductivity,
            k_np,
            phi_total,
            self._temperature
        )
        
        return self._create_result("Patel (Temperature-Dependent)", k_eff, rho_np, cp_np, phi_total)
    
    def calculate_koo_kleinstreuer(self) -> EnhancedSimulationResult:
        """Calculate using Koo-Kleinstreuer model with Brownian motion."""
        self._validate_configuration()
        
        k_np, rho_np, cp_np, d_eff = self._get_effective_nanoparticle_properties()
        phi_total = self._get_total_volume_fraction()
        
        k_eff = koo_kleinstreuer_model(
            self._base_fluid.thermal_conductivity,
            k_np,
            phi_total,
            self._temperature,
            d_eff,
            self._base_fluid.density,
            rho_np,
            self._base_fluid.specific_heat,
            self._base_fluid.viscosity
        )
        
        return self._create_result("Koo-Kleinstreuer (Brownian)", k_eff, rho_np, cp_np, phi_total)
    
    def calculate_hajjar_hybrid(self) -> EnhancedSimulationResult:
        """Calculate using Hajjar hybrid model (requires exactly 2 components)."""
        self._validate_configuration()
        
        if len(self._nanoparticle_components) != 2:
            raise ValueError("Hajjar model requires exactly 2 nanoparticle components")
        
        c1, c2 = self._nanoparticle_components[0], self._nanoparticle_components[1]
        
        k_eff = hajjar_hybrid_model(
            self._base_fluid.thermal_conductivity,
            c1.material.thermal_conductivity,
            c2.material.thermal_conductivity,
            c1.volume_fraction,
            c2.volume_fraction,
            c1.sphericity
        )
        
        _, rho_np, cp_np, _ = self._get_effective_nanoparticle_properties()
        phi_total = self._get_total_volume_fraction()
        
        return self._create_result("Hajjar Hybrid", k_eff, rho_np, cp_np, phi_total)
    
    def calculate_sundar_hybrid(self) -> EnhancedSimulationResult:
        """Calculate using Sundar hybrid model (requires exactly 2 components)."""
        self._validate_configuration()
        
        if len(self._nanoparticle_components) != 2:
            raise ValueError("Sundar model requires exactly 2 nanoparticle components")
        
        c1, c2 = self._nanoparticle_components[0], self._nanoparticle_components[1]
        
        k_eff = sundar_hybrid_model(
            self._base_fluid.thermal_conductivity,
            c1.material.thermal_conductivity,
            c2.material.thermal_conductivity,
            c1.volume_fraction,
            c2.volume_fraction,
            self._temperature,
            c1.particle_size,
            c2.particle_size
        )
        
        _, rho_np, cp_np, _ = self._get_effective_nanoparticle_properties()
        phi_total = self._get_total_volume_fraction()
        
        return self._create_result("Sundar Hybrid", k_eff, rho_np, cp_np, phi_total)
    
    def calculate_xue_interfacial(self, beta: float = 0.1) -> EnhancedSimulationResult:
        """
        Calculate using Xue interfacial layer model.
        
        Args:
            beta: Interfacial layer thickness ratio (h/r), typically 0.05-0.2
        """
        self._validate_configuration()
        
        if len(self._nanoparticle_components) != 1:
            raise ValueError("Xue interfacial model works with mono nanofluids only")
        
        c = self._nanoparticle_components[0]
        
        k_eff = xue_interfacial_layer_model(
            self._base_fluid.thermal_conductivity,
            c.material.thermal_conductivity,
            c.volume_fraction,
            beta=beta
        )
        
        return self._create_result(
            f"Xue Interfacial (β={beta:.2f})", 
            k_eff, 
            c.material.density, 
            c.material.specific_heat,
            c.volume_fraction
        )
    
    def calculate_leong_yang_interfacial(
        self, 
        h: float = 2.0, 
        k_layer_ratio: float = 2.0
    ) -> EnhancedSimulationResult:
        """
        Calculate using Leong-Yang interfacial nanolayer model.
        
        Args:
            h: Nanolayer thickness (nm), typically 0.5-5.0
            k_layer_ratio: Ratio k_layer/k_bf, typically 1.5-5.0 for ordered layers
        """
        self._validate_configuration()
        
        if len(self._nanoparticle_components) != 1:
            raise ValueError("Leong-Yang model works with mono nanofluids only")
        
        c = self._nanoparticle_components[0]
        
        k_eff = leong_yang_interfacial_model(
            self._base_fluid.thermal_conductivity,
            c.material.thermal_conductivity,
            c.volume_fraction,
            c.particle_size,
            h=h,
            k_layer_ratio=k_layer_ratio
        )
        
        return self._create_result(
            f"Leong-Yang (h={h}nm, k_r={k_layer_ratio})", 
            k_eff, 
            c.material.density, 
            c.material.specific_heat,
            c.volume_fraction
        )
    
    def calculate_yu_choi_interfacial(self, beta: float = 0.1) -> EnhancedSimulationResult:
        """
        Calculate using Yu-Choi interfacial layer model.
        
        Args:
            beta: Interfacial layer thickness ratio
        """
        self._validate_configuration()
        
        if len(self._nanoparticle_components) != 1:
            raise ValueError("Yu-Choi interfacial model works with mono nanofluids only")
        
        c = self._nanoparticle_components[0]
        
        k_eff = yu_choi_interfacial_model(
            self._base_fluid.thermal_conductivity,
            c.material.thermal_conductivity,
            c.volume_fraction,
            beta=beta
        )
        
        return self._create_result(
            f"Yu-Choi Interfacial (β={beta:.2f})", 
            k_eff, 
            c.material.density, 
            c.material.specific_heat,
            c.volume_fraction
        )
    
    def calculate_all_applicable_models(self) -> List[EnhancedSimulationResult]:
        """
        Calculate using all applicable models.
        
        Returns:
            List of results from all applicable models
        """
        results = [
            self.calculate_maxwell(),
            self.calculate_patel(),
            self.calculate_koo_kleinstreuer(),
        ]
        
        # Add interfacial layer models for mono nanofluids
        if len(self._nanoparticle_components) == 1:
            try:
                results.append(self.calculate_xue_interfacial())
            except Exception:
                pass
            
            try:
                results.append(self.calculate_leong_yang_interfacial())
            except Exception:
                pass
            
            try:
                results.append(self.calculate_yu_choi_interfacial())
            except Exception:
                pass
        
        # Add hybrid models if applicable
        if len(self._nanoparticle_components) == 2:
            try:
                results.append(self.calculate_hajjar_hybrid())
            except Exception:
                pass
            
            try:
                results.append(self.calculate_sundar_hybrid())
            except Exception:
                pass
        
        return results
    
    def _create_result(
        self,
        model_name: str,
        k_eff: float,
        rho_np: float,
        cp_np: float,
        phi_total: float
    ) -> EnhancedSimulationResult:
        """Create comprehensive result with all properties."""
        
        # Calculate density and specific heat
        rho_eff = nanofluid_density(
            self._base_fluid.density,
            rho_np,
            phi_total
        )
        
        cp_eff = nanofluid_specific_heat(
            self._base_fluid.density,
            rho_np,
            self._base_fluid.specific_heat,
            cp_np,
            phi_total
        )
        
        # Calculate viscosity
        if phi_total < 0.1:
            mu_eff = batchelor_viscosity(self._base_fluid.viscosity, phi_total)
        else:
            from .thermophysical_properties import brinkman_viscosity
            mu_eff = brinkman_viscosity(self._base_fluid.viscosity, phi_total)
        
        # Calculate thermal diffusivity
        alpha_eff = k_eff / (rho_eff * cp_eff)
        
        # Calculate Prandtl number
        pr_eff = mu_eff * cp_eff / k_eff
        
        # Calculate enhancements
        enhancement_k = ((k_eff / self._base_fluid.thermal_conductivity) - 1) * 100
        enhancement_mu = ((mu_eff / self._base_fluid.viscosity) - 1) * 100
        
        return EnhancedSimulationResult(
            model_name=model_name,
            temperature=self._temperature,
            k_effective=k_eff,
            mu_effective=mu_eff,
            rho_effective=rho_eff,
            cp_effective=cp_eff,
            alpha_effective=alpha_eff,
            pr_effective=pr_eff,
            enhancement_k=enhancement_k,
            enhancement_mu=enhancement_mu
        )
    
    def parametric_study_temperature(
        self,
        temperature_range: Tuple[float, float],
        n_points: int = 20,
        models: Optional[List[str]] = None
    ) -> Dict[str, List[EnhancedSimulationResult]]:
        """
        Perform parametric study varying temperature.
        
        Args:
            temperature_range: (T_min, T_max) in Kelvin
            n_points: Number of temperature points
            models: List of model names (default: all applicable)
            
        Returns:
            Dictionary mapping model names to lists of results
        """
        original_temp = self._temperature
        
        temperatures = np.linspace(temperature_range[0], temperature_range[1], n_points)
        
        results_dict = {}
        
        for T in temperatures:
            self.set_temperature(T)
            
            if models is None:
                results = self.calculate_all_applicable_models()
            else:
                results = []
                for model in models:
                    if model.lower() == "maxwell":
                        results.append(self.calculate_maxwell())
                    elif model.lower() == "patel":
                        results.append(self.calculate_patel())
                    elif model.lower() == "koo_kleinstreuer":
                        results.append(self.calculate_koo_kleinstreuer())
            
            for result in results:
                if result.model_name not in results_dict:
                    results_dict[result.model_name] = []
                results_dict[result.model_name].append(result)
        
        # Restore original temperature
        self._temperature = original_temp
        
        return results_dict
    
    def parametric_study_concentration(
        self,
        phi_range: Tuple[float, float],
        n_points: int = 20,
        models: Optional[List[str]] = None
    ) -> Dict[str, List[EnhancedSimulationResult]]:
        """
        Perform parametric study varying concentration.
        
        For hybrid nanofluids, the ratio between components is maintained.
        
        Args:
            phi_range: (phi_min, phi_max) total volume fraction range
            n_points: Number of concentration points
            models: List of model names
            
        Returns:
            Dictionary mapping model names to lists of results
        """
        if not self._nanoparticle_components:
            raise ValueError("No nanoparticle components added")
        
        # Save original volume fractions
        original_fractions = [c.volume_fraction for c in self._nanoparticle_components]
        original_total = sum(original_fractions)
        
        if original_total == 0:
            raise ValueError("Total volume fraction is zero")
        
        # Calculate ratios
        ratios = [f / original_total for f in original_fractions]
        
        phi_values = np.linspace(phi_range[0], phi_range[1], n_points)
        
        results_dict = {}
        
        for phi_total in phi_values:
            # Set volume fractions maintaining ratios
            for i, component in enumerate(self._nanoparticle_components):
                component.volume_fraction = phi_total * ratios[i]
            
            results = self.calculate_all_applicable_models() if models is None else []
            
            for result in results:
                if result.model_name not in results_dict:
                    results_dict[result.model_name] = []
                results_dict[result.model_name].append(result)
        
        # Restore original fractions
        for i, component in enumerate(self._nanoparticle_components):
            component.volume_fraction = original_fractions[i]
        
        return results_dict
    
    def get_configuration_summary(self) -> str:
        """Get formatted summary of current configuration."""
        lines = [
            "=" * 70,
            "NANOFLUID CONFIGURATION",
            "=" * 70,
        ]
        
        if self._base_fluid:
            lines.append(f"Base Fluid: {self._base_fluid.name}")
            lines.append(f"  k = {self._base_fluid.thermal_conductivity:.4f} W/m·K")
            lines.append(f"  ρ = {self._base_fluid.density:.1f} kg/m³")
            lines.append(f"  cp = {self._base_fluid.specific_heat:.1f} J/kg·K")
            lines.append(f"  μ = {self._base_fluid.viscosity:.6f} Pa·s")
        
        lines.append(f"\nTemperature: {self._temperature:.2f} K ({self._temperature-273.15:.2f} °C)")
        
        if self._nanoparticle_components:
            lines.append(f"\nNanoparticle Components ({len(self._nanoparticle_components)}):")
            for i, comp in enumerate(self._nanoparticle_components, 1):
                lines.append(f"  {i}. {comp.material.name} ({comp.material.formula})")
                lines.append(f"     Volume Fraction: {comp.volume_fraction*100:.2f}%")
                lines.append(f"     Particle Size: {comp.particle_size:.1f} nm")
                lines.append(f"     k = {comp.material.thermal_conductivity:.1f} W/m·K")
            
            lines.append(f"\nTotal Volume Fraction: {self._get_total_volume_fraction()*100:.2f}%")
            lines.append(f"Nanofluid Type: {'Hybrid' if self.is_hybrid() else 'Mono'}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def export_to_json(self, results: Optional[List[EnhancedSimulationResult]] = None) -> str:
        """Export configuration and results to JSON."""
        config = {
            "base_fluid": {
                "name": self._base_fluid.name if self._base_fluid else None,
                "thermal_conductivity": self._base_fluid.thermal_conductivity if self._base_fluid else None,
                "density": self._base_fluid.density if self._base_fluid else None,
                "specific_heat": self._base_fluid.specific_heat if self._base_fluid else None,
                "viscosity": self._base_fluid.viscosity if self._base_fluid else None,
            },
            "temperature_K": self._temperature,
            "temperature_C": self._temperature - 273.15,
            "nanoparticles": [
                {
                    "name": c.material.name,
                    "formula": c.material.formula,
                    "volume_fraction": c.volume_fraction,
                    "volume_fraction_percent": c.volume_fraction * 100,
                    "particle_size_nm": c.particle_size,
                    "thermal_conductivity": c.material.thermal_conductivity,
                }
                for c in self._nanoparticle_components
            ],
            "is_hybrid": self.is_hybrid(),
            "total_volume_fraction": self._get_total_volume_fraction(),
        }
        
        data = {"configuration": config}
        
        if results:
            data["results"] = [r.to_dict() for r in results]
        
        return json.dumps(data, indent=2)
