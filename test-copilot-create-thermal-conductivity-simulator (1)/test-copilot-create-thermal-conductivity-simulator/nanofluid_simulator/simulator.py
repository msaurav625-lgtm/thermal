"""
Nanofluid Thermal Conductivity Simulator

This module provides the main simulator class that combines various models
and material databases to calculate and compare thermal conductivity of nanofluids.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import json

from .models import (
    maxwell_model,
    hamilton_crosser_model,
    bruggeman_model,
    yu_choi_model,
    wasp_model,
    pak_cho_correlation,
)
from .nanoparticles import NanoparticleDatabase, NanoparticleMaterial, BaseFluid


@dataclass
class SimulationResult:
    """
    Results from a single thermal conductivity calculation.
    
    Attributes:
        model_name: Name of the model used
        k_effective: Effective thermal conductivity (W/m·K)
        k_base_fluid: Base fluid thermal conductivity (W/m·K)
        k_nanoparticle: Nanoparticle thermal conductivity (W/m·K)
        volume_fraction: Volume fraction of nanoparticles
        enhancement_ratio: k_effective / k_base_fluid
        enhancement_percent: Percentage enhancement in thermal conductivity
    """
    model_name: str
    k_effective: float
    k_base_fluid: float
    k_nanoparticle: float
    volume_fraction: float
    enhancement_ratio: float
    enhancement_percent: float
    
    def __str__(self) -> str:
        return (
            f"{self.model_name}:\n"
            f"  Effective thermal conductivity: {self.k_effective:.4f} W/m·K\n"
            f"  Enhancement: {self.enhancement_percent:.2f}%"
        )
    
    def to_dict(self) -> dict:
        """Convert result to dictionary format."""
        return {
            "model": self.model_name,
            "k_effective": self.k_effective,
            "k_base_fluid": self.k_base_fluid,
            "k_nanoparticle": self.k_nanoparticle,
            "volume_fraction": self.volume_fraction,
            "enhancement_ratio": self.enhancement_ratio,
            "enhancement_percent": self.enhancement_percent,
        }


class NanofluidSimulator:
    """
    Simulator for calculating thermal conductivity of nanofluids.
    
    This class provides methods to calculate the effective thermal conductivity
    of nanofluids using various theoretical models. It can use materials from
    the built-in database or custom material properties.
    
    Example:
        >>> sim = NanofluidSimulator()
        >>> sim.set_base_fluid("water")
        >>> sim.set_nanoparticle("Cu")
        >>> sim.set_volume_fraction(0.01)
        >>> results = sim.calculate_all_models()
        >>> for result in results:
        ...     print(result)
    """
    
    AVAILABLE_MODELS = [
        "maxwell",
        "hamilton_crosser",
        "bruggeman",
        "yu_choi",
        "wasp",
        "pak_cho"
    ]
    
    def __init__(self):
        """Initialize the simulator with default values."""
        self._k_base_fluid: Optional[float] = None
        self._k_nanoparticle: Optional[float] = None
        self._volume_fraction: float = 0.0
        self._nanoparticle_type: Optional[str] = None
        self._base_fluid_name: Optional[str] = None
        self._particle_radius: float = 25.0  # nm
        self._nanolayer_thickness: float = 1.0  # nm
        self._sphericity: float = 1.0  # 1 for spherical particles
        self._temperature: float = 298.15  # K
        
        self._nanoparticle_material: Optional[NanoparticleMaterial] = None
        self._base_fluid: Optional[BaseFluid] = None
    
    def set_base_fluid(
        self,
        name_or_k: Union[str, float],
        temperature: Optional[float] = None
    ) -> "NanofluidSimulator":
        """
        Set the base fluid for the simulation.
        
        Args:
            name_or_k: Either the name of a base fluid from the database
                      (e.g., "water", "ethylene_glycol") or the thermal
                      conductivity value in W/m·K
            temperature: Optional temperature in K (for future temperature-dependent calculations)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> sim = NanofluidSimulator()
            >>> sim.set_base_fluid("water")  # Use database value
            >>> sim.set_base_fluid(0.6)  # Use custom value
        """
        if isinstance(name_or_k, str):
            self._base_fluid = NanoparticleDatabase.get_base_fluid(name_or_k)
            self._k_base_fluid = self._base_fluid.thermal_conductivity
            self._base_fluid_name = name_or_k
        else:
            if name_or_k <= 0:
                raise ValueError("Thermal conductivity must be positive")
            self._k_base_fluid = float(name_or_k)
            self._base_fluid_name = "custom"
            self._base_fluid = None
        
        if temperature is not None:
            self._temperature = temperature
            
        return self
    
    def set_nanoparticle(
        self,
        formula_or_k: Union[str, float],
        particle_radius: Optional[float] = None,
        sphericity: float = 1.0
    ) -> "NanofluidSimulator":
        """
        Set the nanoparticle material for the simulation.
        
        Args:
            formula_or_k: Either the chemical formula of a nanoparticle from
                         the database (e.g., "Cu", "Al2O3") or the thermal
                         conductivity value in W/m·K
            particle_radius: Particle radius in nm (default: 25 nm)
            sphericity: Particle sphericity for non-spherical particles
                       (1 for spheres, <1 for other shapes)
            
        Returns:
            Self for method chaining
        """
        if isinstance(formula_or_k, str):
            self._nanoparticle_material = NanoparticleDatabase.get_nanoparticle(formula_or_k)
            self._k_nanoparticle = self._nanoparticle_material.thermal_conductivity
            self._nanoparticle_type = formula_or_k
        else:
            if formula_or_k <= 0:
                raise ValueError("Thermal conductivity must be positive")
            self._k_nanoparticle = float(formula_or_k)
            self._nanoparticle_type = "custom"
            self._nanoparticle_material = None
        
        if particle_radius is not None:
            if particle_radius <= 0:
                raise ValueError("Particle radius must be positive")
            self._particle_radius = particle_radius
            
        if not 0 < sphericity <= 1:
            raise ValueError("Sphericity must be between 0 (exclusive) and 1 (inclusive)")
        self._sphericity = sphericity
        
        return self
    
    def set_volume_fraction(self, phi: float) -> "NanofluidSimulator":
        """
        Set the volume fraction of nanoparticles.
        
        Args:
            phi: Volume fraction (0 to 1, e.g., 0.01 for 1%)
            
        Returns:
            Self for method chaining
        """
        if not 0 <= phi <= 1:
            raise ValueError("Volume fraction must be between 0 and 1")
        self._volume_fraction = phi
        return self
    
    def set_volume_fraction_percent(self, phi_percent: float) -> "NanofluidSimulator":
        """
        Set the volume fraction of nanoparticles as a percentage.
        
        Args:
            phi_percent: Volume fraction in percent (0 to 100, e.g., 1 for 1%)
            
        Returns:
            Self for method chaining
        """
        if not 0 <= phi_percent <= 100:
            raise ValueError("Volume fraction percent must be between 0 and 100")
        self._volume_fraction = phi_percent / 100.0
        return self
    
    def set_nanolayer_thickness(self, thickness: float) -> "NanofluidSimulator":
        """
        Set the nanolayer thickness for Yu-Choi model.
        
        Args:
            thickness: Nanolayer thickness in nm
            
        Returns:
            Self for method chaining
        """
        if thickness < 0:
            raise ValueError("Nanolayer thickness must be non-negative")
        self._nanolayer_thickness = thickness
        return self
    
    def _validate_configuration(self) -> None:
        """Validate that the simulator is properly configured."""
        if self._k_base_fluid is None:
            raise ValueError(
                "Base fluid not set. Use set_base_fluid() to configure."
            )
        if self._k_nanoparticle is None:
            raise ValueError(
                "Nanoparticle not set. Use set_nanoparticle() to configure."
            )
    
    def _create_result(self, model_name: str, k_effective: float) -> SimulationResult:
        """Create a SimulationResult from calculated thermal conductivity."""
        enhancement_ratio = k_effective / self._k_base_fluid
        enhancement_percent = (enhancement_ratio - 1) * 100
        
        return SimulationResult(
            model_name=model_name,
            k_effective=k_effective,
            k_base_fluid=self._k_base_fluid,
            k_nanoparticle=self._k_nanoparticle,
            volume_fraction=self._volume_fraction,
            enhancement_ratio=enhancement_ratio,
            enhancement_percent=enhancement_percent,
        )
    
    def calculate_maxwell(self) -> SimulationResult:
        """
        Calculate thermal conductivity using Maxwell model.
        
        Returns:
            SimulationResult with Maxwell model prediction
        """
        self._validate_configuration()
        k_eff = maxwell_model(
            self._k_base_fluid,
            self._k_nanoparticle,
            self._volume_fraction
        )
        return self._create_result("Maxwell", k_eff)
    
    def calculate_hamilton_crosser(self) -> SimulationResult:
        """
        Calculate thermal conductivity using Hamilton-Crosser model.
        
        Returns:
            SimulationResult with Hamilton-Crosser model prediction
        """
        self._validate_configuration()
        k_eff = hamilton_crosser_model(
            self._k_base_fluid,
            self._k_nanoparticle,
            self._volume_fraction,
            sphericity=self._sphericity
        )
        return self._create_result("Hamilton-Crosser", k_eff)
    
    def calculate_bruggeman(self) -> SimulationResult:
        """
        Calculate thermal conductivity using Bruggeman model.
        
        Returns:
            SimulationResult with Bruggeman model prediction
        """
        self._validate_configuration()
        k_eff = bruggeman_model(
            self._k_base_fluid,
            self._k_nanoparticle,
            self._volume_fraction
        )
        return self._create_result("Bruggeman", k_eff)
    
    def calculate_yu_choi(self) -> SimulationResult:
        """
        Calculate thermal conductivity using Yu-Choi model.
        
        Returns:
            SimulationResult with Yu-Choi model prediction
        """
        self._validate_configuration()
        k_eff = yu_choi_model(
            self._k_base_fluid,
            self._k_nanoparticle,
            self._volume_fraction,
            layer_thickness=self._nanolayer_thickness,
            particle_radius=self._particle_radius
        )
        return self._create_result("Yu-Choi", k_eff)
    
    def calculate_wasp(self) -> SimulationResult:
        """
        Calculate thermal conductivity using Wasp model.
        
        Returns:
            SimulationResult with Wasp model prediction
        """
        self._validate_configuration()
        k_eff = wasp_model(
            self._k_base_fluid,
            self._k_nanoparticle,
            self._volume_fraction
        )
        return self._create_result("Wasp", k_eff)
    
    def calculate_pak_cho(self) -> SimulationResult:
        """
        Calculate thermal conductivity using Pak-Cho correlation.
        
        Note: This model requires a supported nanoparticle type.
        
        Returns:
            SimulationResult with Pak-Cho correlation prediction
        """
        self._validate_configuration()
        
        if self._nanoparticle_type is None or self._nanoparticle_type == "custom":
            raise ValueError(
                "Pak-Cho correlation requires a named nanoparticle type. "
                "Use set_nanoparticle() with a formula from the database."
            )
        
        k_eff = pak_cho_correlation(
            self._k_base_fluid,
            self._volume_fraction,
            self._nanoparticle_type
        )
        return self._create_result("Pak-Cho", k_eff)
    
    def calculate_all_models(
        self,
        include_pak_cho: bool = True
    ) -> List[SimulationResult]:
        """
        Calculate thermal conductivity using all available models.
        
        Args:
            include_pak_cho: Whether to include Pak-Cho correlation
                           (requires named nanoparticle type)
            
        Returns:
            List of SimulationResults from all models
        """
        results = [
            self.calculate_maxwell(),
            self.calculate_hamilton_crosser(),
            self.calculate_bruggeman(),
            self.calculate_yu_choi(),
            self.calculate_wasp(),
        ]
        
        if include_pak_cho and self._nanoparticle_type not in [None, "custom"]:
            try:
                results.append(self.calculate_pak_cho())
            except ValueError:
                pass  # Skip Pak-Cho if nanoparticle type not supported
        
        return results
    
    def parametric_study(
        self,
        volume_fractions: List[float],
        models: Optional[List[str]] = None
    ) -> Dict[str, List[SimulationResult]]:
        """
        Perform a parametric study varying volume fraction.
        
        Args:
            volume_fractions: List of volume fractions to simulate
            models: List of model names to use (default: all models)
            
        Returns:
            Dictionary mapping model names to lists of results
        """
        if models is None:
            models = ["maxwell", "hamilton_crosser", "bruggeman", "wasp"]
        
        results: Dict[str, List[SimulationResult]] = {model: [] for model in models}
        
        original_phi = self._volume_fraction
        
        for phi in volume_fractions:
            self.set_volume_fraction(phi)
            
            for model in models:
                if model == "maxwell":
                    results[model].append(self.calculate_maxwell())
                elif model == "hamilton_crosser":
                    results[model].append(self.calculate_hamilton_crosser())
                elif model == "bruggeman":
                    results[model].append(self.calculate_bruggeman())
                elif model == "yu_choi":
                    results[model].append(self.calculate_yu_choi())
                elif model == "wasp":
                    results[model].append(self.calculate_wasp())
                elif model == "pak_cho":
                    try:
                        results[model].append(self.calculate_pak_cho())
                    except ValueError:
                        pass
        
        # Restore original volume fraction
        self._volume_fraction = original_phi
        
        return results
    
    def compare_models(self) -> str:
        """
        Generate a comparison table of all models.
        
        Returns:
            Formatted string table comparing model predictions
        """
        results = self.calculate_all_models()
        
        lines = [
            "=" * 70,
            "NANOFLUID THERMAL CONDUCTIVITY COMPARISON",
            "=" * 70,
            f"Base Fluid: {self._base_fluid_name or 'Custom'} "
            f"(k = {self._k_base_fluid:.4f} W/m·K)",
            f"Nanoparticle: {self._nanoparticle_type or 'Custom'} "
            f"(k = {self._k_nanoparticle:.4f} W/m·K)",
            f"Volume Fraction: {self._volume_fraction * 100:.2f}%",
            f"Particle Radius: {self._particle_radius:.1f} nm",
            f"Sphericity: {self._sphericity:.2f}",
            "-" * 70,
            f"{'Model':<20} {'k_eff (W/m·K)':<15} {'Enhancement (%)':<15}",
            "-" * 70,
        ]
        
        for result in results:
            lines.append(
                f"{result.model_name:<20} {result.k_effective:<15.4f} "
                f"{result.enhancement_percent:<15.2f}"
            )
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def to_json(self, results: Optional[List[SimulationResult]] = None) -> str:
        """
        Export simulation results to JSON format.
        
        Args:
            results: List of results to export (default: calculate all models)
            
        Returns:
            JSON string of simulation results
        """
        if results is None:
            results = self.calculate_all_models()
        
        data = {
            "configuration": {
                "base_fluid": self._base_fluid_name,
                "nanoparticle": self._nanoparticle_type,
                "k_base_fluid": self._k_base_fluid,
                "k_nanoparticle": self._k_nanoparticle,
                "volume_fraction": self._volume_fraction,
                "particle_radius_nm": self._particle_radius,
                "sphericity": self._sphericity,
            },
            "results": [r.to_dict() for r in results],
        }
        
        return json.dumps(data, indent=2)
    
    def get_configuration(self) -> dict:
        """Get current simulator configuration as a dictionary."""
        return {
            "base_fluid": self._base_fluid_name,
            "k_base_fluid": self._k_base_fluid,
            "nanoparticle": self._nanoparticle_type,
            "k_nanoparticle": self._k_nanoparticle,
            "volume_fraction": self._volume_fraction,
            "volume_fraction_percent": self._volume_fraction * 100,
            "particle_radius_nm": self._particle_radius,
            "nanolayer_thickness_nm": self._nanolayer_thickness,
            "sphericity": self._sphericity,
            "temperature_K": self._temperature,
        }
    
    def __str__(self) -> str:
        """String representation of the simulator configuration."""
        config = self.get_configuration()
        return (
            f"NanofluidSimulator:\n"
            f"  Base Fluid: {config['base_fluid']} (k={config['k_base_fluid']} W/m·K)\n"
            f"  Nanoparticle: {config['nanoparticle']} (k={config['k_nanoparticle']} W/m·K)\n"
            f"  Volume Fraction: {config['volume_fraction_percent']:.2f}%\n"
            f"  Particle Radius: {config['particle_radius_nm']:.1f} nm"
        )
