"""
Nanoparticle Database

This module provides a database of common nanoparticle materials with their
thermal and physical properties for use in thermal conductivity calculations.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class NanoparticleMaterial:
    """
    Properties of a nanoparticle material.
    
    Attributes:
        name: Full name of the material
        formula: Chemical formula
        thermal_conductivity: Thermal conductivity in W/m·K
        density: Density in kg/m³
        specific_heat: Specific heat capacity in J/kg·K
        particle_size_range: Typical particle size range in nm (min, max)
    """
    name: str
    formula: str
    thermal_conductivity: float  # W/m·K
    density: float  # kg/m³
    specific_heat: float  # J/kg·K
    particle_size_range: tuple = (1, 100)  # nm
    
    def __str__(self) -> str:
        return (
            f"{self.name} ({self.formula}): "
            f"k={self.thermal_conductivity} W/m·K, "
            f"ρ={self.density} kg/m³"
        )


@dataclass
class BaseFluid:
    """
    Properties of a base fluid.
    
    Attributes:
        name: Name of the fluid
        thermal_conductivity: Thermal conductivity in W/m·K at reference temperature
        density: Density in kg/m³
        specific_heat: Specific heat capacity in J/kg·K
        viscosity: Dynamic viscosity in Pa·s
        reference_temperature: Reference temperature in K
    """
    name: str
    thermal_conductivity: float  # W/m·K
    density: float  # kg/m³
    specific_heat: float  # J/kg·K
    viscosity: float  # Pa·s
    reference_temperature: float = 298.15  # K (25°C)
    
    def __str__(self) -> str:
        return (
            f"{self.name}: "
            f"k={self.thermal_conductivity} W/m·K, "
            f"ρ={self.density} kg/m³ "
            f"at {self.reference_temperature-273.15:.1f}°C"
        )


class NanoparticleDatabase:
    """
    Database of nanoparticle materials and base fluids with their properties.
    
    This class provides access to thermal and physical properties of common
    nanoparticle materials and base fluids used in nanofluid research.
    All methods are class methods and can be called without instantiation.
    
    Example:
        >>> copper = NanoparticleDatabase.get_nanoparticle("Cu")
        >>> print(copper.thermal_conductivity)
        401
        >>> water = NanoparticleDatabase.get_base_fluid("water")
        >>> print(water.thermal_conductivity)
        0.613
    """
    
    # Nanoparticle materials database
    NANOPARTICLES: Dict[str, NanoparticleMaterial] = {
        "Cu": NanoparticleMaterial(
            name="Copper",
            formula="Cu",
            thermal_conductivity=401,
            density=8933,
            specific_heat=385,
            particle_size_range=(10, 100)
        ),
        "Ag": NanoparticleMaterial(
            name="Silver",
            formula="Ag",
            thermal_conductivity=429,
            density=10490,
            specific_heat=235,
            particle_size_range=(10, 100)
        ),
        "Au": NanoparticleMaterial(
            name="Gold",
            formula="Au",
            thermal_conductivity=317,
            density=19300,
            specific_heat=129,
            particle_size_range=(5, 50)
        ),
        "Al": NanoparticleMaterial(
            name="Aluminum",
            formula="Al",
            thermal_conductivity=237,
            density=2700,
            specific_heat=897,
            particle_size_range=(20, 100)
        ),
        "Al2O3": NanoparticleMaterial(
            name="Alumina (Aluminum Oxide)",
            formula="Al₂O₃",
            thermal_conductivity=40,
            density=3970,
            specific_heat=765,
            particle_size_range=(10, 100)
        ),
        "TiO2": NanoparticleMaterial(
            name="Titanium Dioxide",
            formula="TiO₂",
            thermal_conductivity=8.9,
            density=4250,
            specific_heat=686,
            particle_size_range=(10, 50)
        ),
        "CuO": NanoparticleMaterial(
            name="Copper Oxide",
            formula="CuO",
            thermal_conductivity=76.5,
            density=6310,
            specific_heat=535,
            particle_size_range=(20, 80)
        ),
        "SiO2": NanoparticleMaterial(
            name="Silicon Dioxide (Silica)",
            formula="SiO₂",
            thermal_conductivity=1.4,
            density=2220,
            specific_heat=745,
            particle_size_range=(10, 100)
        ),
        "Fe2O3": NanoparticleMaterial(
            name="Iron(III) Oxide (Hematite)",
            formula="Fe₂O₃",
            thermal_conductivity=6,
            density=5240,
            specific_heat=650,
            particle_size_range=(20, 100)
        ),
        "Fe3O4": NanoparticleMaterial(
            name="Iron(II,III) Oxide (Magnetite)",
            formula="Fe₃O₄",
            thermal_conductivity=9.7,
            density=5180,
            specific_heat=670,
            particle_size_range=(10, 100)
        ),
        "ZnO": NanoparticleMaterial(
            name="Zinc Oxide",
            formula="ZnO",
            thermal_conductivity=29,
            density=5600,
            specific_heat=494,
            particle_size_range=(20, 100)
        ),
        "MgO": NanoparticleMaterial(
            name="Magnesium Oxide",
            formula="MgO",
            thermal_conductivity=48.4,
            density=3580,
            specific_heat=874,
            particle_size_range=(20, 100)
        ),
        "CNT": NanoparticleMaterial(
            name="Carbon Nanotubes",
            formula="CNT",
            thermal_conductivity=3000,
            density=2100,
            specific_heat=711,
            particle_size_range=(1, 50)
        ),
        "MWCNT": NanoparticleMaterial(
            name="Multi-Walled Carbon Nanotubes",
            formula="MWCNT",
            thermal_conductivity=3000,
            density=2100,
            specific_heat=711,
            particle_size_range=(5, 50)
        ),
        "Graphene": NanoparticleMaterial(
            name="Graphene Nanoplatelets",
            formula="C",
            thermal_conductivity=5000,
            density=2200,
            specific_heat=709,
            particle_size_range=(1, 10)
        ),
        "Diamond": NanoparticleMaterial(
            name="Nanodiamond",
            formula="C",
            thermal_conductivity=2000,
            density=3510,
            specific_heat=509,
            particle_size_range=(4, 10)
        ),
        "SiC": NanoparticleMaterial(
            name="Silicon Carbide",
            formula="SiC",
            thermal_conductivity=120,
            density=3210,
            specific_heat=750,
            particle_size_range=(10, 100)
        ),
        "AlN": NanoparticleMaterial(
            name="Aluminum Nitride",
            formula="AlN",
            thermal_conductivity=285,
            density=3260,
            specific_heat=740,
            particle_size_range=(20, 100)
        ),
        "BN": NanoparticleMaterial(
            name="Boron Nitride",
            formula="BN",
            thermal_conductivity=300,
            density=2100,
            specific_heat=800,
            particle_size_range=(10, 100)
        ),
        "rGO": NanoparticleMaterial(
            name="Reduced Graphene Oxide",
            formula="rGO",
            thermal_conductivity=2500,
            density=2200,
            specific_heat=700,
            particle_size_range=(1, 5)
        ),
    }
    
    # Base fluids database
    BASE_FLUIDS: Dict[str, BaseFluid] = {
        "water": BaseFluid(
            name="Water",
            thermal_conductivity=0.613,
            density=997,
            specific_heat=4182,
            viscosity=0.001,
            reference_temperature=298.15
        ),
        "ethylene_glycol": BaseFluid(
            name="Ethylene Glycol",
            thermal_conductivity=0.252,
            density=1113,
            specific_heat=2415,
            viscosity=0.0161,
            reference_temperature=298.15
        ),
        "propylene_glycol": BaseFluid(
            name="Propylene Glycol",
            thermal_conductivity=0.200,
            density=1036,
            specific_heat=2500,
            viscosity=0.042,
            reference_temperature=298.15
        ),
        "engine_oil": BaseFluid(
            name="Engine Oil",
            thermal_conductivity=0.145,
            density=884,
            specific_heat=1909,
            viscosity=0.486,
            reference_temperature=298.15
        ),
        "mineral_oil": BaseFluid(
            name="Mineral Oil",
            thermal_conductivity=0.130,
            density=870,
            specific_heat=2090,
            viscosity=0.065,
            reference_temperature=298.15
        ),
        "transformer_oil": BaseFluid(
            name="Transformer Oil",
            thermal_conductivity=0.125,
            density=879,
            specific_heat=1860,
            viscosity=0.0125,
            reference_temperature=298.15
        ),
        "kerosene": BaseFluid(
            name="Kerosene",
            thermal_conductivity=0.145,
            density=790,
            specific_heat=2090,
            viscosity=0.00164,
            reference_temperature=298.15
        ),
        "glycerol": BaseFluid(
            name="Glycerol",
            thermal_conductivity=0.285,
            density=1261,
            specific_heat=2427,
            viscosity=1.412,
            reference_temperature=298.15
        ),
        "water_eg_50_50": BaseFluid(
            name="Water-Ethylene Glycol (50:50)",
            thermal_conductivity=0.415,
            density=1055,
            specific_heat=3298,
            viscosity=0.00405,
            reference_temperature=298.15
        ),
        "water_eg_60_40": BaseFluid(
            name="Water-Ethylene Glycol (60:40)",
            thermal_conductivity=0.445,
            density=1033,
            specific_heat=3485,
            viscosity=0.00295,
            reference_temperature=298.15
        ),
        "water_pg_50_50": BaseFluid(
            name="Water-Propylene Glycol (50:50)",
            thermal_conductivity=0.395,
            density=1017,
            specific_heat=3341,
            viscosity=0.00385,
            reference_temperature=298.15
        ),
    }
    
    @classmethod
    def get_nanoparticle(cls, formula: str) -> NanoparticleMaterial:
        """
        Get nanoparticle properties by chemical formula.
        
        Args:
            formula: Chemical formula (e.g., "Cu", "Al2O3", "TiO2")
            
        Returns:
            NanoparticleMaterial dataclass with properties
            
        Raises:
            KeyError: If the nanoparticle is not in the database
        """
        formula_upper = formula.upper()
        for key, material in cls.NANOPARTICLES.items():
            if key.upper() == formula_upper:
                return material
        raise KeyError(
            f"Nanoparticle '{formula}' not found. "
            f"Available: {', '.join(cls.NANOPARTICLES.keys())}"
        )
    
    @classmethod
    def get_base_fluid(cls, name: str) -> BaseFluid:
        """
        Get base fluid properties by name.
        
        Args:
            name: Name of the base fluid (e.g., "water", "ethylene_glycol")
            
        Returns:
            BaseFluid dataclass with properties
            
        Raises:
            KeyError: If the base fluid is not in the database
        """
        name_lower = name.lower().replace(" ", "_").replace("-", "_")
        for key, fluid in cls.BASE_FLUIDS.items():
            if key.lower() == name_lower:
                return fluid
        raise KeyError(
            f"Base fluid '{name}' not found. "
            f"Available: {', '.join(cls.BASE_FLUIDS.keys())}"
        )
    
    @classmethod
    def list_nanoparticles(cls) -> list:
        """List all available nanoparticle materials."""
        return list(cls.NANOPARTICLES.keys())
    
    @classmethod
    def list_base_fluids(cls) -> list:
        """List all available base fluids."""
        return list(cls.BASE_FLUIDS.keys())
    
    @classmethod
    def add_nanoparticle(
        cls,
        formula: str,
        name: str,
        thermal_conductivity: float,
        density: float,
        specific_heat: float,
        particle_size_range: tuple = (1, 100)
    ) -> None:
        """
        Add a custom nanoparticle material to the database.
        
        Args:
            formula: Chemical formula
            name: Full name of the material
            thermal_conductivity: Thermal conductivity in W/m·K
            density: Density in kg/m³
            specific_heat: Specific heat capacity in J/kg·K
            particle_size_range: Typical particle size range in nm
        """
        cls.NANOPARTICLES[formula] = NanoparticleMaterial(
            name=name,
            formula=formula,
            thermal_conductivity=thermal_conductivity,
            density=density,
            specific_heat=specific_heat,
            particle_size_range=particle_size_range
        )
    
    @classmethod
    def add_base_fluid(
        cls,
        key: str,
        name: str,
        thermal_conductivity: float,
        density: float,
        specific_heat: float,
        viscosity: float,
        reference_temperature: float = 298.15
    ) -> None:
        """
        Add a custom base fluid to the database.
        
        Args:
            key: Key identifier for the fluid
            name: Full name of the fluid
            thermal_conductivity: Thermal conductivity in W/m·K
            density: Density in kg/m³
            specific_heat: Specific heat capacity in J/kg·K
            viscosity: Dynamic viscosity in Pa·s
            reference_temperature: Reference temperature in K
        """
        cls.BASE_FLUIDS[key] = BaseFluid(
            name=name,
            thermal_conductivity=thermal_conductivity,
            density=density,
            specific_heat=specific_heat,
            viscosity=viscosity,
            reference_temperature=reference_temperature
        )
