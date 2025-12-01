"""
Thermophysical Properties of Nanofluids

This module calculates various thermophysical properties beyond thermal conductivity,
including viscosity, density, specific heat, and Prandtl number.

References:
1. Einstein, A. (1906). Annalen der Physik.
2. Batchelor, G.K. (1977). Journal of Fluid Mechanics.
3. Brinkman, H.C. (1952). The Journal of Chemical Physics.
4. Pak, B.C. and Cho, Y.I. (1998). Experimental Heat Transfer.
5. Xuan, Y. and Roetzel, W. (2000). International Journal of Heat and Mass Transfer.
"""

import math
from typing import Optional, Tuple


def einstein_viscosity(mu_bf: float, phi: float) -> float:
    """
    Einstein Model for effective viscosity (dilute suspensions).
    
    Valid for very low volume fractions (φ < 0.02) of spherical particles.
    
    Formula:
        μ_eff = μ_bf * (1 + 2.5*φ)
    
    Args:
        mu_bf: Dynamic viscosity of base fluid (Pa·s)
        phi: Volume fraction of nanoparticles (0 to 1)
        
    Returns:
        Effective dynamic viscosity (Pa·s)
    """
    if mu_bf <= 0:
        raise ValueError("Viscosity must be positive")
    if not 0 <= phi <= 1:
        raise ValueError("Volume fraction must be between 0 and 1")
    
    return mu_bf * (1 + 2.5 * phi)


def batchelor_viscosity(mu_bf: float, phi: float) -> float:
    """
    Batchelor Model for effective viscosity.
    
    Extends Einstein's model to include particle-particle interactions.
    Valid for φ < 0.1.
    
    Formula:
        μ_eff = μ_bf * (1 + 2.5*φ + 6.2*φ²)
    
    Args:
        mu_bf: Dynamic viscosity of base fluid (Pa·s)
        phi: Volume fraction of nanoparticles (0 to 1)
        
    Returns:
        Effective dynamic viscosity (Pa·s)
    """
    if mu_bf <= 0:
        raise ValueError("Viscosity must be positive")
    if not 0 <= phi <= 1:
        raise ValueError("Volume fraction must be between 0 and 1")
    
    return mu_bf * (1 + 2.5 * phi + 6.2 * phi**2)


def brinkman_viscosity(mu_bf: float, phi: float) -> float:
    """
    Brinkman Model for effective viscosity.
    
    Valid for moderate volume fractions (φ < 0.4).
    
    Formula:
        μ_eff = μ_bf / (1 - φ)^2.5
    
    Args:
        mu_bf: Dynamic viscosity of base fluid (Pa·s)
        phi: Volume fraction of nanoparticles (0 to 1)
        
    Returns:
        Effective dynamic viscosity (Pa·s)
    """
    if mu_bf <= 0:
        raise ValueError("Viscosity must be positive")
    if not 0 <= phi < 1:
        raise ValueError("Volume fraction must be between 0 and 1 (exclusive)")
    
    return mu_bf / ((1 - phi) ** 2.5)


def krieger_dougherty_viscosity(
    mu_bf: float,
    phi: float,
    phi_max: float = 0.605
) -> float:
    """
    Krieger-Dougherty Model for effective viscosity.
    
    Accurate for a wide range of volume fractions up to maximum packing.
    
    Formula:
        μ_eff = μ_bf * (1 - φ/φ_max)^(-[μ]*φ_max)
        where [μ] = 2.5 (intrinsic viscosity for spheres)
    
    Args:
        mu_bf: Dynamic viscosity of base fluid (Pa·s)
        phi: Volume fraction of nanoparticles (0 to 1)
        phi_max: Maximum packing fraction (default 0.605 for random packing)
        
    Returns:
        Effective dynamic viscosity (Pa·s)
    """
    if mu_bf <= 0:
        raise ValueError("Viscosity must be positive")
    if not 0 <= phi < phi_max:
        raise ValueError(f"Volume fraction must be between 0 and {phi_max}")
    
    intrinsic_viscosity = 2.5  # For spherical particles
    exponent = -intrinsic_viscosity * phi_max
    
    return mu_bf * ((1 - phi / phi_max) ** exponent)


def temperature_dependent_viscosity(
    mu_ref: float,
    T: float,
    T_ref: float = 298.15,
    fluid_type: str = "water"
) -> float:
    """
    Calculate temperature-dependent viscosity of base fluid.
    
    Uses Vogel-Fulcher-Tammann equation for temperature dependence.
    
    Args:
        mu_ref: Reference viscosity at T_ref (Pa·s)
        T: Temperature (K)
        T_ref: Reference temperature (K)
        fluid_type: Type of fluid ("water", "ethylene_glycol", "oil")
        
    Returns:
        Viscosity at temperature T (Pa·s)
    """
    if mu_ref <= 0:
        raise ValueError("Reference viscosity must be positive")
    if T <= 0:
        raise ValueError("Temperature must be positive")
    
    # Temperature coefficients for different fluids
    coefficients = {
        "water": {"A": 2.414e-5, "B": 247.8, "C": 140},
        "ethylene_glycol": {"A": 6.51e-3, "B": 1790, "C": 112},
        "oil": {"A": 0.01, "B": 1500, "C": 100},
    }
    
    fluid_type_lower = fluid_type.lower().replace(" ", "_").replace("-", "_")
    
    if fluid_type_lower in coefficients:
        coef = coefficients[fluid_type_lower]
        # Vogel equation: μ = A * exp(B / (T - C))
        mu_T = coef["A"] * math.exp(coef["B"] / (T - coef["C"]))
        
        # Scale to match reference value
        mu_ref_calc = coef["A"] * math.exp(coef["B"] / (T_ref - coef["C"]))
        return mu_T * (mu_ref / mu_ref_calc)
    else:
        # Simple exponential approximation
        alpha = 0.025  # Temperature coefficient (1/K)
        return mu_ref * math.exp(-alpha * (T - T_ref))


def nanofluid_density(
    rho_bf: float,
    rho_np: float,
    phi: float
) -> float:
    """
    Calculate effective density of nanofluid using mixing rule.
    
    Formula:
        ρ_eff = φ*ρ_np + (1-φ)*ρ_bf
    
    Args:
        rho_bf: Density of base fluid (kg/m³)
        rho_np: Density of nanoparticles (kg/m³)
        phi: Volume fraction of nanoparticles (0 to 1)
        
    Returns:
        Effective density of nanofluid (kg/m³)
    """
    if rho_bf <= 0 or rho_np <= 0:
        raise ValueError("Densities must be positive")
    if not 0 <= phi <= 1:
        raise ValueError("Volume fraction must be between 0 and 1")
    
    return phi * rho_np + (1 - phi) * rho_bf


def nanofluid_specific_heat(
    rho_bf: float,
    rho_np: float,
    cp_bf: float,
    cp_np: float,
    phi: float
) -> float:
    """
    Calculate effective specific heat of nanofluid.
    
    Uses thermal equilibrium model (Xuan and Roetzel).
    
    Formula:
        (ρ*cp)_eff = φ*(ρ*cp)_np + (1-φ)*(ρ*cp)_bf
        cp_eff = (ρ*cp)_eff / ρ_eff
    
    Args:
        rho_bf: Density of base fluid (kg/m³)
        rho_np: Density of nanoparticles (kg/m³)
        cp_bf: Specific heat of base fluid (J/kg·K)
        cp_np: Specific heat of nanoparticles (J/kg·K)
        phi: Volume fraction of nanoparticles (0 to 1)
        
    Returns:
        Effective specific heat of nanofluid (J/kg·K)
    """
    if any(x <= 0 for x in [rho_bf, rho_np, cp_bf, cp_np]):
        raise ValueError("All properties must be positive")
    if not 0 <= phi <= 1:
        raise ValueError("Volume fraction must be between 0 and 1")
    
    rho_eff = nanofluid_density(rho_bf, rho_np, phi)
    
    rho_cp_eff = phi * rho_np * cp_np + (1 - phi) * rho_bf * cp_bf
    
    return rho_cp_eff / rho_eff


def thermal_diffusivity(
    k_eff: float,
    rho_eff: float,
    cp_eff: float
) -> float:
    """
    Calculate thermal diffusivity of nanofluid.
    
    Formula:
        α = k / (ρ*cp)
    
    Args:
        k_eff: Thermal conductivity (W/m·K)
        rho_eff: Density (kg/m³)
        cp_eff: Specific heat (J/kg·K)
        
    Returns:
        Thermal diffusivity (m²/s)
    """
    if any(x <= 0 for x in [k_eff, rho_eff, cp_eff]):
        raise ValueError("All properties must be positive")
    
    return k_eff / (rho_eff * cp_eff)


def prandtl_number(
    mu_eff: float,
    cp_eff: float,
    k_eff: float
) -> float:
    """
    Calculate Prandtl number of nanofluid.
    
    Formula:
        Pr = μ*cp / k
    
    Args:
        mu_eff: Dynamic viscosity (Pa·s)
        cp_eff: Specific heat (J/kg·K)
        k_eff: Thermal conductivity (W/m·K)
        
    Returns:
        Prandtl number (dimensionless)
    """
    if any(x <= 0 for x in [mu_eff, cp_eff, k_eff]):
        raise ValueError("All properties must be positive")
    
    return mu_eff * cp_eff / k_eff


def reynolds_number(
    rho: float,
    v: float,
    L: float,
    mu: float
) -> float:
    """
    Calculate Reynolds number.
    
    Formula:
        Re = ρ*v*L / μ
    
    Args:
        rho: Density (kg/m³)
        v: Velocity (m/s)
        L: Characteristic length (m)
        mu: Dynamic viscosity (Pa·s)
        
    Returns:
        Reynolds number (dimensionless)
    """
    if rho <= 0 or L <= 0 or mu <= 0:
        raise ValueError("Density, length, and viscosity must be positive")
    
    return rho * v * L / mu


def nusselt_number(
    h: float,
    L: float,
    k: float
) -> float:
    """
    Calculate Nusselt number.
    
    Formula:
        Nu = h*L / k
    
    Args:
        h: Heat transfer coefficient (W/m²·K)
        L: Characteristic length (m)
        k: Thermal conductivity (W/m·K)
        
    Returns:
        Nusselt number (dimensionless)
    """
    if L <= 0 or k <= 0:
        raise ValueError("Length and thermal conductivity must be positive")
    
    return h * L / k


def heat_transfer_coefficient_enhancement(
    k_nf: float,
    k_bf: float,
    mu_nf: float,
    mu_bf: float,
    exponent: float = 0.33
) -> float:
    """
    Calculate enhancement in heat transfer coefficient.
    
    Empirical correlation for convective heat transfer enhancement.
    
    Formula:
        h_nf/h_bf ≈ (k_nf/k_bf) * (μ_bf/μ_nf)^exponent
    
    Args:
        k_nf: Nanofluid thermal conductivity (W/m·K)
        k_bf: Base fluid thermal conductivity (W/m·K)
        mu_nf: Nanofluid viscosity (Pa·s)
        mu_bf: Base fluid viscosity (Pa·s)
        exponent: Empirical exponent (typically 0.33 for turbulent flow)
        
    Returns:
        Heat transfer coefficient ratio h_nf/h_bf
    """
    if any(x <= 0 for x in [k_nf, k_bf, mu_nf, mu_bf]):
        raise ValueError("All properties must be positive")
    
    k_ratio = k_nf / k_bf
    mu_ratio = mu_bf / mu_nf
    
    return k_ratio * (mu_ratio ** exponent)


class ThermophysicalProperties:
    """
    Comprehensive calculator for all thermophysical properties of nanofluids.
    
    This class provides a unified interface to calculate thermal conductivity,
    viscosity, density, specific heat, and derived properties.
    """
    
    def __init__(
        self,
        k_bf: float,
        k_np: float,
        rho_bf: float,
        rho_np: float,
        cp_bf: float,
        cp_np: float,
        mu_bf: float,
        phi: float,
        T: float = 298.15
    ):
        """
        Initialize with base fluid and nanoparticle properties.
        
        Args:
            k_bf: Base fluid thermal conductivity (W/m·K)
            k_np: Nanoparticle thermal conductivity (W/m·K)
            rho_bf: Base fluid density (kg/m³)
            rho_np: Nanoparticle density (kg/m³)
            cp_bf: Base fluid specific heat (J/kg·K)
            cp_np: Nanoparticle specific heat (J/kg·K)
            mu_bf: Base fluid viscosity (Pa·s)
            phi: Volume fraction (0 to 1)
            T: Temperature (K)
        """
        self.k_bf = k_bf
        self.k_np = k_np
        self.rho_bf = rho_bf
        self.rho_np = rho_np
        self.cp_bf = cp_bf
        self.cp_np = cp_np
        self.mu_bf = mu_bf
        self.phi = phi
        self.T = T
        
        # Calculate effective properties
        self._calculate_properties()
    
    def _calculate_properties(self):
        """Calculate all effective properties."""
        # Density
        self.rho_eff = nanofluid_density(self.rho_bf, self.rho_np, self.phi)
        
        # Specific heat
        self.cp_eff = nanofluid_specific_heat(
            self.rho_bf, self.rho_np,
            self.cp_bf, self.cp_np,
            self.phi
        )
        
        # Viscosity (using Batchelor for moderate concentrations)
        if self.phi < 0.02:
            self.mu_eff = einstein_viscosity(self.mu_bf, self.phi)
        elif self.phi < 0.1:
            self.mu_eff = batchelor_viscosity(self.mu_bf, self.phi)
        else:
            self.mu_eff = brinkman_viscosity(self.mu_bf, self.phi)
    
    def set_thermal_conductivity(self, k_eff: float):
        """Set the effective thermal conductivity from external calculation."""
        self.k_eff = k_eff
    
    def get_all_properties(self) -> dict:
        """Get all calculated properties as a dictionary."""
        properties = {
            "thermal_conductivity": self.k_eff if hasattr(self, 'k_eff') else None,
            "density": self.rho_eff,
            "specific_heat": self.cp_eff,
            "dynamic_viscosity": self.mu_eff,
            "kinematic_viscosity": self.mu_eff / self.rho_eff,
        }
        
        if hasattr(self, 'k_eff'):
            properties["thermal_diffusivity"] = thermal_diffusivity(
                self.k_eff, self.rho_eff, self.cp_eff
            )
            properties["prandtl_number"] = prandtl_number(
                self.mu_eff, self.cp_eff, self.k_eff
            )
        
        return properties
    
    def print_summary(self):
        """Print a formatted summary of all properties."""
        props = self.get_all_properties()
        
        print("=" * 60)
        print("NANOFLUID THERMOPHYSICAL PROPERTIES")
        print("=" * 60)
        print(f"Temperature: {self.T:.2f} K ({self.T-273.15:.2f} °C)")
        print(f"Volume Fraction: {self.phi*100:.2f}%")
        print("-" * 60)
        
        if props["thermal_conductivity"]:
            print(f"Thermal Conductivity: {props['thermal_conductivity']:.4f} W/m·K")
            print(f"  Enhancement: {(props['thermal_conductivity']/self.k_bf - 1)*100:.2f}%")
        
        print(f"Density: {props['density']:.2f} kg/m³")
        print(f"  Enhancement: {(props['density']/self.rho_bf - 1)*100:.2f}%")
        
        print(f"Specific Heat: {props['specific_heat']:.2f} J/kg·K")
        print(f"  Change: {(props['specific_heat']/self.cp_bf - 1)*100:.2f}%")
        
        print(f"Dynamic Viscosity: {props['dynamic_viscosity']:.6f} Pa·s")
        print(f"  Enhancement: {(props['dynamic_viscosity']/self.mu_bf - 1)*100:.2f}%")
        
        print(f"Kinematic Viscosity: {props['kinematic_viscosity']*1e6:.4f} mm²/s")
        
        if "thermal_diffusivity" in props:
            print(f"Thermal Diffusivity: {props['thermal_diffusivity']*1e7:.4f} × 10⁻⁷ m²/s")
        
        if "prandtl_number" in props:
            print(f"Prandtl Number: {props['prandtl_number']:.4f}")
        
        print("=" * 60)
