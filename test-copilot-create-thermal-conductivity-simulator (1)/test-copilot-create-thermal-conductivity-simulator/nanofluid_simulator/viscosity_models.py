"""
Advanced Viscosity Models for Nanofluids

This module implements comprehensive viscosity models accounting for:
- Temperature dependence (Arrhenius-type relations)
- Concentration dependence
- Shear-rate dependence (non-Newtonian behavior)
- Flow regime transitions

Key Physics:
- Viscosity decreases exponentially with temperature
- Viscosity increases with nanoparticle concentration
- High concentrations can exhibit shear-thinning or shear-thickening
- Aggregation dramatically increases viscosity

References:
1. Einstein, A. (1906). "Eine neue Bestimmung der Moleküldimensionen"
2. Batchelor, G.K. (1977). "The effect of Brownian motion on bulk stress"
3. Krieger, I.M. & Dougherty, T.J. (1959). "A mechanism for non-Newtonian flow"
4. Vogel, H. (1921). "Das Temperaturabhängigkeitsgesetz der Viskosität"
5. Carreau, P.J. (1972). "Rheological equations from molecular network theories"
"""

import math
from typing import Optional, Dict, Tuple
import numpy as np


# ============================================================================
# BASE FLUID TEMPERATURE-DEPENDENT VISCOSITY
# ============================================================================

class BaseFluidViscosity:
    """
    Temperature-dependent viscosity correlations for common base fluids.
    
    Most fluids follow Vogel-Fulcher-Tammann (VFT) or Arrhenius equations:
    μ(T) = A * exp(B/(T - C)) or μ(T) = A * exp(B/T)
    """
    
    @staticmethod
    def water_viscosity(T: float) -> float:
        """
        Water dynamic viscosity (Pa·s) as function of temperature (K).
        
        Valid range: 273.15 K to 413.15 K (0°C to 140°C)
        Correlation: Vogel-Fulcher-Tammann equation
        
        Args:
            T: Temperature (K)
            
        Returns:
            Dynamic viscosity (Pa·s)
        """
        # Vogel equation parameters for water
        A = 2.414e-5  # Pa·s
        B = 247.8     # K
        C = 140       # K
        
        mu = A * 10 ** (B / (T - C))
        return mu
    
    @staticmethod
    def ethylene_glycol_viscosity(T: float) -> float:
        """
        Ethylene glycol dynamic viscosity (Pa·s) vs temperature (K).
        
        Valid range: 273 K to 393 K
        """
        # Andrade equation: μ = A * exp(B/T)
        A = 6.65e-6
        B = 2300
        
        mu = A * math.exp(B / T)
        return mu
    
    @staticmethod
    def propylene_glycol_viscosity(T: float) -> float:
        """Propylene glycol viscosity (Pa·s) vs temperature (K)."""
        A = 1.2e-5
        B = 2100
        
        mu = A * math.exp(B / T)
        return mu
    
    @staticmethod
    def engine_oil_viscosity(T: float, SAE_grade: int = 40) -> float:
        """
        Engine oil viscosity (Pa·s) vs temperature (K).
        
        Args:
            T: Temperature (K)
            SAE_grade: SAE viscosity grade (10, 20, 30, 40, 50)
        """
        # Walther equation for engine oils
        # Base parameters for SAE 40
        if SAE_grade == 10:
            A, B = 3.5e-4, 1200
        elif SAE_grade == 20:
            A, B = 5.0e-4, 1400
        elif SAE_grade == 30:
            A, B = 7.5e-4, 1600
        elif SAE_grade == 40:
            A, B = 1.2e-3, 1800
        elif SAE_grade == 50:
            A, B = 2.0e-3, 2000
        else:
            A, B = 1.2e-3, 1800  # Default to SAE 40
        
        mu = A * math.exp(B / T)
        return mu
    
    @staticmethod
    def get_base_fluid_viscosity(fluid_name: str, T: float) -> float:
        """
        Get viscosity for any base fluid by name.
        
        Args:
            fluid_name: Name of base fluid
            T: Temperature (K)
            
        Returns:
            Dynamic viscosity (Pa·s)
        """
        fluid_map = {
            'water': BaseFluidViscosity.water_viscosity,
            'ethylene_glycol': BaseFluidViscosity.ethylene_glycol_viscosity,
            'propylene_glycol': BaseFluidViscosity.propylene_glycol_viscosity,
            'engine_oil': BaseFluidViscosity.engine_oil_viscosity,
        }
        
        # Handle mixture fluids
        if 'water_eg' in fluid_name:
            # Water-ethylene glycol mixture
            parts = fluid_name.split('_')
            if len(parts) >= 3:
                try:
                    water_pct = int(parts[2])
                    eg_pct = 100 - water_pct if len(parts) < 4 else int(parts[3])
                    
                    mu_water = BaseFluidViscosity.water_viscosity(T)
                    mu_eg = BaseFluidViscosity.ethylene_glycol_viscosity(T)
                    
                    # Log-linear mixing rule (better than linear for viscosity)
                    ln_mu = (water_pct/100) * math.log(mu_water) + \
                           (eg_pct/100) * math.log(mu_eg)
                    return math.exp(ln_mu)
                except:
                    pass
            return BaseFluidViscosity.water_viscosity(T)
        
        # Get function and calculate
        func = fluid_map.get(fluid_name, BaseFluidViscosity.water_viscosity)
        return func(T)


# ============================================================================
# NANOFLUID VISCOSITY MODELS
# ============================================================================

def einstein_viscosity_temp(
    mu_bf: float,
    phi: float,
    T: float,
    T_ref: float = 298.15
) -> float:
    """
    Temperature-corrected Einstein viscosity model.
    
    Valid for φ < 0.02 (dilute suspensions)
    
    Args:
        mu_bf: Base fluid viscosity at temperature T (Pa·s)
        phi: Volume fraction (0 to 1)
        T: Temperature (K)
        T_ref: Reference temperature (K)
        
    Returns:
        Effective viscosity (Pa·s)
    """
    # Einstein equation: μ_eff = μ_bf * (1 + 2.5*φ)
    mu_eff = mu_bf * (1 + 2.5 * phi)
    
    return mu_eff


def batchelor_viscosity_temp(
    mu_bf: float,
    phi: float
) -> float:
    """
    Batchelor viscosity model accounting for pair interactions.
    
    Valid for φ < 0.10
    Includes two-body hydrodynamic interactions
    
    Args:
        mu_bf: Base fluid viscosity (Pa·s)
        phi: Volume fraction (0 to 1)
        
    Returns:
        Effective viscosity (Pa·s)
    """
    mu_eff = mu_bf * (1 + 2.5 * phi + 6.5 * phi**2)
    return mu_eff


def brinkman_viscosity_temp(
    mu_bf: float,
    phi: float
) -> float:
    """
    Brinkman viscosity model for moderate concentrations.
    
    Valid for φ < 0.25
    
    Args:
        mu_bf: Base fluid viscosity (Pa·s)
        phi: Volume fraction (0 to 1)
        
    Returns:
        Effective viscosity (Pa·s)
    """
    mu_eff = mu_bf / (1 - phi) ** 2.5
    return mu_eff


def krieger_dougherty_viscosity(
    mu_bf: float,
    phi: float,
    phi_max: float = 0.605,
    intrinsic_viscosity: float = 2.5
) -> float:
    """
    Krieger-Dougherty equation for concentrated suspensions.
    
    This is the most accurate model for high volume fractions,
    accounting for particle crowding and percolation effects.
    
    Valid for all φ < φ_max
    
    Args:
        mu_bf: Base fluid viscosity (Pa·s)
        phi: Volume fraction (0 to 1)
        phi_max: Maximum packing fraction (default 0.605 for spheres)
                 Typical values: 0.52-0.68 depending on particle shape
        intrinsic_viscosity: [μ] parameter (default 2.5 for spheres)
        
    Returns:
        Effective viscosity (Pa·s)
    """
    if phi >= phi_max:
        # At maximum packing, viscosity approaches infinity
        return mu_bf * 1e6
    
    exponent = -intrinsic_viscosity * phi_max
    mu_eff = mu_bf * (1 - phi / phi_max) ** exponent
    
    return mu_eff


def corcione_viscosity_model(
    mu_bf: float,
    phi: float,
    T: float,
    d_p: float,
    d_bf: float = 0.3
) -> Tuple[float, dict]:
    """
    Corcione empirical viscosity model with temperature and size effects.
    
    Based on 200+ experimental datasets.
    
    Formula:
        μ_nf/μ_bf = 1 / (1 - 34.87*(d_p/d_bf)^(-0.3) * φ^1.03)
    
    Args:
        mu_bf: Base fluid viscosity (Pa·s)
        phi: Volume fraction (0 to 1)
        T: Temperature (K)
        d_p: Particle diameter (nm)
        d_bf: Base fluid molecule diameter (nm, default 0.3 for water)
        
    Returns:
        Tuple of (effective_viscosity, diagnostics_dict)
    """
    # Size ratio effect
    size_ratio = (d_p / d_bf) ** (-0.3)
    
    # Concentration effect
    phi_term = phi ** 1.03
    
    # Combined effect
    denominator = 1 - 34.87 * size_ratio * phi_term
    
    if denominator <= 0:
        # Approaching maximum packing
        mu_eff = mu_bf * 1e5
    else:
        mu_eff = mu_bf / denominator
    
    diagnostics = {
        'size_ratio_factor': size_ratio,
        'phi_term': phi_term,
        'viscosity_ratio': mu_eff / mu_bf
    }
    
    return mu_eff, diagnostics


# ============================================================================
# SHEAR-RATE DEPENDENT VISCOSITY (NON-NEWTONIAN BEHAVIOR)
# ============================================================================

def carreau_model(
    mu_bf: float,
    phi: float,
    shear_rate: float,
    mu_0_ratio: float = 2.0,
    mu_inf_ratio: float = 1.0,
    lambda_time: float = 1.0,
    n: float = 0.5
) -> Tuple[float, str]:
    """
    Carreau model for shear-thinning/thickening behavior.
    
    Most nanofluids exhibit shear-thinning at high concentrations:
    - Low shear: particles form aggregates (high viscosity)
    - High shear: aggregates break apart (lower viscosity)
    
    Formula:
        μ(γ̇) = μ_∞ + (μ_0 - μ_∞) * [1 + (λ*γ̇)²]^((n-1)/2)
    
    Args:
        mu_bf: Base fluid viscosity (Pa·s)
        phi: Volume fraction (0 to 1)
        shear_rate: Shear rate γ̇ (1/s)
        mu_0_ratio: Zero-shear viscosity ratio μ_0/μ_bf (default 2.0)
        mu_inf_ratio: Infinite-shear viscosity ratio μ_∞/μ_bf (default 1.0)
        lambda_time: Relaxation time constant (s)
        n: Power-law index (n<1: shear-thinning, n>1: shear-thickening)
        
    Returns:
        Tuple of (effective_viscosity, behavior_type)
    """
    # Zero-shear and infinite-shear viscosities
    mu_0 = mu_bf * mu_0_ratio * (1 + 2.5 * phi)
    mu_inf = mu_bf * mu_inf_ratio * (1 + 1.5 * phi)
    
    # Carreau equation
    lambda_gamma = lambda_time * shear_rate
    mu_eff = mu_inf + (mu_0 - mu_inf) * (1 + lambda_gamma**2) ** ((n - 1) / 2)
    
    # Determine behavior
    if n < 1:
        behavior = "Shear-thinning"
    elif n > 1:
        behavior = "Shear-thickening"
    else:
        behavior = "Newtonian"
    
    return mu_eff, behavior


def power_law_viscosity(
    K: float,
    n: float,
    shear_rate: float
) -> Tuple[float, str]:
    """
    Simple power-law model for non-Newtonian fluids.
    
    Formula:
        μ(γ̇) = K * γ̇^(n-1)
    
    Args:
        K: Consistency index (Pa·s^n)
        n: Power-law index
        shear_rate: Shear rate (1/s)
        
    Returns:
        Tuple of (apparent_viscosity, behavior_type)
    """
    if shear_rate == 0:
        return K, "Undefined"
    
    mu_app = K * shear_rate ** (n - 1)
    
    if n < 1:
        behavior = "Shear-thinning"
    elif n > 1:
        behavior = "Shear-thickening"
    else:
        behavior = "Newtonian"
    
    return mu_app, behavior


# ============================================================================
# AGGREGATION EFFECTS ON VISCOSITY
# ============================================================================

def aggregated_nanofluid_viscosity(
    mu_bf: float,
    phi: float,
    aggregation_factor: float = 1.5,
    fractal_dimension: float = 2.3
) -> Tuple[float, float]:
    """
    Viscosity model accounting for nanoparticle aggregation.
    
    Aggregation increases effective volume fraction and viscosity:
    - Stable nanofluid: individual particles (low viscosity)
    - Aggregated: clusters with trapped fluid (high viscosity)
    
    Key Physics:
    - Fractal aggregates trap base fluid inside
    - Effective volume fraction increases
    - Viscosity can increase 2-10x due to aggregation
    
    Args:
        mu_bf: Base fluid viscosity (Pa·s)
        phi: True volume fraction of nanoparticles (0 to 1)
        aggregation_factor: Aggregate size / primary particle size (1-5)
                          1.0 = no aggregation (stable)
                          2-3 = moderate aggregation
                          >3 = severe aggregation
        fractal_dimension: D_f (typical range 1.8-2.5)
                          Lower D_f = more open, fractal structures
        
    Returns:
        Tuple of (effective_viscosity, effective_phi)
    """
    # Effective volume fraction accounting for aggregation
    # Based on: φ_eff = φ * (R_agg/R_p)^(3-D_f)
    phi_eff = phi * aggregation_factor ** (3 - fractal_dimension)
    
    # Limit effective phi to physical maximum
    phi_eff = min(phi_eff, 0.6)
    
    # Use Krieger-Dougherty with effective phi
    mu_eff = krieger_dougherty_viscosity(mu_bf, phi_eff)
    
    return mu_eff, phi_eff


# ============================================================================
# COMPREHENSIVE VISCOSITY CALCULATOR
# ============================================================================

class NanofluidViscosityCalculator:
    """
    Comprehensive calculator for nanofluid viscosity with all effects.
    """
    
    @staticmethod
    def calculate_complete(
        base_fluid: str,
        T: float,
        phi: float,
        d_p: float = 50.0,
        shear_rate: float = 0.0,
        aggregation_state: str = "stable"
    ) -> Dict[str, float]:
        """
        Calculate viscosity with all effects included.
        
        Args:
            base_fluid: Base fluid name
            T: Temperature (K)
            phi: Volume fraction (0 to 1)
            d_p: Particle diameter (nm)
            shear_rate: Shear rate (1/s, 0 for static)
            aggregation_state: "stable", "moderate", or "severe"
            
        Returns:
            Dictionary with multiple viscosity predictions
        """
        # 1. Base fluid viscosity at temperature T
        mu_bf = BaseFluidViscosity.get_base_fluid_viscosity(base_fluid, T)
        
        # 2. Calculate with different models
        results = {}
        
        # Einstein (dilute)
        if phi < 0.02:
            results['Einstein'] = einstein_viscosity_temp(mu_bf, phi, T)
        
        # Batchelor (low concentration)
        if phi < 0.10:
            results['Batchelor'] = batchelor_viscosity_temp(mu_bf, phi)
        
        # Brinkman (moderate)
        if phi < 0.25:
            results['Brinkman'] = brinkman_viscosity_temp(mu_bf, phi)
        
        # Krieger-Dougherty (all concentrations)
        results['Krieger-Dougherty'] = krieger_dougherty_viscosity(mu_bf, phi)
        
        # Corcione (empirical)
        mu_corcione, _ = corcione_viscosity_model(mu_bf, phi, T, d_p)
        results['Corcione'] = mu_corcione
        
        # 3. Aggregation effects
        if aggregation_state == "moderate":
            mu_agg, phi_eff = aggregated_nanofluid_viscosity(mu_bf, phi, 2.0)
            results['Aggregated'] = mu_agg
        elif aggregation_state == "severe":
            mu_agg, phi_eff = aggregated_nanofluid_viscosity(mu_bf, phi, 3.5)
            results['Aggregated'] = mu_agg
        
        # 4. Shear-rate effects
        if shear_rate > 0:
            mu_shear, behavior = carreau_model(mu_bf, phi, shear_rate)
            results['Shear-dependent'] = mu_shear
            results['_behavior'] = behavior
        
        # Add base fluid reference
        results['Base_Fluid'] = mu_bf
        
        return results
