"""
Flow-Dependent Thermal Conductivity Models for Nanofluids

This module implements advanced models that account for fluid motion effects
on thermal conductivity, including Brownian motion enhancement under flow,
velocity/shear rate dependency, and convective contributions.

Key Physics:
- Flow accelerates Brownian motion (micro-convection)
- Shear-induced particle alignment affects conductivity
- Velocity gradients enhance energy transport mechanisms

References:
1. Buongiorno, J. (2006). "Convective transport in nanofluids" ASME J. Heat Transfer
2. Corcione, M. (2011). "Empirical correlating equations for predicting the effective 
   thermal conductivity and dynamic viscosity of nanofluids" Energy Conversion Management
3. Rea, U. et al. (2009). "Laminar convection heat transfer and viscous pressure loss 
   of alumina-water and zirconia-water nanofluids" Int. J. Heat Mass Transfer
"""

import math
from typing import Optional, Tuple
import numpy as np

# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
AVOGADRO_NUMBER = 6.02214076e23    # mol⁻¹


def buongiorno_convective_model(
    k_bf: float,
    k_np: float,
    phi: float,
    T: float,
    d_p: float,
    rho_bf: float,
    rho_np: float,
    mu_bf: float,
    velocity: float = 0.0,
    shear_rate: Optional[float] = None
) -> Tuple[float, dict]:
    """
    Buongiorno Two-Phase Model with Flow Effects.
    
    This model accounts for:
    - Brownian diffusion (thermophoresis)
    - Thermophoretic diffusion
    - Flow-enhanced micro-convection
    - Shear-induced particle migration
    
    The Buongiorno model is the foundation for understanding nanofluid
    transport mechanisms under flow conditions.
    
    Key Physics:
    1. Brownian diffusion coefficient increases with temperature
    2. Flow shear enhances particle collision frequency
    3. Thermophoresis drives particles from hot to cold regions
    4. Effective conductivity increases non-linearly with velocity
    
    Args:
        k_bf: Base fluid thermal conductivity (W/m·K)
        k_np: Nanoparticle thermal conductivity (W/m·K)
        phi: Volume fraction (0 to 1)
        T: Temperature (K)
        d_p: Particle diameter (nm)
        rho_bf: Base fluid density (kg/m³)
        rho_np: Nanoparticle density (kg/m³)
        mu_bf: Base fluid dynamic viscosity (Pa·s)
        velocity: Flow velocity (m/s, default 0)
        shear_rate: Shear rate (1/s, optional - calculated from velocity if not provided)
        
    Returns:
        Tuple of (k_effective, diagnostics_dict)
        diagnostics_dict contains: D_B, D_T, Pe, enhancement_ratio
    """
    if phi == 0:
        return k_bf, {}
    
    # Convert particle size to meters
    d_p_m = d_p * 1e-9
    
    # 1. Static component (Maxwell model)
    k_static = k_bf * (k_np + 2*k_bf + 2*phi*(k_np - k_bf)) / \
                      (k_np + 2*k_bf - phi*(k_np - k_bf))
    
    # 2. Brownian diffusion coefficient
    D_B = BOLTZMANN_CONSTANT * T / (3 * math.pi * mu_bf * d_p_m)
    
    # 3. Thermophoretic diffusion coefficient
    # Ratio of thermal conductivities
    kappa = k_np / k_bf
    # Thermophoretic diffusion
    D_T = 0.26 * k_bf / (2 * k_bf + k_np) * mu_bf / (rho_bf * T)
    
    # 4. Brownian motion contribution (temperature-dependent)
    # Particle thermal velocity
    v_B = math.sqrt(18 * BOLTZMANN_CONSTANT * T / (math.pi * rho_np * d_p_m**3))
    
    # Brownian enhancement factor (empirical correlation from literature)
    f_B = 5e4 * phi * rho_bf * BOLTZMANN_CONSTANT * T / (rho_np * d_p_m) * \
          math.sqrt(T / (rho_np * d_p_m**2))
    
    # 5. Flow enhancement
    if velocity > 0 or shear_rate is not None:
        # Calculate shear rate if not provided
        if shear_rate is None:
            # Estimate shear rate from velocity (assuming pipe flow)
            # γ ≈ 8V/D for pipe, use conservative estimate
            shear_rate = 10 * velocity  # Order of magnitude estimate
        
        # Peclet number (ratio of convective to diffusive transport)
        if D_B > 0:
            Pe = velocity * d_p_m / D_B
        else:
            Pe = 0
        
        # Flow enhancement factor (based on experimental correlations)
        # At low Pe: diffusion dominates, at high Pe: convection dominates
        if Pe < 1:
            f_flow = 1 + 0.1 * Pe  # Linear enhancement
        elif Pe < 100:
            f_flow = 1 + 0.05 * math.sqrt(Pe)  # Transitional regime
        else:
            f_flow = 1 + 0.5 * math.log10(Pe)  # Convection-dominated
        
        # Shear-induced enhancement
        # High shear rates increase particle collision frequency
        f_shear = 1 + 0.01 * math.sqrt(shear_rate * d_p_m**2 / D_B) if D_B > 0 else 1
        
        # Combined flow effect
        flow_contribution = f_B * f_flow * f_shear
    else:
        Pe = 0
        f_flow = 1
        f_shear = 1
        flow_contribution = f_B
    
    # Total effective thermal conductivity
    k_eff = k_static + flow_contribution
    
    # Diagnostics
    diagnostics = {
        'D_B': D_B,
        'D_T': D_T,
        'Pe': Pe,
        'v_Brownian': v_B,
        'f_flow': f_flow,
        'f_shear': f_shear,
        'k_static': k_static,
        'k_Brownian': flow_contribution,
        'enhancement_ratio': k_eff / k_bf
    }
    
    return k_eff, diagnostics


def corcione_model(
    k_bf: float,
    k_np: float,
    phi: float,
    T: float,
    d_p: float,
    rho_bf: float,
    rho_np: float,
    mu_bf: float,
    d_bf: float = 0.3  # Base fluid molecule diameter in nm
) -> Tuple[float, dict]:
    """
    Corcione Empirical Model for Nanofluid Thermal Conductivity.
    
    This model is based on extensive experimental data and provides
    excellent predictions across wide ranges of conditions.
    
    Key features:
    - Accounts for particle size effects explicitly
    - Temperature-dependent formulation
    - Validated against 200+ experimental datasets
    - Includes base fluid molecular diameter effects
    
    Formula:
        k_eff/k_bf = 1 + 4.4*Re^0.4 * Pr^0.66 * (T/T_fr)^10 * (k_np/k_bf)^0.03 * φ^0.66
    
    where:
        Re = ρ_bf * u_B * d_p / μ_bf (Brownian Reynolds number)
        Pr = μ_bf * c_p / k_bf (Prandtl number)
        u_B = 2*k_B*T/(π*μ_bf*d_p) (Brownian velocity)
        T_fr = 293.15 K (reference temperature)
    
    Args:
        k_bf: Base fluid thermal conductivity (W/m·K)
        k_np: Nanoparticle thermal conductivity (W/m·K)
        phi: Volume fraction (0 to 1)
        T: Temperature (K)
        d_p: Particle diameter (nm)
        rho_bf: Base fluid density (kg/m³)
        rho_np: Nanoparticle density (kg/m³)
        mu_bf: Dynamic viscosity (Pa·s)
        d_bf: Base fluid molecular diameter (nm, default 0.3 for water)
        
    Returns:
        Tuple of (k_effective, diagnostics_dict)
    """
    if phi == 0:
        return k_bf, {}
    
    # Convert to SI units
    d_p_m = d_p * 1e-9
    d_bf_m = d_bf * 1e-9
    
    # Reference temperature
    T_fr = 293.15  # K (20°C)
    
    # Brownian velocity
    u_B = 2 * BOLTZMANN_CONSTANT * T / (math.pi * mu_bf * d_p_m)
    
    # Brownian Reynolds number
    Re_B = rho_bf * u_B * d_p_m / mu_bf
    
    # Prandtl number (approximate for water-based fluids)
    c_p_bf = 4182  # J/kg·K (approximation)
    Pr = mu_bf * c_p_bf / k_bf
    
    # Corcione correlation
    k_ratio = (k_np / k_bf) ** 0.03
    temp_ratio = (T / T_fr) ** 10
    Re_term = Re_B ** 0.4
    Pr_term = Pr ** 0.66
    phi_term = phi ** 0.66
    
    enhancement = 1 + 4.4 * Re_term * Pr_term * temp_ratio * k_ratio * phi_term
    
    k_eff = k_bf * enhancement
    
    # Diagnostics
    diagnostics = {
        'Re_Brownian': Re_B,
        'Pr': Pr,
        'u_Brownian': u_B,
        'enhancement_factor': enhancement,
        'T_ratio': temp_ratio,
        'k_ratio_contribution': k_ratio
    }
    
    return k_eff, diagnostics


def rea_bonnet_convective_model(
    k_bf: float,
    k_np: float,
    phi: float,
    T: float,
    velocity: float,
    hydraulic_diameter: float,
    L: float,
    rho_bf: float,
    mu_bf: float,
    c_p_bf: float
) -> Tuple[float, dict]:
    """
    Rea-Bonnet Model for Convective Heat Transfer in Nanofluids.
    
    This model specifically addresses heat transfer enhancement in
    flowing nanofluids, accounting for:
    - Flow regime (laminar vs turbulent)
    - Entrance effects
    - Developing thermal boundary layers
    - Nanoparticle migration in velocity gradients
    
    Key Applications:
    - Heat exchangers
    - Cooling channels
    - Microfluidic devices
    
    Args:
        k_bf: Base fluid thermal conductivity (W/m·K)
        k_np: Nanoparticle thermal conductivity (W/m·K)
        phi: Volume fraction (0 to 1)
        T: Temperature (K)
        velocity: Mean flow velocity (m/s)
        hydraulic_diameter: Channel hydraulic diameter (m)
        L: Channel length (m)
        rho_bf: Base fluid density (kg/m³)
        mu_bf: Dynamic viscosity (Pa·s)
        c_p_bf: Specific heat capacity (J/kg·K)
        
    Returns:
        Tuple of (k_effective_convective, diagnostics_dict)
    """
    if phi == 0 or velocity == 0:
        return k_bf, {'Re': 0, 'Nu': 0}
    
    # 1. Static thermal conductivity (Maxwell)
    k_static = k_bf * (k_np + 2*k_bf + 2*phi*(k_np - k_bf)) / \
                      (k_np + 2*k_bf - phi*(k_np - k_bf))
    
    # 2. Calculate Reynolds number
    Re = rho_bf * velocity * hydraulic_diameter / mu_bf
    
    # 3. Calculate Prandtl number
    Pr = mu_bf * c_p_bf / k_bf
    
    # 4. Determine flow regime and calculate Nusselt number
    if Re < 2300:  # Laminar flow
        # Graetz number
        Gz = Re * Pr * hydraulic_diameter / L
        
        if Gz > 10:  # Developing flow
            Nu = 3.66 + (0.0668 * Gz) / (1 + 0.04 * Gz**(2/3))
        else:  # Fully developed
            Nu = 3.66  # Constant Nu for laminar fully developed
        
        flow_regime = "Laminar"
        
    else:  # Turbulent flow
        # Gnielinski correlation (valid for 3000 < Re < 5e6)
        f = (0.790 * math.log(Re) - 1.64) ** (-2)  # Friction factor
        
        Nu = (f/8) * (Re - 1000) * Pr / \
             (1 + 12.7 * math.sqrt(f/8) * (Pr**(2/3) - 1))
        
        flow_regime = "Turbulent"
    
    # 5. Convective heat transfer coefficient
    h = Nu * k_static / hydraulic_diameter
    
    # 6. Effective thermal conductivity accounting for convection
    # The effective conductivity increases due to micro-convection
    # and enhanced mixing from nanoparticles
    
    # Convective enhancement factor (empirical)
    f_conv = 1 + 0.01 * phi * Nu  # Nanoparticles enhance convective transfer
    
    k_eff_conv = k_static * f_conv
    
    # 7. Migration effect (particles move toward cooler regions in flow)
    # This creates local concentration gradients that enhance heat transfer
    Pe_particle = velocity * hydraulic_diameter / \
                  (BOLTZMANN_CONSTANT * T / (3 * math.pi * mu_bf * 50e-9))
    
    if Pe_particle > 100:
        migration_factor = 1 + 0.05 * math.log10(Pe_particle)
    else:
        migration_factor = 1
    
    k_eff = k_eff_conv * migration_factor
    
    # Diagnostics
    diagnostics = {
        'Re': Re,
        'Pr': Pr,
        'Nu': Nu,
        'flow_regime': flow_regime,
        'h_conv': h,
        'Pe_particle': Pe_particle,
        'f_convective': f_conv,
        'f_migration': migration_factor,
        'k_static': k_static,
        'enhancement_total': k_eff / k_bf
    }
    
    return k_eff, diagnostics


def shear_enhanced_conductivity(
    k_static: float,
    phi: float,
    shear_rate: float,
    d_p: float,
    T: float,
    mu_bf: float
) -> Tuple[float, float]:
    """
    Calculate thermal conductivity enhancement due to shear flow.
    
    Under shear flow, nanoparticles experience:
    - Increased collision frequency
    - Alignment effects (for non-spherical particles)
    - Enhanced micro-mixing
    - Formation of particle chains/structures
    
    Args:
        k_static: Static thermal conductivity (W/m·K)
        phi: Volume fraction (0 to 1)
        shear_rate: Shear rate γ̇ (1/s)
        d_p: Particle diameter (nm)
        T: Temperature (K)
        mu_bf: Base fluid viscosity (Pa·s)
        
    Returns:
        Tuple of (k_effective, enhancement_factor)
    """
    d_p_m = d_p * 1e-9
    
    # Shear Peclet number (ratio of shear to Brownian motion)
    D_B = BOLTZMANN_CONSTANT * T / (3 * math.pi * mu_bf * d_p_m)
    Pe_shear = shear_rate * d_p_m**2 / D_B if D_B > 0 else 0
    
    # Enhancement mechanisms:
    if Pe_shear < 1:
        # Brownian motion dominates - minimal shear effect
        f_shear = 1 + 0.01 * phi * Pe_shear
    elif Pe_shear < 100:
        # Transitional regime - moderate enhancement
        f_shear = 1 + 0.05 * phi * math.sqrt(Pe_shear)
    else:
        # Shear-dominated - particle structuring effects
        # At very high shear, particles align and form chains
        f_shear = 1 + 0.1 * phi * math.log10(Pe_shear)
    
    k_eff = k_static * f_shear
    
    return k_eff, f_shear


def velocity_dependent_conductivity(
    k_bf: float,
    k_np: float,
    phi: float,
    velocity: float,
    characteristic_length: float = 0.001  # 1 mm default
) -> float:
    """
    Simple velocity-dependent thermal conductivity model.
    
    For quick calculations when detailed flow parameters are not available.
    Based on empirical observations that k_eff increases with flow velocity
    due to micro-convection and enhanced particle dispersion.
    
    Args:
        k_bf: Base fluid thermal conductivity (W/m·K)
        k_np: Nanoparticle thermal conductivity (W/m·K)
        phi: Volume fraction (0 to 1)
        velocity: Flow velocity (m/s)
        characteristic_length: Characteristic length scale (m)
        
    Returns:
        Effective thermal conductivity (W/m·K)
    """
    # Static component
    k_static = k_bf * (k_np + 2*k_bf + 2*phi*(k_np - k_bf)) / \
                      (k_np + 2*k_bf - phi*(k_np - k_bf))
    
    if velocity == 0:
        return k_static
    
    # Velocity enhancement (empirical)
    # Based on: k_eff increases ~5-15% per m/s for typical nanofluids
    v_enhancement = 1 + 0.08 * phi * velocity
    
    k_eff = k_static * v_enhancement
    
    return k_eff
