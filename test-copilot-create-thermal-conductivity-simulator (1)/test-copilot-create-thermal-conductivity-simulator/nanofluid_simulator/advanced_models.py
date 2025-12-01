"""
Advanced Thermal Conductivity Models for Nanofluids

This module implements state-of-the-art models including temperature-dependent,
Brownian motion, and hybrid nanofluid models.

References:
1. Patel, H.E. et al. (2003). Applied Physics Letters.
2. Koo, J. and Kleinstreuer, C. (2004). Journal of Nanoparticle Research.
3. Hajjar, Z. et al. (2014). International Communications in Heat and Mass Transfer.
4. Sundar, L.S. et al. (2017). International Communications in Heat and Mass Transfer.
5. Esfe, M.H. et al. (2015). International Communications in Heat and Mass Transfer.
6. Xue, Q.Z. (2003). Physics Letters A - Interfacial layer model.
7. Leong, K.C. and Yang, C. (2006). International Journal of Heat and Mass Transfer.
"""

import math
from typing import Optional, Dict, List, Tuple


# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K


def patel_model(
    k_bf: float,
    k_np: float,
    phi: float,
    T: float,
    T_ref: float = 298.15,
    alpha: float = 0.001
) -> float:
    """
    Patel Temperature-Dependent Model for thermal conductivity.
    
    This model accounts for temperature effects on thermal conductivity
    enhancement, showing that enhancement increases with temperature
    due to intensified Brownian motion.
    
    Formula:
        k_eff/k_bf = (k_np + 2*k_bf + 2*phi*(k_np - k_bf)) / 
                     (k_np + 2*k_bf - phi*(k_np - k_bf)) * 
                     [1 + alpha*(T - T_ref)]
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np: Thermal conductivity of nanoparticles (W/m·K)
        phi: Volume fraction of nanoparticles (0 to 1)
        T: Temperature (K)
        T_ref: Reference temperature (K, default 298.15 K = 25°C)
        alpha: Temperature coefficient (default 0.001 K⁻¹)
        
    Returns:
        Effective thermal conductivity of the nanofluid (W/m·K)
    """
    if k_bf <= 0 or k_np <= 0:
        raise ValueError("Thermal conductivities must be positive")
    if not 0 <= phi <= 1:
        raise ValueError("Volume fraction must be between 0 and 1")
    if T <= 0:
        raise ValueError("Temperature must be positive")
    
    # Base Maxwell model
    numerator = k_np + 2 * k_bf + 2 * phi * (k_np - k_bf)
    denominator = k_np + 2 * k_bf - phi * (k_np - k_bf)
    k_maxwell = k_bf * (numerator / denominator)
    
    # Temperature enhancement factor
    temp_factor = 1 + alpha * (T - T_ref)
    
    return k_maxwell * temp_factor


def koo_kleinstreuer_model(
    k_bf: float,
    k_np: float,
    phi: float,
    T: float,
    particle_diameter: float,
    rho_bf: float,
    rho_np: float,
    cp_bf: float,
    mu_bf: float
) -> float:
    """
    Koo-Kleinstreuer Model including Brownian motion effects.
    
    This model explicitly accounts for Brownian motion of nanoparticles,
    which creates micro-convection and enhances thermal conductivity.
    
    Formula:
        k_eff = k_static + k_Brownian
        k_static: Classical Maxwell model
        k_Brownian: Contribution from Brownian motion
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np: Thermal conductivity of nanoparticles (W/m·K)
        phi: Volume fraction of nanoparticles (0 to 1)
        T: Temperature (K)
        particle_diameter: Particle diameter (nm)
        rho_bf: Density of base fluid (kg/m³)
        rho_np: Density of nanoparticles (kg/m³)
        cp_bf: Specific heat of base fluid (J/kg·K)
        mu_bf: Dynamic viscosity of base fluid (Pa·s)
        
    Returns:
        Effective thermal conductivity of the nanofluid (W/m·K)
    """
    if any(x <= 0 for x in [k_bf, k_np, T, particle_diameter, rho_bf, rho_np, cp_bf, mu_bf]):
        raise ValueError("All physical properties must be positive")
    if not 0 <= phi <= 1:
        raise ValueError("Volume fraction must be between 0 and 1")
    
    # Convert particle diameter from nm to m
    d_p = particle_diameter * 1e-9
    
    # Static component (Maxwell model)
    numerator = k_np + 2 * k_bf + 2 * phi * (k_np - k_bf)
    denominator = k_np + 2 * k_bf - phi * (k_np - k_bf)
    k_static = k_bf * (numerator / denominator)
    
    # Brownian motion component
    # Mean free path of fluid molecules (approximation)
    # For liquids, this is typically on the order of angstroms
    # Using simplified approximation
    
    # Brownian velocity
    v_B = math.sqrt(18 * BOLTZMANN_CONSTANT * T / (math.pi * rho_np * d_p**3))
    
    # Reynolds number for Brownian motion
    Re_B = rho_bf * v_B * d_p / mu_bf
    
    # Prandtl number
    Pr = mu_bf * cp_bf / k_bf
    
    # Empirical function for Brownian motion contribution
    # Based on Koo & Kleinstreuer's work
    beta = 0.0017  # Empirical constant
    
    # Brownian contribution
    k_Brownian = 5e4 * beta * phi * rho_bf * cp_bf * v_B * \
                 math.sqrt(BOLTZMANN_CONSTANT * T / (rho_np * d_p))
    
    return k_static + k_Brownian


def chon_model(
    k_bf: float,
    k_np: float,
    phi: float,
    T: float,
    particle_diameter: float,
    mu_bf: float,
    rho_bf: float
) -> float:
    """
    Chon et al. Model combining particle size and temperature effects.
    
    This empirical model accounts for both particle size and temperature
    through Prandtl number, Reynolds number, and particle volume fraction.
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np: Thermal conductivity of nanoparticles (W/m·K)
        phi: Volume fraction of nanoparticles (0 to 1)
        T: Temperature (K)
        particle_diameter: Particle diameter (nm)
        mu_bf: Dynamic viscosity of base fluid (Pa·s)
        rho_bf: Density of base fluid (kg/m³)
        
    Returns:
        Effective thermal conductivity of the nanofluid (W/m·K)
    """
    if any(x <= 0 for x in [k_bf, k_np, T, particle_diameter, mu_bf, rho_bf]):
        raise ValueError("All physical properties must be positive")
    if not 0 <= phi <= 1:
        raise ValueError("Volume fraction must be between 0 and 1")
    
    # Convert particle diameter from nm to m
    d_p = particle_diameter * 1e-9
    
    # Molecular mean free path for water (approximation)
    lambda_bf = 0.17e-9  # m
    
    # Prandtl number (approximation for water)
    cp_bf = 4182  # J/kg·K (water)
    Pr = mu_bf * cp_bf / k_bf
    
    # Brownian velocity
    rho_np = 8933  # Approximation (copper)
    v_B = math.sqrt(18 * BOLTZMANN_CONSTANT * T / (math.pi * rho_np * d_p**3))
    
    # Reynolds number
    Re = rho_bf * v_B * d_p / mu_bf
    
    # Chon correlation
    k_ratio = k_np / k_bf
    
    k_eff = k_bf * (1 + 64.7 * phi**0.746 * (d_p / lambda_bf)**0.369 * 
                    k_ratio**0.7476 * Pr**0.9955 * Re**1.2321)
    
    return k_eff


def hybrid_nanofluid_model(
    k_bf: float,
    nanoparticles: List[Dict[str, float]],
    model_type: str = "maxwell"
) -> float:
    """
    Model for hybrid nanofluids with multiple nanoparticle types.
    
    This function calculates thermal conductivity for hybrid nanofluids
    containing two or more types of nanoparticles.
    
    Formula (simplified approach):
        Calculate equivalent nanoparticle properties based on mixture rule,
        then apply selected model with equivalent properties.
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        nanoparticles: List of dictionaries, each containing:
                      - 'k': thermal conductivity (W/m·K)
                      - 'phi': volume fraction (0 to 1)
                      - 'density': density (kg/m³)
                      - 'cp': specific heat (J/kg·K)
        model_type: Model to use for calculation ("maxwell", "hamilton_crosser")
        
    Returns:
        Effective thermal conductivity of the hybrid nanofluid (W/m·K)
    """
    if not nanoparticles:
        raise ValueError("At least one nanoparticle must be specified")
    
    # Calculate total volume fraction
    phi_total = sum(np['phi'] for np in nanoparticles)
    
    if phi_total <= 0:
        return k_bf
    
    if phi_total > 1:
        raise ValueError("Total volume fraction cannot exceed 1")
    
    # Calculate effective nanoparticle properties (weighted average)
    k_np_eff = 0
    rho_np_eff = 0
    cp_np_eff = 0
    
    for np in nanoparticles:
        weight = np['phi'] / phi_total
        k_np_eff += np['k'] * weight
        rho_np_eff += np['density'] * weight
        cp_np_eff += np['cp'] * weight
    
    # Apply selected model with effective properties
    if model_type.lower() == "maxwell":
        numerator = k_np_eff + 2 * k_bf + 2 * phi_total * (k_np_eff - k_bf)
        denominator = k_np_eff + 2 * k_bf - phi_total * (k_np_eff - k_bf)
        k_eff = k_bf * (numerator / denominator)
    elif model_type.lower() == "hamilton_crosser":
        n = 3  # Spherical particles
        numerator = k_np_eff + (n - 1) * k_bf - (n - 1) * phi_total * (k_bf - k_np_eff)
        denominator = k_np_eff + (n - 1) * k_bf + phi_total * (k_bf - k_np_eff)
        k_eff = k_bf * (numerator / denominator)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return k_eff


def hajjar_hybrid_model(
    k_bf: float,
    k_np1: float,
    k_np2: float,
    phi1: float,
    phi2: float,
    sphericity: float = 1.0
) -> float:
    """
    Hajjar et al. Model for binary hybrid nanofluids.
    
    Specifically designed for hybrid nanofluids with two nanoparticle types.
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np1: Thermal conductivity of first nanoparticle (W/m·K)
        k_np2: Thermal conductivity of second nanoparticle (W/m·K)
        phi1: Volume fraction of first nanoparticle (0 to 1)
        phi2: Volume fraction of second nanoparticle (0 to 1)
        sphericity: Particle sphericity (0 to 1, default 1)
        
    Returns:
        Effective thermal conductivity of the hybrid nanofluid (W/m·K)
    """
    if any(k <= 0 for k in [k_bf, k_np1, k_np2]):
        raise ValueError("Thermal conductivities must be positive")
    if not 0 <= phi1 <= 1 or not 0 <= phi2 <= 1:
        raise ValueError("Volume fractions must be between 0 and 1")
    if phi1 + phi2 > 1:
        raise ValueError("Total volume fraction cannot exceed 1")
    if not 0 < sphericity <= 1:
        raise ValueError("Sphericity must be between 0 (exclusive) and 1 (inclusive)")
    
    phi_total = phi1 + phi2
    
    if phi_total == 0:
        return k_bf
    
    # Weighted average thermal conductivity
    k_np_eff = (phi1 * k_np1 + phi2 * k_np2) / phi_total
    
    # Shape factor
    n = 3 / sphericity
    
    # Modified Hamilton-Crosser for hybrid nanofluid
    numerator = k_np_eff + (n - 1) * k_bf - (n - 1) * phi_total * (k_bf - k_np_eff)
    denominator = k_np_eff + (n - 1) * k_bf + phi_total * (k_bf - k_np_eff)
    
    k_eff = k_bf * (numerator / denominator)
    
    return k_eff


def esfe_hybrid_model(
    k_bf: float,
    k_np1: float,
    k_np2: float,
    phi1: float,
    phi2: float,
    T: float,
    T_ref: float = 298.15
) -> float:
    """
    Esfe et al. Temperature-Dependent Model for hybrid nanofluids.
    
    Empirical model that includes temperature effects in hybrid nanofluids.
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np1: Thermal conductivity of first nanoparticle (W/m·K)
        k_np2: Thermal conductivity of second nanoparticle (W/m·K)
        phi1: Volume fraction of first nanoparticle (0 to 1)
        phi2: Volume fraction of second nanoparticle (0 to 1)
        T: Temperature (K)
        T_ref: Reference temperature (K, default 298.15 K)
        
    Returns:
        Effective thermal conductivity of the hybrid nanofluid (W/m·K)
    """
    if any(k <= 0 for k in [k_bf, k_np1, k_np2]):
        raise ValueError("Thermal conductivities must be positive")
    if not 0 <= phi1 <= 1 or not 0 <= phi2 <= 1:
        raise ValueError("Volume fractions must be between 0 and 1")
    if phi1 + phi2 > 1:
        raise ValueError("Total volume fraction cannot exceed 1")
    if T <= 0:
        raise ValueError("Temperature must be positive")
    
    phi_total = phi1 + phi2
    
    if phi_total == 0:
        return k_bf
    
    # Effective nanoparticle thermal conductivity
    k_np_eff = (phi1 * k_np1 + phi2 * k_np2) / phi_total
    
    # Base Maxwell model
    numerator = k_np_eff + 2 * k_bf + 2 * phi_total * (k_np_eff - k_bf)
    denominator = k_np_eff + 2 * k_bf - phi_total * (k_np_eff - k_bf)
    k_base = k_bf * (numerator / denominator)
    
    # Temperature enhancement (empirical)
    temp_enhancement = 1 + 0.0025 * (T - T_ref) * phi_total
    
    return k_base * temp_enhancement


def sundar_hybrid_model(
    k_bf: float,
    k_np1: float,
    k_np2: float,
    phi1: float,
    phi2: float,
    T: float,
    particle_size1: float = 25.0,
    particle_size2: float = 25.0
) -> float:
    """
    Sundar et al. Comprehensive Model for hybrid nanofluids.
    
    This model considers particle size distribution along with temperature
    and composition effects for hybrid nanofluids.
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np1: Thermal conductivity of first nanoparticle (W/m·K)
        k_np2: Thermal conductivity of second nanoparticle (W/m·K)
        phi1: Volume fraction of first nanoparticle (0 to 1)
        phi2: Volume fraction of second nanoparticle (0 to 1)
        T: Temperature (K)
        particle_size1: Average particle size of first nanoparticle (nm)
        particle_size2: Average particle size of second nanoparticle (nm)
        
    Returns:
        Effective thermal conductivity of the hybrid nanofluid (W/m·K)
    """
    if any(k <= 0 for k in [k_bf, k_np1, k_np2]):
        raise ValueError("Thermal conductivities must be positive")
    if not 0 <= phi1 <= 1 or not 0 <= phi2 <= 1:
        raise ValueError("Volume fractions must be between 0 and 1")
    if phi1 + phi2 > 1:
        raise ValueError("Total volume fraction cannot exceed 1")
    if T <= 0:
        raise ValueError("Temperature must be positive")
    if particle_size1 <= 0 or particle_size2 <= 0:
        raise ValueError("Particle sizes must be positive")
    
    phi_total = phi1 + phi2
    
    if phi_total == 0:
        return k_bf
    
    # Weighted properties
    k_np_eff = (phi1 * k_np1 + phi2 * k_np2) / phi_total
    d_eff = (phi1 * particle_size1 + phi2 * particle_size2) / phi_total
    
    # Base enhancement
    numerator = k_np_eff + 2 * k_bf + 2 * phi_total * (k_np_eff - k_bf)
    denominator = k_np_eff + 2 * k_bf - phi_total * (k_np_eff - k_bf)
    k_base = k_bf * (numerator / denominator)
    
    # Size-dependent correction (smaller particles = higher enhancement)
    size_factor = 1 + 0.01 * (50.0 / d_eff)
    
    # Temperature factor
    temp_factor = 1 + 0.002 * (T - 298.15)
    
    # Synergistic effect for hybrid (empirical)
    synergy = 1.0
    if phi1 > 0 and phi2 > 0:
        synergy = 1 + 0.15 * min(phi1, phi2) / phi_total
    
    return k_base * size_factor * temp_factor * synergy


def takabi_salehi_model(
    k_bf: float,
    k_np1: float,
    k_np2: float,
    phi1: float,
    phi2: float,
    T: float,
    mu_bf: float,
    rho_bf: float,
    rho_np1: float,
    rho_np2: float,
    d_p1: float,
    d_p2: float
) -> float:
    """
    Takabi and Salehi Model for hybrid nanofluids with Brownian motion.
    
    Advanced model incorporating Brownian motion effects for hybrid nanofluids.
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np1: Thermal conductivity of first nanoparticle (W/m·K)
        k_np2: Thermal conductivity of second nanoparticle (W/m·K)
        phi1: Volume fraction of first nanoparticle (0 to 1)
        phi2: Volume fraction of second nanoparticle (0 to 1)
        T: Temperature (K)
        mu_bf: Dynamic viscosity of base fluid (Pa·s)
        rho_bf: Density of base fluid (kg/m³)
        rho_np1: Density of first nanoparticle (kg/m³)
        rho_np2: Density of second nanoparticle (kg/m³)
        d_p1: Diameter of first nanoparticle (nm)
        d_p2: Diameter of second nanoparticle (nm)
        
    Returns:
        Effective thermal conductivity of the hybrid nanofluid (W/m·K)
    """
    phi_total = phi1 + phi2
    
    if phi_total == 0:
        return k_bf
    
    # Effective properties
    k_np_eff = (phi1 * k_np1 + phi2 * k_np2) / phi_total
    rho_np_eff = (phi1 * rho_np1 + phi2 * rho_np2) / phi_total
    d_p_eff = (phi1 * d_p1 + phi2 * d_p2) / phi_total
    
    # Convert diameter to meters
    d_p = d_p_eff * 1e-9
    
    # Static component (Maxwell)
    numerator = k_np_eff + 2 * k_bf + 2 * phi_total * (k_np_eff - k_bf)
    denominator = k_np_eff + 2 * k_bf - phi_total * (k_np_eff - k_bf)
    k_static = k_bf * (numerator / denominator)
    
    # Brownian component
    v_B = math.sqrt(18 * BOLTZMANN_CONSTANT * T / (math.pi * rho_np_eff * d_p**3))
    
    # Empirical Brownian enhancement (simplified)
    c_p_bf = 4182  # J/kg·K (approximation for water-based fluids)
    k_Brownian = 5e4 * 0.001 * phi_total * rho_bf * c_p_bf * v_B * \
                 math.sqrt(BOLTZMANN_CONSTANT * T / (rho_np_eff * d_p))
    
    return k_static + k_Brownian


def xue_interfacial_layer_model(
    k_bf: float,
    k_np: float,
    phi: float,
    beta: float = 0.1
) -> float:
    """
    Xue Interfacial Layer Model for thermal conductivity.
    
    This model accounts for the interfacial nanolayer between nanoparticles
    and base fluid. The nanolayer has different thermal properties than both
    the bulk nanoparticle and bulk fluid, significantly affecting heat transfer.
    
    Key Physics:
    - Interfacial layer thickness typically 1-10 nm
    - Layer conductivity intermediate between particle and fluid
    - Critical for understanding enhancement mechanisms
    
    Formula:
        k_eff/k_bf = [k_np(1+2β)³ + 2k_bf + 2φ(1+β)³(k_np - k_bf)] /
                     [k_np(1+2β)³ + 2k_bf - φ(1+β)³(k_np - k_bf)]
    
    where β = h/r is the ratio of nanolayer thickness (h) to particle radius (r)
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np: Thermal conductivity of nanoparticles (W/m·K)
        phi: Volume fraction of nanoparticles (0 to 1)
        beta: Interfacial layer thickness ratio (h/r, default 0.1)
              Typical values: 0.05-0.2
        
    Returns:
        Effective thermal conductivity (W/m·K)
        
    Reference:
        Xue, Q.Z. (2003). "Model for effective thermal conductivity of nanofluids"
        Physics Letters A, 307(5-6), 313-317.
    """
    if phi == 0:
        return k_bf
    
    # Composite particle volume fraction (particle + layer)
    gamma = (1 + beta) ** 3
    phi_c = gamma * phi
    
    # Composite particle thermal conductivity
    # Assumes layer conductivity is geometric mean of k_bf and k_np
    k_layer = math.sqrt(k_bf * k_np)
    
    # Two-layer composite sphere
    numerator_inner = k_np + 2 * k_layer - 2 * (k_layer - k_np)
    denominator_inner = k_np + 2 * k_layer + (k_layer - k_np)
    k_composite = k_layer * (numerator_inner / denominator_inner)
    
    # Maxwell equation with composite particles
    numerator = k_composite + 2 * k_bf + 2 * phi_c * (k_composite - k_bf)
    denominator = k_composite + 2 * k_bf - phi_c * (k_composite - k_bf)
    
    k_eff = k_bf * (numerator / denominator)
    
    return k_eff


def leong_yang_interfacial_model(
    k_bf: float,
    k_np: float,
    phi: float,
    d_p: float,
    h: float = 2.0,
    k_layer_ratio: float = 2.0
) -> float:
    """
    Leong-Yang Interfacial Nanolayer Model with explicit layer properties.
    
    This model explicitly considers the interfacial nanolayer with its own
    thermal conductivity, which can be higher or lower than the base fluid
    depending on molecular ordering at the interface.
    
    Key Features:
    - Explicit nanolayer thickness (typically 1-5 nm)
    - Adjustable layer conductivity
    - Accounts for particle size effects
    - Suitable for oxide nanoparticles in polar fluids
    
    Formula:
        k_eff = k_bf * [k_np(1+2h/d_p)³ + 2k_layer + 2φ(1+h/d_p)³(k_layer - k_bf)] /
                       [k_np(1+2h/d_p)³ + 2k_layer - φ(1+h/d_p)³(k_layer - k_bf)]
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np: Thermal conductivity of nanoparticles (W/m·K)
        phi: Volume fraction of nanoparticles (0 to 1)
        d_p: Particle diameter (nm)
        h: Nanolayer thickness (nm, default 2.0)
           Typical values: 0.5-5.0 nm
        k_layer_ratio: Ratio of k_layer/k_bf (default 2.0)
                      Values: 1.5-5.0 for ordered layers
                              0.5-1.0 for disordered layers
        
    Returns:
        Effective thermal conductivity (W/m·K)
        
    Reference:
        Leong, K.C., Yang, C., & Murshed, S.M.S. (2006). 
        "A model for the thermal conductivity of nanofluids"
        International Journal of Heat and Mass Transfer, 49(21-22), 4317-4323.
    """
    if phi == 0:
        return k_bf
    
    # Interfacial layer conductivity
    k_layer = k_layer_ratio * k_bf
    
    # Beta: layer thickness ratio
    beta = 2 * h / d_p  # Factor of 2 because h is thickness, need ratio to radius
    
    # Volume ratio of composite particle
    gamma = (1 + beta) ** 3
    
    # Equivalent thermal conductivity of composite sphere (particle + layer)
    # Using series resistance model
    r_p = d_p / 2  # particle radius (nm)
    r_c = r_p + h  # composite radius (nm)
    
    # Thermal resistance approach
    R_p = r_p / (k_np * 4 * math.pi * r_p**2)
    R_layer = h / (k_layer * 4 * math.pi * ((r_p + h)**2))
    R_total = R_p + R_layer
    
    # Equivalent conductivity
    k_eq = r_c / (R_total * 4 * math.pi * r_c**2)
    
    # Modified Maxwell model with composite particles
    phi_eff = gamma * phi
    
    numerator = k_eq + 2 * k_bf + 2 * phi_eff * (k_eq - k_bf)
    denominator = k_eq + 2 * k_bf - phi_eff * (k_eq - k_bf)
    
    k_eff = k_bf * (numerator / denominator)
    
    return k_eff


def yu_choi_interfacial_model(
    k_bf: float,
    k_np: float,
    phi: float,
    beta: float = 0.1
) -> float:
    """
    Yu-Choi Modified Model with Interfacial Layer.
    
    Extension of Yu-Choi model accounting for nanolayer effects.
    Particularly accurate for metal oxide nanoparticles in water.
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np: Thermal conductivity of nanoparticles (W/m·K)
        phi: Volume fraction of nanoparticles (0 to 1)
        beta: Nanolayer thickness ratio (default 0.1)
        
    Returns:
        Effective thermal conductivity (W/m·K)
    """
    if phi == 0:
        return k_bf
    
    # Effective volume fraction including nanolayer
    gamma = (1 + beta) ** 3
    phi_eff = gamma * phi
    
    # Shape factor for spherical particles
    n = 3
    
    # Modified Yu-Choi equation
    numerator = k_np + (n - 1) * k_bf + (n - 1) * phi_eff * (k_np - k_bf)
    denominator = k_np + (n - 1) * k_bf - phi_eff * (k_np - k_bf)
    
    k_eff = k_bf * (numerator / denominator)
    
    return k_eff

