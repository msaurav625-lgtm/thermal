"""
Thermal Conductivity Models for Nanofluids

This module implements various theoretical and empirical models for calculating
the effective thermal conductivity of nanofluids (suspensions of nanoparticles
in a base fluid).

References:
1. Maxwell, J.C. (1881). A Treatise on Electricity and Magnetism.
2. Hamilton, R.L. and Crosser, O.K. (1962). Industrial & Engineering Chemistry Fundamentals.
3. Bruggeman, D.A.G. (1935). Annalen der Physik.
4. Yu, W. and Choi, S.U.S. (2003). Journal of Nanoparticle Research.
5. Wasp, E.J. et al. (1977). Solid-Liquid Flow Slurry Pipeline Transportation.
6. Pak, B.C. and Cho, Y.I. (1998). Experimental Heat Transfer.
"""

import math
from typing import Optional


def validate_inputs(
    k_bf: float,
    k_np: float,
    phi: float,
    name: str = "model"
) -> None:
    """
    Validate input parameters for thermal conductivity models.
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np: Thermal conductivity of nanoparticles (W/m·K)
        phi: Volume fraction of nanoparticles (0 to 1)
        name: Name of the model for error messages
        
    Raises:
        ValueError: If any input parameter is invalid
    """
    if k_bf <= 0:
        raise ValueError(f"{name}: Base fluid thermal conductivity must be positive")
    if k_np <= 0:
        raise ValueError(f"{name}: Nanoparticle thermal conductivity must be positive")
    if not 0 <= phi <= 1:
        raise ValueError(f"{name}: Volume fraction must be between 0 and 1")


def maxwell_model(k_bf: float, k_np: float, phi: float) -> float:
    """
    Maxwell Model for effective thermal conductivity of nanofluids.
    
    The Maxwell model is valid for low volume fractions of spherical particles
    with no particle interaction. It is one of the earliest and most widely
    used models for predicting thermal conductivity of dilute suspensions.
    
    Formula:
        k_eff/k_bf = (k_np + 2*k_bf + 2*phi*(k_np - k_bf)) / 
                     (k_np + 2*k_bf - phi*(k_np - k_bf))
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np: Thermal conductivity of nanoparticles (W/m·K)
        phi: Volume fraction of nanoparticles (0 to 1)
        
    Returns:
        Effective thermal conductivity of the nanofluid (W/m·K)
        
    Example:
        >>> k_eff = maxwell_model(k_bf=0.613, k_np=401, phi=0.01)
        >>> print(f"Effective thermal conductivity: {k_eff:.4f} W/m·K")
    """
    validate_inputs(k_bf, k_np, phi, "Maxwell model")
    
    numerator = k_np + 2 * k_bf + 2 * phi * (k_np - k_bf)
    denominator = k_np + 2 * k_bf - phi * (k_np - k_bf)
    
    return k_bf * (numerator / denominator)


def hamilton_crosser_model(
    k_bf: float,
    k_np: float,
    phi: float,
    n: Optional[float] = None,
    sphericity: float = 1.0
) -> float:
    """
    Hamilton-Crosser Model for effective thermal conductivity.
    
    This model extends Maxwell's model to account for non-spherical particles
    through a shape factor. It is widely used for both spherical and
    non-spherical nanoparticles.
    
    Formula:
        k_eff/k_bf = (k_np + (n-1)*k_bf - (n-1)*phi*(k_bf - k_np)) /
                     (k_np + (n-1)*k_bf + phi*(k_bf - k_np))
        
        where n = 3/sphericity for non-spherical particles
        For spheres: n = 3, sphericity = 1
        For cylinders: sphericity ≈ 0.5, n ≈ 6
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np: Thermal conductivity of nanoparticles (W/m·K)
        phi: Volume fraction of nanoparticles (0 to 1)
        n: Shape factor (optional, calculated from sphericity if not provided)
        sphericity: Particle sphericity (0 to 1, default 1 for spheres)
        
    Returns:
        Effective thermal conductivity of the nanofluid (W/m·K)
        
    Example:
        >>> k_eff = hamilton_crosser_model(k_bf=0.613, k_np=35, phi=0.02, sphericity=0.5)
        >>> print(f"Effective thermal conductivity: {k_eff:.4f} W/m·K")
    """
    validate_inputs(k_bf, k_np, phi, "Hamilton-Crosser model")
    
    if not 0 < sphericity <= 1:
        raise ValueError("Sphericity must be between 0 (exclusive) and 1 (inclusive)")
    
    if n is None:
        n = 3 / sphericity
    
    numerator = k_np + (n - 1) * k_bf - (n - 1) * phi * (k_bf - k_np)
    denominator = k_np + (n - 1) * k_bf + phi * (k_bf - k_np)
    
    return k_bf * (numerator / denominator)


def bruggeman_model(k_bf: float, k_np: float, phi: float) -> float:
    """
    Bruggeman Model for effective thermal conductivity.
    
    The Bruggeman model considers random distribution of spherical particles
    and includes the interaction between particles. It is valid for higher
    volume fractions compared to Maxwell's model.
    
    Implicit Formula:
        (1 - phi) * (k_bf - k_eff)/(k_bf + 2*k_eff) + 
        phi * (k_np - k_eff)/(k_np + 2*k_eff) = 0
    
    This implementation solves the equation iteratively.
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np: Thermal conductivity of nanoparticles (W/m·K)
        phi: Volume fraction of nanoparticles (0 to 1)
        
    Returns:
        Effective thermal conductivity of the nanofluid (W/m·K)
        
    Example:
        >>> k_eff = bruggeman_model(k_bf=0.613, k_np=401, phi=0.05)
        >>> print(f"Effective thermal conductivity: {k_eff:.4f} W/m·K")
    """
    validate_inputs(k_bf, k_np, phi, "Bruggeman model")
    
    if phi == 0:
        return k_bf
    
    # Use Newton-Raphson method to solve the implicit equation
    k_eff = k_bf  # Initial guess
    tolerance = 1e-10
    max_iterations = 100
    
    for _ in range(max_iterations):
        # Bruggeman equation: f(k_eff) = 0
        term1 = (1 - phi) * (k_bf - k_eff) / (k_bf + 2 * k_eff)
        term2 = phi * (k_np - k_eff) / (k_np + 2 * k_eff)
        f = term1 + term2
        
        # Derivative of f with respect to k_eff
        d_term1 = (1 - phi) * (-3 * k_bf) / (k_bf + 2 * k_eff) ** 2
        d_term2 = phi * (-3 * k_np) / (k_np + 2 * k_eff) ** 2
        df = d_term1 + d_term2
        
        if abs(df) < 1e-15:
            break
            
        k_new = k_eff - f / df
        
        if k_new <= 0:
            k_new = k_eff / 2
            
        if abs(k_new - k_eff) < tolerance:
            return k_new
            
        k_eff = k_new
    
    return k_eff


def yu_choi_model(
    k_bf: float,
    k_np: float,
    phi: float,
    layer_thickness: float = 1.0,
    particle_radius: float = 10.0,
    layer_conductivity_factor: float = 10.0
) -> float:
    """
    Yu and Choi Model for effective thermal conductivity.
    
    This model accounts for the nanolayer (interfacial layer) effect around
    nanoparticles. The nanolayer is a semi-solid layer of base fluid molecules
    around the nanoparticle that has higher thermal conductivity than bulk fluid.
    
    Formula:
        k_eff/k_bf = (k_np_eff + 2*k_bf + 2*(k_np_eff - k_bf)*(1 + beta)^3 * phi) /
                     (k_np_eff + 2*k_bf - (k_np_eff - k_bf)*(1 + beta)^3 * phi)
        
        where:
        beta = layer_thickness / particle_radius
        k_np_eff = effective conductivity of particle + nanolayer system
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np: Thermal conductivity of nanoparticles (W/m·K)
        phi: Volume fraction of nanoparticles (0 to 1)
        layer_thickness: Thickness of the nanolayer (nm)
        particle_radius: Radius of nanoparticles (nm)
        layer_conductivity_factor: Factor relating nanolayer conductivity to base 
                                   fluid conductivity (k_layer = factor * k_bf).
                                   Typically ranges from 2 to 100. Default is 10.
        
    Returns:
        Effective thermal conductivity of the nanofluid (W/m·K)
        
    Example:
        >>> k_eff = yu_choi_model(k_bf=0.613, k_np=401, phi=0.01, 
        ...                       layer_thickness=2, particle_radius=25)
        >>> print(f"Effective thermal conductivity: {k_eff:.4f} W/m·K")
    """
    validate_inputs(k_bf, k_np, phi, "Yu and Choi model")
    
    if layer_thickness < 0:
        raise ValueError("Layer thickness must be non-negative")
    if particle_radius <= 0:
        raise ValueError("Particle radius must be positive")
    if layer_conductivity_factor <= 0:
        raise ValueError("Layer conductivity factor must be positive")
    
    beta = layer_thickness / particle_radius
    
    # Effective volume fraction including nanolayer
    phi_eff = phi * (1 + beta) ** 3
    
    # Nanolayer conductivity based on configurable factor
    k_layer = layer_conductivity_factor * k_bf
    
    # Effective conductivity of particle + layer composite
    gamma = k_layer / k_np
    
    k_np_eff = k_np * (
        (2 * (1 - gamma) + (1 + beta) ** 3 * (1 + 2 * gamma) * gamma) /
        ((1 - gamma) + (1 + beta) ** 3 * (1 + 2 * gamma))
    ) if k_np != k_layer else k_np
    
    # Modified Maxwell equation with effective values
    numerator = k_np_eff + 2 * k_bf + 2 * (k_np_eff - k_bf) * phi_eff
    denominator = k_np_eff + 2 * k_bf - (k_np_eff - k_bf) * phi_eff
    
    return k_bf * (numerator / denominator)


def wasp_model(k_bf: float, k_np: float, phi: float) -> float:
    """
    Wasp Model for effective thermal conductivity.
    
    The Wasp model is a simplified version of Hamilton-Crosser for spherical
    particles. It is identical to Maxwell's model for spherical particles.
    
    Formula:
        k_eff/k_bf = (k_np + 2*k_bf - 2*phi*(k_bf - k_np)) /
                     (k_np + 2*k_bf + phi*(k_bf - k_np))
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        k_np: Thermal conductivity of nanoparticles (W/m·K)
        phi: Volume fraction of nanoparticles (0 to 1)
        
    Returns:
        Effective thermal conductivity of the nanofluid (W/m·K)
        
    Example:
        >>> k_eff = wasp_model(k_bf=0.613, k_np=401, phi=0.01)
        >>> print(f"Effective thermal conductivity: {k_eff:.4f} W/m·K")
    """
    validate_inputs(k_bf, k_np, phi, "Wasp model")
    
    numerator = k_np + 2 * k_bf - 2 * phi * (k_bf - k_np)
    denominator = k_np + 2 * k_bf + phi * (k_bf - k_np)
    
    return k_bf * (numerator / denominator)


def pak_cho_correlation(
    k_bf: float,
    phi: float,
    nanoparticle_type: str = "Al2O3"
) -> float:
    """
    Pak and Cho Empirical Correlation for thermal conductivity.
    
    This is an empirical correlation based on experimental data for
    specific nanofluid systems. It provides a linear relationship
    between thermal conductivity enhancement and volume fraction.
    
    Formula:
        k_eff/k_bf = 1 + C * phi
        
        where C is an empirical constant depending on the nanoparticle type:
        - Al2O3 (Alumina): C ≈ 7.47
        - TiO2 (Titanium dioxide): C ≈ 2.92
        - CuO (Copper oxide): C ≈ 9.0
        - Cu (Copper): C ≈ 15.0
        - Ag (Silver): C ≈ 18.0
    
    Args:
        k_bf: Thermal conductivity of base fluid (W/m·K)
        phi: Volume fraction of nanoparticles (0 to 1)
        nanoparticle_type: Type of nanoparticle ("Al2O3", "TiO2", "CuO", "Cu", "Ag")
        
    Returns:
        Effective thermal conductivity of the nanofluid (W/m·K)
        
    Example:
        >>> k_eff = pak_cho_correlation(k_bf=0.613, phi=0.03, nanoparticle_type="Cu")
        >>> print(f"Effective thermal conductivity: {k_eff:.4f} W/m·K")
    """
    if k_bf <= 0:
        raise ValueError("Base fluid thermal conductivity must be positive")
    if not 0 <= phi <= 1:
        raise ValueError("Volume fraction must be between 0 and 1")
    
    # Empirical constants for different nanoparticle types
    empirical_constants = {
        "Al2O3": 7.47,
        "TiO2": 2.92,
        "CuO": 9.0,
        "Cu": 15.0,
        "Ag": 18.0,
        "SiO2": 3.0,
        "Fe2O3": 6.0,
        "ZnO": 5.5,
        "CNT": 25.0,  # Carbon nanotubes
        "Graphene": 30.0,
    }
    
    nanoparticle_upper = nanoparticle_type.upper()
    matching_key = None
    
    for key in empirical_constants:
        if key.upper() == nanoparticle_upper:
            matching_key = key
            break
    
    if matching_key is None:
        available = ", ".join(empirical_constants.keys())
        raise ValueError(
            f"Unknown nanoparticle type: {nanoparticle_type}. "
            f"Available types: {available}"
        )
    
    C = empirical_constants[matching_key]
    
    return k_bf * (1 + C * phi)
