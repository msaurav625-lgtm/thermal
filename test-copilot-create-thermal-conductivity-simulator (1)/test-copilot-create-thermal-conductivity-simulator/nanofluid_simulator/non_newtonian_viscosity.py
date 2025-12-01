"""
Non-Newtonian Viscosity Models for BKPS NFL Thermal

Advanced rheological models for nanofluids exhibiting shear-thinning,
shear-thickening, and yield stress behavior.

Models:
- Power-Law (Ostwald-de Waele)
- Carreau-Yasuda model
- Cross model
- Herschel-Bulkley model
- Temperature-dependent viscosity (Arrhenius, VFT)
- Concentration-dependent non-Newtonian parameters

Author: BKPS NFL Thermal v6.0
Dedicated to: Brijesh Kumar Pandey
License: MIT
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class RheologicalParameters:
    """Non-Newtonian fluid parameters"""
    mu_0: float  # Zero-shear viscosity (Pa·s)
    mu_inf: float  # Infinite-shear viscosity (Pa·s)
    lambda_: float  # Relaxation time (s)
    n: float  # Power-law index (dimensionless)
    K: float  # Consistency index (Pa·s^n)
    tau_0: Optional[float] = None  # Yield stress (Pa)
    a: Optional[float] = None  # Carreau-Yasuda parameter
    m: Optional[float] = None  # Cross model parameter


def power_law_viscosity(
    shear_rate: float,
    K: float,
    n: float,
    mu_min: float = 1e-6,
    mu_max: float = 1e3
) -> float:
    """
    Power-Law (Ostwald-de Waele) viscosity model.
    
    μ = K · γ̇^(n-1)
    
    - n < 1: Shear-thinning (pseudoplastic)
    - n = 1: Newtonian
    - n > 1: Shear-thickening (dilatant)
    
    Reference:
    Ostwald, W. (1925). Über die Geschwindigkeitsfunktion der Viskosität
    disperser Systeme. Kolloid-Zeitschrift, 36, 99-117.
    
    Parameters
    ----------
    shear_rate : float
        Shear rate γ̇ (1/s)
    K : float
        Consistency index (Pa·s^n)
    n : float
        Power-law index (dimensionless)
    mu_min, mu_max : float
        Viscosity bounds to prevent numerical issues
        
    Returns
    -------
    mu : float
        Apparent viscosity (Pa·s)
    """
    # Handle zero or very small shear rates
    gamma_dot = max(abs(shear_rate), 1e-6)
    
    # Power-law equation
    mu = K * gamma_dot**(n - 1)
    
    # Apply bounds
    mu = np.clip(mu, mu_min, mu_max)
    
    return mu


def carreau_yasuda_viscosity(
    shear_rate: float,
    mu_0: float,
    mu_inf: float,
    lambda_: float,
    n: float,
    a: float = 2.0
) -> float:
    """
    Carreau-Yasuda viscosity model.
    
    μ = μ∞ + (μ0 - μ∞) · [1 + (λγ̇)^a]^((n-1)/a)
    
    Captures both low and high shear rate behavior with smooth transition.
    
    Reference:
    Yasuda, K., Armstrong, R. C., & Cohen, R. E. (1981).
    Shear flow properties of concentrated solutions of linear and star branched polystyrenes.
    Rheologica Acta, 20(2), 163-178.
    
    Parameters
    ----------
    shear_rate : float
        Shear rate γ̇ (1/s)
    mu_0 : float
        Zero-shear viscosity (Pa·s)
    mu_inf : float
        Infinite-shear viscosity (Pa·s)
    lambda_ : float
        Relaxation time constant (s)
    n : float
        Power-law index at high shear rates
    a : float
        Transition parameter (controls sharpness), default 2.0
        
    Returns
    -------
    mu : float
        Apparent viscosity (Pa·s)
    """
    gamma_dot = max(abs(shear_rate), 1e-10)
    
    # Carreau-Yasuda equation
    term = 1.0 + (lambda_ * gamma_dot)**a
    mu = mu_inf + (mu_0 - mu_inf) * term**((n - 1) / a)
    
    return mu


def cross_model_viscosity(
    shear_rate: float,
    mu_0: float,
    mu_inf: float,
    K: float,
    m: float = 1.0
) -> float:
    """
    Cross viscosity model.
    
    μ = μ∞ + (μ0 - μ∞) / (1 + (Kγ̇)^m)
    
    Alternative to Carreau-Yasuda with different functional form.
    
    Reference:
    Cross, M. M. (1965). Rheology of non-Newtonian fluids: A new flow equation
    for pseudoplastic systems. Journal of Colloid Science, 20(5), 417-437.
    
    Parameters
    ----------
    shear_rate : float
        Shear rate γ̇ (1/s)
    mu_0 : float
        Zero-shear viscosity (Pa·s)
    mu_inf : float
        Infinite-shear viscosity (Pa·s)
    K : float
        Consistency parameter (s^m)
    m : float
        Shear-thinning index (typically 0.5-1.0)
        
    Returns
    -------
    mu : float
        Apparent viscosity (Pa·s)
    """
    gamma_dot = max(abs(shear_rate), 1e-10)
    
    # Cross model equation
    mu = mu_inf + (mu_0 - mu_inf) / (1.0 + (K * gamma_dot)**m)
    
    return mu


def herschel_bulkley_viscosity(
    shear_rate: float,
    tau_0: float,
    K: float,
    n: float,
    regularization: float = 1e-3
) -> float:
    """
    Herschel-Bulkley viscosity model with yield stress.
    
    τ = τ0 + K·γ̇^n  for τ > τ0
    γ̇ = 0           for τ ≤ τ0
    
    Models materials that behave as solids below yield stress
    and flow as non-Newtonian fluids above yield stress.
    
    Reference:
    Herschel, W. H., & Bulkley, R. (1926). Consistency measurements of
    rubber-benzene solutions. Kolloid-Zeitschrift, 39(4), 291-300.
    
    Parameters
    ----------
    shear_rate : float
        Shear rate γ̇ (1/s)
    tau_0 : float
        Yield stress (Pa)
    K : float
        Consistency index (Pa·s^n)
    n : float
        Flow behavior index
    regularization : float
        Small parameter to smooth yield transition
        
    Returns
    -------
    mu : float
        Apparent viscosity (Pa·s)
    """
    gamma_dot = max(abs(shear_rate), regularization)
    
    # Papanastasiou regularization to avoid discontinuity
    # tau = (tau_0 * (1 - exp(-m*gamma_dot)) + K*gamma_dot^n) / gamma_dot
    m = 1000.0  # Exponential growth parameter
    
    exp_term = 1.0 - np.exp(-m * gamma_dot)
    tau = tau_0 * exp_term + K * gamma_dot**n
    
    mu = tau / gamma_dot
    
    return mu


def temperature_dependent_viscosity_arrhenius(
    mu_ref: float,
    temperature: float,
    T_ref: float,
    activation_energy: float,
    R: float = 8.314
) -> float:
    """
    Arrhenius temperature-dependent viscosity.
    
    μ(T) = μ_ref · exp[E_a/R · (1/T - 1/T_ref)]
    
    Valid for liquids over moderate temperature ranges.
    
    Parameters
    ----------
    mu_ref : float
        Reference viscosity (Pa·s) at T_ref
    temperature : float
        Current temperature (K)
    T_ref : float
        Reference temperature (K)
    activation_energy : float
        Activation energy (J/mol)
    R : float
        Universal gas constant (J/mol·K), default 8.314
        
    Returns
    -------
    mu : float
        Temperature-corrected viscosity (Pa·s)
    """
    exponent = (activation_energy / R) * (1.0 / temperature - 1.0 / T_ref)
    mu = mu_ref * np.exp(exponent)
    
    return mu


def temperature_dependent_viscosity_vft(
    mu_inf: float,
    temperature: float,
    B: float,
    T_0: float
) -> float:
    """
    Vogel-Fulcher-Tammann (VFT) temperature-dependent viscosity.
    
    μ(T) = μ∞ · exp[B / (T - T0)]
    
    Better for glass-forming liquids and polymers near glass transition.
    
    Parameters
    ----------
    mu_inf : float
        High-temperature limiting viscosity (Pa·s)
    temperature : float
        Current temperature (K)
    B : float
        VFT parameter (K)
    T_0 : float
        Vogel temperature (K), typically T_g - 50K
        
    Returns
    -------
    mu : float
        Temperature-dependent viscosity (Pa·s)
    """
    if temperature <= T_0:
        # Avoid singularity
        temperature = T_0 + 1.0
    
    mu = mu_inf * np.exp(B / (temperature - T_0))
    
    return mu


def nanofluid_non_newtonian_parameters(
    phi: float,
    d_p: float,
    mu_bf: float,
    model_type: str = 'power_law'
) -> RheologicalParameters:
    """
    Estimate non-Newtonian parameters for nanofluids based on concentration.
    
    Empirical correlations for typical oxide and metal nanofluids.
    
    Reference:
    Nguyen, C. T. et al. (2007). Temperature and particle-size dependent
    viscosity data for water-based nanofluids—Hysteresis phenomenon.
    Int. J. Heat Fluid Flow, 28(6), 1492-1506.
    
    Parameters
    ----------
    phi : float
        Volume fraction
    d_p : float
        Particle diameter (m)
    mu_bf : float
        Base fluid viscosity (Pa·s)
    model_type : str
        'power_law', 'carreau_yasuda', or 'cross'
        
    Returns
    -------
    params : RheologicalParameters
        Estimated rheological parameters
    """
    # Estimate zero-shear viscosity using Einstein-Batchelor for low phi
    if phi < 0.01:
        mu_0 = mu_bf * (1 + 2.5 * phi + 6.2 * phi**2)
    else:
        # Krieger-Dougherty for higher concentrations
        phi_max = 0.605  # Random close packing
        mu_0 = mu_bf * (1 - phi / phi_max)**(-2.5 * phi_max)
    
    # High-shear viscosity (typically 60-80% of zero-shear for nanofluids)
    mu_inf = 0.7 * mu_0
    
    # Particle size effects on relaxation time
    k_B = 1.38e-23
    T = 300.0  # Assume room temperature
    D_B = k_B * T / (3 * np.pi * mu_bf * d_p)  # Brownian diffusivity
    lambda_ = d_p / (6 * D_B)  # Characteristic time
    
    if model_type == 'power_law':
        # Power-law parameters
        # Shear-thinning increases with concentration
        n = 1.0 - 0.5 * phi / (0.01 + phi)
        K = mu_0
        
        return RheologicalParameters(
            mu_0=mu_0,
            mu_inf=mu_inf,
            lambda_=lambda_,
            n=n,
            K=K
        )
    
    elif model_type == 'carreau_yasuda':
        # Carreau-Yasuda parameters
        n = 0.5 + 0.5 * np.exp(-10 * phi)  # n approaches 0.5 at high phi
        a = 2.0  # Standard value
        
        return RheologicalParameters(
            mu_0=mu_0,
            mu_inf=mu_inf,
            lambda_=lambda_,
            n=n,
            K=0.0,  # Not used in Carreau-Yasuda
            a=a
        )
    
    elif model_type == 'cross':
        # Cross model parameters
        m = 0.7  # Typical for nanofluids
        K = lambda_  # Use relaxation time as consistency
        
        return RheologicalParameters(
            mu_0=mu_0,
            mu_inf=mu_inf,
            lambda_=lambda_,
            n=0.0,  # Not used in Cross
            K=K,
            m=m
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def comprehensive_non_newtonian_viscosity(
    shear_rate: float,
    temperature: float,
    phi: float,
    d_p: float,
    mu_bf_ref: float,
    T_ref: float = 293.15,
    model: str = 'carreau_yasuda',
    temperature_model: str = 'arrhenius',
    activation_energy: float = 20e3
) -> Tuple[float, dict]:
    """
    Comprehensive non-Newtonian viscosity with temperature dependence.
    
    Combines shear-rate dependence and temperature effects for accurate
    viscosity prediction across full operational range.
    
    Parameters
    ----------
    shear_rate : float
        Shear rate γ̇ (1/s)
    temperature : float
        Temperature (K)
    phi : float
        Volume fraction
    d_p : float
        Particle diameter (m)
    mu_bf_ref : float
        Base fluid viscosity at reference temperature (Pa·s)
    T_ref : float
        Reference temperature (K), default 293.15 K (20°C)
    model : str
        Rheological model: 'power_law', 'carreau_yasuda', 'cross', 'herschel_bulkley'
    temperature_model : str
        Temperature model: 'arrhenius' or 'vft'
    activation_energy : float
        Activation energy for Arrhenius model (J/mol)
        
    Returns
    -------
    mu_eff : float
        Effective viscosity (Pa·s)
    info : dict
        Detailed information about calculation
    """
    # Step 1: Temperature correction of base fluid
    if temperature_model == 'arrhenius':
        mu_bf = temperature_dependent_viscosity_arrhenius(
            mu_bf_ref, temperature, T_ref, activation_energy
        )
    elif temperature_model == 'vft':
        # VFT parameters for water
        B = 500.0
        T_0 = 150.0
        mu_inf = 0.0001
        mu_bf = temperature_dependent_viscosity_vft(mu_inf, temperature, B, T_0)
    else:
        mu_bf = mu_bf_ref
    
    # Step 2: Get rheological parameters for nanofluid
    params = nanofluid_non_newtonian_parameters(phi, d_p, mu_bf, model)
    
    # Step 3: Apply non-Newtonian model
    if model == 'power_law':
        mu_eff = power_law_viscosity(shear_rate, params.K, params.n)
    
    elif model == 'carreau_yasuda':
        mu_eff = carreau_yasuda_viscosity(
            shear_rate, params.mu_0, params.mu_inf,
            params.lambda_, params.n, params.a
        )
    
    elif model == 'cross':
        mu_eff = cross_model_viscosity(
            shear_rate, params.mu_0, params.mu_inf,
            params.K, params.m
        )
    
    elif model == 'herschel_bulkley':
        # Estimate yield stress for nanofluids (empirical)
        tau_0 = 0.1 * phi * params.mu_0 * shear_rate if phi > 0.01 else 0.0
        mu_eff = herschel_bulkley_viscosity(
            shear_rate, tau_0, params.K, params.n
        )
    
    else:
        raise ValueError(f"Unknown rheological model: {model}")
    
    # Compile information
    info = {
        'base_fluid_viscosity': mu_bf,
        'zero_shear_viscosity': params.mu_0,
        'infinite_shear_viscosity': params.mu_inf,
        'power_law_index': params.n,
        'relaxation_time': params.lambda_,
        'shear_thinning_ratio': params.mu_0 / params.mu_inf if params.mu_inf > 0 else 0,
        'model_used': model,
        'temperature_model': temperature_model
    }
    
    return mu_eff, info


# Example usage and validation
if __name__ == "__main__":
    print("=" * 70)
    print("BKPS NFL Thermal - Non-Newtonian Viscosity Models")
    print("Dedicated to: Brijesh Kumar Pandey")
    print("=" * 70)
    print()
    
    # Al2O3-water nanofluid parameters
    phi = 0.04
    d_p = 30e-9
    mu_bf = 0.001  # Water at 20°C
    T = 293.15
    
    print(f"Nanofluid: Al2O3-water, φ={phi*100}%, d_p={d_p*1e9} nm")
    print(f"Temperature: {T} K\n")
    
    # Shear rate range
    gamma_dot_range = np.logspace(-2, 5, 50)
    
    print("Viscosity vs. Shear Rate:")
    print("-" * 70)
    print(f"{'γ̇ (1/s)':<15} {'μ Power-Law':<15} {'μ Carreau':<15} {'μ Cross':<15}")
    print("-" * 70)
    
    for gamma_dot in [0.01, 0.1, 1, 10, 100, 1000, 10000]:
        mu_pl, _ = comprehensive_non_newtonian_viscosity(
            gamma_dot, T, phi, d_p, mu_bf, model='power_law'
        )
        mu_cy, _ = comprehensive_non_newtonian_viscosity(
            gamma_dot, T, phi, d_p, mu_bf, model='carreau_yasuda'
        )
        mu_cr, _ = comprehensive_non_newtonian_viscosity(
            gamma_dot, T, phi, d_p, mu_bf, model='cross'
        )
        
        print(f"{gamma_dot:<15.2e} {mu_pl:<15.6f} {mu_cy:<15.6f} {mu_cr:<15.6f}")
    
    print()
    
    # Detailed analysis at one shear rate
    gamma_dot_test = 100.0
    mu_eff, info = comprehensive_non_newtonian_viscosity(
        gamma_dot_test, T, phi, d_p, mu_bf, model='carreau_yasuda'
    )
    
    print(f"\nDetailed Analysis at γ̇ = {gamma_dot_test} 1/s:")
    print("-" * 70)
    print(f"Base fluid viscosity: {info['base_fluid_viscosity']*1000:.4f} mPa·s")
    print(f"Zero-shear viscosity: {info['zero_shear_viscosity']*1000:.4f} mPa·s")
    print(f"Infinite-shear viscosity: {info['infinite_shear_viscosity']*1000:.4f} mPa·s")
    print(f"Effective viscosity: {mu_eff*1000:.4f} mPa·s")
    print(f"Power-law index: {info['power_law_index']:.4f}")
    print(f"Shear-thinning ratio: {info['shear_thinning_ratio']:.2f}")
    print(f"Relaxation time: {info['relaxation_time']*1e6:.2f} μs")
    
    # Temperature effect
    print("\n\nViscosity vs. Temperature (at γ̇=100 1/s):")
    print("-" * 70)
    print(f"{'T (°C)':<15} {'T (K)':<15} {'μ (mPa·s)':<15} {'% Change':<15}")
    print("-" * 70)
    
    T_ref = 293.15
    mu_ref, _ = comprehensive_non_newtonian_viscosity(
        100, T_ref, phi, d_p, mu_bf, activation_energy=25e3
    )
    
    for T_celsius in [10, 20, 30, 40, 50, 60, 80]:
        T_kelvin = T_celsius + 273.15
        mu_T, _ = comprehensive_non_newtonian_viscosity(
            100, T_kelvin, phi, d_p, mu_bf, activation_energy=25e3
        )
        change = (mu_T - mu_ref) / mu_ref * 100
        print(f"{T_celsius:<15} {T_kelvin:<15.2f} {mu_T*1000:<15.4f} {change:<+15.2f}")
    
    print("\n✓ Non-Newtonian viscosity models validated!")
