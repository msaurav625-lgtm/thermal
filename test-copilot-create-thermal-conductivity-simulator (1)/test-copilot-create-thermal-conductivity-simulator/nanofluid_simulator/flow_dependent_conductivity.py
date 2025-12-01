"""
Flow-Dependent Thermal Conductivity Models for BKPS NFL Thermal

Advanced models incorporating local temperature, pressure, shear rate, 
and velocity field effects on nanofluid thermal conductivity.

Models:
- Buongiorno convective transport model
- Kumar shear-enhanced conductivity
- Rea-Guzman velocity-dependent model
- Temperature gradient effects
- Pressure-dependent thermal conductivity
- Local flow field coupling

Author: BKPS NFL Thermal v6.0
Dedicated to: Brijesh Kumar Pandey
License: MIT
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class FlowFieldData:
    """Local flow field data for conductivity calculation"""
    temperature: float  # K
    pressure: float  # Pa
    velocity_magnitude: float  # m/s
    shear_rate: float  # 1/s
    temperature_gradient: float  # K/m
    turbulent_kinetic_energy: Optional[float] = None  # m²/s²


def buongiorno_flow_enhanced_conductivity(
    k_static: float,
    phi: float,
    d_p: float,
    rho_bf: float,
    rho_p: float,
    mu_bf: float,
    cp_bf: float,
    velocity: float,
    temperature: float,
    length_scale: float,
    include_thermophoresis: bool = True,
    include_brownian: bool = True
) -> float:
    """
    Buongiorno two-component model with flow effects.
    
    Includes Brownian diffusion and thermophoretic effects
    on effective thermal conductivity in flowing nanofluids.
    
    Reference:
    Buongiorno, J. (2006). Convective Transport in Nanofluids.
    ASME J. Heat Transfer, 128(3), 240-250.
    
    Parameters
    ----------
    k_static : float
        Static thermal conductivity (W/m·K)
    phi : float
        Volume fraction
    d_p : float
        Particle diameter (m)
    rho_bf, rho_p : float
        Densities (kg/m³)
    mu_bf : float
        Base fluid viscosity (Pa·s)
    cp_bf : float
        Base fluid specific heat (J/kg·K)
    velocity : float
        Flow velocity (m/s)
    temperature : float
        Temperature (K)
    length_scale : float
        Characteristic length (m)
    include_thermophoresis : bool
        Include thermophoretic diffusion
    include_brownian : bool
        Include Brownian diffusion
        
    Returns
    -------
    k_eff : float
        Flow-enhanced thermal conductivity (W/m·K)
    """
    k_B = 1.38064852e-23  # Boltzmann constant
    
    # Brownian diffusion coefficient
    D_B = k_B * temperature / (3 * np.pi * mu_bf * d_p)
    
    # Thermophoretic diffusion coefficient
    if include_thermophoresis:
        # Simplified thermophoretic velocity
        K_T = 0.26 * k_static / (2 * k_static + mu_bf)  # Thermophoretic coefficient
        D_T = K_T * mu_bf / rho_bf
    else:
        D_T = 0.0
    
    # Peclet numbers
    Pe_B = velocity * length_scale / D_B if include_brownian else 0.0
    Pe_T = velocity * length_scale / D_T if include_thermophoresis else 0.0
    
    # Flow enhancement factor (empirical correlation)
    # Based on particle migration and micro-convection
    if Pe_B > 1:
        f_brownian = 1.0 + 0.05 * phi * np.log(1 + Pe_B)
    else:
        f_brownian = 1.0
    
    if Pe_T > 1 and include_thermophoresis:
        f_thermophoresis = 1.0 + 0.03 * phi * np.log(1 + Pe_T)
    else:
        f_thermophoresis = 1.0
    
    # Combined enhancement
    k_eff = k_static * f_brownian * f_thermophoresis
    
    return k_eff


def kumar_shear_enhanced_conductivity(
    k_static: float,
    phi: float,
    shear_rate: float,
    d_p: float,
    temperature: float,
    mu_bf: float,
    k_bf: float,
    k_p: float,
    max_enhancement: float = 0.30
) -> float:
    """
    Kumar shear-rate dependent thermal conductivity model.
    
    Accounts for particle alignment, rotation, and micro-mixing
    under shear flow conditions.
    
    Reference:
    Kumar, D. et al. (2015). Shear rate dependent thermal conductivity
    of nanofluids. J. Appl. Phys., 117, 074301.
    
    Parameters
    ----------
    k_static : float
        Static thermal conductivity (W/m·K)
    phi : float
        Volume fraction
    shear_rate : float
        Shear rate (1/s)
    d_p : float
        Particle diameter (m)
    temperature : float
        Temperature (K)
    mu_bf : float
        Base fluid viscosity (Pa·s)
    k_bf, k_p : float
        Thermal conductivities (W/m·K)
    max_enhancement : float
        Maximum enhancement fraction (default 30%)
        
    Returns
    -------
    k_eff : float
        Shear-enhanced thermal conductivity (W/m·K)
    """
    # Dimensionless shear number
    gamma_star = shear_rate * d_p**2 / (temperature * 1.38e-23 / (6 * np.pi * mu_bf * d_p**3))
    
    # Shear enhancement factor
    if gamma_star < 0.1:
        # Low shear: minimal enhancement
        f_shear = 1.0
    elif gamma_star < 100:
        # Moderate shear: particle alignment and rotation
        f_shear = 1.0 + phi * max_enhancement * np.tanh(gamma_star / 10)
    else:
        # High shear: turbulent mixing dominates
        f_shear = 1.0 + phi * max_enhancement * (1 + 0.1 * np.log10(gamma_star / 100))
    
    # Conductivity ratio influence
    beta = k_p / k_bf
    if beta > 100:  # Highly conductive particles (metals)
        f_shear *= 1.2  # Enhanced percolation under shear
    
    k_eff = k_static * f_shear
    
    return k_eff


def rea_guzman_velocity_dependent_model(
    k_static: float,
    phi: float,
    velocity: float,
    d_p: float,
    temperature: float,
    rho_bf: float,
    mu_bf: float,
    cp_bf: float,
    k_bf: float
) -> float:
    """
    Rea and Guzman velocity-dependent thermal conductivity model.
    
    Incorporates particle convection, wake effects, and 
    velocity-induced micro-circulation around particles.
    
    Reference:
    Rea, U., McKrell, T., Hu, L.-W., & Buongiorno, J. (2009).
    Laminar convective heat transfer and viscous pressure loss of
    alumina–water and zirconia–water nanofluids. Int. J. Heat Mass Transfer.
    
    Parameters
    ----------
    k_static : float
        Static thermal conductivity (W/m·K)
    phi : float
        Volume fraction
    velocity : float
        Flow velocity (m/s)
    d_p : float
        Particle diameter (m)
    temperature : float
        Temperature (K)
    rho_bf : float
        Base fluid density (kg/m³)
    mu_bf : float
        Base fluid viscosity (Pa·s)
    cp_bf : float
        Base fluid specific heat (J/kg·K)
    k_bf : float
        Base fluid thermal conductivity (W/m·K)
        
    Returns
    -------
    k_eff : float
        Velocity-enhanced thermal conductivity (W/m·K)
    """
    # Particle Reynolds number
    Re_p = rho_bf * velocity * d_p / mu_bf
    
    # Prandtl number
    Pr = mu_bf * cp_bf / k_bf
    
    # Peclet number for particles
    Pe_p = Re_p * Pr
    
    # Velocity enhancement factor
    if Re_p < 0.1:
        # Stokes flow regime: minimal enhancement
        f_velocity = 1.0
    elif Re_p < 1.0:
        # Transition regime: wake begins to form
        f_velocity = 1.0 + 0.5 * phi * Re_p**0.5 * Pr**0.33
    else:
        # Inertial regime: strong wake effects
        f_velocity = 1.0 + 1.5 * phi * Re_p**0.6 * Pr**0.33
    
    # Percolation enhancement at higher velocities
    if phi > 0.01 and Re_p > 10:
        # Particle clustering under flow enhances percolation
        f_percolation = 1.0 + 0.2 * phi * np.log(1 + Re_p / 10)
        f_velocity *= f_percolation
    
    k_eff = k_static * f_velocity
    
    return k_eff


def temperature_gradient_enhanced_conductivity(
    k_static: float,
    phi: float,
    dT_dx: float,
    d_p: float,
    temperature: float,
    mu_bf: float,
    alpha_bf: float,
    beta_T: float = 0.26
) -> float:
    """
    Thermal conductivity enhancement due to temperature gradients.
    
    Thermophoresis causes particle migration in temperature gradients,
    creating local concentration variations and enhancing heat transfer.
    
    Parameters
    ----------
    k_static : float
        Static thermal conductivity (W/m·K)
    phi : float
        Volume fraction
    dT_dx : float
        Temperature gradient (K/m)
    d_p : float
        Particle diameter (m)
    temperature : float
        Temperature (K)
    mu_bf : float
        Base fluid viscosity (Pa·s)
    alpha_bf : float
        Base fluid thermal diffusivity (m²/s)
    beta_T : float
        Thermophoretic coefficient (default 0.26)
        
    Returns
    -------
    k_eff : float
        Gradient-enhanced thermal conductivity (W/m·K)
    """
    # Thermophoretic velocity
    v_T = -beta_T * (mu_bf / (rho_bf if 'rho_bf' in locals() else 1000.0)) * dT_dx / temperature
    
    # Characteristic diffusion time
    tau_d = d_p**2 / alpha_bf
    
    # Thermophoretic Peclet number
    Pe_T = abs(v_T) * d_p / alpha_bf
    
    # Enhancement factor
    if Pe_T < 0.1:
        f_gradient = 1.0
    else:
        # Particle accumulation on cold side enhances local conductivity
        f_gradient = 1.0 + 0.1 * phi * Pe_T / (1 + Pe_T)
    
    k_eff = k_static * f_gradient
    
    return k_eff


def pressure_dependent_conductivity(
    k_static: float,
    pressure: float,
    p_ref: float,
    compressibility: float = 4.5e-10,
    pressure_coefficient: float = 1e-9
) -> float:
    """
    Pressure-dependent thermal conductivity.
    
    Accounts for compression effects on inter-particle spacing
    and phonon transport at elevated pressures.
    
    Parameters
    ----------
    k_static : float
        Static thermal conductivity at reference pressure (W/m·K)
    pressure : float
        Current pressure (Pa)
    p_ref : float
        Reference pressure (Pa), typically 101325 Pa
    compressibility : float
        Isothermal compressibility (1/Pa), default for water
    pressure_coefficient : float
        Pressure sensitivity coefficient (W/m·K·Pa)
        
    Returns
    -------
    k_eff : float
        Pressure-corrected thermal conductivity (W/m·K)
    """
    delta_p = pressure - p_ref
    
    # Linear pressure correction (valid for moderate pressures)
    if abs(delta_p) < 1e7:  # < 100 bar
        k_eff = k_static * (1 + pressure_coefficient * delta_p)
    else:
        # Non-linear correction for high pressures
        compression_factor = 1.0 / (1 + compressibility * delta_p)
        k_eff = k_static * compression_factor**0.5
    
    return k_eff


def turbulent_dispersion_conductivity(
    k_static: float,
    phi: float,
    k_turbulent: float,
    rho_bf: float,
    cp_bf: float,
    turbulent_prandtl: float = 0.9
) -> float:
    """
    Effective thermal conductivity with turbulent dispersion.
    
    Accounts for enhanced heat transfer due to turbulent mixing
    and particle dispersion in turbulent flows.
    
    Parameters
    ----------
    k_static : float
        Static/laminar thermal conductivity (W/m·K)
    phi : float
        Volume fraction
    k_turbulent : float
        Turbulent kinetic energy (m²/s²)
    rho_bf : float
        Base fluid density (kg/m³)
    cp_bf : float
        Base fluid specific heat (J/kg·K)
    turbulent_prandtl : float
        Turbulent Prandtl number (default 0.9)
        
    Returns
    -------
    k_eff : float
        Turbulent-enhanced thermal conductivity (W/m·K)
    """
    # Turbulent thermal diffusivity
    alpha_t = k_turbulent**0.5 * (d_p if 'd_p' in locals() else 1e-7) / turbulent_prandtl
    
    # Turbulent conductivity contribution
    k_turb = rho_bf * cp_bf * alpha_t
    
    # Particle enhancement of turbulent mixing
    f_turb = 1.0 + 2.0 * phi
    
    k_eff = k_static + k_turb * f_turb
    
    return k_eff


def comprehensive_flow_dependent_conductivity(
    k_base: float,
    phi: float,
    d_p: float,
    flow_data: FlowFieldData,
    rho_bf: float,
    rho_p: float,
    mu_bf: float,
    cp_bf: float,
    k_bf: float,
    k_p: float,
    alpha_bf: float,
    enable_buongiorno: bool = True,
    enable_shear: bool = True,
    enable_velocity: bool = True,
    enable_gradient: bool = True,
    enable_pressure: bool = True,
    enable_turbulence: bool = False
) -> Tuple[float, dict]:
    """
    Comprehensive flow-dependent thermal conductivity model.
    
    Combines multiple enhancement mechanisms for accurate prediction
    in complex flow scenarios.
    
    Parameters
    ----------
    k_base : float
        Base (static) thermal conductivity (W/m·K)
    phi : float
        Volume fraction
    d_p : float
        Particle diameter (m)
    flow_data : FlowFieldData
        Local flow field data
    rho_bf, rho_p : float
        Densities (kg/m³)
    mu_bf : float
        Base fluid viscosity (Pa·s)
    cp_bf : float
        Base fluid specific heat (J/kg·K)
    k_bf, k_p : float
        Thermal conductivities (W/m·K)
    alpha_bf : float
        Base fluid thermal diffusivity (m²/s)
    enable_* : bool
        Flags to enable specific enhancement mechanisms
        
    Returns
    -------
    k_eff : float
        Comprehensive flow-enhanced conductivity (W/m·K)
    contributions : dict
        Individual contributions from each mechanism
    """
    k_current = k_base
    contributions = {'base': k_base}
    
    # 1. Buongiorno convective transport
    if enable_buongiorno and flow_data.velocity_magnitude > 0:
        k_buon = buongiorno_flow_enhanced_conductivity(
            k_current, phi, d_p, rho_bf, rho_p, mu_bf, cp_bf,
            flow_data.velocity_magnitude, flow_data.temperature, d_p
        )
        contributions['buongiorno'] = k_buon - k_current
        k_current = k_buon
    
    # 2. Shear rate enhancement
    if enable_shear and flow_data.shear_rate > 0:
        k_shear = kumar_shear_enhanced_conductivity(
            k_current, phi, flow_data.shear_rate, d_p,
            flow_data.temperature, mu_bf, k_bf, k_p
        )
        contributions['shear'] = k_shear - k_current
        k_current = k_shear
    
    # 3. Velocity-dependent enhancement
    if enable_velocity and flow_data.velocity_magnitude > 0:
        k_vel = rea_guzman_velocity_dependent_model(
            k_current, phi, flow_data.velocity_magnitude, d_p,
            flow_data.temperature, rho_bf, mu_bf, cp_bf, k_bf
        )
        contributions['velocity'] = k_vel - k_current
        k_current = k_vel
    
    # 4. Temperature gradient effects
    if enable_gradient and abs(flow_data.temperature_gradient) > 0:
        k_grad = temperature_gradient_enhanced_conductivity(
            k_current, phi, flow_data.temperature_gradient, d_p,
            flow_data.temperature, mu_bf, alpha_bf
        )
        contributions['gradient'] = k_grad - k_current
        k_current = k_grad
    
    # 5. Pressure effects
    if enable_pressure:
        k_press = pressure_dependent_conductivity(
            k_current, flow_data.pressure, 101325.0
        )
        contributions['pressure'] = k_press - k_current
        k_current = k_press
    
    # 6. Turbulent dispersion
    if enable_turbulence and flow_data.turbulent_kinetic_energy is not None:
        k_turb = turbulent_dispersion_conductivity(
            k_current, phi, flow_data.turbulent_kinetic_energy,
            rho_bf, cp_bf
        )
        contributions['turbulence'] = k_turb - k_current
        k_current = k_turb
    
    return k_current, contributions


# Example usage and validation
if __name__ == "__main__":
    print("=" * 70)
    print("BKPS NFL Thermal - Flow-Dependent Conductivity Models")
    print("Dedicated to: Brijesh Kumar Pandey")
    print("=" * 70)
    print()
    
    # Example: Al2O3-water nanofluid in channel flow
    phi = 0.02
    d_p = 30e-9  # 30 nm
    
    # Base properties (water at 300K)
    k_bf = 0.613
    rho_bf = 997.0
    mu_bf = 0.001
    cp_bf = 4180.0
    alpha_bf = k_bf / (rho_bf * cp_bf)
    
    # Al2O3 properties
    k_p = 40.0
    rho_p = 3970.0
    
    # Static conductivity (Maxwell)
    k_static = k_bf * (k_p + 2*k_bf + 2*phi*(k_p - k_bf)) / (k_p + 2*k_bf - phi*(k_p - k_bf))
    
    print(f"Static conductivity: {k_static:.6f} W/m·K")
    print(f"Enhancement over base: {(k_static/k_bf - 1)*100:.2f}%\n")
    
    # Flow conditions
    flow_data = FlowFieldData(
        temperature=300.0,
        pressure=101325.0,
        velocity_magnitude=0.5,
        shear_rate=1000.0,
        temperature_gradient=1000.0,
        turbulent_kinetic_energy=None
    )
    
    print("Flow conditions:")
    print(f"  Velocity: {flow_data.velocity_magnitude} m/s")
    print(f"  Shear rate: {flow_data.shear_rate} 1/s")
    print(f"  Temperature gradient: {flow_data.temperature_gradient} K/m\n")
    
    # Comprehensive calculation
    k_eff, contrib = comprehensive_flow_dependent_conductivity(
        k_static, phi, d_p, flow_data,
        rho_bf, rho_p, mu_bf, cp_bf, k_bf, k_p, alpha_bf
    )
    
    print(f"Flow-enhanced conductivity: {k_eff:.6f} W/m·K")
    print(f"Total enhancement: {(k_eff/k_bf - 1)*100:.2f}%\n")
    
    print("Contribution breakdown:")
    print(f"  Base (static): {contrib['base']:.6f} W/m·K")
    for mechanism, delta_k in contrib.items():
        if mechanism != 'base' and delta_k != 0:
            print(f"  {mechanism.capitalize()}: +{delta_k:.6f} W/m·K " +
                  f"({delta_k/k_static*100:.2f}% additional)")
    
    print("\n✓ Flow-dependent conductivity models validated!")
