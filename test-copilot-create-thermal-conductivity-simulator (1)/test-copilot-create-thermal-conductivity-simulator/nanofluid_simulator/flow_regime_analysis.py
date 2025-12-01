"""
Flow Regime Analysis and Heat Transfer Performance

This module calculates flow-related parameters and heat transfer performance:
- Reynolds number and flow regime identification
- Nusselt number correlations
- Pressure drop and pumping power
- Thermal-hydraulic performance evaluation
- Friction factor calculations

Key Applications:
- Heat exchanger design
- Cooling system optimization
- Energy efficiency analysis
- Performance vs pumping cost trade-offs

References:
1. Gnielinski, V. (1976). "New equations for heat and mass transfer in turbulent pipe flow"
2. Shah, R.K. & London, A.L. (1978). "Laminar Flow Forced Convection in Ducts"
3. Pak, B.C. & Cho, Y.I. (1998). "Hydrodynamic and heat transfer study of dispersed fluids"
"""

import math
from typing import Tuple, Dict, Optional
import numpy as np


# ============================================================================
# DIMENSIONLESS NUMBERS
# ============================================================================

def reynolds_number(
    rho: float,
    velocity: float,
    diameter: float,
    mu: float
) -> float:
    """
    Calculate Reynolds number.
    
    Re = ρVD/μ
    
    Flow regimes:
    - Re < 2300: Laminar flow
    - 2300 < Re < 4000: Transitional
    - Re > 4000: Turbulent flow
    
    Args:
        rho: Density (kg/m³)
        velocity: Flow velocity (m/s)
        diameter: Hydraulic diameter (m)
        mu: Dynamic viscosity (Pa·s)
        
    Returns:
        Reynolds number (dimensionless)
    """
    Re = rho * velocity * diameter / mu
    return Re


def prandtl_number(
    mu: float,
    c_p: float,
    k: float
) -> float:
    """
    Calculate Prandtl number.
    
    Pr = μ·c_p / k
    
    Physical meaning: ratio of momentum diffusivity to thermal diffusivity
    
    Typical values:
    - Water: Pr ≈ 5-7
    - Oils: Pr ≈ 100-1000
    - Liquid metals: Pr ≈ 0.01
    
    Args:
        mu: Dynamic viscosity (Pa·s)
        c_p: Specific heat capacity (J/kg·K)
        k: Thermal conductivity (W/m·K)
        
    Returns:
        Prandtl number (dimensionless)
    """
    Pr = mu * c_p / k
    return Pr


def peclet_number(
    velocity: float,
    diameter: float,
    alpha: float
) -> float:
    """
    Calculate Peclet number.
    
    Pe = VD/α = Re·Pr
    
    Ratio of advective to diffusive heat transport
    
    Args:
        velocity: Flow velocity (m/s)
        diameter: Characteristic length (m)
        alpha: Thermal diffusivity (m²/s)
        
    Returns:
        Peclet number (dimensionless)
    """
    Pe = velocity * diameter / alpha
    return Pe


def graetz_number(
    Re: float,
    Pr: float,
    diameter: float,
    length: float
) -> float:
    """
    Calculate Graetz number.
    
    Gz = (D/L) · Re · Pr
    
    Important for developing thermal boundary layers
    
    Args:
        Re: Reynolds number
        Pr: Prandtl number
        diameter: Hydraulic diameter (m)
        length: Channel length (m)
        
    Returns:
        Graetz number (dimensionless)
    """
    Gz = (diameter / length) * Re * Pr
    return Gz


# ============================================================================
# NUSSELT NUMBER CORRELATIONS
# ============================================================================

def nusselt_laminar_developing(
    Re: float,
    Pr: float,
    diameter: float,
    length: float,
    boundary_condition: str = "constant_heat_flux"
) -> float:
    """
    Nusselt number for laminar developing flow in circular tubes.
    
    Args:
        Re: Reynolds number
        Pr: Prandtl number
        diameter: Tube diameter (m)
        length: Tube length (m)
        boundary_condition: "constant_heat_flux" or "constant_temperature"
        
    Returns:
        Average Nusselt number
    """
    Gz = graetz_number(Re, Pr, diameter, length)
    
    if boundary_condition == "constant_heat_flux":
        # Constant heat flux condition
        if Gz > 10:
            # Developing flow
            Nu = 4.36 + 0.0668 * Gz / (1 + 0.04 * Gz**(2/3))
        else:
            # Fully developed
            Nu = 4.36
    else:
        # Constant temperature condition
        if Gz > 10:
            Nu = 3.66 + 0.065 * Gz / (1 + 0.04 * Gz**(2/3))
        else:
            Nu = 3.66
    
    return Nu


def nusselt_turbulent_gnielinski(
    Re: float,
    Pr: float,
    diameter: float,
    length: float
) -> Tuple[float, float]:
    """
    Gnielinski correlation for turbulent flow in tubes.
    
    Valid for:
    - 3000 < Re < 5×10⁶
    - 0.5 < Pr < 2000
    - L/D > 10
    
    Args:
        Re: Reynolds number
        Pr: Prandtl number
        diameter: Tube diameter (m)
        length: Tube length (m)
        
    Returns:
        Tuple of (Nu_average, friction_factor)
    """
    # Friction factor (Petukhov correlation)
    f = (0.790 * math.log(Re) - 1.64) ** (-2)
    
    # Gnielinski correlation
    numerator = (f / 8) * (Re - 1000) * Pr
    denominator = 1 + 12.7 * math.sqrt(f / 8) * (Pr**(2/3) - 1)
    
    Nu = numerator / denominator
    
    # Entrance effect correction
    if length / diameter < 60:
        Nu *= (1 + (diameter / length)**(2/3))
    
    return Nu, f


def nusselt_transitional(
    Re: float,
    Pr: float,
    diameter: float,
    length: float
) -> float:
    """
    Nusselt number for transitional flow (2300 < Re < 4000).
    
    Interpolation between laminar and turbulent values.
    
    Args:
        Re: Reynolds number
        Pr: Prandtl number
        diameter: Tube diameter (m)
        length: Tube length (m)
        
    Returns:
        Nusselt number
    """
    # Laminar value at Re = 2300
    Nu_lam = nusselt_laminar_developing(2300, Pr, diameter, length)
    
    # Turbulent value at Re = 4000
    Nu_turb, _ = nusselt_turbulent_gnielinski(4000, Pr, diameter, length)
    
    # Linear interpolation
    Nu = Nu_lam + (Nu_turb - Nu_lam) * (Re - 2300) / (4000 - 2300)
    
    return Nu


def nusselt_number_complete(
    Re: float,
    Pr: float,
    diameter: float,
    length: float
) -> Tuple[float, str]:
    """
    Calculate Nusselt number for any flow regime.
    
    Args:
        Re: Reynolds number
        Pr: Prandtl number
        diameter: Tube diameter (m)
        length: Tube length (m)
        
    Returns:
        Tuple of (Nu, flow_regime)
    """
    if Re < 2300:
        Nu = nusselt_laminar_developing(Re, Pr, diameter, length)
        regime = "Laminar"
    elif Re < 4000:
        Nu = nusselt_transitional(Re, Pr, diameter, length)
        regime = "Transitional"
    else:
        Nu, _ = nusselt_turbulent_gnielinski(Re, Pr, diameter, length)
        regime = "Turbulent"
    
    return Nu, regime


# ============================================================================
# PRESSURE DROP AND FRICTION FACTOR
# ============================================================================

def friction_factor_laminar(
    Re: float
) -> float:
    """
    Friction factor for laminar flow in circular pipe.
    
    f = 64 / Re
    
    Args:
        Re: Reynolds number
        
    Returns:
        Darcy friction factor
    """
    f = 64 / Re if Re > 0 else 0
    return f


def friction_factor_turbulent(
    Re: float,
    roughness: float = 0.0,
    diameter: float = 1.0
) -> float:
    """
    Friction factor for turbulent flow (Colebrook-White equation).
    
    Solved iteratively or using explicit Haaland approximation.
    
    Args:
        Re: Reynolds number
        roughness: Absolute roughness (m)
        diameter: Pipe diameter (m)
        
    Returns:
        Darcy friction factor
    """
    # Relative roughness
    epsilon = roughness / diameter
    
    if epsilon == 0:
        # Smooth pipe (Petukhov correlation)
        f = (0.790 * math.log(Re) - 1.64) ** (-2)
    else:
        # Haaland explicit approximation
        term1 = (epsilon / 3.7)
        term2 = 6.9 / Re
        f = (-1.8 * math.log10(term1**1.11 + term2)) ** (-2)
    
    return f


def pressure_drop(
    f: float,
    length: float,
    diameter: float,
    rho: float,
    velocity: float
) -> float:
    """
    Calculate pressure drop in pipe.
    
    ΔP = f · (L/D) · (ρV²/2)
    
    Args:
        f: Darcy friction factor
        length: Pipe length (m)
        diameter: Pipe diameter (m)
        rho: Fluid density (kg/m³)
        velocity: Flow velocity (m/s)
        
    Returns:
        Pressure drop (Pa)
    """
    dP = f * (length / diameter) * (rho * velocity**2 / 2)
    return dP


def pumping_power(
    dP: float,
    volume_flow_rate: float,
    pump_efficiency: float = 0.75
) -> float:
    """
    Calculate required pumping power.
    
    P = ΔP · Q̇ / η
    
    Args:
        dP: Pressure drop (Pa)
        volume_flow_rate: Volumetric flow rate (m³/s)
        pump_efficiency: Pump efficiency (0 to 1)
        
    Returns:
        Pumping power (W)
    """
    P = dP * volume_flow_rate / pump_efficiency
    return P


# ============================================================================
# THERMAL-HYDRAULIC PERFORMANCE
# ============================================================================

def heat_transfer_coefficient(
    Nu: float,
    k: float,
    diameter: float
) -> float:
    """
    Calculate convective heat transfer coefficient.
    
    h = Nu · k / D
    
    Args:
        Nu: Nusselt number
        k: Thermal conductivity (W/m·K)
        diameter: Hydraulic diameter (m)
        
    Returns:
        Heat transfer coefficient (W/m²·K)
    """
    h = Nu * k / diameter
    return h


def thermal_resistance(
    h: float,
    area: float
) -> float:
    """
    Calculate convective thermal resistance.
    
    R = 1 / (h·A)
    
    Args:
        h: Heat transfer coefficient (W/m²·K)
        area: Heat transfer area (m²)
        
    Returns:
        Thermal resistance (K/W)
    """
    R = 1 / (h * area) if h * area > 0 else np.inf
    return R


def heat_transfer_rate(
    h: float,
    area: float,
    T_hot: float,
    T_cold: float
) -> float:
    """
    Calculate heat transfer rate.
    
    Q̇ = h·A·ΔT
    
    Args:
        h: Heat transfer coefficient (W/m²·K)
        area: Heat transfer area (m²)
        T_hot: Hot surface temperature (K)
        T_cold: Cold fluid temperature (K)
        
    Returns:
        Heat transfer rate (W)
    """
    Q = h * area * abs(T_hot - T_cold)
    return Q


def performance_index(
    h_nf: float,
    h_bf: float,
    P_nf: float,
    P_bf: float
) -> float:
    """
    Calculate thermal-hydraulic performance index.
    
    PI = (h_nf/h_bf) / (P_nf/P_bf)^(1/3)
    
    PI > 1: Nanofluid beneficial (heat transfer gain > pumping penalty)
    PI < 1: Base fluid better
    
    Args:
        h_nf: Nanofluid heat transfer coefficient (W/m²·K)
        h_bf: Base fluid heat transfer coefficient (W/m²·K)
        P_nf: Nanofluid pumping power (W)
        P_bf: Base fluid pumping power (W)
        
    Returns:
        Performance index (dimensionless)
    """
    heat_ratio = h_nf / h_bf if h_bf > 0 else 1
    power_ratio = P_nf / P_bf if P_bf > 0 else 1
    
    PI = heat_ratio / (power_ratio ** (1/3))
    
    return PI


# ============================================================================
# COMPREHENSIVE FLOW ANALYZER
# ============================================================================

class FlowRegimeAnalyzer:
    """
    Comprehensive analyzer for flow regime and thermal-hydraulic performance.
    """
    
    @staticmethod
    def analyze_complete(
        rho: float,
        mu: float,
        k: float,
        c_p: float,
        velocity: float,
        diameter: float,
        length: float,
        roughness: float = 0.0,
        pump_efficiency: float = 0.75
    ) -> Dict[str, float]:
        """
        Complete flow and heat transfer analysis.
        
        Args:
            rho: Density (kg/m³)
            mu: Dynamic viscosity (Pa·s)
            k: Thermal conductivity (W/m·K)
            c_p: Specific heat (J/kg·K)
            velocity: Flow velocity (m/s)
            diameter: Pipe diameter (m)
            length: Pipe length (m)
            roughness: Surface roughness (m)
            pump_efficiency: Pump efficiency (0-1)
            
        Returns:
            Comprehensive diagnostics dictionary
        """
        results = {}
        
        # 1. Dimensionless numbers
        Re = reynolds_number(rho, velocity, diameter, mu)
        Pr = prandtl_number(mu, c_p, k)
        alpha = k / (rho * c_p)
        Pe = peclet_number(velocity, diameter, alpha)
        
        results['Reynolds'] = Re
        results['Prandtl'] = Pr
        results['Peclet'] = Pe
        
        # 2. Flow regime
        if Re < 2300:
            regime = "Laminar"
        elif Re < 4000:
            regime = "Transitional"
        else:
            regime = "Turbulent"
        results['flow_regime'] = regime
        
        # 3. Nusselt number
        Nu, _ = nusselt_number_complete(Re, Pr, diameter, length)
        results['Nusselt'] = Nu
        
        # 4. Heat transfer coefficient
        h = heat_transfer_coefficient(Nu, k, diameter)
        results['h_conv'] = h
        
        # 5. Friction factor
        if Re < 2300:
            f = friction_factor_laminar(Re)
        else:
            f = friction_factor_turbulent(Re, roughness, diameter)
        results['friction_factor'] = f
        
        # 6. Pressure drop
        dP = pressure_drop(f, length, diameter, rho, velocity)
        results['pressure_drop_Pa'] = dP
        results['pressure_drop_kPa'] = dP / 1000
        
        # 7. Volumetric flow rate
        area = math.pi * (diameter / 2) ** 2
        Q = velocity * area
        results['volume_flow_m3_s'] = Q
        results['volume_flow_L_min'] = Q * 60000
        
        # 8. Mass flow rate
        m_dot = rho * Q
        results['mass_flow_kg_s'] = m_dot
        
        # 9. Pumping power
        P_pump = pumping_power(dP, Q, pump_efficiency)
        results['pumping_power_W'] = P_pump
        results['pumping_power_kW'] = P_pump / 1000
        
        # 10. Thermal resistance (per meter length)
        perimeter = math.pi * diameter
        area_per_meter = perimeter * 1.0
        R_th = thermal_resistance(h, area_per_meter)
        results['thermal_resistance_K_W_per_m'] = R_th
        
        return results
    
    @staticmethod
    def compare_nanofluid_vs_basefluid(
        # Nanofluid properties
        rho_nf: float, mu_nf: float, k_nf: float, c_p_nf: float,
        # Base fluid properties
        rho_bf: float, mu_bf: float, k_bf: float, c_p_bf: float,
        # Flow conditions
        velocity: float, diameter: float, length: float,
        roughness: float = 0.0
    ) -> Dict[str, any]:
        """
        Compare thermal-hydraulic performance: nanofluid vs base fluid.
        
        Returns:
            Comparison dictionary with performance metrics
        """
        # Analyze both fluids
        nf = FlowRegimeAnalyzer.analyze_complete(
            rho_nf, mu_nf, k_nf, c_p_nf, velocity, diameter, length, roughness
        )
        bf = FlowRegimeAnalyzer.analyze_complete(
            rho_bf, mu_bf, k_bf, c_p_bf, velocity, diameter, length, roughness
        )
        
        # Calculate enhancements/penalties
        comparison = {
            'nanofluid': nf,
            'basefluid': bf,
            
            # Heat transfer enhancement
            'h_enhancement_%': ((nf['h_conv'] / bf['h_conv']) - 1) * 100,
            'Nu_enhancement_%': ((nf['Nusselt'] / bf['Nusselt']) - 1) * 100,
            
            # Pressure drop penalty
            'dP_increase_%': ((nf['pressure_drop_Pa'] / bf['pressure_drop_Pa']) - 1) * 100,
            'pumping_power_increase_%': ((nf['pumping_power_W'] / bf['pumping_power_W']) - 1) * 100,
            
            # Overall performance
            'performance_index': performance_index(
                nf['h_conv'], bf['h_conv'],
                nf['pumping_power_W'], bf['pumping_power_W']
            ),
            
            # Recommendation
            'recommendation': "Use nanofluid" if performance_index(
                nf['h_conv'], bf['h_conv'],
                nf['pumping_power_W'], bf['pumping_power_W']
            ) > 1.1 else "Use base fluid"
        }
        
        return comparison
