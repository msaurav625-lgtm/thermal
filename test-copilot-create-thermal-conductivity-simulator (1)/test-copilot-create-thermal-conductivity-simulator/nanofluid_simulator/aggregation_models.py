"""
Nanoparticle Aggregation and Interaction Physics

This module implements models for:
- DLVO theory (Derjaguin-Landau-Verwey-Overbeek)
- Particle-particle collision dynamics
- Aggregation kinetics
- Nanolayer interaction effects
- Mean free path corrections

Key Physics:
- Van der Waals attraction between particles
- Electrostatic repulsion (double-layer forces)
- Brownian collision frequency
- Shear-induced aggregation/breakup
- Surface chemistry effects on thermal transport

References:
1. Derjaguin, B. & Landau, L. (1941). "Theory of stability of colloids"
2. Verwey, E.J.W. & Overbeek, J.T.G. (1948). "Theory of stability"
3. Smoluchowski, M. (1917). "Kinetics of coagulation of colloidal solutions"
4. Batchelor, G.K. (1976). "Brownian diffusion of particles with hydrodynamic interaction"
"""

import math
from typing import Tuple, Dict, Optional
import numpy as np

# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
ELEMENTARY_CHARGE = 1.602176634e-19  # C
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
AVOGADRO_NUMBER = 6.02214076e23  # mol⁻¹


# ============================================================================
# DLVO THEORY - COLLOIDAL STABILITY
# ============================================================================

def calculate_van_der_waals_potential(
    A_Hamaker: float,
    d_p: float,
    h: float
) -> float:
    """
    Calculate Van der Waals attraction potential between two spheres.
    
    Van der Waals forces cause particles to attract each other,
    leading to aggregation if not counterbalanced by repulsion.
    
    Formula (for h << d_p):
        V_vdw = -A_H * d_p / (12 * h)
    
    Args:
        A_Hamaker: Hamaker constant (J)
                   Typical values: 0.4-4.0 × 10⁻²⁰ J
                   - Higher for metals (1-4 × 10⁻²⁰ J)
                   - Lower for oxides (0.4-1 × 10⁻²⁰ J)
        d_p: Particle diameter (nm)
        h: Surface-to-surface separation distance (nm)
        
    Returns:
        Van der Waals potential (J)
    """
    if h <= 0:
        return -np.inf  # Infinite attraction at contact
    
    # Convert to meters
    d_p_m = d_p * 1e-9
    h_m = h * 1e-9
    
    # Van der Waals potential
    V_vdw = -A_Hamaker * d_p_m / (12 * h_m)
    
    return V_vdw


def calculate_electrostatic_potential(
    d_p: float,
    h: float,
    zeta_potential: float,
    ionic_strength: float,
    T: float,
    epsilon_r: float = 78.5
) -> float:
    """
    Calculate electrostatic repulsion potential (double-layer repulsion).
    
    Electric double layer around charged particles creates repulsion,
    stabilizing the suspension against aggregation.
    
    Key factors:
    - High zeta potential → stronger repulsion → stable suspension
    - High ionic strength → compressed double layer → weak repulsion
    
    Formula (constant potential approximation):
        V_elec = 2πε₀εᵣRψ₀² * exp(-κh)
    
    Args:
        d_p: Particle diameter (nm)
        h: Surface separation distance (nm)
        zeta_potential: Zeta potential (mV)
        ionic_strength: Ionic strength (mol/L)
        T: Temperature (K)
        epsilon_r: Relative permittivity (78.5 for water at 25°C)
        
    Returns:
        Electrostatic potential (J)
    """
    # Convert units
    d_p_m = d_p * 1e-9
    h_m = h * 1e-9
    zeta_V = zeta_potential * 1e-3  # mV to V
    I = ionic_strength * 1000  # mol/L to mol/m³
    
    # Debye length (inverse Debye parameter κ)
    kappa = math.sqrt(
        2 * AVOGADRO_NUMBER * ELEMENTARY_CHARGE**2 * I /
        (VACUUM_PERMITTIVITY * epsilon_r * BOLTZMANN_CONSTANT * T)
    )
    
    # Debye length
    lambda_D = 1 / kappa
    
    # Electrostatic potential (constant potential model)
    R = d_p_m / 2  # Particle radius
    
    V_elec = 2 * math.pi * VACUUM_PERMITTIVITY * epsilon_r * R * zeta_V**2 * \
             math.exp(-kappa * h_m)
    
    return V_elec


def dlvo_total_potential(
    A_Hamaker: float,
    d_p: float,
    h: float,
    zeta_potential: float,
    ionic_strength: float,
    T: float
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate total DLVO interaction potential.
    
    Total = Van der Waals attraction + Electrostatic repulsion
    
    Stability analysis:
    - V_total > 0: Energy barrier prevents aggregation (stable)
    - V_total < 0: Particles attract (aggregation likely)
    - Max(V_total) > 15 kT: Suspension is stable
    - Max(V_total) < 5 kT: Rapid aggregation
    
    Args:
        A_Hamaker: Hamaker constant (J)
        d_p: Particle diameter (nm)
        h: Separation distance (nm)
        zeta_potential: Zeta potential (mV)
        ionic_strength: Ionic strength (mol/L)
        T: Temperature (K)
        
    Returns:
        Tuple of (V_total, components_dict)
    """
    V_vdw = calculate_van_der_waals_potential(A_Hamaker, d_p, h)
    V_elec = calculate_electrostatic_potential(d_p, h, zeta_potential, 
                                               ionic_strength, T)
    
    V_total = V_vdw + V_elec
    
    # Thermal energy for comparison
    kT = BOLTZMANN_CONSTANT * T
    
    components = {
        'V_vdw': V_vdw,
        'V_elec': V_elec,
        'V_total': V_total,
        'V_total_over_kT': V_total / kT,
        'kT': kT
    }
    
    return V_total, components


def assess_colloidal_stability(
    d_p: float,
    zeta_potential: float,
    ionic_strength: float,
    T: float = 298.15,
    A_Hamaker: float = 1e-20
) -> Dict[str, any]:
    """
    Assess colloidal stability and predict aggregation tendency.
    
    Returns stability assessment, energy barrier height, and
    aggregation probability.
    
    Args:
        d_p: Particle diameter (nm)
        zeta_potential: Zeta potential (mV)
        ionic_strength: Ionic strength (mol/L)
        T: Temperature (K)
        A_Hamaker: Hamaker constant (J)
        
    Returns:
        Dictionary with stability assessment
    """
    # Calculate potential energy profile
    h_range = np.logspace(-1, 2, 100)  # 0.1 to 100 nm
    V_profile = []
    
    for h in h_range:
        V, _ = dlvo_total_potential(A_Hamaker, d_p, h, zeta_potential, 
                                    ionic_strength, T)
        V_profile.append(V)
    
    V_profile = np.array(V_profile)
    kT = BOLTZMANN_CONSTANT * T
    
    # Find energy barrier
    V_max = np.max(V_profile)
    V_max_over_kT = V_max / kT
    
    # Determine stability
    if V_max_over_kT > 15:
        stability = "Stable"
        aggregation_risk = "Low"
    elif V_max_over_kT > 5:
        stability = "Marginally stable"
        aggregation_risk = "Moderate"
    else:
        stability = "Unstable"
        aggregation_risk = "High"
    
    # Debye length
    epsilon_r = 78.5
    I = ionic_strength * 1000
    kappa = math.sqrt(
        2 * AVOGADRO_NUMBER * ELEMENTARY_CHARGE**2 * I /
        (VACUUM_PERMITTIVITY * epsilon_r * BOLTZMANN_CONSTANT * T)
    )
    debye_length = 1 / kappa * 1e9  # in nm
    
    results = {
        'stability_status': stability,
        'aggregation_risk': aggregation_risk,
        'energy_barrier_kT': V_max_over_kT,
        'energy_barrier_J': V_max,
        'debye_length_nm': debye_length,
        'recommendation': "Add surfactant" if aggregation_risk == "High" 
                         else "Stable suspension"
    }
    
    return results


# ============================================================================
# AGGREGATION KINETICS
# ============================================================================

def brownian_collision_frequency(
    n: float,
    d_p: float,
    T: float,
    mu_bf: float
) -> float:
    """
    Calculate Brownian collision frequency (Smoluchowski theory).
    
    Predicts how often particles collide due to Brownian motion.
    
    Formula:
        J = (4 * k_B * T) / (3 * μ) * n²
    
    Args:
        n: Number concentration of particles (particles/m³)
        d_p: Particle diameter (nm)
        T: Temperature (K)
        mu_bf: Base fluid viscosity (Pa·s)
        
    Returns:
        Collision frequency (collisions/m³/s)
    """
    d_p_m = d_p * 1e-9
    
    # Smoluchowski collision kernel for Brownian motion
    K_Brownian = (4 * BOLTZMANN_CONSTANT * T) / (3 * mu_bf)
    
    # Collision frequency
    J = K_Brownian * n**2
    
    return J


def shear_induced_collision_frequency(
    n: float,
    d_p: float,
    shear_rate: float
) -> float:
    """
    Calculate collision frequency due to shear flow.
    
    In flowing systems, velocity gradients bring particles together.
    
    Formula:
        J = (4/3) * γ̇ * d³ * n²
    
    Args:
        n: Number concentration (particles/m³)
        d_p: Particle diameter (nm)
        shear_rate: Shear rate γ̇ (1/s)
        
    Returns:
        Collision frequency (collisions/m³/s)
    """
    d_p_m = d_p * 1e-9
    
    # Shear collision kernel
    K_shear = (4/3) * shear_rate * d_p_m**3
    
    # Collision frequency
    J = K_shear * n**2
    
    return J


def aggregate_fractal_dimension(
    aggregation_regime: str = "DLCA"
) -> float:
    """
    Get fractal dimension based on aggregation mechanism.
    
    Fractal dimension determines aggregate structure:
    - Higher D_f: compact aggregates
    - Lower D_f: open, chain-like aggregates
    
    Args:
        aggregation_regime: "DLCA", "RLCA", or "shear"
                           DLCA: Diffusion-Limited Cluster Aggregation
                           RLCA: Reaction-Limited Cluster Aggregation
        
    Returns:
        Fractal dimension D_f
    """
    regimes = {
        "DLCA": 1.8,   # Rapid aggregation, open structures
        "RLCA": 2.1,   # Slow aggregation, more compact
        "shear": 2.3,  # Shear-induced, compact aggregates
        "compact": 2.5  # Nearly spherical aggregates
    }
    
    return regimes.get(aggregation_regime, 2.0)


# ============================================================================
# NANOLAYER EFFECTS
# ============================================================================

def interfacial_layer_thickness(
    material_type: str,
    base_fluid: str
) -> float:
    """
    Estimate interfacial nanolayer thickness.
    
    The nanolayer is an ordered molecular layer of base fluid
    molecules at the particle surface. Its properties differ
    from bulk fluid.
    
    Thickness depends on:
    - Surface chemistry
    - Fluid polarity
    - Temperature
    - Surface roughness
    
    Args:
        material_type: "metal", "oxide", "carbon"
        base_fluid: "water", "oil", "glycol"
        
    Returns:
        Layer thickness (nm)
    """
    # Empirical values from literature
    thickness_map = {
        ('metal', 'water'): 2.0,
        ('oxide', 'water'): 3.5,
        ('carbon', 'water'): 1.5,
        ('metal', 'oil'): 1.0,
        ('oxide', 'oil'): 1.5,
        ('carbon', 'oil'): 0.8,
        ('metal', 'glycol'): 2.5,
        ('oxide', 'glycol'): 4.0,
        ('carbon', 'glycol'): 2.0,
    }
    
    return thickness_map.get((material_type, base_fluid), 2.0)


def interfacial_layer_conductivity(
    k_bf: float,
    k_np: float,
    ordering_factor: float = 1.5
) -> float:
    """
    Calculate thermal conductivity of interfacial layer.
    
    The nanolayer has intermediate properties between bulk
    fluid and solid particle. Molecular ordering typically
    increases conductivity.
    
    Args:
        k_bf: Base fluid conductivity (W/m·K)
        k_np: Nanoparticle conductivity (W/m·K)
        ordering_factor: Enhancement due to molecular ordering (1-3)
                        1.0 = no enhancement (disordered)
                        2.0 = moderate ordering (typical)
                        3.0 = highly ordered crystalline layer
        
    Returns:
        Layer thermal conductivity (W/m·K)
    """
    # Geometric mean with ordering enhancement
    k_layer = math.sqrt(k_bf * k_np) * ordering_factor
    
    # Limit to physical bounds
    k_layer = min(k_layer, k_np)
    k_layer = max(k_layer, k_bf)
    
    return k_layer


# ============================================================================
# PARTICLE COLLISION DYNAMICS UNDER FLOW
# ============================================================================

def particle_collision_efficiency(
    d_p: float,
    velocity: float,
    T: float,
    mu_bf: float,
    energy_barrier_kT: float = 5.0
) -> float:
    """
    Calculate collision efficiency (probability of sticking upon collision).
    
    Not all collisions lead to aggregation. Particles must overcome
    energy barrier.
    
    Args:
        d_p: Particle diameter (nm)
        velocity: Flow velocity (m/s)
        T: Temperature (K)
        mu_bf: Viscosity (Pa·s)
        energy_barrier_kT: DLVO energy barrier (in units of kT)
        
    Returns:
        Collision efficiency α (0 to 1)
    """
    kT = BOLTZMANN_CONSTANT * T
    
    # Kinetic energy from flow
    d_p_m = d_p * 1e-9
    rho_p = 8900  # Approximate particle density (kg/m³)
    m_p = rho_p * (4/3) * math.pi * (d_p_m/2)**3
    E_kinetic = 0.5 * m_p * velocity**2
    
    # Thermal energy
    E_thermal = 1.5 * kT
    
    # Total energy available
    E_total = E_kinetic + E_thermal
    
    # Energy barrier
    E_barrier = energy_barrier_kT * kT
    
    # Collision efficiency (Boltzmann factor)
    if E_total > E_barrier:
        alpha = 1.0  # All collisions lead to aggregation
    else:
        alpha = math.exp(-(E_barrier - E_total) / kT)
    
    return alpha


def mean_free_path_correction(
    d_p: float,
    phi: float,
    aggregation_factor: float = 1.0
) -> float:
    """
    Calculate mean free path between particles.
    
    As concentration increases, particles are closer together,
    reducing the mean distance between collisions.
    
    Args:
        d_p: Particle diameter (nm)
        phi: Volume fraction (0 to 1)
        aggregation_factor: Effective size multiplier due to aggregation
        
    Returns:
        Mean free path (nm)
    """
    if phi == 0:
        return np.inf
    
    # Effective particle diameter
    d_eff = d_p * aggregation_factor
    
    # Number density
    V_p = (4/3) * math.pi * (d_eff/2)**3  # nm³
    n = phi / V_p  # particles per nm³
    
    # Mean free path (simplified kinetic theory)
    if n > 0:
        lambda_mfp = 1 / (math.sqrt(2) * math.pi * d_eff**2 * n)
    else:
        lambda_mfp = np.inf
    
    return lambda_mfp


# ============================================================================
# COMPREHENSIVE INTERACTION ANALYZER
# ============================================================================

class ParticleInteractionAnalyzer:
    """
    Comprehensive analyzer for particle interactions and aggregation.
    """
    
    @staticmethod
    def analyze_stability(
        d_p: float,
        phi: float,
        T: float,
        mu_bf: float,
        zeta_potential: float = 30.0,
        ionic_strength: float = 0.001,
        velocity: float = 0.0,
        shear_rate: float = 0.0
    ) -> Dict[str, any]:
        """
        Complete analysis of particle stability and interactions.
        
        Returns:
            Comprehensive diagnostics dictionary
        """
        results = {}
        
        # 1. DLVO stability analysis
        stability = assess_colloidal_stability(d_p, zeta_potential, 
                                               ionic_strength, T)
        results['stability'] = stability
        
        # 2. Collision frequencies
        # Number concentration from volume fraction
        V_p = (4/3) * math.pi * ((d_p*1e-9)/2)**3
        n = phi / V_p
        
        J_Brownian = brownian_collision_frequency(n, d_p, T, mu_bf)
        results['collision_freq_Brownian'] = J_Brownian
        
        if shear_rate > 0:
            J_shear = shear_induced_collision_frequency(n, d_p, shear_rate)
            results['collision_freq_shear'] = J_shear
            results['collision_freq_total'] = J_Brownian + J_shear
        
        # 3. Collision efficiency
        alpha = particle_collision_efficiency(
            d_p, velocity, T, mu_bf, 
            energy_barrier_kT=stability['energy_barrier_kT']
        )
        results['collision_efficiency'] = alpha
        
        # 4. Mean free path
        mfp = mean_free_path_correction(d_p, phi)
        results['mean_free_path_nm'] = mfp
        
        # 5. Aggregation prediction
        if alpha > 0.5 and stability['aggregation_risk'] == "High":
            results['aggregation_prediction'] = "Rapid aggregation expected"
        elif alpha > 0.1:
            results['aggregation_prediction'] = "Slow aggregation possible"
        else:
            results['aggregation_prediction'] = "Stable suspension"
        
        return results
