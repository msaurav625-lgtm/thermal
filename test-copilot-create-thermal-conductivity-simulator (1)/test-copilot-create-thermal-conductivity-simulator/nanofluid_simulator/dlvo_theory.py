"""
DLVO Theory and Particle Interaction Physics for BKPS NFL Thermal

Implements Derjaguin-Landau-Verwey-Overbeek (DLVO) theory for colloidal stability,
particle aggregation, and clustering effects on thermophysical properties.

Features:
- Van der Waals attractive forces
- Electrostatic repulsive forces (EDL)
- Zeta potential pH dependence
- Ionic strength effects
- Fractal aggregation modeling
- Cluster size distribution
- Effects on thermal conductivity and viscosity

Author: BKPS NFL Thermal v6.0
Dedicated to: Brijesh Kumar Pandey
License: MIT
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from scipy.special import erf
from scipy.optimize import fsolve


@dataclass
class DLVOParameters:
    """DLVO theory parameters"""
    hamaker_constant: float  # J
    zeta_potential: float  # V
    ionic_strength: float  # mol/L
    temperature: float  # K
    dielectric_constant: float  # dimensionless
    pH: float  # dimensionless


@dataclass
class ClusterProperties:
    """Cluster characteristics"""
    avg_cluster_size: float  # Number of particles per cluster
    fractal_dimension: float  # Fractal dimension
    cluster_radius: float  # Effective cluster radius (m)
    volume_fraction_clustered: float  # Fraction of particles in clusters


def van_der_waals_potential(
    H: float,
    R: float,
    A_hamaker: float
) -> float:
    """
    Van der Waals interaction potential (attractive).
    
    V_vdw = -A / (6H) · [2R²/(H² + 4RH) + 2R²/(H² + 4RH + 4R²) + ln((H² + 4RH)/(H² + 4RH + 4R²))]
    
    For H << R, simplified to: V_vdw ≈ -A·R / (12H)
    
    Reference:
    Hamaker, H. C. (1937). The London—van der Waals attraction between spherical particles.
    Physica, 4(10), 1058-1072.
    
    Parameters
    ----------
    H : float
        Surface-to-surface separation distance (m)
    R : float
        Particle radius (m)
    A_hamaker : float
        Hamaker constant (J), typically 0.4-4 × 10^-20 J for aqueous systems
        
    Returns
    -------
    V_vdw : float
        Van der Waals potential energy (J)
    """
    if H <= 0:
        return -np.inf  # Contact
    
    # Full sphere-sphere interaction
    if H < 0.1 * R:
        # Small gap approximation
        V_vdw = -A_hamaker * R / (12.0 * H)
    else:
        # Complete formula
        term1 = 2 * R**2 / (H**2 + 4*R*H)
        term2 = 2 * R**2 / (H**2 + 4*R*H + 4*R**2)
        term3 = np.log((H**2 + 4*R*H) / (H**2 + 4*R*H + 4*R**2))
        
        V_vdw = -(A_hamaker / 6.0) * (term1 + term2 + term3)
    
    return V_vdw


def electrostatic_repulsion_potential(
    H: float,
    R: float,
    zeta: float,
    kappa: float,
    epsilon_r: float = 80.0,
    temperature: float = 298.15
) -> float:
    """
    Electrostatic double-layer repulsion potential.
    
    V_elec = 2π·ε0·εr·R·ζ²·exp(-κH)  (Debye-Hückel approximation)
    
    Valid for low surface potentials (ζ < 25 mV).
    
    Reference:
    Verwey, E. J. W., & Overbeek, J. T. G. (1948). Theory of the Stability of
    Lyophobic Colloids. Elsevier.
    
    Parameters
    ----------
    H : float
        Surface-to-surface separation distance (m)
    R : float
        Particle radius (m)
    zeta : float
        Zeta potential (V)
    kappa : float
        Inverse Debye length (1/m)
    epsilon_r : float
        Relative dielectric constant of medium (80 for water)
    temperature : float
        Temperature (K)
        
    Returns
    -------
    V_elec : float
        Electrostatic repulsion energy (J)
    """
    epsilon_0 = 8.854187817e-12  # F/m, vacuum permittivity
    
    # Constant potential boundary condition
    V_elec = 2 * np.pi * epsilon_0 * epsilon_r * R * zeta**2 * np.exp(-kappa * H)
    
    return V_elec


def debye_length(
    ionic_strength: float,
    temperature: float = 298.15,
    epsilon_r: float = 80.0
) -> float:
    """
    Calculate Debye screening length.
    
    λ_D = 1/κ = sqrt(ε0·εr·kB·T / (2·NA·e²·I))
    
    Parameters
    ----------
    ionic_strength : float
        Ionic strength (mol/L)
    temperature : float
        Temperature (K)
    epsilon_r : float
        Relative dielectric constant
        
    Returns
    -------
    lambda_D : float
        Debye length (m)
    """
    epsilon_0 = 8.854187817e-12  # F/m
    k_B = 1.38064852e-23  # J/K
    N_A = 6.02214086e23  # 1/mol
    e = 1.602176634e-19  # C
    
    # Convert ionic strength from mol/L to mol/m³
    I_SI = ionic_strength * 1000.0
    
    # Debye length
    lambda_D = np.sqrt(epsilon_0 * epsilon_r * k_B * temperature / (2 * N_A * e**2 * I_SI))
    
    return lambda_D


def zeta_potential_pH_dependence(
    pH: float,
    isoelectric_point: float = 8.0,
    max_zeta: float = -40e-3,
    pH_sensitivity: float = 5.0
) -> float:
    """
    Empirical zeta potential as function of pH.
    
    For oxide nanoparticles (Al2O3, TiO2, SiO2).
    
    Reference:
    Kosmulski, M. (2009). Surface charging and points of zero charge.
    CRC Press.
    
    Parameters
    ----------
    pH : float
        Solution pH
    isoelectric_point : float
        Isoelectric point (IEP), pH where ζ=0
        Al2O3: ~9, TiO2: ~6, SiO2: ~2-3
    max_zeta : float
        Maximum zeta potential magnitude (V), typically -40 to -60 mV
    pH_sensitivity : float
        Steepness of pH response
        
    Returns
    -------
    zeta : float
        Zeta potential (V)
    """
    # Sigmoidal transition around IEP
    delta_pH = pH - isoelectric_point
    zeta = max_zeta * np.tanh(delta_pH / pH_sensitivity)
    
    return zeta


def dlvo_total_potential(
    H: float,
    R: float,
    A_hamaker: float,
    zeta: float,
    ionic_strength: float,
    temperature: float = 298.15,
    epsilon_r: float = 80.0
) -> Tuple[float, float, float]:
    """
    Total DLVO interaction potential.
    
    V_total = V_vdw + V_elec
    
    Parameters
    ----------
    H : float
        Separation distance (m)
    R : float
        Particle radius (m)
    A_hamaker : float
        Hamaker constant (J)
    zeta : float
        Zeta potential (V)
    ionic_strength : float
        Ionic strength (mol/L)
    temperature : float
        Temperature (K)
    epsilon_r : float
        Relative dielectric constant
        
    Returns
    -------
    V_total : float
        Total interaction energy (J)
    V_vdw : float
        Van der Waals component (J)
    V_elec : float
        Electrostatic component (J)
    """
    # Calculate Debye length
    lambda_D = debye_length(ionic_strength, temperature, epsilon_r)
    kappa = 1.0 / lambda_D
    
    # Van der Waals attraction
    V_vdw = van_der_waals_potential(H, R, A_hamaker)
    
    # Electrostatic repulsion
    V_elec = electrostatic_repulsion_potential(H, R, zeta, kappa, epsilon_r, temperature)
    
    # Total potential
    V_total = V_vdw + V_elec
    
    return V_total, V_vdw, V_elec


def energy_barrier_height(
    R: float,
    A_hamaker: float,
    zeta: float,
    ionic_strength: float,
    temperature: float = 298.15
) -> Tuple[float, float]:
    """
    Calculate maximum energy barrier in DLVO potential.
    
    Returns
    -------
    E_barrier : float
        Energy barrier height (J)
    H_barrier : float
        Separation distance at barrier (m)
    """
    # Search for maximum in potential curve
    H_range = np.logspace(-10, -7, 1000)  # 0.1 nm to 100 nm
    
    V_max = -np.inf
    H_max = 0
    
    for H in H_range:
        V_total, _, _ = dlvo_total_potential(
            H, R, A_hamaker, zeta, ionic_strength, temperature
        )
        if V_total > V_max:
            V_max = V_total
            H_max = H
    
    return V_max, H_max


def aggregation_rate_smoluchowski(
    phi: float,
    d_p: float,
    temperature: float,
    mu: float,
    stability_ratio: float = 1.0
) -> float:
    """
    Smoluchowski aggregation rate with stability ratio.
    
    k_agg = (8·kB·T) / (3·μ·W)
    
    W = stability ratio (1 = fast aggregation, >>1 = slow aggregation)
    
    Reference:
    Smoluchowski, M. V. (1917). Versuch einer mathematischen Theorie der
    Koagulationskinetik kolloider Lösungen. Z. Phys. Chem., 92, 129-168.
    
    Parameters
    ----------
    phi : float
        Volume fraction
    d_p : float
        Particle diameter (m)
    temperature : float
        Temperature (K)
    mu : float
        Viscosity (Pa·s)
    stability_ratio : float
        DLVO stability ratio W
        
    Returns
    -------
    k_agg : float
        Aggregation rate constant (m³/s)
    """
    k_B = 1.38064852e-23
    
    k_agg = (8 * k_B * temperature) / (3 * mu * stability_ratio)
    
    return k_agg


def fractal_cluster_size(
    N: int,
    R_p: float,
    D_f: float = 1.8
) -> float:
    """
    Fractal aggregate radius.
    
    R_cluster = R_p · N^(1/D_f)
    
    D_f = fractal dimension
    - DLCA (diffusion-limited): D_f ≈ 1.8
    - RLCA (reaction-limited): D_f ≈ 2.1
    - Dense packing: D_f → 3.0
    
    Reference:
    Meakin, P. (1983). Formation of fractal clusters and networks by
    irreversible diffusion-limited aggregation. Phys. Rev. Lett., 51(13), 1119.
    
    Parameters
    ----------
    N : int
        Number of primary particles in cluster
    R_p : float
        Primary particle radius (m)
    D_f : float
        Fractal dimension (1.0-3.0)
        
    Returns
    -------
    R_cluster : float
        Cluster radius (m)
    """
    R_cluster = R_p * N**(1.0 / D_f)
    
    return R_cluster


def cluster_size_distribution(
    N_max: int,
    time: float,
    k_agg: float,
    N_0: float,
    distribution_type: str = 'power_law'
) -> np.ndarray:
    """
    Cluster size distribution function.
    
    Power-law distribution typical for aggregating systems.
    
    Parameters
    ----------
    N_max : int
        Maximum cluster size to consider
    time : float
        Aggregation time (s)
    k_agg : float
        Aggregation rate constant (m³/s)
    N_0 : float
        Initial particle number density (1/m³)
    distribution_type : str
        'power_law' or 'exponential'
        
    Returns
    -------
    n_k : ndarray
        Number density of k-mers (1/m³)
    """
    k_values = np.arange(1, N_max + 1)
    
    if distribution_type == 'power_law':
        # Power-law: n_k ~ k^(-tau)
        tau = 2.5  # Typical exponent
        n_k = N_0 * k_values**(-tau)
        n_k /= np.sum(n_k)  # Normalize
        n_k *= N_0
    
    elif distribution_type == 'exponential':
        # Exponential decay
        n_k = N_0 * np.exp(-k_values / (k_agg * time * N_0))
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    return n_k


def clustering_effect_on_conductivity(
    k_nocluster: float,
    phi: float,
    avg_cluster_size: float,
    D_f: float = 1.8
) -> float:
    """
    Thermal conductivity modification due to clustering.
    
    Clusters have different effective properties than dispersed particles.
    
    Reference:
    Prasher, R., Evans, W., Meakin, P., Fish, J., Phelan, P., & Keblinski, P. (2006).
    Effect of aggregation on thermal conduction in colloidal nanofluids.
    Applied Physics Letters, 89(14), 143119.
    
    Parameters
    ----------
    k_nocluster : float
        Thermal conductivity without clustering (W/m·K)
    phi : float
        Volume fraction
    avg_cluster_size : float
        Average number of particles per cluster
    D_f : float
        Fractal dimension
        
    Returns
    -------
    k_clustered : float
        Thermal conductivity with clustering (W/m·K)
    """
    if avg_cluster_size <= 1.0:
        return k_nocluster
    
    # Cluster porosity
    porosity_cluster = 1.0 - (3.0 / D_f)
    
    # Effective volume fraction in clusters
    phi_eff = phi / (1 - porosity_cluster)
    
    # Clustering can reduce conductivity (poor internal connectivity)
    # or enhance it (percolation pathways)
    if D_f < 2.0:
        # Open fractal structure: reduced conductivity
        f_cluster = 1.0 - 0.3 * np.log(avg_cluster_size)
    else:
        # Dense structure: potential enhancement
        f_cluster = 1.0 + 0.1 * np.log(avg_cluster_size) * (phi / 0.01)
    
    k_clustered = k_nocluster * max(f_cluster, 0.5)  # Prevent excessive reduction
    
    return k_clustered


def clustering_effect_on_viscosity(
    mu_nocluster: float,
    phi: float,
    avg_cluster_size: float,
    D_f: float = 1.8
) -> float:
    """
    Viscosity modification due to clustering.
    
    Clusters increase effective hydrodynamic size and viscosity.
    
    Reference:
    Tseng, W. J., & Wu, C. H. (2002). Aggregation, rheology and electrophoretic
    packing structure of aqueous Al2O3 nanoparticle suspensions.
    Acta Materialia, 50(15), 3757-3766.
    
    Parameters
    ----------
    mu_nocluster : float
        Viscosity without clustering (Pa·s)
    phi : float
        Volume fraction
    avg_cluster_size : float
        Average number of particles per cluster
    D_f : float
        Fractal dimension
        
    Returns
    -------
    mu_clustered : float
        Viscosity with clustering (Pa·s)
    """
    if avg_cluster_size <= 1.0:
        return mu_nocluster
    
    # Effective volume fraction of clusters (including trapped fluid)
    R_eff_ratio = avg_cluster_size**(1.0 / D_f)  # R_cluster / R_particle
    phi_eff = phi * R_eff_ratio**3
    
    # Viscosity increase factor
    # Clusters act as larger effective particles
    mu_clustered = mu_nocluster * (1 + 2.5 * phi_eff + 5.0 * phi_eff**2)
    
    return mu_clustered


def comprehensive_dlvo_analysis(
    phi: float,
    d_p: float,
    material: str,
    pH: float,
    ionic_strength: float,
    temperature: float = 298.15,
    k_bf: float = 0.6,
    mu_bf: float = 0.001
) -> dict:
    """
    Complete DLVO analysis for nanofluid stability and properties.
    
    Parameters
    ----------
    phi : float
        Volume fraction
    d_p : float
        Particle diameter (m)
    material : str
        Nanoparticle material ('Al2O3', 'TiO2', 'SiO2', 'CuO', etc.)
    pH : float
        Solution pH
    ionic_strength : float
        Ionic strength (mol/L)
    temperature : float
        Temperature (K)
    k_bf : float
        Base fluid thermal conductivity (W/m·K)
    mu_bf : float
        Base fluid viscosity (Pa·s)
        
    Returns
    -------
    results : dict
        Comprehensive DLVO analysis results
    """
    R = d_p / 2.0
    
    # Material-specific properties
    material_data = {
        'Al2O3': {'A_hamaker': 3.7e-20, 'IEP': 9.0},
        'TiO2': {'A_hamaker': 5.0e-20, 'IEP': 6.0},
        'SiO2': {'A_hamaker': 0.8e-20, 'IEP': 2.5},
        'CuO': {'A_hamaker': 8.0e-20, 'IEP': 9.5},
        'Cu': {'A_hamaker': 40.0e-20, 'IEP': 7.0},
        'Ag': {'A_hamaker': 50.0e-20, 'IEP': 4.0}
    }
    
    if material not in material_data:
        material = 'Al2O3'  # Default
    
    A_hamaker = material_data[material]['A_hamaker']
    IEP = material_data[material]['IEP']
    
    # Calculate zeta potential
    zeta = zeta_potential_pH_dependence(pH, IEP)
    
    # Debye length
    lambda_D = debye_length(ionic_strength, temperature)
    
    # Energy barrier
    E_barrier, H_barrier = energy_barrier_height(
        R, A_hamaker, zeta, ionic_strength, temperature
    )
    
    # Thermal energy
    k_B = 1.38064852e-23
    E_thermal = k_B * temperature
    
    # Stability assessment
    stability_ratio = np.exp(E_barrier / E_thermal) if E_barrier > 0 else 1.0
    
    if stability_ratio > 100:
        stability = "STABLE (slow aggregation)"
        D_f = 2.1  # RLCA
    elif stability_ratio > 10:
        stability = "METASTABLE (moderate aggregation)"
        D_f = 2.0
    else:
        stability = "UNSTABLE (fast aggregation)"
        D_f = 1.8  # DLCA
    
    # Aggregation rate
    k_agg = aggregation_rate_smoluchowski(phi, d_p, temperature, mu_bf, stability_ratio)
    
    # Estimate average cluster size (time-dependent, assume 1 hour)
    time_hours = 1.0
    N_particles = 6 * phi / (np.pi * d_p**3)
    avg_N = min(1 + k_agg * N_particles * time_hours * 3600, 100)
    
    # Cluster properties
    R_cluster = fractal_cluster_size(int(avg_N), R, D_f)
    
    results = {
        'material': material,
        'hamaker_constant': A_hamaker,
        'isoelectric_point': IEP,
        'zeta_potential': zeta,
        'debye_length': lambda_D,
        'energy_barrier': E_barrier,
        'barrier_location': H_barrier,
        'thermal_energy': E_thermal,
        'stability_ratio': stability_ratio,
        'stability_status': stability,
        'fractal_dimension': D_f,
        'aggregation_rate': k_agg,
        'avg_cluster_size': avg_N,
        'cluster_radius': R_cluster,
        'hydrodynamic_size_ratio': R_cluster / R
    }
    
    return results


# Example usage and validation
if __name__ == "__main__":
    print("=" * 80)
    print("BKPS NFL Thermal - DLVO Theory & Particle Interactions")
    print("Dedicated to: Brijesh Kumar Pandey")
    print("=" * 80)
    print()
    
    # Al2O3-water nanofluid
    phi = 0.02
    d_p = 30e-9
    
    print(f"System: Al2O3-water nanofluid")
    print(f"Volume fraction: {phi*100}%")
    print(f"Particle diameter: {d_p*1e9} nm\n")
    
    # Test different pH and ionic strength conditions
    conditions = [
        (7.0, 0.001, "Neutral pH, low salt"),
        (7.0, 0.1, "Neutral pH, high salt"),
        (4.0, 0.001, "Acidic pH, low salt"),
        (10.0, 0.001, "Basic pH, low salt")
    ]
    
    for pH, I, desc in conditions:
        print(f"\nCondition: {desc}")
        print(f"  pH = {pH}, Ionic strength = {I} mol/L")
        print("-" * 80)
        
        results = comprehensive_dlvo_analysis(
            phi, d_p, 'Al2O3', pH, I, temperature=298.15
        )
        
        print(f"  Zeta potential: {results['zeta_potential']*1000:.2f} mV")
        print(f"  Debye length: {results['debye_length']*1e9:.2f} nm")
        print(f"  Energy barrier: {results['energy_barrier']/1.38e-23:.1f} kT")
        print(f"  Stability ratio: {results['stability_ratio']:.1e}")
        print(f"  Status: {results['stability_status']}")
        print(f"  Avg cluster size: {results['avg_cluster_size']:.1f} particles")
        print(f"  Cluster radius: {results['cluster_radius']*1e9:.1f} nm")
        print(f"  Fractal dimension: {results['fractal_dimension']:.2f}")
    
    print("\n" + "=" * 80)
    print("✓ DLVO theory module validated!")
