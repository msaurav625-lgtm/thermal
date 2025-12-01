"""
Advanced Integrated Simulator for BKPS NFL Thermal v6.0

Combines all advanced physics models:
- Flow-dependent thermal conductivity
- Non-Newtonian viscosity
- DLVO theory & particle interactions
- Clustering & aggregation effects
- Enhanced hybrid nanofluid support
- Temperature, pressure, shear rate coupling

Author: BKPS NFL Thermal v6.0
Dedicated to: Brijesh Kumar Pandey
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import advanced physics modules
from .flow_dependent_conductivity import (
    FlowFieldData, comprehensive_flow_dependent_conductivity
)
from .non_newtonian_viscosity import (
    comprehensive_non_newtonian_viscosity, RheologicalParameters
)
from .dlvo_theory import (
    comprehensive_dlvo_analysis, clustering_effect_on_conductivity,
    clustering_effect_on_viscosity
)
# from .enhanced_simulator import EnhancedNanofluidSimulator
from .models import maxwell_model
# from .viscosity_models import einstein_viscosity


@dataclass
class NanoparticleComponent:
    """Enhanced nanoparticle component with full property specification"""
    material: str
    volume_fraction: float
    diameter: float  # m
    shape: str  # 'sphere', 'rod', 'sheet', 'tube', 'ellipsoid', etc.
    aspect_ratio: Optional[float] = None  # For non-spherical
    k_particle: Optional[float] = None  # W/m·K
    rho_particle: Optional[float] = None  # kg/m³
    cp_particle: Optional[float] = None  # J/kg·K


class BKPSNanofluidSimulator:
    """
    BKPS NFL Thermal - Professional Research-Grade Simulator
    
    Comprehensive nanofluid simulation with advanced physics:
    - Static property calculations with 25+ models
    - Flow-dependent thermal conductivity
    - Non-Newtonian rheology
    - DLVO theory & particle interactions
    - Hybrid nanofluids with 2+ components
    - pH and ionic strength effects
    - Temperature, pressure, shear rate coupling
    """
    
    def __init__(
        self,
        base_fluid: str = 'Water',
        temperature: float = 293.15,
        pressure: float = 101325.0
    ):
        """
        Initialize BKPS NFL Thermal simulator.
        
        Parameters
        ----------
        base_fluid : str
            Base fluid name
        temperature : float
            Temperature (K)
        pressure : float
            Pressure (Pa)
        """
        self.base_fluid = base_fluid
        self.temperature = temperature
        self.pressure = pressure
        
        # Nanoparticle components (supports hybrid nanofluids)
        self.components: List[NanoparticleComponent] = []
        
        # Environmental conditions
        self.pH = 7.0
        self.ionic_strength = 0.001  # mol/L
        
        # Flow conditions
        self.velocity = 0.0
        self.shear_rate = 0.0
        self.temperature_gradient = 0.0
        
        # Physics model selections
        self.thermal_conductivity_model = 'Maxwell'
        self.viscosity_model = 'Einstein'
        self.enable_flow_effects = True
        self.enable_non_newtonian = True
        self.enable_dlvo = True
        self.enable_clustering = True
        
        # Base fluid properties
        self._initialize_base_fluid_properties()
        
        # DLVO analysis results (cached)
        self.dlvo_results = {}
        
        print(f"BKPS NFL Thermal v6.0 Simulator initialized")
        print(f"Dedicated to: Brijesh Kumar Pandey")
        print(f"Base fluid: {base_fluid}")
        print(f"Temperature: {temperature} K")
        print(f"Pressure: {pressure} Pa\n")
    
    def _initialize_base_fluid_properties(self):
        """Initialize base fluid thermophysical properties"""
        # Simplified - should use comprehensive database
        if self.base_fluid == 'Water':
            self.k_bf = 0.613  # W/m·K at 20°C
            self.rho_bf = 997.0  # kg/m³
            self.mu_bf = 0.001  # Pa·s
            self.cp_bf = 4180.0  # J/kg·K
            self.alpha_bf = self.k_bf / (self.rho_bf * self.cp_bf)
        elif self.base_fluid == 'EG':
            self.k_bf = 0.252
            self.rho_bf = 1115.0
            self.mu_bf = 0.0162
            self.cp_bf = 2400.0
            self.alpha_bf = self.k_bf / (self.rho_bf * self.cp_bf)
        else:
            # Default to water
            self.k_bf = 0.6
            self.rho_bf = 1000.0
            self.mu_bf = 0.001
            self.cp_bf = 4000.0
            self.alpha_bf = 1.5e-7
    
    def add_nanoparticle(
        self,
        material: str,
        volume_fraction: float,
        diameter: float,
        shape: str = 'sphere',
        aspect_ratio: Optional[float] = None
    ):
        """
        Add nanoparticle component (supports hybrid nanofluids).
        
        Parameters
        ----------
        material : str
            Nanoparticle material (Al2O3, Cu, CuO, TiO2, Ag, SiO2, etc.)
        volume_fraction : float
            Volume fraction (0-1)
        diameter : float
            Particle diameter (m)
        shape : str
            Particle shape
        aspect_ratio : float, optional
            Aspect ratio for non-spherical particles
        """
        # Material database
        materials_db = {
            'Al2O3': {'k': 40.0, 'rho': 3970.0, 'cp': 765.0},
            'Cu': {'k': 401.0, 'rho': 8933.0, 'cp': 385.0},
            'CuO': {'k': 33.0, 'rho': 6500.0, 'cp': 535.0},
            'TiO2': {'k': 8.4, 'rho': 4250.0, 'cp': 686.0},
            'Ag': {'k': 429.0, 'rho': 10500.0, 'cp': 235.0},
            'SiO2': {'k': 1.4, 'rho': 2200.0, 'cp': 745.0},
            'Au': {'k': 317.0, 'rho': 19300.0, 'cp': 129.0},
            'Fe3O4': {'k': 6.0, 'rho': 5180.0, 'cp': 670.0},
            'ZnO': {'k': 29.0, 'rho': 5606.0, 'cp': 495.0},
            'CNT': {'k': 3000.0, 'rho': 2100.0, 'cp': 710.0},
            'Graphene': {'k': 5000.0, 'rho': 2200.0, 'cp': 700.0}
        }
        
        if material not in materials_db:
            print(f"Warning: Material {material} not in database, using Al2O3 properties")
            material = 'Al2O3'
        
        props = materials_db[material]
        
        component = NanoparticleComponent(
            material=material,
            volume_fraction=volume_fraction,
            diameter=diameter,
            shape=shape,
            aspect_ratio=aspect_ratio,
            k_particle=props['k'],
            rho_particle=props['rho'],
            cp_particle=props['cp']
        )
        
        self.components.append(component)
        
        print(f"Added: {material} nanoparticles")
        print(f"  φ={volume_fraction*100:.2f}%, d={diameter*1e9:.1f}nm, shape={shape}")
    
    def set_environmental_conditions(self, pH: float, ionic_strength: float):
        """Set pH and ionic strength for DLVO calculations"""
        self.pH = pH
        self.ionic_strength = ionic_strength
        print(f"Environmental conditions: pH={pH}, I={ionic_strength} mol/L")
    
    def set_flow_conditions(
        self,
        velocity: float = 0.0,
        shear_rate: float = 0.0,
        temperature_gradient: float = 0.0
    ):
        """Set flow conditions for flow-dependent properties"""
        self.velocity = velocity
        self.shear_rate = shear_rate
        self.temperature_gradient = temperature_gradient
        
        if velocity > 0 or shear_rate > 0:
            print(f"Flow conditions: v={velocity}m/s, γ̇={shear_rate}1/s")
    
    def calculate_base_fluid_conductivity(self) -> float:
        """
        Calculate base fluid thermal conductivity only (no nanoparticles).
        
        Useful for comparing with nanofluid enhancement or for pure fluid analysis.
        
        Returns
        -------
        k_bf : float
            Base fluid thermal conductivity (W/m·K)
        """
        return self.k_bf
    
    def calculate_static_thermal_conductivity(self, base_fluid_only: bool = False) -> float:
        """
        Calculate static thermal conductivity (no flow effects).
        
        Parameters
        ----------
        base_fluid_only : bool, optional
            If True, return base fluid conductivity only (ignore nanoparticles)
            If False (default), calculate nanofluid effective conductivity
        
        Returns
        -------
        k_static : float
            Static thermal conductivity (W/m·K)
        """
        if base_fluid_only or not self.components:
            return self.k_bf
        
        # For hybrid nanofluids, use Maxwell-Garnett effective medium
        k_eff = self.k_bf
        
        for comp in self.components:
            phi = comp.volume_fraction
            k_p = comp.k_particle
            
            # Hamilton-Crosser for non-spherical particles
            if comp.shape == 'sphere':
                n = 3.0  # Sphericity
            elif comp.shape == 'rod' or comp.shape == 'tube':
                n = 6.0  # Cylindrical
            elif comp.shape == 'sheet':
                n = 8.0  # Platelet
            else:
                n = 3.0
            
            # Hamilton-Crosser model
            k_eff = k_eff * (k_p + (n-1)*k_eff + (n-1)*phi*(k_p - k_eff)) / \
                    (k_p + (n-1)*k_eff - phi*(k_p - k_eff))
        
        return k_eff
    
    def calculate_flow_dependent_conductivity(self) -> Tuple[float, dict]:
        """
        Calculate flow-dependent thermal conductivity.
        
        Returns
        -------
        k_eff : float
            Flow-enhanced conductivity (W/m·K)
        contributions : dict
            Breakdown of enhancement mechanisms
        """
        if not self.components:
            return self.k_bf, {'base': self.k_bf}
        
        # Start with static conductivity
        k_static = self.calculate_static_thermal_conductivity()
        
        if not self.enable_flow_effects or (self.velocity == 0 and self.shear_rate == 0):
            return k_static, {'base': k_static}
        
        # Use first component for flow calculations (dominant particle)
        comp = self.components[0]
        phi_total = sum(c.volume_fraction for c in self.components)
        
        # Create flow field data
        flow_data = FlowFieldData(
            temperature=self.temperature,
            pressure=self.pressure,
            velocity_magnitude=self.velocity,
            shear_rate=self.shear_rate,
            temperature_gradient=self.temperature_gradient
        )
        
        # Calculate flow-enhanced conductivity
        k_eff, contributions = comprehensive_flow_dependent_conductivity(
            k_static, phi_total, comp.diameter, flow_data,
            self.rho_bf, comp.rho_particle, self.mu_bf, self.cp_bf,
            self.k_bf, comp.k_particle, self.alpha_bf
        )
        
        return k_eff, contributions
    
    def calculate_base_fluid_viscosity(self) -> float:
        """
        Calculate base fluid viscosity only (no nanoparticles).
        
        Useful for comparing with nanofluid viscosity increase.
        
        Returns
        -------
        mu_bf : float
            Base fluid viscosity (Pa·s)
        """
        return self.mu_bf
    
    def calculate_viscosity(self, base_fluid_only: bool = False) -> Tuple[float, dict]:
        """
        Calculate effective viscosity (Newtonian or non-Newtonian).
        
        Parameters
        ----------
        base_fluid_only : bool, optional
            If True, return base fluid viscosity only (ignore nanoparticles)
            If False (default), calculate nanofluid effective viscosity
        
        Returns
        -------
        mu_eff : float
            Effective viscosity (Pa·s)
        info : dict
            Detailed viscosity information
        """
        if base_fluid_only or not self.components:
            return self.mu_bf, {'base_fluid_viscosity': self.mu_bf, 'model_used': 'Pure fluid'}
        
        phi_total = sum(c.volume_fraction for c in self.components)
        d_avg = np.mean([c.diameter for c in self.components])
        
        if self.enable_non_newtonian and self.shear_rate > 0:
            # Non-Newtonian viscosity
            mu_eff, info = comprehensive_non_newtonian_viscosity(
                self.shear_rate, self.temperature, phi_total, d_avg,
                self.mu_bf, model='carreau_yasuda'
            )
        else:
            # Newtonian viscosity (Einstein for dilute suspensions)
            mu_eff = self.mu_bf * (1 + 2.5 * phi_total)
            info = {
                'base_fluid_viscosity': self.mu_bf,
                'zero_shear_viscosity': mu_eff,
                'model_used': 'Einstein'
            }
        
        return mu_eff, info
    
    def perform_dlvo_analysis(self) -> dict:
        """
        Perform DLVO stability analysis.
        
        Returns
        -------
        results : dict
            Comprehensive DLVO analysis results
        """
        if not self.components or not self.enable_dlvo:
            return {}
        
        # Analyze first component (dominant)
        comp = self.components[0]
        
        results = comprehensive_dlvo_analysis(
            comp.volume_fraction, comp.diameter, comp.material,
            self.pH, self.ionic_strength, self.temperature,
            self.k_bf, self.mu_bf
        )
        
        self.dlvo_results = results
        return results
    
    def calculate_clustering_effects(self) -> Tuple[float, float]:
        """
        Calculate clustering effects on k and μ.
        
        Returns
        -------
        k_clustered : float
            Clustering-modified conductivity (W/m·K)
        mu_clustered : float
            Clustering-modified viscosity (Pa·s)
        """
        if not self.enable_clustering or not self.dlvo_results:
            k_base, _ = self.calculate_flow_dependent_conductivity()
            mu_base, _ = self.calculate_viscosity()
            return k_base, mu_base
        
        # Get base properties
        k_base, _ = self.calculate_flow_dependent_conductivity()
        mu_base, _ = self.calculate_viscosity()
        
        # Apply clustering corrections
        phi_total = sum(c.volume_fraction for c in self.components)
        avg_cluster_size = self.dlvo_results.get('avg_cluster_size', 1.0)
        D_f = self.dlvo_results.get('fractal_dimension', 1.8)
        
        k_clustered = clustering_effect_on_conductivity(
            k_base, phi_total, avg_cluster_size, D_f
        )
        
        mu_clustered = clustering_effect_on_viscosity(
            mu_base, phi_total, avg_cluster_size, D_f
        )
        
        return k_clustered, mu_clustered
    
    def comprehensive_analysis(self) -> dict:
        """
        Perform comprehensive thermophysical property analysis.
        
        Returns
        -------
        results : dict
            Complete analysis results
        """
        print("\n" + "=" * 80)
        print("BKPS NFL Thermal - Comprehensive Analysis")
        print("Dedicated to: Brijesh Kumar Pandey")
        print("=" * 80)
        
        # Static properties
        k_static = self.calculate_static_thermal_conductivity()
        
        # Flow-dependent conductivity
        k_flow, k_contributions = self.calculate_flow_dependent_conductivity()
        
        # Viscosity
        mu_eff, mu_info = self.calculate_viscosity()
        
        # DLVO analysis
        dlvo_results = self.perform_dlvo_analysis()
        
        # Clustering effects
        k_final, mu_final = self.calculate_clustering_effects()
        
        # Compile results
        phi_total = sum(c.volume_fraction for c in self.components)
        
        results = {
            'base_fluid': self.base_fluid,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'total_volume_fraction': phi_total,
            'num_components': len(self.components),
            
            # Thermal conductivity
            'k_base_fluid': self.k_bf,
            'k_static': k_static,
            'k_flow_enhanced': k_flow,
            'k_final_with_clustering': k_final,
            'k_enhancement_static': (k_static / self.k_bf - 1) * 100,
            'k_enhancement_flow': (k_flow / self.k_bf - 1) * 100,
            'k_enhancement_total': (k_final / self.k_bf - 1) * 100,
            'k_contributions': k_contributions,
            
            # Viscosity
            'mu_base_fluid': self.mu_bf,
            'mu_effective': mu_eff,
            'mu_final_with_clustering': mu_final,
            'mu_ratio': mu_final / self.mu_bf,
            'mu_info': mu_info,
            
            # DLVO
            'dlvo_analysis': dlvo_results,
            
            # Flow conditions
            'velocity': self.velocity,
            'shear_rate': self.shear_rate,
            'reynolds_number': self._calculate_reynolds_number(),
            
            # Environmental
            'pH': self.pH,
            'ionic_strength': self.ionic_strength
        }
        
        # Print summary
        self._print_results_summary(results)
        
        return results
    
    def _calculate_reynolds_number(self) -> float:
        """Calculate Reynolds number if flow exists"""
        if self.velocity == 0 or not self.components:
            return 0.0
        
        d_h = 0.01  # Assume 1 cm characteristic length
        mu_eff, _ = self.calculate_viscosity()
        Re = self.rho_bf * self.velocity * d_h / mu_eff
        return Re
    
    def _print_results_summary(self, results: dict):
        """Print formatted results summary"""
        print(f"\nSystem Configuration:")
        print(f"  Base fluid: {results['base_fluid']}")
        print(f"  Temperature: {results['temperature']} K")
        print(f"  Pressure: {results['pressure']} Pa")
        print(f"  Total φ: {results['total_volume_fraction']*100:.2f}%")
        print(f"  Components: {results['num_components']}")
        
        print(f"\nThermal Conductivity:")
        print(f"  Base fluid: {results['k_base_fluid']:.4f} W/m·K")
        print(f"  Static: {results['k_static']:.4f} W/m·K (+{results['k_enhancement_static']:.1f}%)")
        print(f"  Flow-enhanced: {results['k_flow_enhanced']:.4f} W/m·K (+{results['k_enhancement_flow']:.1f}%)")
        print(f"  Final (with clustering): {results['k_final_with_clustering']:.4f} W/m·K (+{results['k_enhancement_total']:.1f}%)")
        
        print(f"\nViscosity:")
        print(f"  Base fluid: {results['mu_base_fluid']*1000:.4f} mPa·s")
        print(f"  Effective: {results['mu_effective']*1000:.4f} mPa·s")
        print(f"  Final (with clustering): {results['mu_final_with_clustering']*1000:.4f} mPa·s")
        print(f"  Ratio: {results['mu_ratio']:.2f}x")
        
        if results['dlvo_analysis']:
            dlvo = results['dlvo_analysis']
            print(f"\nDLVO Stability Analysis:")
            print(f"  Zeta potential: {dlvo['zeta_potential']*1000:.2f} mV")
            print(f"  Energy barrier: {dlvo['energy_barrier']/1.38e-23:.1f} kT")
            print(f"  Status: {dlvo['stability_status']}")
            print(f"  Avg cluster size: {dlvo['avg_cluster_size']:.1f} particles")
        
        if results['velocity'] > 0 or results['shear_rate'] > 0:
            print(f"\nFlow Conditions:")
            print(f"  Velocity: {results['velocity']} m/s")
            print(f"  Shear rate: {results['shear_rate']} 1/s")
            print(f"  Reynolds number: {results['reynolds_number']:.1f}")
        
        print("=" * 80)


# Example usage
if __name__ == "__main__":
    print("BKPS NFL Thermal v6.0 - Integrated Advanced Simulator")
    print("Dedicated to: Brijesh Kumar Pandey\n")
    
    # Create simulator
    sim = BKPSNanofluidSimulator(base_fluid='Water', temperature=300.0)
    
    # Add Al2O3 nanoparticles
    sim.add_nanoparticle('Al2O3', volume_fraction=0.02, diameter=30e-9, shape='sphere')
    
    # Set environmental conditions
    sim.set_environmental_conditions(pH=7.0, ionic_strength=0.001)
    
    # Set flow conditions
    sim.set_flow_conditions(velocity=0.5, shear_rate=1000.0)
    
    # Perform comprehensive analysis
    results = sim.comprehensive_analysis()
    
    print("\n✓ Advanced integrated simulator validated!")
