"""
Advanced Flow-Dependent Property Calculator with Full User Control

Features:
- Flow-dependent thermal conductivity (Buongiorno, Kumar, Rea-Guzman models)
- Flow-dependent viscosity (shear-thinning/thickening)
- Single, multiple, or no nanoparticle selection
- Range-based parameter sweeps
- Comparison mode for multiple configurations

Author: BKPS NFL Thermal v7.1
Dedicated to: Brijesh Kumar Pandey
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings


@dataclass
class FlowConditions:
    """Flow condition parameters"""
    velocity: float = 0.05  # m/s
    shear_rate: Optional[float] = None  # 1/s (calculated if None)
    reynolds_number: Optional[float] = None  # Calculated
    temperature: float = 300.0  # K
    pressure: float = 101325.0  # Pa
    characteristic_length: float = 0.01  # m (hydraulic diameter)


@dataclass
class NanoparticleSpec:
    """Nanoparticle specification with optional ranges"""
    material: str
    volume_fraction: Union[float, Tuple[float, float, int]] = 0.02  # Single or (min, max, steps)
    diameter: Union[float, Tuple[float, float, int]] = 30e-9  # Single or (min, max, steps)
    shape: str = 'sphere'
    k_particle: Optional[float] = None  # W/m·K
    rho_particle: Optional[float] = None  # kg/m³
    enabled: bool = True  # User can disable without removing


@dataclass
class FlowDependentConfig:
    """Complete configuration for flow-dependent calculations"""
    base_fluid: str = 'Water'
    nanoparticles: List[NanoparticleSpec] = field(default_factory=list)
    flow_conditions: FlowConditions = field(default_factory=FlowConditions)
    
    # Model selections
    conductivity_models: List[str] = field(default_factory=lambda: ['buongiorno', 'kumar', 'static'])
    viscosity_models: List[str] = field(default_factory=lambda: ['brinkman', 'batchelor'])
    
    # Physics options
    include_brownian: bool = True
    include_thermophoresis: bool = True
    include_shear_effects: bool = True
    include_clustering: bool = False
    
    # Range sweep options
    sweep_velocity: Optional[Tuple[float, float, int]] = None  # (min, max, steps)
    sweep_temperature: Optional[Tuple[float, float, int]] = None


class AdvancedFlowCalculator:
    """
    Advanced calculator for flow-dependent thermal conductivity and viscosity
    
    User Controls:
    1. Select 0, 1, or multiple nanoparticles
    2. Define parameter ranges or single values
    3. Choose which models to compute
    4. Enable/disable physics effects
    5. Sweep over velocity, temperature, etc.
    """
    
    # Material database
    MATERIAL_DATABASE = {
        'Al2O3': {'k': 40.0, 'rho': 3970, 'cp': 880},
        'CuO': {'k': 76.5, 'rho': 6310, 'cp': 540},
        'TiO2': {'k': 8.9, 'rho': 4250, 'cp': 686},
        'SiO2': {'k': 1.4, 'rho': 2220, 'cp': 745},
        'Cu': {'k': 401.0, 'rho': 8933, 'cp': 385},
        'Ag': {'k': 429.0, 'rho': 10500, 'cp': 235},
        'Au': {'k': 318.0, 'rho': 19300, 'cp': 129},
        'Fe3O4': {'k': 9.7, 'rho': 5180, 'cp': 670},
        'ZnO': {'k': 29.0, 'rho': 5606, 'cp': 495},
        'CNT': {'k': 3000.0, 'rho': 2100, 'cp': 700},
        'Graphene': {'k': 5000.0, 'rho': 2200, 'cp': 710},
    }
    
    BASE_FLUID_DATABASE = {
        'Water': {'k': 0.613, 'rho': 998.0, 'mu': 0.001, 'cp': 4182},
        'EG': {'k': 0.252, 'rho': 1115, 'mu': 0.0157, 'cp': 2430},  # Ethylene glycol
        'Oil': {'k': 0.145, 'rho': 880, 'mu': 0.05, 'cp': 1900},
    }
    
    def __init__(self, config: FlowDependentConfig):
        """Initialize with configuration"""
        self.config = config
        self._validate_config()
        
        # Get base fluid properties
        if config.base_fluid in self.BASE_FLUID_DATABASE:
            self.bf_props = self.BASE_FLUID_DATABASE[config.base_fluid]
        else:
            raise ValueError(f"Unknown base fluid: {config.base_fluid}")
    
    def _validate_config(self):
        """Validate configuration"""
        # Check at least one model selected
        if not self.config.conductivity_models and not self.config.viscosity_models:
            warnings.warn("No models selected. Using defaults.")
            self.config.conductivity_models = ['buongiorno']
            self.config.viscosity_models = ['brinkman']
        
        # Validate nanoparticles
        enabled_count = sum(1 for np in self.config.nanoparticles if np.enabled)
        if enabled_count == 0:
            warnings.warn("No nanoparticles enabled. Will calculate base fluid properties only.")
    
    def calculate_static_conductivity(
        self,
        phi: float,
        k_p: float,
        model: str = 'maxwell'
    ) -> float:
        """
        Calculate static thermal conductivity
        
        Parameters
        ----------
        phi : float
            Volume fraction
        k_p : float
            Particle conductivity
        model : str
            Model name
            
        Returns
        -------
        k_nf : float
            Nanofluid conductivity (W/m·K)
        """
        k_bf = self.bf_props['k']
        
        if model.lower() == 'maxwell':
            # Maxwell model
            k_nf = k_bf * (k_p + 2*k_bf + 2*phi*(k_p - k_bf)) / (k_p + 2*k_bf - phi*(k_p - k_bf))
        
        elif model.lower() == 'hamilton-crosser':
            # Hamilton-Crosser for spherical particles (n=3)
            n = 3
            k_nf = k_bf * (k_p + (n-1)*k_bf + (n-1)*phi*(k_p - k_bf)) / (k_p + (n-1)*k_bf - phi*(k_p - k_bf))
        
        elif model.lower() == 'bruggeman':
            # Bruggeman model (implicit solution)
            # Simplified explicit approximation
            k_nf = k_bf * (1 + 3*phi*(k_p/k_bf - 1) / (k_p/k_bf + 2 - phi*(k_p/k_bf - 1)))
        
        else:
            k_nf = k_bf * (1 + 3*phi)  # Simple linear
        
        return k_nf
    
    def calculate_buongiorno_conductivity(
        self,
        phi: float,
        d_p: float,
        k_static: float,
        velocity: float,
        temperature: float
    ) -> float:
        """
        Buongiorno model for flow-enhanced conductivity
        
        Reference: Buongiorno, J. (2006). ASME J. Heat Transfer, 128(3), 240-250.
        """
        k_B = 1.38064852e-23  # Boltzmann constant
        rho_bf = self.bf_props['rho']
        mu_bf = self.bf_props['mu']
        cp_bf = self.bf_props['cp']
        
        # Brownian diffusion coefficient
        D_B = k_B * temperature / (3 * np.pi * mu_bf * d_p)
        
        # Thermophoretic enhancement (simplified)
        if self.config.include_thermophoresis:
            K_T = 0.26 * k_static / (2 * k_static + mu_bf)
            D_T = K_T * mu_bf / rho_bf
        else:
            D_T = 0.0
        
        # Peclet number for flow
        L = self.config.flow_conditions.characteristic_length
        Pe = velocity * L / D_B if D_B > 0 else 0
        
        # Flow enhancement factor (empirical)
        if Pe > 1:
            enhancement = 1 + 0.1 * np.log10(Pe) * phi
        else:
            enhancement = 1.0
        
        # Brownian motion enhancement
        if self.config.include_brownian:
            # Koo-Kleinstreuer model component
            brownian_enhancement = 1 + 5e4 * phi * (k_B * temperature) / (d_p * mu_bf)
        else:
            brownian_enhancement = 1.0
        
        k_flow = k_static * enhancement * brownian_enhancement
        
        return k_flow
    
    def calculate_kumar_conductivity(
        self,
        phi: float,
        k_static: float,
        shear_rate: float
    ) -> float:
        """
        Kumar shear-enhanced conductivity model
        
        Reference: Kumar et al. (2019). Shear rate dependent thermal conductivity
        """
        if not self.config.include_shear_effects or shear_rate == 0:
            return k_static
        
        # Shear enhancement factor
        # Higher shear → better particle distribution → higher k
        shear_factor = 1 + 0.05 * phi * np.log10(1 + shear_rate / 100)
        
        k_shear = k_static * shear_factor
        
        return k_shear
    
    def calculate_rea_guzman_conductivity(
        self,
        phi: float,
        d_p: float,
        k_static: float,
        reynolds: float
    ) -> float:
        """
        Rea-Guzman velocity-dependent model
        
        Reference: Rea et al. (2009). Int. J. Heat Fluid Flow
        """
        # Reynolds-dependent enhancement
        if reynolds < 1:
            re_factor = 1.0
        elif reynolds < 100:
            re_factor = 1 + 0.02 * phi * np.log10(reynolds)
        else:
            re_factor = 1 + 0.04 * phi * np.log10(reynolds)
        
        k_flow = k_static * re_factor
        
        return k_flow
    
    def calculate_viscosity_einstein(self, phi: float) -> float:
        """Einstein viscosity model (dilute, Newtonian)"""
        mu_bf = self.bf_props['mu']
        mu_nf = mu_bf * (1 + 2.5 * phi)
        return mu_nf
    
    def calculate_viscosity_brinkman(self, phi: float) -> float:
        """Brinkman viscosity model (moderate concentrations)"""
        mu_bf = self.bf_props['mu']
        mu_nf = mu_bf / (1 - phi)**2.5
        return mu_nf
    
    def calculate_viscosity_batchelor(self, phi: float) -> float:
        """Batchelor model (includes Brownian effects)"""
        mu_bf = self.bf_props['mu']
        mu_nf = mu_bf * (1 + 2.5*phi + 6.2*phi**2)
        return mu_nf
    
    def calculate_viscosity_shear_dependent(
        self,
        phi: float,
        shear_rate: float,
        model: str = 'power_law'
    ) -> float:
        """
        Shear-dependent viscosity (non-Newtonian)
        
        Parameters
        ----------
        phi : float
            Volume fraction
        shear_rate : float
            Shear rate (1/s)
        model : str
            'power_law', 'carreau', or 'cross'
        """
        mu_bf = self.bf_props['mu']
        
        # Base nanofluid viscosity (Brinkman)
        mu_0 = mu_bf / (1 - phi)**2.5
        
        if not self.config.include_shear_effects or shear_rate == 0:
            return mu_0
        
        if model == 'power_law':
            # Power-law: μ = K * γ^(n-1)
            # For shear-thinning: n < 1
            # For shear-thickening: n > 1
            if phi < 0.02:
                n = 1.0  # Newtonian
            elif phi < 0.05:
                n = 0.95  # Slight shear-thinning
            else:
                n = 0.85  # Shear-thinning
            
            K = mu_0  # Consistency index
            mu_eff = K * shear_rate**(n - 1)
        
        elif model == 'carreau':
            # Carreau model: (μ - μ_inf) / (μ_0 - μ_inf) = [1 + (λγ)^2]^((n-1)/2)
            mu_inf = mu_bf  # Infinite-shear viscosity
            lambda_c = 0.1  # Time constant (s)
            n = 0.9
            
            mu_eff = mu_inf + (mu_0 - mu_inf) / (1 + (lambda_c * shear_rate)**2)**((1-n)/2)
        
        else:  # cross
            # Cross model
            mu_inf = mu_bf
            K = 1.0
            m = 0.5
            
            mu_eff = mu_inf + (mu_0 - mu_inf) / (1 + (K * shear_rate)**m)
        
        return mu_eff
    
    def calculate_reynolds_number(
        self,
        velocity: float,
        viscosity: float
    ) -> float:
        """Calculate Reynolds number"""
        rho = self.bf_props['rho']
        D = self.config.flow_conditions.characteristic_length
        Re = rho * velocity * D / viscosity
        return Re
    
    def calculate_shear_rate(
        self,
        velocity: float,
        geometry: str = 'channel'
    ) -> float:
        """
        Estimate shear rate from velocity and geometry
        
        Parameters
        ----------
        velocity : float
            Flow velocity (m/s)
        geometry : str
            'channel', 'pipe', or 'custom'
        """
        D = self.config.flow_conditions.characteristic_length
        
        if geometry == 'channel':
            # Parallel plate: γ ≈ 6*V/H
            shear_rate = 6 * velocity / D
        elif geometry == 'pipe':
            # Pipe: γ ≈ 8*V/D
            shear_rate = 8 * velocity / D
        else:
            # Generic: γ ≈ V/L
            shear_rate = velocity / D
        
        return shear_rate
    
    def calculate_single_condition(
        self,
        nanoparticle: NanoparticleSpec,
        flow_conditions: FlowConditions
    ) -> Dict:
        """
        Calculate properties for a single condition
        
        Returns
        -------
        results : dict
            Contains k_static, k_flow (per model), mu_static, mu_flow (per model), Re, etc.
        """
        if not nanoparticle.enabled:
            return self._base_fluid_only_results(flow_conditions)
        
        # Get particle properties
        if nanoparticle.k_particle is None:
            if nanoparticle.material in self.MATERIAL_DATABASE:
                k_p = self.MATERIAL_DATABASE[nanoparticle.material]['k']
                rho_p = self.MATERIAL_DATABASE[nanoparticle.material]['rho']
            else:
                raise ValueError(f"Unknown material: {nanoparticle.material}. Provide k_particle.")
        else:
            k_p = nanoparticle.k_particle
            rho_p = nanoparticle.rho_particle or 3000.0
        
        # Handle range or single value
        phi = nanoparticle.volume_fraction if isinstance(nanoparticle.volume_fraction, float) else nanoparticle.volume_fraction[0]
        d_p = nanoparticle.diameter if isinstance(nanoparticle.diameter, float) else nanoparticle.diameter[0]
        
        results = {
            'material': nanoparticle.material,
            'volume_fraction': phi,
            'diameter': d_p,
            'temperature': flow_conditions.temperature,
            'velocity': flow_conditions.velocity,
        }
        
        # Calculate static properties
        k_static = self.calculate_static_conductivity(phi, k_p, 'maxwell')
        results['k_static'] = k_static
        results['k_base'] = self.bf_props['k']
        results['enhancement_static'] = (k_static / self.bf_props['k'] - 1) * 100
        
        # Calculate shear rate
        shear_rate = flow_conditions.shear_rate
        if shear_rate is None:
            shear_rate = self.calculate_shear_rate(flow_conditions.velocity)
        results['shear_rate'] = shear_rate
        
        # Calculate flow-dependent conductivity
        results['conductivity'] = {}
        for model in self.config.conductivity_models:
            if model == 'buongiorno':
                k_flow = self.calculate_buongiorno_conductivity(
                    phi, d_p, k_static, 
                    flow_conditions.velocity, 
                    flow_conditions.temperature
                )
                results['conductivity']['buongiorno'] = k_flow
            
            elif model == 'kumar':
                k_flow = self.calculate_kumar_conductivity(phi, k_static, shear_rate)
                results['conductivity']['kumar'] = k_flow
            
            elif model == 'static':
                results['conductivity']['static'] = k_static
        
        # Calculate viscosity
        results['viscosity'] = {}
        for model in self.config.viscosity_models:
            if model == 'einstein':
                mu = self.calculate_viscosity_einstein(phi)
            elif model == 'brinkman':
                mu = self.calculate_viscosity_brinkman(phi)
            elif model == 'batchelor':
                mu = self.calculate_viscosity_batchelor(phi)
            elif model == 'shear_dependent':
                mu = self.calculate_viscosity_shear_dependent(phi, shear_rate)
            else:
                mu = self.calculate_viscosity_brinkman(phi)
            
            results['viscosity'][model] = mu
        
        # Calculate Reynolds number (using Brinkman viscosity)
        mu_nf = results['viscosity'].get('brinkman', self.bf_props['mu'])
        results['reynolds'] = self.calculate_reynolds_number(flow_conditions.velocity, mu_nf)
        
        # Add if Rea-Guzman model requested
        if 'rea_guzman' in self.config.conductivity_models:
            k_rg = self.calculate_rea_guzman_conductivity(
                phi, d_p, k_static, results['reynolds']
            )
            results['conductivity']['rea_guzman'] = k_rg
        
        return results
    
    def _base_fluid_only_results(self, flow_conditions: FlowConditions) -> Dict:
        """Results for base fluid only (no nanoparticles)"""
        return {
            'material': 'None',
            'volume_fraction': 0.0,
            'diameter': 0.0,
            'temperature': flow_conditions.temperature,
            'velocity': flow_conditions.velocity,
            'k_static': self.bf_props['k'],
            'k_base': self.bf_props['k'],
            'enhancement_static': 0.0,
            'shear_rate': self.calculate_shear_rate(flow_conditions.velocity),
            'conductivity': {'static': self.bf_props['k']},
            'viscosity': {'base': self.bf_props['mu']},
            'reynolds': self.calculate_reynolds_number(flow_conditions.velocity, self.bf_props['mu'])
        }
    
    def calculate_parametric_sweep(self) -> List[Dict]:
        """
        Perform parametric sweep over specified ranges
        
        Returns
        -------
        results : list of dict
            Results for each parameter combination
        """
        all_results = []
        
        # Get enabled nanoparticles
        enabled_nps = [np for np in self.config.nanoparticles if np.enabled]
        
        if not enabled_nps:
            # Base fluid only
            if self.config.sweep_velocity:
                v_min, v_max, v_steps = self.config.sweep_velocity
                velocities = np.linspace(v_min, v_max, v_steps)
            else:
                velocities = [self.config.flow_conditions.velocity]
            
            for v in velocities:
                flow_cond = FlowConditions(
                    velocity=v,
                    temperature=self.config.flow_conditions.temperature,
                    characteristic_length=self.config.flow_conditions.characteristic_length
                )
                result = self._base_fluid_only_results(flow_cond)
                all_results.append(result)
            
            return all_results
        
        # Process each nanoparticle
        for np_spec in enabled_nps:
            # Expand volume fraction range
            if isinstance(np_spec.volume_fraction, tuple):
                phi_min, phi_max, phi_steps = np_spec.volume_fraction
                phis = np.linspace(phi_min, phi_max, phi_steps)
            else:
                phis = [np_spec.volume_fraction]
            
            # Expand diameter range
            if isinstance(np_spec.diameter, tuple):
                d_min, d_max, d_steps = np_spec.diameter
                diameters = np.linspace(d_min, d_max, d_steps)
            else:
                diameters = [np_spec.diameter]
            
            # Expand velocity range if specified
            if self.config.sweep_velocity:
                v_min, v_max, v_steps = self.config.sweep_velocity
                velocities = np.linspace(v_min, v_max, v_steps)
            else:
                velocities = [self.config.flow_conditions.velocity]
            
            # Expand temperature range if specified
            if self.config.sweep_temperature:
                t_min, t_max, t_steps = self.config.sweep_temperature
                temperatures = np.linspace(t_min, t_max, t_steps)
            else:
                temperatures = [self.config.flow_conditions.temperature]
            
            # Calculate for all combinations
            for phi in phis:
                for d_p in diameters:
                    for v in velocities:
                        for T in temperatures:
                            # Create temporary spec with single values
                            temp_np = NanoparticleSpec(
                                material=np_spec.material,
                                volume_fraction=phi,
                                diameter=d_p,
                                shape=np_spec.shape,
                                k_particle=np_spec.k_particle,
                                rho_particle=np_spec.rho_particle,
                                enabled=True
                            )
                            
                            flow_cond = FlowConditions(
                                velocity=v,
                                temperature=T,
                                characteristic_length=self.config.flow_conditions.characteristic_length
                            )
                            
                            result = self.calculate_single_condition(temp_np, flow_cond)
                            all_results.append(result)
        
        return all_results
    
    def calculate_comparison(self) -> Dict[str, List[Dict]]:
        """
        Compare multiple nanoparticle configurations
        
        Returns
        -------
        comparison : dict
            Results grouped by nanoparticle material
        """
        comparison = {}
        
        enabled_nps = [np for np in self.config.nanoparticles if np.enabled]
        
        if not enabled_nps:
            comparison['BaseFluid'] = [self._base_fluid_only_results(self.config.flow_conditions)]
            return comparison
        
        for np_spec in enabled_nps:
            result = self.calculate_single_condition(np_spec, self.config.flow_conditions)
            if np_spec.material not in comparison:
                comparison[np_spec.material] = []
            comparison[np_spec.material].append(result)
        
        return comparison


# ============================================================================
# Convenience Functions
# ============================================================================

def calculate_flow_properties(
    base_fluid: str = 'Water',
    nanoparticles: Optional[List[Dict]] = None,
    velocity: float = 0.05,
    temperature: float = 300.0,
    models: Optional[List[str]] = None
) -> Dict:
    """
    Quick calculation of flow-dependent properties
    
    Parameters
    ----------
    base_fluid : str
        Base fluid name
    nanoparticles : list of dict, optional
        List of nanoparticle specs, each dict with keys:
        'material', 'volume_fraction', 'diameter', 'enabled' (optional)
    velocity : float
        Flow velocity (m/s)
    temperature : float
        Temperature (K)
    models : list of str, optional
        Models to calculate
    
    Returns
    -------
    results : dict
        Calculation results
    """
    # Create nanoparticle specs
    np_specs = []
    if nanoparticles:
        for np_dict in nanoparticles:
            np_specs.append(NanoparticleSpec(
                material=np_dict['material'],
                volume_fraction=np_dict.get('volume_fraction', 0.02),
                diameter=np_dict.get('diameter', 30e-9),
                shape=np_dict.get('shape', 'sphere'),
                enabled=np_dict.get('enabled', True)
            ))
    
    # Create config
    config = FlowDependentConfig(
        base_fluid=base_fluid,
        nanoparticles=np_specs,
        flow_conditions=FlowConditions(velocity=velocity, temperature=temperature),
        conductivity_models=models or ['buongiorno', 'kumar', 'static'],
        viscosity_models=['brinkman', 'batchelor']
    )
    
    # Calculate
    calculator = AdvancedFlowCalculator(config)
    
    if len(np_specs) <= 1:
        # Single or no nanoparticle
        if np_specs:
            return calculator.calculate_single_condition(np_specs[0], config.flow_conditions)
        else:
            return calculator._base_fluid_only_results(config.flow_conditions)
    else:
        # Multiple nanoparticles - return comparison
        return calculator.calculate_comparison()


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("ADVANCED FLOW-DEPENDENT CALCULATOR - Demo")
    print("="*70)
    
    # Example 1: Single nanoparticle
    print("\n1. Single Nanoparticle (Al2O3)")
    result = calculate_flow_properties(
        base_fluid='Water',
        nanoparticles=[{'material': 'Al2O3', 'volume_fraction': 0.02, 'diameter': 30e-9}],
        velocity=0.1,
        temperature=300
    )
    print(f"   Static k: {result['k_static']:.4f} W/m·K")
    print(f"   Buongiorno k: {result['conductivity'].get('buongiorno', 0):.4f} W/m·K")
    print(f"   Reynolds: {result['reynolds']:.1f}")
    
    # Example 2: Multiple nanoparticles comparison
    print("\n2. Multiple Nanoparticles (Al2O3, CuO, Cu)")
    comparison = calculate_flow_properties(
        base_fluid='Water',
        nanoparticles=[
            {'material': 'Al2O3', 'volume_fraction': 0.02, 'diameter': 30e-9},
            {'material': 'CuO', 'volume_fraction': 0.02, 'diameter': 30e-9},
            {'material': 'Cu', 'volume_fraction': 0.02, 'diameter': 30e-9},
        ],
        velocity=0.1
    )
    for material, results in comparison.items():
        r = results[0]
        print(f"   {material}: k_buongiorno={r['conductivity'].get('buongiorno', 0):.4f} W/m·K")
    
    print("\n" + "="*70)
