#!/usr/bin/env python3
"""
BKPS NFL Thermal Pro 7.0 - Unified Engine
Dedicated to: Brijesh Kumar Pandey

Single entry point for all simulation modes:
- Static thermal properties
- Flow-dependent properties
- Full CFD simulation
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import json
import logging
from datetime import datetime
from enum import Enum
import warnings

# Import all physics engines
from .integrated_simulator_v6 import BKPSNanofluidSimulator
from .cfd_solver import NavierStokesSolver
from .cfd_mesh import StructuredMesh2D
from .flow_simulator import FlowNanofluidSimulator  # Renamed from FlowSimulator
from .ai_recommender import AIRecommendationEngine  # Renamed from AIRecommender

# Version info
__version__ = "7.0.0"
__release_date__ = "2025-11-30"
__codename__ = "BKPS NFL Thermal Pro"

# Setup logging
logger = logging.getLogger(__name__)


class SimulationMode(Enum):
    """Simulation mode selection"""
    STATIC = "static"
    FLOW = "flow"
    CFD = "cfd"
    HYBRID = "hybrid"


class SolverBackend(Enum):
    """Computational backend selection"""
    NUMPY = "numpy"
    NUMBA = "numba"
    PYTORCH = "pytorch"
    CUPY = "cupy"


@dataclass
class NanoparticleConfig:
    """Nanoparticle configuration"""
    material: str
    volume_fraction: float
    diameter: float  # meters
    shape: str = "sphere"
    custom_properties: Optional[Dict[str, float]] = None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []
        if self.volume_fraction < 0 or self.volume_fraction > 0.15:
            errors.append(f"Volume fraction {self.volume_fraction} out of range [0, 0.15]")
        if self.diameter <= 0 or self.diameter > 1e-6:
            errors.append(f"Diameter {self.diameter} out of range (0, 1e-6]")
        if self.shape not in ["sphere", "cylinder", "platelet", "tube"]:
            errors.append(f"Shape {self.shape} not recognized")
        return len(errors) == 0, errors


@dataclass
class BaseFluidConfig:
    """Base fluid configuration"""
    name: str
    temperature: float = 300.0  # Kelvin
    pressure: float = 101325.0  # Pa
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []
        if self.temperature < 250 or self.temperature > 500:
            errors.append(f"Temperature {self.temperature} out of range [250, 500] K")
        if self.pressure < 1e4 or self.pressure > 1e7:
            errors.append(f"Pressure {self.pressure} out of range")
        return len(errors) == 0, errors


@dataclass
class GeometryConfig:
    """Flow geometry configuration"""
    geometry_type: str = "channel"  # channel, pipe, custom
    length: float = 0.1  # meters
    height: float = 0.01  # meters
    width: Optional[float] = None  # meters (for 3D)
    diameter: Optional[float] = None  # meters (for pipe)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []
        if self.geometry_type not in ["channel", "pipe", "custom"]:
            errors.append(f"Geometry type {self.geometry_type} not recognized")
        if self.length <= 0:
            errors.append("Length must be positive")
        if self.geometry_type == "pipe" and (self.diameter is None or self.diameter <= 0):
            errors.append("Pipe geometry requires positive diameter")
        return len(errors) == 0, errors


@dataclass
class MeshConfig:
    """Mesh configuration for CFD"""
    nx: int = 50
    ny: int = 50
    nz: Optional[int] = None
    mesh_type: str = "structured"
    refinement_zones: List[Dict] = field(default_factory=list)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []
        if self.nx < 10 or self.ny < 10:
            errors.append("Mesh resolution too coarse (min 10x10)")
        if self.nx > 500 or self.ny > 500:
            errors.append("Mesh resolution too fine (max 500x500)")
        return len(errors) == 0, errors


@dataclass
@dataclass
class SolverConfig:
    """Solver configuration"""
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    relaxation_factor: float = 0.7
    time_step: Optional[float] = None
    enable_turbulence: bool = False
    turbulence_model: str = "k-epsilon"
    backend: SolverBackend = SolverBackend.NUMPY
    num_threads: int = -1  # -1 = auto
    enable_caching: bool = True
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []
        if self.max_iterations < 1:
            errors.append("max_iterations must be positive")
        if self.convergence_tolerance <= 0:
            errors.append("convergence_tolerance must be positive")
        if not 0 < self.relaxation_factor <= 1:
            errors.append("relaxation_factor must be in (0, 1]")
        return len(errors) == 0, errors


@dataclass
class FlowConfig:
    """Flow conditions configuration"""
    velocity: float = 1.0  # m/s (or inlet velocity)
    reynolds_number: Optional[float] = None
    flow_rate: Optional[float] = None  # m³/s
    inlet_temperature: float = 300.0  # K
    wall_temperature: Optional[float] = None  # K (for heat transfer)
    heat_flux: Optional[float] = None  # W/m²
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []
        if self.velocity < 0:
            errors.append("Velocity cannot be negative")
        if self.inlet_temperature < 250 or self.inlet_temperature > 500:
            errors.append("Inlet temperature out of range")
        return len(errors) == 0, errors


@dataclass
@dataclass
class UnifiedConfig:
    """
    Unified configuration for all simulation modes
    
    Usage:
        config = UnifiedConfig(
            mode=SimulationMode.STATIC,
            base_fluid=BaseFluidConfig(name="Water", temperature=300),
            nanoparticles=[NanoparticleConfig(material="Al2O3", volume_fraction=0.02, diameter=30e-9)]
        )
    """
    # Required fields (no defaults)
    mode: SimulationMode
    base_fluid: BaseFluidConfig
    nanoparticles: List[NanoparticleConfig]
    
    # Fields with default_factory (must be together)
    solver: SolverConfig = field(default_factory=SolverConfig)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Optional configurations with simple defaults
    geometry: Optional[GeometryConfig] = None
    mesh: Optional[MeshConfig] = None
    flow: Optional[FlowConfig] = None
    
    # Analysis options
    enable_dlvo: bool = True
    enable_aggregation: bool = True
    enable_non_newtonian: bool = True
    enable_interfacial_layer: bool = True
    enable_ai_recommendations: bool = True
    enable_sensitivity_analysis: bool = False
    enable_uq: bool = False  # Research-grade uncertainty quantification
    
    # Metadata
    project_name: str = "Untitled"
    description: str = ""
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate entire configuration"""
        errors = []
        
        # Validate base fluid
        valid, errs = self.base_fluid.validate()
        errors.extend(errs)
        
        # Validate nanoparticles
        if not self.nanoparticles:
            errors.append("At least one nanoparticle required")
        for i, np_config in enumerate(self.nanoparticles):
            valid, errs = np_config.validate()
            errors.extend([f"Nanoparticle {i}: {e}" for e in errs])
        
        # Mode-specific validation
        if self.mode in [SimulationMode.FLOW, SimulationMode.CFD]:
            if self.geometry is None:
                errors.append("Geometry required for flow/CFD modes")
            else:
                valid, errs = self.geometry.validate()
                errors.extend(errs)
            
            if self.flow is None:
                errors.append("Flow conditions required for flow/CFD modes")
            else:
                valid, errs = self.flow.validate()
                errors.extend(errs)
        
        if self.mode == SimulationMode.CFD:
            if self.mesh is None:
                errors.append("Mesh required for CFD mode")
            else:
                valid, errs = self.mesh.validate()
                errors.extend(errs)
        
        # Validate solver
        valid, errs = self.solver.validate()
        errors.extend(errs)
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['mode'] = self.mode.value
        data['solver']['backend'] = self.solver.backend.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UnifiedConfig':
        """Load from dictionary"""
        # Convert mode
        data['mode'] = SimulationMode(data['mode'])
        
        # Convert backend
        if 'solver' in data and 'backend' in data['solver']:
            data['solver']['backend'] = SolverBackend(data['solver']['backend'])
        
        # Reconstruct nested dataclasses
        data['base_fluid'] = BaseFluidConfig(**data['base_fluid'])
        data['nanoparticles'] = [NanoparticleConfig(**np) for np in data['nanoparticles']]
        
        if data.get('geometry'):
            data['geometry'] = GeometryConfig(**data['geometry'])
        if data.get('mesh'):
            data['mesh'] = MeshConfig(**data['mesh'])
        if data.get('flow'):
            data['flow'] = FlowConfig(**data['flow'])
        if data.get('solver'):
            data['solver'] = SolverConfig(**data['solver'])
        
        return cls(**data)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file"""
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'UnifiedConfig':
        """Load configuration from JSON file"""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Configuration loaded from {filepath}")
        return cls.from_dict(data)


class BKPSNanofluidEngine:
    """
    Unified entry point for BKPS NFL Thermal Pro 7.0
    
    Usage:
        # Static mode
        engine = BKPSNanofluidEngine(config)
        results = engine.run()
        
        # Or quick start
        engine = BKPSNanofluidEngine.quick_start(
            mode="static",
            base_fluid="Water",
            nanoparticle="Al2O3",
            volume_fraction=0.02,
            temperature=300
        )
    """
    
    VERSION = __version__
    RELEASE_DATE = __release_date__
    CODENAME = __codename__
    
    def __init__(self, config: UnifiedConfig):
        """
        Initialize unified engine
        
        Args:
            config: UnifiedConfig object with all parameters
        """
        self.config = config
        self.results = None
        self._simulator = None
        self._cfd_solver = None
        self._flow_simulator = None
        self._ai_recommender = None
        
        # Validate configuration
        valid, errors = config.validate()
        if not valid:
            raise ValueError(f"Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors))
        
        # Log startup
        logger.info("="*80)
        logger.info(f"{self.CODENAME} v{self.VERSION}")
        logger.info(f"Release Date: {self.RELEASE_DATE}")
        logger.info(f"Dedicated to: Brijesh Kumar Pandey")
        logger.info("="*80)
        logger.info(f"Mode: {config.mode.value}")
        logger.info(f"Base Fluid: {config.base_fluid.name} @ {config.base_fluid.temperature}K")
        logger.info(f"Nanoparticles: {len(config.nanoparticles)} components")
        logger.info(f"Backend: {config.solver.backend.value}")
        logger.info("="*80)
        
        # Initialize physics engines
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize physics engines based on mode"""
        # Always initialize base simulator
        self._simulator = BKPSNanofluidSimulator(
            base_fluid=self.config.base_fluid.name,
            temperature=self.config.base_fluid.temperature,
            pressure=self.config.base_fluid.pressure
        )
        
        # Add nanoparticles
        for np_config in self.config.nanoparticles:
            if np_config.custom_properties:
                # Custom nanoparticle (would need to extend simulator)
                logger.warning("Custom nanoparticle properties not fully implemented yet")
            
            self._simulator.add_nanoparticle(
                material=np_config.material,
                volume_fraction=np_config.volume_fraction,
                diameter=np_config.diameter,
                shape=np_config.shape
            )
        
        # Initialize mode-specific engines
        if self.config.mode == SimulationMode.FLOW:
            self._flow_simulator = FlowNanofluidSimulator(
                base_fluid=self.config.base_fluid.name,
                temperature=self.config.base_fluid.temperature
            )
        
        elif self.config.mode == SimulationMode.CFD:
            # Create mesh
            geom = self.config.geometry
            mesh_cfg = self.config.mesh
            
            if geom.geometry_type == "channel":
                mesh = StructuredMesh2D(
                    x_range=(0, geom.length),
                    y_range=(0, geom.height),
                    nx=mesh_cfg.nx,
                    ny=mesh_cfg.ny
                )
            else:
                raise NotImplementedError(f"Geometry {geom.geometry_type} not yet implemented")
            
            self._cfd_solver = NavierStokesSolver(mesh)
        
        # Initialize AI recommender if enabled
        if self.config.enable_ai_recommendations:
            self._ai_recommender = AIRecommendationEngine()
    
    @classmethod
    def quick_start(cls,
                    mode: str = "static",
                    base_fluid: str = "Water",
                    nanoparticle: str = "Al2O3",
                    volume_fraction: float = 0.02,
                    temperature: float = 300,
                    diameter: float = 30e-9,
                    **kwargs) -> 'BKPSNanofluidEngine':
        """
        Quick start with minimal parameters
        
        Args:
            mode: "static", "flow", or "cfd"
            base_fluid: Base fluid name
            nanoparticle: Nanoparticle material
            volume_fraction: Volume fraction
            temperature: Temperature in Kelvin
            diameter: Particle diameter in meters
            **kwargs: Additional configuration parameters
        
        Returns:
            Initialized engine
        """
        config = UnifiedConfig(
            mode=SimulationMode(mode),
            base_fluid=BaseFluidConfig(name=base_fluid, temperature=temperature),
            nanoparticles=[
                NanoparticleConfig(
                    material=nanoparticle,
                    volume_fraction=volume_fraction,
                    diameter=diameter
                )
            ]
        )
        
        # Add optional parameters
        if mode in ["flow", "cfd"]:
            config.geometry = GeometryConfig(**kwargs.get('geometry', {}))
            config.flow = FlowConfig(**kwargs.get('flow', {}))
        
        if mode == "cfd":
            config.mesh = MeshConfig(**kwargs.get('mesh', {}))
        
        return cls(config)
    
    def run(self, progress_callback=None) -> Dict[str, Any]:
        """
        Run simulation based on configured mode
        
        Args:
            progress_callback: Optional callback(percentage: int)
        
        Returns:
            Dictionary with results
        """
        logger.info(f"Starting {self.config.mode.value} simulation...")
        
        try:
            if self.config.mode == SimulationMode.STATIC:
                results = self._run_static(progress_callback)
            elif self.config.mode == SimulationMode.FLOW:
                results = self._run_flow(progress_callback)
            elif self.config.mode == SimulationMode.CFD:
                results = self._run_cfd(progress_callback)
            elif self.config.mode == SimulationMode.HYBRID:
                results = self._run_hybrid(progress_callback)
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")
            
            # Add AI recommendations if enabled
            if self.config.enable_ai_recommendations and self._ai_recommender:
                results['ai_recommendations'] = self._generate_ai_recommendations(results)
            
            # Add metadata
            results['metadata'] = {
                'version': self.VERSION,
                'mode': self.config.mode.value,
                'timestamp': datetime.now().isoformat(),
                'configuration': self.config.to_dict()
            }
            
            self.results = results
            logger.info("Simulation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
    
    def _run_static(self, progress_callback=None) -> Dict[str, Any]:
        """Run static property calculation"""
        # Calculate base fluid properties
        k_base = self._simulator.calculate_base_fluid_conductivity()
        mu_base = self._simulator.calculate_base_fluid_viscosity()
        
        # Calculate nanofluid properties
        k_static = self._simulator.calculate_static_thermal_conductivity()
        
        # Viscosity (returns tuple)
        mu_result = self._simulator.calculate_viscosity()
        if isinstance(mu_result, tuple):
            mu_nf = mu_result[0]
        else:
            mu_nf = mu_result
        
        # Try to get density and specific heat (may not be implemented)
        try:
            if hasattr(self._simulator, 'calculate_density'):
                rho_nf = self._simulator.calculate_density()
            else:
                rho_nf = 1000.0  # Default water density
        except:
            rho_nf = 1000.0
        
        try:
            if hasattr(self._simulator, 'calculate_specific_heat'):
                cp_nf = self._simulator.calculate_specific_heat()
            else:
                cp_nf = 4180.0  # Default water specific heat
        except:
            cp_nf = 4180.0
        
        results = {
            'k_base': k_base,
            'k_static': k_static,
            'mu_base': mu_base,
            'mu_nf': mu_nf,
            'rho_nf': rho_nf,
            'cp_nf': cp_nf
        }
        
        # Calculate enhancement
        results['enhancement_k'] = (results['k_static'] / results['k_base'] - 1) * 100
        results['viscosity_ratio'] = results['mu_nf'] / results['mu_base']
        
        # Calculate stability (DLVO)
        if self.config.enable_dlvo:
            try:
                dlvo_analysis = self._simulator.perform_dlvo_analysis()
                if dlvo_analysis:
                    # Store key DLVO metrics
                    results['dlvo_potential'] = dlvo_analysis.get('zeta_potential', None)
                    results['dlvo_barrier'] = dlvo_analysis.get('energy_barrier', None)
                    results['stability_ratio'] = dlvo_analysis.get('stability_ratio', None)
                    results['stability_status'] = dlvo_analysis.get('stability_status', None)
                else:
                    results['dlvo_potential'] = None
            except Exception as e:
                logger.warning(f"DLVO analysis failed: {e}")
                results['dlvo_potential'] = None
        
        if progress_callback:
            progress_callback(100)
        
        return results
    
    def _run_flow(self, progress_callback=None) -> Dict[str, Any]:
        """Run flow-dependent property calculation"""
        if not self._flow_simulator:
            raise RuntimeError("Flow simulator not initialized")
        
        # Get flow-dependent properties
        velocity = self.config.flow.velocity
        
        results = self._run_static(None)  # Get base properties
        
        # Add flow-dependent thermal conductivity
        results['k_flow'] = self._simulator.calculate_flow_dependent_conductivity(velocity)
        results['enhancement_k_flow'] = (results['k_flow'] / results['k_base'] - 1) * 100
        
        if progress_callback:
            progress_callback(100)
        
        return results
    
    def _run_cfd(self, progress_callback=None) -> Dict[str, Any]:
        """Run full CFD simulation"""
        if not self._cfd_solver:
            raise RuntimeError("CFD solver not initialized")
        
        # Get nanofluid properties using correct method names
        mu_result = self._simulator.calculate_viscosity()
        mu_nf = mu_result[0] if isinstance(mu_result, tuple) else mu_result
        
        # Calculate or estimate other properties
        k_nf = self._simulator.calculate_static_thermal_conductivity()
        
        # Estimate density (mixture rule)
        phi_total = sum(c.volume_fraction for c in self._simulator.components)
        rho_bf = self._simulator.rho_bf
        rho_p = self._simulator.components[0].rho_particle if self._simulator.components else 3970
        rho_nf = phi_total * rho_p + (1 - phi_total) * rho_bf
        
        # Estimate specific heat (mixture rule)
        cp_bf = self._simulator.cp_bf
        cp_p = self._simulator.components[0].cp_particle if self._simulator.components else 880
        cp_nf = (phi_total * rho_p * cp_p + (1 - phi_total) * rho_bf * cp_bf) / rho_nf
        
        # Set properties in solver
        self._cfd_solver.set_fluid_properties(
            viscosity=mu_nf,
            density=rho_nf,
            thermal_conductivity=k_nf,
            specific_heat=cp_nf
        )
        
        # Set boundary conditions (example - inlet/outlet)
        from nanofluid_simulator.cfd_solver import BoundaryType, BoundaryCondition
        
        # Inlet velocity (left boundary)
        inlet_velocity = self.config.flow.velocity if self.config.flow else 1.0
        # Get temperature from flow config or base fluid config
        if self.config.flow and hasattr(self.config.flow, 'inlet_temperature'):
            inlet_temp = self.config.flow.inlet_temperature
        else:
            inlet_temp = self.config.base_fluid.temperature
        
        self._cfd_solver.set_boundary_condition(
            BoundaryType.INLET,
            BoundaryCondition(
                bc_type=BoundaryType.INLET,
                velocity=(inlet_velocity, 0.0),
                temperature=inlet_temp
            )
        )
        
        # Outlet pressure (right boundary)
        self._cfd_solver.set_boundary_condition(
            BoundaryType.OUTLET,
            BoundaryCondition(
                bc_type=BoundaryType.OUTLET,
                pressure=0.0
            )
        )
        
        # Wall boundaries (top/bottom) - no-slip, adiabatic
        self._cfd_solver.set_boundary_condition(
            BoundaryType.WALL,
            BoundaryCondition(
                bc_type=BoundaryType.WALL,
                velocity=(0.0, 0.0)
            )
        )
        
        if progress_callback:
            progress_callback(20)
        
        # Solve using SIMPLE algorithm
        logger.info("Running SIMPLE algorithm for coupled Navier-Stokes equations...")
        converged = self._cfd_solver.solve(
            max_iterations=self.config.solver.max_iterations,
            verbose=True
        )
        
        if progress_callback:
            progress_callback(80)
        
        # Get converged results
        flow_field = self._cfd_solver.get_results()
        
        if progress_callback:
            progress_callback(100)
        
        results = {
            'converged': converged,
            'velocity_u': flow_field.u,
            'velocity_v': flow_field.v,
            'pressure': flow_field.p,
            'temperature': flow_field.T,
            'mesh': self._cfd_solver.mesh,
            'residuals': self._cfd_solver.residuals,
            'properties': {
                'mu_nf': mu_nf,
                'rho_nf': rho_nf,
                'k_nf': k_nf,
                'cp_nf': cp_nf
            }
        }
        
        return results
    
    def _run_hybrid(self, progress_callback=None) -> Dict[str, Any]:
        """Run hybrid simulation (static + flow + CFD)"""
        results = {}
        
        # Static properties
        results['static'] = self._run_static(lambda p: progress_callback(p//3) if progress_callback else None)
        
        # Flow properties
        if self.config.flow:
            results['flow'] = self._run_flow(lambda p: progress_callback(33 + p//3) if progress_callback else None)
        
        # CFD if mesh provided
        if self.config.mesh:
            results['cfd'] = self._run_cfd(lambda p: progress_callback(66 + p//3) if progress_callback else None)
        
        return results
    
    def _generate_ai_recommendations(self, results: Dict) -> Dict:
        """Generate AI-based recommendations"""
        if not self._ai_recommender:
            return {}
        
        # Extract key metrics
        if 'enhancement_k' in results:
            enhancement = results['enhancement_k']
        elif 'static' in results and 'enhancement_k' in results['static']:
            enhancement = results['static']['enhancement_k']
        else:
            enhancement = 0
        
        recommendations = {
            'optimal_conditions': {},
            'warnings': [],
            'suggestions': []
        }
        
        # Generate recommendations based on results
        if enhancement < 5:
            recommendations['suggestions'].append(
                "Low enhancement detected. Consider higher thermal conductivity particles (Cu, CNT)"
            )
        
        if 'viscosity_ratio' in results and results['viscosity_ratio'] > 2:
            recommendations['warnings'].append(
                "High viscosity increase (>100%). Consider pumping power requirements."
            )
        
        return recommendations
    
    def export_results(self, filepath: Union[str, Path], format: str = 'json') -> None:
        """
        Export results to file
        
        Args:
            filepath: Output file path
            format: 'json', 'csv', or 'hdf5'
        """
        if self.results is None:
            raise RuntimeError("No results to export. Run simulation first.")
        
        filepath = Path(filepath)
        
        if format == 'json':
            # Convert numpy arrays to lists for JSON
            def convert_arrays(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_arrays(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_arrays(item) for item in obj]
                return obj
            
            with open(filepath, 'w') as f:
                json.dump(convert_arrays(self.results), f, indent=2)
        
        elif format == 'csv':
            # CSV export for tabular data only
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write simple key-value pairs
                for key, value in self.results.items():
                    if isinstance(value, (int, float, str)):
                        writer.writerow([key, value])
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Results exported to {filepath}")
    
    def __repr__(self) -> str:
        return (f"BKPSNanofluidEngine(mode={self.config.mode.value}, "
                f"base_fluid={self.config.base_fluid.name}, "
                f"nanoparticles={len(self.config.nanoparticles)})")


# Convenience functions
def create_static_config(base_fluid: str = "Water",
                        nanoparticle: str = "Al2O3",
                        volume_fraction: float = 0.02,
                        temperature: float = 300,
                        diameter: float = 30e-9) -> UnifiedConfig:
    """Create configuration for static mode"""
    return UnifiedConfig(
        mode=SimulationMode.STATIC,
        base_fluid=BaseFluidConfig(name=base_fluid, temperature=temperature),
        nanoparticles=[
            NanoparticleConfig(
                material=nanoparticle,
                volume_fraction=volume_fraction,
                diameter=diameter
            )
        ]
    )


def create_flow_config(base_fluid: str = "Water",
                      nanoparticle: str = "Al2O3",
                      volume_fraction: float = 0.02,
                      temperature: float = 300,
                      velocity: float = 1.0,
                      diameter: float = 30e-9) -> UnifiedConfig:
    """Create configuration for flow mode"""
    return UnifiedConfig(
        mode=SimulationMode.FLOW,
        base_fluid=BaseFluidConfig(name=base_fluid, temperature=temperature),
        nanoparticles=[
            NanoparticleConfig(
                material=nanoparticle,
                volume_fraction=volume_fraction,
                diameter=diameter
            )
        ],
        geometry=GeometryConfig(geometry_type="channel", length=0.1, height=0.01),
        flow=FlowConfig(velocity=velocity, inlet_temperature=temperature)
    )


def create_cfd_config(base_fluid: str = "Water",
                     nanoparticle: str = "Al2O3",
                     volume_fraction: float = 0.02,
                     temperature: float = 300,
                     velocity: float = 1.0,
                     diameter: float = 30e-9,
                     nx: int = 50,
                     ny: int = 50) -> UnifiedConfig:
    """Create configuration for CFD mode"""
    return UnifiedConfig(
        mode=SimulationMode.CFD,
        base_fluid=BaseFluidConfig(name=base_fluid, temperature=temperature),
        nanoparticles=[
            NanoparticleConfig(
                material=nanoparticle,
                volume_fraction=volume_fraction,
                diameter=diameter
            )
        ],
        geometry=GeometryConfig(geometry_type="channel", length=0.1, height=0.01),
        flow=FlowConfig(velocity=velocity, inlet_temperature=temperature),
        mesh=MeshConfig(nx=nx, ny=ny)
    )
