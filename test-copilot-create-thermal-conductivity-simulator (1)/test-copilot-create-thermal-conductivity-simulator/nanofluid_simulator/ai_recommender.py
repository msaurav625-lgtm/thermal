"""
AI Recommendation Engine for Nanofluid Configuration Optimization

This module provides intelligent recommendations for optimal nanofluid
configuration based on application requirements, using multi-objective
optimization and physics-based constraints.

Features:
- Nanoparticle selection based on thermal performance and cost
- Optimal concentration recommendation
- Flow velocity optimization
- Stability prediction and warnings
- Multi-objective optimization (maximize heat transfer, minimize cost)
- Application-specific recommendations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .nanoparticles import NanoparticleDatabase
from .aggregation_models import assess_colloidal_stability
from .flow_simulator import FlowNanofluidSimulator
from .enhanced_simulator import EnhancedNanofluidSimulator


class ApplicationType(Enum):
    """Types of nanofluid applications."""
    HEAT_EXCHANGER = "heat_exchanger"
    CPU_COOLING = "cpu_cooling"
    EV_BATTERY = "ev_battery"
    SOLAR_THERMAL = "solar_thermal"
    HVAC = "hvac"
    RADIATOR = "radiator"
    GENERAL = "general"


class OptimizationObjective(Enum):
    """Optimization objectives."""
    MAXIMIZE_HEAT_TRANSFER = "max_heat_transfer"
    MINIMIZE_COST = "min_cost"
    MAXIMIZE_EFFICIENCY = "max_efficiency"
    BALANCE = "balance"


@dataclass
class MaterialCostData:
    """Cost information for nanoparticles (relative scale)."""
    formula: str
    relative_cost: float  # 1.0 = reference, higher = more expensive
    availability: float   # 0-1, higher = more available
    thermal_conductivity: float
    density: float


@dataclass
class RecommendationConstraints:
    """User-defined constraints for recommendations."""
    min_temperature: float = 273.15  # K
    max_temperature: float = 373.15  # K
    max_volume_fraction: float = 0.05
    max_viscosity_ratio: float = 2.0  # max μ_nf/μ_bf
    max_pressure_drop: Optional[float] = None  # Pa
    max_pumping_power: Optional[float] = None  # W
    min_performance_index: float = 1.0
    budget_constraint: Optional[float] = None  # relative scale
    require_stability: bool = True
    prefer_common_materials: bool = True


@dataclass
class NanofluidRecommendation:
    """Complete nanofluid configuration recommendation."""
    # Recommended configuration
    nanoparticle: str
    volume_fraction: float
    particle_size: float  # nm
    temperature: float  # K
    
    # Predicted performance
    thermal_conductivity: float  # W/m·K
    enhancement: float  # %
    viscosity: float  # Pa·s
    viscosity_ratio: float
    
    # Optional fields with defaults
    flow_velocity: Optional[float] = None  # m/s (if flow mode)
    
    # Flow metrics (if applicable)
    reynolds: Optional[float] = None
    nusselt: Optional[float] = None
    heat_transfer_coeff: Optional[float] = None
    pressure_drop: Optional[float] = None
    pumping_power: Optional[float] = None
    performance_index: Optional[float] = None
    
    # Stability assessment
    stability_status: str = "Unknown"
    aggregation_risk: str = "Unknown"
    
    # Scoring
    overall_score: float = 0.0
    thermal_score: float = 0.0
    cost_score: float = 0.0
    stability_score: float = 0.0
    efficiency_score: float = 0.0
    
    # Recommendations and warnings
    recommendation_text: str = ""
    warnings: List[str] = None
    alternatives: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.alternatives is None:
            self.alternatives = []


class AIRecommendationEngine:
    """
    AI-powered recommendation engine for optimal nanofluid configuration.
    
    Uses physics-based models combined with optimization algorithms to
    suggest the best nanofluid configuration for specific applications.
    """
    
    # Material cost database (relative scale, 1.0 = reference)
    MATERIAL_COSTS = {
        "Cu": MaterialCostData("Cu", 3.0, 0.95, 401, 8933),
        "Al": MaterialCostData("Al", 1.5, 0.98, 237, 2702),
        "Ag": MaterialCostData("Ag", 10.0, 0.70, 429, 10500),
        "Au": MaterialCostData("Au", 50.0, 0.60, 317, 19300),
        "Al2O3": MaterialCostData("Al2O3", 1.0, 0.99, 40, 3970),
        "TiO2": MaterialCostData("TiO2", 1.2, 0.95, 8.4, 4230),
        "CuO": MaterialCostData("CuO", 2.0, 0.90, 33, 6500),
        "SiO2": MaterialCostData("SiO2", 0.8, 0.99, 1.4, 2220),
        "Fe3O4": MaterialCostData("Fe3O4", 1.5, 0.92, 6, 5200),
        "ZnO": MaterialCostData("ZnO", 1.3, 0.93, 29, 5606),
        "CNT": MaterialCostData("CNT", 20.0, 0.50, 3000, 1350),
        "Graphene": MaterialCostData("Graphene", 30.0, 0.40, 5000, 2267),
        "Diamond": MaterialCostData("Diamond", 100.0, 0.30, 2200, 3515),
    }
    
    def __init__(self, base_fluid: str = "water"):
        """Initialize the recommendation engine."""
        self.base_fluid = base_fluid
        self.simulator_static = EnhancedNanofluidSimulator()
        self.simulator_flow = FlowNanofluidSimulator()
        
    def recommend_configuration(
        self,
        application: ApplicationType,
        temperature_celsius: float,
        objective: OptimizationObjective = OptimizationObjective.BALANCE,
        constraints: Optional[RecommendationConstraints] = None,
        flow_conditions: Optional[Dict[str, float]] = None,
        top_n: int = 5
    ) -> List[NanofluidRecommendation]:
        """
        Recommend optimal nanofluid configurations.
        
        Parameters:
            application: Type of application
            temperature_celsius: Operating temperature (°C)
            objective: Optimization objective
            constraints: User-defined constraints
            flow_conditions: Dict with 'velocity', 'diameter', 'length' (if flow mode)
            top_n: Number of recommendations to return
            
        Returns:
            List of recommendations, sorted by overall score
        """
        if constraints is None:
            constraints = RecommendationConstraints()
        
        # Get application-specific parameters
        app_params = self._get_application_parameters(application)
        
        # Determine if flow mode
        use_flow_mode = flow_conditions is not None
        
        # Generate candidate configurations
        candidates = self._generate_candidates(
            temperature_celsius,
            constraints,
            app_params,
            use_flow_mode
        )
        
        # Evaluate each candidate
        recommendations = []
        for candidate in candidates:
            rec = self._evaluate_candidate(
                candidate,
                temperature_celsius,
                objective,
                constraints,
                flow_conditions,
                app_params
            )
            if rec is not None:
                recommendations.append(rec)
        
        # Sort by overall score
        recommendations.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Deduplicate: keep only unique (nanoparticle, volume_fraction) pairs
        seen = set()
        unique_recs = []
        for rec in recommendations:
            key = (rec.nanoparticle, round(rec.volume_fraction, 3))
            if key not in seen:
                seen.add(key)
                unique_recs.append(rec)
        
        # Add alternatives to top recommendations
        for i, rec in enumerate(unique_recs[:top_n]):
            rec.alternatives = [
                f"{r.nanoparticle} (φ={r.volume_fraction:.3f})"
                for r in unique_recs[i+1:i+4]
            ]
        
        return unique_recs[:top_n]
    
    def optimize_concentration(
        self,
        nanoparticle: str,
        temperature_celsius: float,
        objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_EFFICIENCY,
        constraints: Optional[RecommendationConstraints] = None,
        flow_conditions: Optional[Dict[str, float]] = None
    ) -> NanofluidRecommendation:
        """
        Optimize volume fraction for a specific nanoparticle.
        
        Returns the optimal concentration that maximizes the objective
        while satisfying all constraints.
        """
        if constraints is None:
            constraints = RecommendationConstraints()
        
        # Test range of concentrations
        phi_values = np.linspace(0.005, constraints.max_volume_fraction, 20)
        
        best_score = -np.inf
        best_rec = None
        
        for phi in phi_values:
            candidate = {
                'nanoparticle': nanoparticle,
                'volume_fraction': phi,
                'particle_size': 30.0  # Default size
            }
            
            rec = self._evaluate_candidate(
                candidate,
                temperature_celsius,
                objective,
                constraints,
                flow_conditions,
                {}
            )
            
            if rec is not None and rec.overall_score > best_score:
                best_score = rec.overall_score
                best_rec = rec
        
        if best_rec is not None:
            best_rec.recommendation_text = (
                f"Optimal concentration for {nanoparticle}: "
                f"{best_rec.volume_fraction:.3f} ({best_rec.volume_fraction*100:.1f}%)"
            )
        
        return best_rec
    
    def optimize_velocity(
        self,
        nanoparticle: str,
        volume_fraction: float,
        temperature_celsius: float,
        channel_diameter: float,
        channel_length: float,
        constraints: Optional[RecommendationConstraints] = None
    ) -> Tuple[float, NanofluidRecommendation]:
        """
        Optimize flow velocity for maximum Performance Index.
        
        Returns optimal velocity and corresponding recommendation.
        """
        if constraints is None:
            constraints = RecommendationConstraints()
        
        # Test range of velocities
        velocities = np.linspace(0.1, 5.0, 30)
        
        best_pi = -np.inf
        best_velocity = None
        best_rec = None
        
        for v in velocities:
            flow_conditions = {
                'velocity': v,
                'diameter': channel_diameter,
                'length': channel_length
            }
            
            candidate = {
                'nanoparticle': nanoparticle,
                'volume_fraction': volume_fraction,
                'particle_size': 30.0
            }
            
            rec = self._evaluate_candidate(
                candidate,
                temperature_celsius,
                OptimizationObjective.MAXIMIZE_EFFICIENCY,
                constraints,
                flow_conditions,
                {}
            )
            
            if rec is not None and rec.performance_index is not None:
                if rec.performance_index > best_pi:
                    # Check constraints
                    if constraints.max_pressure_drop and rec.pressure_drop > constraints.max_pressure_drop:
                        continue
                    if constraints.max_pumping_power and rec.pumping_power > constraints.max_pumping_power:
                        continue
                    
                    best_pi = rec.performance_index
                    best_velocity = v
                    best_rec = rec
        
        if best_rec is not None:
            best_rec.recommendation_text = (
                f"Optimal velocity: {best_velocity:.2f} m/s "
                f"(PI = {best_pi:.3f})"
            )
        
        return best_velocity, best_rec
    
    def _get_application_parameters(self, application: ApplicationType) -> Dict:
        """Get typical parameters for different applications."""
        params = {
            ApplicationType.HEAT_EXCHANGER: {
                'typical_temp_range': (40, 90),
                'preferred_materials': ['Cu', 'CuO', 'Al2O3'],
                'max_phi': 0.05,
                'prioritize': 'thermal_performance'
            },
            ApplicationType.CPU_COOLING: {
                'typical_temp_range': (30, 70),
                'preferred_materials': ['Cu', 'Al2O3', 'CNT'],
                'max_phi': 0.03,
                'prioritize': 'thermal_performance'
            },
            ApplicationType.EV_BATTERY: {
                'typical_temp_range': (20, 50),
                'preferred_materials': ['Al2O3', 'SiO2', 'CuO'],
                'max_phi': 0.04,
                'prioritize': 'balance'
            },
            ApplicationType.SOLAR_THERMAL: {
                'typical_temp_range': (60, 120),
                'preferred_materials': ['Cu', 'CuO', 'CNT'],
                'max_phi': 0.05,
                'prioritize': 'thermal_performance'
            },
            ApplicationType.HVAC: {
                'typical_temp_range': (0, 50),
                'preferred_materials': ['Al2O3', 'TiO2', 'SiO2'],
                'max_phi': 0.03,
                'prioritize': 'cost'
            },
            ApplicationType.RADIATOR: {
                'typical_temp_range': (70, 110),
                'preferred_materials': ['Cu', 'CuO', 'Al2O3'],
                'max_phi': 0.04,
                'prioritize': 'thermal_performance'
            },
            ApplicationType.GENERAL: {
                'typical_temp_range': (20, 80),
                'preferred_materials': ['Al2O3', 'CuO', 'TiO2'],
                'max_phi': 0.04,
                'prioritize': 'balance'
            }
        }
        return params.get(application, params[ApplicationType.GENERAL])
    
    def _generate_candidates(
        self,
        temperature_celsius: float,
        constraints: RecommendationConstraints,
        app_params: Dict,
        use_flow_mode: bool
    ) -> List[Dict]:
        """Generate candidate configurations to evaluate."""
        candidates = []
        
        # Get available nanoparticles
        available_particles = NanoparticleDatabase.list_nanoparticles()
        
        # Prioritize based on application
        if app_params.get('preferred_materials'):
            # Put preferred materials first
            preferred = [p for p in available_particles if p in app_params['preferred_materials']]
            others = [p for p in available_particles if p not in app_params['preferred_materials']]
            particles_to_test = preferred + others[:5]  # Test preferred + 5 others
        else:
            particles_to_test = available_particles[:10]  # Test top 10
        
        # Volume fractions to test
        max_phi = min(constraints.max_volume_fraction, app_params.get('max_phi', 0.05))
        phi_values = [0.01, 0.02, 0.03, 0.04, max_phi]
        
        # Particle sizes to test
        sizes = [20.0, 30.0, 50.0]
        
        for particle in particles_to_test:
            for phi in phi_values:
                for size in sizes:
                    candidates.append({
                        'nanoparticle': particle,
                        'volume_fraction': phi,
                        'particle_size': size
                    })
        
        return candidates
    
    def _evaluate_candidate(
        self,
        candidate: Dict,
        temperature_celsius: float,
        objective: OptimizationObjective,
        constraints: RecommendationConstraints,
        flow_conditions: Optional[Dict],
        app_params: Dict
    ) -> Optional[NanofluidRecommendation]:
        """Evaluate a candidate configuration and return recommendation."""
        try:
            # Choose simulator based on flow conditions
            use_flow = flow_conditions is not None
            
            # Create fresh simulator instance
            if use_flow:
                sim = FlowNanofluidSimulator()
            else:
                sim = EnhancedNanofluidSimulator()
            
            # Configure simulator
            sim.set_base_fluid(self.base_fluid)
            sim.set_temperature_celsius(temperature_celsius)
            
            # Add nanoparticle
            try:
                sim.add_nanoparticle(
                    candidate['nanoparticle'],
                    candidate['volume_fraction'],
                    candidate['particle_size']
                )
            except Exception as e:
                # Skip if nanoparticle not available
                return None
            
            # Set flow conditions if applicable
            if use_flow:
                sim.set_flow_velocity(flow_conditions['velocity'])
                sim.set_channel_geometry(
                    diameter=flow_conditions['diameter'],
                    length=flow_conditions['length']
                )
            
            # Calculate properties
            if use_flow:
                results = sim.calculate_all_flow_models()
                if not results:
                    return None
                result = results[0]  # Use first model (Buongiorno)
            else:
                results = sim.calculate_all_applicable_models()
                if not results:
                    return None
                # Use Patel model if available, otherwise first model
                result = next((r for r in results if 'Patel' in r.model_name), results[0])
            
            # Get base fluid viscosity for ratios
            base_visc = sim._base_fluid.viscosity if sim._base_fluid else 0.001
            
            # Check hard constraints
            visc_ratio = result.mu_effective / base_visc
            if visc_ratio > constraints.max_viscosity_ratio:
                return None
            
            if use_flow:
                if constraints.max_pressure_drop and result.pressure_drop_Pa > constraints.max_pressure_drop:
                    return None
                if constraints.max_pumping_power and result.pumping_power_W > constraints.max_pumping_power:
                    return None
                if result.performance_index < constraints.min_performance_index:
                    return None
            
            # Assess stability
            try:
                stability_info = assess_colloidal_stability(
                    particle_size=candidate['particle_size'] * 1e-9,
                    temperature=temperature_celsius + 273.15,
                    ionic_strength=0.001,
                    surface_potential=-30e-3,
                    hamaker_constant=1e-20
                )
            except:
                # If stability assessment fails, use default values
                stability_info = {
                    'stability_status': 'Unknown',
                    'aggregation_risk': 'Moderate',
                    'energy_barrier_kT': 10.0
                }
            
            if constraints.require_stability and stability_info['aggregation_risk'] == "High":
                return None
            
            # Calculate scores
            thermal_score = self._calculate_thermal_score(result, use_flow)
            cost_score = self._calculate_cost_score(candidate)
            stability_score = self._calculate_stability_score(stability_info)
            efficiency_score = self._calculate_efficiency_score(result, use_flow)
            
            # Calculate overall score based on objective
            overall_score = self._calculate_overall_score(
                thermal_score, cost_score, stability_score, efficiency_score, objective
            )
            
            # Create recommendation
            rec = NanofluidRecommendation(
                nanoparticle=candidate['nanoparticle'],
                volume_fraction=candidate['volume_fraction'],
                particle_size=candidate['particle_size'],
                temperature=temperature_celsius + 273.15,
                thermal_conductivity=result.k_effective,
                enhancement=result.enhancement_k,
                viscosity=result.mu_effective,
                viscosity_ratio=visc_ratio,
                stability_status=stability_info['stability_status'],
                aggregation_risk=stability_info['aggregation_risk'],
                overall_score=overall_score,
                thermal_score=thermal_score,
                cost_score=cost_score,
                stability_score=stability_score,
                efficiency_score=efficiency_score
            )
            
            # Add flow metrics if applicable
            if use_flow:
                rec.flow_velocity = flow_conditions['velocity']
                rec.reynolds = result.Reynolds
                rec.nusselt = result.Nusselt
                rec.heat_transfer_coeff = result.h_convective
                rec.pressure_drop = result.pressure_drop_Pa
                rec.pumping_power = result.pumping_power_W
                rec.performance_index = result.performance_index
            
            # Generate recommendation text
            rec.recommendation_text = self._generate_recommendation_text(rec, use_flow)
            
            # Add warnings
            rec.warnings = self._generate_warnings(rec, constraints, stability_info)
            
            return rec
            
        except Exception as e:
            # Silently skip failed candidates
            return None
    
    def _calculate_thermal_score(self, result, use_flow: bool) -> float:
        """Calculate thermal performance score (0-1)."""
        if use_flow:
            # Normalize heat transfer coefficient (typical range: 1000-10000 W/m²·K)
            h_norm = min(result.h_convective / 10000.0, 1.0)
            # Normalize enhancement
            enh_norm = min(result.enhancement_k / 100.0, 1.0)
            return 0.6 * h_norm + 0.4 * enh_norm
        else:
            # Normalize enhancement (0-50%)
            return min(result.enhancement_k / 50.0, 1.0)
    
    def _calculate_cost_score(self, candidate: Dict) -> float:
        """Calculate cost score (0-1, higher = cheaper)."""
        material = self.MATERIAL_COSTS.get(candidate['nanoparticle'])
        if material is None:
            return 0.5  # Default score for unknown materials
        
        # Lower cost = higher score
        cost_score = 1.0 / material.relative_cost
        # Higher availability = higher score
        availability_score = material.availability
        
        # Concentration penalty (more material = lower score)
        phi_penalty = 1.0 - (candidate['volume_fraction'] / 0.05)
        
        return 0.5 * cost_score + 0.3 * availability_score + 0.2 * phi_penalty
    
    def _calculate_stability_score(self, stability_info: Dict) -> float:
        """Calculate stability score (0-1)."""
        risk_map = {
            "Low": 1.0,
            "Moderate": 0.7,
            "Marginally stable": 0.5,
            "High": 0.2
        }
        return risk_map.get(stability_info['aggregation_risk'], 0.5)
    
    def _calculate_efficiency_score(self, result, use_flow: bool) -> float:
        """Calculate efficiency score (0-1)."""
        if use_flow:
            # Performance Index score (PI > 1 is good, normalize around PI = 1.0-1.5)
            pi_score = min((result.performance_index - 1.0) / 0.5, 1.0) if result.performance_index >= 1.0 else 0.0
            return max(pi_score, 0.0)
        else:
            # For static mode, use enhancement per viscosity increase
            visc_ratio = result.mu_effective / 0.001  # Assuming water baseline ~0.001 Pa·s
            if visc_ratio > 1.0:
                efficiency = result.enhancement_k / ((visc_ratio - 1.0) * 100)
                return min(efficiency / 10.0, 1.0)
            return 0.5
    
    def _calculate_overall_score(
        self,
        thermal: float,
        cost: float,
        stability: float,
        efficiency: float,
        objective: OptimizationObjective
    ) -> float:
        """Calculate weighted overall score based on objective."""
        weights = {
            OptimizationObjective.MAXIMIZE_HEAT_TRANSFER: [0.6, 0.1, 0.2, 0.1],
            OptimizationObjective.MINIMIZE_COST: [0.2, 0.5, 0.2, 0.1],
            OptimizationObjective.MAXIMIZE_EFFICIENCY: [0.3, 0.2, 0.2, 0.3],
            OptimizationObjective.BALANCE: [0.3, 0.25, 0.25, 0.2]
        }
        
        w = weights.get(objective, weights[OptimizationObjective.BALANCE])
        return w[0] * thermal + w[1] * cost + w[2] * stability + w[3] * efficiency
    
    def _generate_recommendation_text(self, rec: NanofluidRecommendation, use_flow: bool) -> str:
        """Generate human-readable recommendation text."""
        text = f"Recommended: {rec.nanoparticle} at {rec.volume_fraction*100:.1f}% concentration"
        
        if use_flow:
            text += f" with velocity {rec.flow_velocity:.2f} m/s"
        
        text += f"\n  • Thermal conductivity: {rec.thermal_conductivity:.4f} W/m·K ({rec.enhancement:+.1f}% enhancement)"
        text += f"\n  • Viscosity increase: {(rec.viscosity_ratio-1)*100:.1f}%"
        
        if use_flow:
            text += f"\n  • Heat transfer coefficient: {rec.heat_transfer_coeff:.1f} W/m²·K"
            text += f"\n  • Performance Index: {rec.performance_index:.3f}"
            if rec.performance_index > 1.0:
                benefit = (rec.performance_index - 1.0) * 100
                text += f" ({benefit:.1f}% net benefit)"
        
        text += f"\n  • Stability: {rec.stability_status}"
        
        return text
    
    def _generate_warnings(
        self,
        rec: NanofluidRecommendation,
        constraints: RecommendationConstraints,
        stability_info: Dict
    ) -> List[str]:
        """Generate warnings for the recommendation."""
        warnings = []
        
        if rec.viscosity_ratio > 1.5:
            warnings.append(f"High viscosity increase ({(rec.viscosity_ratio-1)*100:.0f}%) may affect pumping")
        
        if stability_info['aggregation_risk'] in ["Moderate", "High"]:
            warnings.append(f"Aggregation risk: {stability_info['aggregation_risk']} - consider surfactants")
        
        if rec.performance_index and rec.performance_index < 1.1:
            warnings.append("Performance Index close to 1.0 - marginal benefit")
        
        material = self.MATERIAL_COSTS.get(rec.nanoparticle)
        if material and material.relative_cost > 5.0:
            warnings.append(f"{rec.nanoparticle} is expensive - consider alternatives")
        
        if rec.volume_fraction > 0.04:
            warnings.append("High concentration may cause stability issues long-term")
        
        return warnings
