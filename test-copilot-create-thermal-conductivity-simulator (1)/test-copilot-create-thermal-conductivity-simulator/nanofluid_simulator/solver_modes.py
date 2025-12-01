"""
Solver Mode Management for Nanofluid Simulator

This module defines solver modes and manages physics model selection.
"""

from enum import Enum
from typing import List, Dict, Set


class SolverMode(Enum):
    """
    Solver operating modes for the nanofluid simulator.
    """
    STATIC = "static"
    FLOW = "flow"


class SolverModeConfig:
    """
    Configuration and model filtering for different solver modes.
    """
    
    # Models available in each mode
    STATIC_MODELS = {
        # Classical static models
        "maxwell",
        "hamilton_crosser",
        "bruggeman",
        "yu_choi",
        "wasp",
        "pak_cho",
        
        # Advanced static models
        "patel",
        "koo_kleinstreuer",
        "chon",
        "takabi_salehi",
        
        # Interfacial layer models
        "xue_interfacial",
        "leong_yang_interfacial",
        "yu_choi_interfacial",
        
        # Hybrid models
        "hajjar_hybrid",
        "sundar_hybrid",
        "esfe_hybrid",
    }
    
    FLOW_MODELS = {
        # All static models
        *STATIC_MODELS,
        
        # Flow-enhanced models
        "buongiorno_convective",
        "corcione_flow",
        "rea_bonnet_convective",
        "shear_enhanced",
        "velocity_dependent",
    }
    
    @staticmethod
    def get_available_models(mode: SolverMode) -> Set[str]:
        """Get list of models available for the given solver mode."""
        if mode == SolverMode.STATIC:
            return SolverModeConfig.STATIC_MODELS.copy()
        elif mode == SolverMode.FLOW:
            return SolverModeConfig.FLOW_MODELS.copy()
        else:
            return set()
    
    @staticmethod
    def is_model_available(model_name: str, mode: SolverMode) -> bool:
        """Check if a model is available in the given solver mode."""
        return model_name in SolverModeConfig.get_available_models(mode)
    
    @staticmethod
    def get_mode_description(mode: SolverMode) -> Dict[str, str]:
        """Get detailed description of solver mode."""
        descriptions = {
            SolverMode.STATIC: {
                "name": "Static Property Solver",
                "icon": "🔬",
                "description": (
                    "Flow velocity ignored. Only temperature + concentration + "
                    "particle physics. Classical and advanced conductivity & "
                    "viscosity models."
                ),
                "best_for": (
                    "Stationary applications, property estimation, "
                    "material characterization, batch processing"
                ),
                "features": [
                    "21+ validated thermal conductivity models",
                    "Temperature-dependent properties (0-200°C)",
                    "Hybrid nanofluid support",
                    "Interfacial layer effects",
                    "Viscosity modeling (concentration-dependent)",
                    "Fast computation"
                ],
                "limitations": [
                    "No flow velocity effects",
                    "No shear-rate dependent viscosity",
                    "No convective enhancement",
                    "No pumping power analysis"
                ]
            },
            SolverMode.FLOW: {
                "name": "Flow-Enhanced Solver (Dynamic Mode)",
                "icon": "🌊",
                "description": (
                    "Includes flow velocity effects, convective enhancement of "
                    "thermal conductivity, shear-rate dependent viscosity models, "
                    "Brownian motion interaction corrections, nanoparticle "
                    "aggregation under flow."
                ),
                "best_for": (
                    "Heat exchangers, cooling channels, pipe flow, "
                    "thermal systems with flow, performance optimization"
                ),
                "features": [
                    "All static models PLUS flow physics",
                    "Flow-enhanced thermal conductivity (Buongiorno, Corcione, Rea-Bonnet)",
                    "Shear-rate dependent viscosity (0-10,000 s⁻¹)",
                    "Reynolds, Nusselt, Prandtl number analysis",
                    "Pressure drop & pumping power calculation",
                    "Performance Index optimization (PI = h_ratio / P_ratio^(1/3))",
                    "DLVO aggregation stability prediction",
                    "Flow regime identification (laminar/transitional/turbulent)",
                    "Heat transfer coefficient calculation"
                ],
                "limitations": [
                    "Requires flow parameters (velocity, geometry)",
                    "More computationally intensive",
                    "Needs careful parameter selection"
                ]
            }
        }
        return descriptions.get(mode, {})
    
    @staticmethod
    def get_required_parameters(mode: SolverMode) -> Dict[str, List[str]]:
        """Get required and optional parameters for each mode."""
        params = {
            SolverMode.STATIC: {
                "required": [
                    "base_fluid",
                    "temperature",
                    "nanoparticle(s)",
                    "volume_fraction",
                    "particle_size"
                ],
                "optional": [
                    "particle_shape_factor",
                    "interfacial_layer_thickness",
                    "aggregation_state"
                ]
            },
            SolverMode.FLOW: {
                "required": [
                    "base_fluid",
                    "temperature",
                    "nanoparticle(s)",
                    "volume_fraction",
                    "particle_size",
                    "flow_velocity",
                    "channel_diameter",
                    "channel_length"
                ],
                "optional": [
                    "particle_shape_factor",
                    "shear_rate",
                    "surface_potential",
                    "ionic_strength",
                    "hamaker_constant",
                    "aggregation_state"
                ]
            }
        }
        return params.get(mode, {"required": [], "optional": []})
    
    @staticmethod
    def get_visualization_modes(mode: SolverMode) -> List[str]:
        """Get available visualization modes for each solver."""
        viz = {
            SolverMode.STATIC: [
                "Thermal Conductivity vs Temperature",
                "Thermal Conductivity vs Concentration",
                "Viscosity vs Temperature",
                "Viscosity vs Concentration",
                "All Properties (k, μ, ρ, Pr)",
                "Model Comparison Bar Chart"
            ],
            SolverMode.FLOW: [
                # All static visualizations
                "Thermal Conductivity vs Temperature",
                "Thermal Conductivity vs Concentration",
                "Viscosity vs Temperature",
                "Viscosity vs Concentration",
                "All Properties (k, μ, ρ, Pr)",
                "Model Comparison Bar Chart",
                
                # Flow-specific visualizations
                "k_eff vs Flow Velocity",
                "Viscosity vs Shear Rate",
                "Reynolds Number vs Velocity",
                "Nusselt Number vs Reynolds",
                "Heat Transfer Coefficient vs Velocity",
                "Pressure Drop vs Velocity",
                "Pumping Power vs Velocity",
                "Performance Index vs Velocity",
                "Flow Regime Map",
                "Aggregation Stability Analysis"
            ]
        }
        return viz.get(mode, [])


class SolverModeManager:
    """
    Manages solver mode state and transitions.
    """
    
    def __init__(self, initial_mode: SolverMode = SolverMode.STATIC):
        self._mode = initial_mode
        self._callbacks = []
    
    @property
    def mode(self) -> SolverMode:
        """Get current solver mode."""
        return self._mode
    
    @mode.setter
    def mode(self, new_mode: SolverMode):
        """Set solver mode and notify callbacks."""
        if new_mode != self._mode:
            old_mode = self._mode
            self._mode = new_mode
            self._notify_mode_changed(old_mode, new_mode)
    
    def register_callback(self, callback):
        """Register a callback to be notified when mode changes."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def unregister_callback(self, callback):
        """Unregister a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _notify_mode_changed(self, old_mode: SolverMode, new_mode: SolverMode):
        """Notify all registered callbacks of mode change."""
        for callback in self._callbacks:
            try:
                callback(old_mode, new_mode)
            except Exception as e:
                print(f"Error in mode change callback: {e}")
    
    def get_mode_info(self) -> Dict:
        """Get comprehensive information about current mode."""
        return {
            "mode": self._mode,
            "description": SolverModeConfig.get_mode_description(self._mode),
            "available_models": list(SolverModeConfig.get_available_models(self._mode)),
            "parameters": SolverModeConfig.get_required_parameters(self._mode),
            "visualizations": SolverModeConfig.get_visualization_modes(self._mode)
        }
    
    def can_use_model(self, model_name: str) -> bool:
        """Check if model can be used in current mode."""
        return SolverModeConfig.is_model_available(model_name, self._mode)
