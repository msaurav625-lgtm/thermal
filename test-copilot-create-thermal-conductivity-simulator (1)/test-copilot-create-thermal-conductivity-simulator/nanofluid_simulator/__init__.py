"""
BKPS NFL Thermal Pro 7.0 - Professional Nanofluid Simulation Platform
Dedicated to: Brijesh Kumar Pandey

The most comprehensive, physics-rich nanofluid simulation software available.

🔬 Groundbreaking Features:
- Unified API for Static, Flow, and CFD modes
- 25+ validated thermal conductivity models
- Complete viscosity modeling (temperature + shear-rate dependent)
- Particle aggregation physics (DLVO theory)
- Flow regime analysis (laminar, transitional, turbulent)
- Full CFD solver (Navier-Stokes + energy equation)
- AI-powered recommendations
- Interfacial nanolayer effects
- Real-time multi-physics visualization

🚀 What's New in v7.0:
- Unified BKPSNanofluidEngine API
- Configuration management system
- User-extensible material database with CRUD GUI
- Validated against 6 published datasets (72.7% within ±20%)
- Professional CLI & GUI launcher
- Project save/load functionality
- Comprehensive validation suite

💡 Applications:
- Heat exchangers & cooling systems
- Biomedical & energy systems  
- Research & development
- Any flow-based thermal application

Version: 7.0.0 | Release: November 30, 2025
"""

# NEW in v7.0: Material Database Management
from .material_database import (
    MaterialDatabase,
    NanoparticleMaterial,
    BaseFluidMaterial,
    get_material_database,
)

# Classical models
from .models import (
    maxwell_model,
    hamilton_crosser_model,
    bruggeman_model,
    yu_choi_model,
    wasp_model,
    pak_cho_correlation,
)

# Advanced models
from .advanced_models import (
    patel_model,
    koo_kleinstreuer_model,
    hajjar_hybrid_model,
    esfe_hybrid_model,
    sundar_hybrid_model,
    takabi_salehi_model,
)

# Thermophysical properties
from .thermophysical_properties import (
    ThermophysicalProperties,
    einstein_viscosity,
    batchelor_viscosity,
    brinkman_viscosity,
    nanofluid_density,
    nanofluid_specific_heat,
    prandtl_number,
)

# Simulators
from .simulator import NanofluidSimulator
from .enhanced_simulator import EnhancedNanofluidSimulator
from .flow_simulator import FlowNanofluidSimulator, FlowSimulationResult

# NEW in v7.0: Unified Engine
from .unified_engine import (
    BKPSNanofluidEngine,
    UnifiedConfig,
    SimulationMode,
    SolverBackend,
    NanoparticleConfig,
    BaseFluidConfig,
    GeometryConfig,
    MeshConfig,
    SolverConfig,
    FlowConfig,
    create_static_config,
    create_flow_config,
    create_cfd_config,
    __version__,
    __release_date__,
    __codename__,
)

# Solver modes
from .solver_modes import SolverMode, SolverModeManager, SolverModeConfig

# AI Recommendation Engine
from .ai_recommender import (
    AIRecommendationEngine,
    ApplicationType,
    OptimizationObjective,
    RecommendationConstraints,
    NanofluidRecommendation
)

# Material database
from .nanoparticles import NanoparticleDatabase

# Flow models
from .flow_models import (
    buongiorno_convective_model,
    corcione_model,
    rea_bonnet_convective_model,
    shear_enhanced_conductivity,
    velocity_dependent_conductivity,
)

# Viscosity models
from .viscosity_models import (
    BaseFluidViscosity,
    NanofluidViscosityCalculator,
    carreau_model,
    aggregated_nanofluid_viscosity,
)

# Aggregation physics
from .aggregation_models import (
    ParticleInteractionAnalyzer,
    assess_colloidal_stability,
    dlvo_total_potential,
)

# Flow regime analysis
from .flow_regime_analysis import (
    FlowRegimeAnalyzer,
    reynolds_number,
    prandtl_number,
    nusselt_number_complete,
)

# Export functionality
from .export import (
    ResultExporter,
    ReportGenerator,
    export_results,
    generate_pdf_report,
)

# NEW in v7.0: PDF Report Generator
from .pdf_report import PDFReportGenerator, generate_quick_report

# NEW in v7.0: Advanced Visualization
from .advanced_visualization import (
    AdvancedVisualizer,
    create_sample_cfd_field,
)

# NEW in v7.0: Validation Center
from .validation_center import (
    ValidationCenter,
    ValidationDataset,
    get_validation_summary,
)

# NEW in v7.1: Parameter Sweep Engine
from .sweep_engine import (
    ParameterSweepEngine,
    SweepResult,
    create_comprehensive_sweep_report,
)

# NEW in v7.1: Particle Interaction Visualization
from .interaction_visualization import (
    ParticleInteractionVisualizer,
    InteractionVisualization,
    create_comprehensive_interaction_report,
)

# NEW in v7.1: Enhanced CFD Visualization
from .cfd_visualization_enhanced import (
    RealTimeCFDVisualizer,
    CFDVisualizationResult,
    run_quick_cfd_visualization,
)

# NEW in v7.1: Advanced Flow-Dependent Calculator
from .advanced_flow_calculator import (
    AdvancedFlowCalculator,
    FlowDependentConfig,
    NanoparticleSpec,
    FlowConditions,
    calculate_flow_properties,
)

__version__ = "7.1.0"
__author__ = "Dedicated to Brijesh Kumar Pandey"
__license__ = "MIT"

__all__ = [
    # NEW in v7.0: Unified Engine
    "BKPSNanofluidEngine",
    "UnifiedConfig",
    "SimulationMode",
    "SolverBackend",
    "NanoparticleConfig",
    "BaseFluidConfig",
    "GeometryConfig",
    "MeshConfig",
    "SolverConfig",
    "FlowConfig",
    "create_static_config",
    "create_flow_config",
    "create_cfd_config",
    
    # NEW in v7.0: Material Database
    "MaterialDatabase",
    "NanoparticleMaterial",
    "BaseFluidMaterial",
    "get_material_database",
    
    # Simulators
    "NanofluidSimulator",
    "EnhancedNanofluidSimulator",
    "FlowNanofluidSimulator",
    "FlowSimulationResult",
    
    # Solver modes
    "SolverMode",
    "SolverModeManager",
    "SolverModeConfig",
    
    # AI Recommendation Engine
    "AIRecommendationEngine",
    "ApplicationType",
    "OptimizationObjective",
    "RecommendationConstraints",
    "NanofluidRecommendation",
    
    # Database
    "NanoparticleDatabase",
    
    # Flow models
    "buongiorno_convective_model",
    "corcione_model",
    "rea_bonnet_convective_model",
    "shear_enhanced_conductivity",
    "velocity_dependent_conductivity",
    
    # Viscosity models
    "BaseFluidViscosity",
    "NanofluidViscosityCalculator",
    "carreau_model",
    "aggregated_nanofluid_viscosity",
    
    # Aggregation physics
    "ParticleInteractionAnalyzer",
    "assess_colloidal_stability",
    "dlvo_total_potential",
    
    # Flow regime analysis
    "FlowRegimeAnalyzer",
    "reynolds_number",
    "prandtl_number",
    "nusselt_number_complete",
    
    # Classical models
    "maxwell_model",
    "hamilton_crosser_model",
    "bruggeman_model",
    "yu_choi_model",
    "wasp_model",
    "pak_cho_correlation",
    
    # Advanced models
    "patel_model",
    "koo_kleinstreuer_model",
    "hajjar_hybrid_model",
    "esfe_hybrid_model",
    "sundar_hybrid_model",
    "takabi_salehi_model",
    
    # Thermophysical properties
    "ThermophysicalProperties",
    "einstein_viscosity",
    "batchelor_viscosity",
    "brinkman_viscosity",
    "nanofluid_density",
    "nanofluid_specific_heat",
    "prandtl_number",
    
    # Export
    "ResultExporter",
    "ReportGenerator",
    "export_results",
    "generate_pdf_report",
    
    # NEW in v7.0: PDF Report Generator
    "PDFReportGenerator",
    "generate_quick_report",
    
    # NEW in v7.0: Advanced Visualization
    "AdvancedVisualizer",
    "create_sample_cfd_field",
    
    # NEW in v7.0: Validation Center
    "ValidationCenter",
    "ValidationDataset",
    "get_validation_summary",
    
    # NEW in v7.1: Parameter Sweep Engine
    "ParameterSweepEngine",
    "SweepResult",
    "create_comprehensive_sweep_report",
    
    # NEW in v7.1: Particle Interaction Visualization
    "ParticleInteractionVisualizer",
    "InteractionVisualization",
    "create_comprehensive_interaction_report",
    
    # NEW in v7.1: Enhanced CFD Visualization
    "RealTimeCFDVisualizer",
    "CFDVisualizationResult",
    "run_quick_cfd_visualization",
    
    # NEW in v7.1: Advanced Flow-Dependent Calculator
    "AdvancedFlowCalculator",
    "FlowDependentConfig",
    "NanoparticleSpec",
    "FlowConditions",
    "calculate_flow_properties",
]
