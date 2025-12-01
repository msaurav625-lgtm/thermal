#!/usr/bin/env python3
"""
Material Database Manager for BKPS NFL Thermal Pro 7.0
Handles user-defined nanoparticles and base fluids with CRUD operations

Dedicated to: Brijesh Kumar Pandey
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

# Default data directory
USER_DATA_DIR = Path.home() / "Documents" / "BKPS_NFL"
USER_MATERIALS_FILE = USER_DATA_DIR / "user_materials.json"
USER_BASEFLUIDS_FILE = USER_DATA_DIR / "user_basefluids.json"


@dataclass
class NanoparticleMaterial:
    """Nanoparticle material properties"""
    name: str
    thermal_conductivity: float  # W/m·K
    density: float  # kg/m³
    specific_heat: float  # J/kg·K
    description: str = ""
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate material properties"""
        errors = []
        
        if not self.name or len(self.name) < 2:
            errors.append("Material name must be at least 2 characters")
        
        if self.thermal_conductivity <= 0 or self.thermal_conductivity > 10000:
            errors.append(f"Thermal conductivity {self.thermal_conductivity} out of realistic range (0, 10000] W/m·K")
        
        if self.density <= 0 or self.density > 50000:
            errors.append(f"Density {self.density} out of realistic range (0, 50000] kg/m³")
        
        if self.specific_heat <= 0 or self.specific_heat > 10000:
            errors.append(f"Specific heat {self.specific_heat} out of realistic range (0, 10000] J/kg·K")
        
        return len(errors) == 0, errors


@dataclass
class BaseFluidMaterial:
    """Base fluid properties"""
    name: str
    thermal_conductivity: float  # W/m·K at reference temperature
    density: float  # kg/m³ at reference temperature
    specific_heat: float  # J/kg·K at reference temperature
    viscosity: float  # Pa·s at reference temperature
    reference_temperature: float = 300.0  # K
    description: str = ""
    
    # Optional: temperature-dependent correlations
    k_coefficients: Optional[List[float]] = None  # [a, b, c] for k = a + b*T + c*T²
    rho_coefficients: Optional[List[float]] = None
    cp_coefficients: Optional[List[float]] = None
    mu_coefficients: Optional[List[float]] = None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate fluid properties"""
        errors = []
        
        if not self.name or len(self.name) < 2:
            errors.append("Fluid name must be at least 2 characters")
        
        if self.thermal_conductivity <= 0 or self.thermal_conductivity > 10:
            errors.append(f"Thermal conductivity {self.thermal_conductivity} out of realistic range (0, 10] W/m·K for fluids")
        
        if self.density <= 0 or self.density > 20000:
            errors.append(f"Density {self.density} out of realistic range (0, 20000] kg/m³")
        
        if self.specific_heat <= 0 or self.specific_heat > 10000:
            errors.append(f"Specific heat {self.specific_heat} out of realistic range (0, 10000] J/kg·K")
        
        if self.viscosity <= 0 or self.viscosity > 1:
            errors.append(f"Viscosity {self.viscosity} out of realistic range (0, 1] Pa·s")
        
        if self.reference_temperature < 200 or self.reference_temperature > 500:
            errors.append(f"Reference temperature {self.reference_temperature} out of range [200, 500] K")
        
        return len(errors) == 0, errors
    
    def get_property_at_temperature(self, prop_name: str, temperature: float) -> float:
        """Calculate temperature-dependent property"""
        # Get base value
        base_value = getattr(self, prop_name)
        
        # Get coefficients
        coeff_name = f"{prop_name}_coefficients"
        coefficients = getattr(self, coeff_name, None)
        
        if coefficients and len(coefficients) >= 2:
            # Polynomial evaluation: prop = a + b*T + c*T²
            T = temperature
            value = coefficients[0]
            if len(coefficients) > 1:
                value += coefficients[1] * T
            if len(coefficients) > 2:
                value += coefficients[2] * T**2
            return value
        else:
            # No temperature dependence, return base value
            return base_value


class MaterialDatabase:
    """
    Manager for user-defined materials and base fluids
    
    Provides CRUD operations with validation and persistence
    """
    
    def __init__(self):
        """Initialize database"""
        # Ensure user data directory exists
        USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self.nanoparticles: Dict[str, NanoparticleMaterial] = self._load_nanoparticles()
        self.base_fluids: Dict[str, BaseFluidMaterial] = self._load_base_fluids()
        
        # Initialize with defaults if empty
        if not self.nanoparticles:
            self._initialize_default_nanoparticles()
        
        if not self.base_fluids:
            self._initialize_default_base_fluids()
    
    def _load_nanoparticles(self) -> Dict[str, NanoparticleMaterial]:
        """Load nanoparticles from file"""
        if not USER_MATERIALS_FILE.exists():
            return {}
        
        try:
            with open(USER_MATERIALS_FILE, 'r') as f:
                data = json.load(f)
            
            materials = {}
            for name, props in data.items():
                materials[name] = NanoparticleMaterial(**props)
            
            logger.info(f"Loaded {len(materials)} nanoparticle materials")
            return materials
        except Exception as e:
            logger.error(f"Failed to load nanoparticles: {e}")
            return {}
    
    def _load_base_fluids(self) -> Dict[str, BaseFluidMaterial]:
        """Load base fluids from file"""
        if not USER_BASEFLUIDS_FILE.exists():
            return {}
        
        try:
            with open(USER_BASEFLUIDS_FILE, 'r') as f:
                data = json.load(f)
            
            fluids = {}
            for name, props in data.items():
                fluids[name] = BaseFluidMaterial(**props)
            
            logger.info(f"Loaded {len(fluids)} base fluid materials")
            return fluids
        except Exception as e:
            logger.error(f"Failed to load base fluids: {e}")
            return {}
    
    def _initialize_default_nanoparticles(self):
        """Initialize with default nanoparticle materials"""
        defaults = {
            "Al2O3": NanoparticleMaterial(
                name="Al2O3",
                thermal_conductivity=40.0,
                density=3970,
                specific_heat=765,
                description="Aluminum Oxide (alumina)"
            ),
            "CuO": NanoparticleMaterial(
                name="CuO",
                thermal_conductivity=18.0,
                density=6500,
                specific_heat=531,
                description="Copper Oxide"
            ),
            "TiO2": NanoparticleMaterial(
                name="TiO2",
                thermal_conductivity=8.4,
                density=4250,
                specific_heat=686,
                description="Titanium Dioxide"
            ),
            "SiO2": NanoparticleMaterial(
                name="SiO2",
                thermal_conductivity=1.4,
                density=2220,
                specific_heat=745,
                description="Silicon Dioxide (silica)"
            ),
            "Cu": NanoparticleMaterial(
                name="Cu",
                thermal_conductivity=401,
                density=8933,
                specific_heat=385,
                description="Copper (metal)"
            ),
            "Ag": NanoparticleMaterial(
                name="Ag",
                thermal_conductivity=429,
                density=10500,
                specific_heat=235,
                description="Silver (metal)"
            ),
            "Au": NanoparticleMaterial(
                name="Au",
                thermal_conductivity=317,
                density=19300,
                specific_heat=129,
                description="Gold (metal)"
            ),
            "Fe3O4": NanoparticleMaterial(
                name="Fe3O4",
                thermal_conductivity=9.7,
                density=5180,
                specific_heat=670,
                description="Magnetite (iron oxide)"
            ),
            "ZnO": NanoparticleMaterial(
                name="ZnO",
                thermal_conductivity=29.0,
                density=5606,
                specific_heat=494,
                description="Zinc Oxide"
            ),
            "CNT": NanoparticleMaterial(
                name="CNT",
                thermal_conductivity=3000,
                density=2100,
                specific_heat=710,
                description="Carbon Nanotubes"
            ),
            "Graphene": NanoparticleMaterial(
                name="Graphene",
                thermal_conductivity=5000,
                density=2267,
                specific_heat=710,
                description="Graphene sheets"
            ),
        }
        
        self.nanoparticles = defaults
        self._save_nanoparticles()
        logger.info("Initialized default nanoparticle materials")
    
    def _initialize_default_base_fluids(self):
        """Initialize with default base fluid materials"""
        defaults = {
            "Water": BaseFluidMaterial(
                name="Water",
                thermal_conductivity=0.613,
                density=997,
                specific_heat=4182,
                viscosity=0.001,
                reference_temperature=300,
                description="Pure water",
                k_coefficients=[0.5650, 0.002, -1e-5],  # k = 0.565 + 0.002*T - 1e-5*T²
                mu_coefficients=[0.002414, -4.6e-5, 2.87e-8]  # Simplified
            ),
            "EG": BaseFluidMaterial(
                name="EG",
                thermal_conductivity=0.252,
                density=1113,
                specific_heat=2415,
                viscosity=0.0157,
                reference_temperature=300,
                description="Ethylene Glycol",
                k_coefficients=[0.242, 0.0003, -5e-7],
                mu_coefficients=[0.0651, -3.9e-4, 5.8e-7]
            ),
            "EG-Water-50/50": BaseFluidMaterial(
                name="EG-Water-50/50",
                thermal_conductivity=0.415,
                density=1070,
                specific_heat=3300,
                viscosity=0.0039,
                reference_temperature=300,
                description="50% Ethylene Glycol, 50% Water mixture"
            ),
        }
        
        self.base_fluids = defaults
        self._save_base_fluids()
        logger.info("Initialized default base fluid materials")
    
    def _save_nanoparticles(self):
        """Save nanoparticles to file"""
        try:
            data = {name: asdict(mat) for name, mat in self.nanoparticles.items()}
            with open(USER_MATERIALS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.nanoparticles)} nanoparticle materials")
        except Exception as e:
            logger.error(f"Failed to save nanoparticles: {e}")
            raise
    
    def _save_base_fluids(self):
        """Save base fluids to file"""
        try:
            data = {name: asdict(mat) for name, mat in self.base_fluids.items()}
            with open(USER_BASEFLUIDS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.base_fluids)} base fluid materials")
        except Exception as e:
            logger.error(f"Failed to save base fluids: {e}")
            raise
    
    # ===== NANOPARTICLE CRUD =====
    
    def add_nanoparticle(self, material: NanoparticleMaterial) -> Tuple[bool, str]:
        """Add new nanoparticle material"""
        # Validate
        valid, errors = material.validate()
        if not valid:
            return False, "\n".join(errors)
        
        # Check for duplicates
        if material.name in self.nanoparticles:
            return False, f"Material '{material.name}' already exists. Use update() instead."
        
        # Add and save
        self.nanoparticles[material.name] = material
        self._save_nanoparticles()
        
        logger.info(f"Added nanoparticle: {material.name}")
        return True, f"Successfully added {material.name}"
    
    def get_nanoparticle(self, name: str) -> Optional[NanoparticleMaterial]:
        """Get nanoparticle by name"""
        return self.nanoparticles.get(name)
    
    def update_nanoparticle(self, material: NanoparticleMaterial) -> Tuple[bool, str]:
        """Update existing nanoparticle"""
        # Validate
        valid, errors = material.validate()
        if not valid:
            return False, "\n".join(errors)
        
        # Check exists
        if material.name not in self.nanoparticles:
            return False, f"Material '{material.name}' does not exist. Use add() instead."
        
        # Update and save
        self.nanoparticles[material.name] = material
        self._save_nanoparticles()
        
        logger.info(f"Updated nanoparticle: {material.name}")
        return True, f"Successfully updated {material.name}"
    
    def delete_nanoparticle(self, name: str) -> Tuple[bool, str]:
        """Delete nanoparticle"""
        if name not in self.nanoparticles:
            return False, f"Material '{name}' does not exist"
        
        del self.nanoparticles[name]
        self._save_nanoparticles()
        
        logger.info(f"Deleted nanoparticle: {name}")
        return True, f"Successfully deleted {name}"
    
    def list_nanoparticles(self) -> List[str]:
        """List all nanoparticle names"""
        return sorted(self.nanoparticles.keys())
    
    # ===== BASE FLUID CRUD =====
    
    def add_base_fluid(self, fluid: BaseFluidMaterial) -> Tuple[bool, str]:
        """Add new base fluid"""
        # Validate
        valid, errors = fluid.validate()
        if not valid:
            return False, "\n".join(errors)
        
        # Check for duplicates
        if fluid.name in self.base_fluids:
            return False, f"Fluid '{fluid.name}' already exists. Use update() instead."
        
        # Add and save
        self.base_fluids[fluid.name] = fluid
        self._save_base_fluids()
        
        logger.info(f"Added base fluid: {fluid.name}")
        return True, f"Successfully added {fluid.name}"
    
    def get_base_fluid(self, name: str) -> Optional[BaseFluidMaterial]:
        """Get base fluid by name"""
        return self.base_fluids.get(name)
    
    def update_base_fluid(self, fluid: BaseFluidMaterial) -> Tuple[bool, str]:
        """Update existing base fluid"""
        # Validate
        valid, errors = fluid.validate()
        if not valid:
            return False, "\n".join(errors)
        
        # Check exists
        if fluid.name not in self.base_fluids:
            return False, f"Fluid '{fluid.name}' does not exist. Use add() instead."
        
        # Update and save
        self.base_fluids[fluid.name] = fluid
        self._save_base_fluids()
        
        logger.info(f"Updated base fluid: {fluid.name}")
        return True, f"Successfully updated {fluid.name}"
    
    def delete_base_fluid(self, name: str) -> Tuple[bool, str]:
        """Delete base fluid"""
        if name not in self.base_fluids:
            return False, f"Fluid '{name}' does not exist"
        
        del self.base_fluids[name]
        self._save_base_fluids()
        
        logger.info(f"Deleted base fluid: {name}")
        return True, f"Successfully deleted {name}"
    
    def list_base_fluids(self) -> List[str]:
        """List all base fluid names"""
        return sorted(self.base_fluids.keys())
    
    def export_database(self, filepath: Path) -> Tuple[bool, str]:
        """Export entire database to JSON"""
        try:
            from datetime import datetime
            
            data = {
                'nanoparticles': {name: asdict(mat) for name, mat in self.nanoparticles.items()},
                'base_fluids': {name: asdict(mat) for name, mat in self.base_fluids.items()},
                'version': '7.0',
                'exported': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True, f"Database exported to {filepath}"
        except Exception as e:
            return False, f"Export failed: {e}"
    
    def import_database(self, filepath: Path, merge: bool = False) -> Tuple[bool, str]:
        """Import database from JSON"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if not merge:
                self.nanoparticles = {}
                self.base_fluids = {}
            
            # Import nanoparticles
            if 'nanoparticles' in data:
                for name, props in data['nanoparticles'].items():
                    self.nanoparticles[name] = NanoparticleMaterial(**props)
            
            # Import base fluids
            if 'base_fluids' in data:
                for name, props in data['base_fluids'].items():
                    self.base_fluids[name] = BaseFluidMaterial(**props)
            
            self._save_nanoparticles()
            self._save_base_fluids()
            
            return True, f"Database imported from {filepath}"
        except Exception as e:
            return False, f"Import failed: {e}"


# Global instance
_global_database = None

def get_material_database() -> MaterialDatabase:
    """Get global material database instance"""
    global _global_database
    if _global_database is None:
        _global_database = MaterialDatabase()
    return _global_database
