"""
Custom Nanoparticle Shape Definitions

Allows users to define and add new nanoparticle geometries for
thermal conductivity calculations beyond standard shapes.

Features:
- Define custom particle shapes
- Calculate shape factors
- Integrate with existing models
- User-extensible framework

Author: Nanofluid Simulator v4.0
"""

import numpy as np
from typing import Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class ParticleShape:
    """
    Custom particle shape definition.
    
    Attributes
    ----------
    name : str
        Shape identifier
    shape_factor : float
        Empirical shape factor (n) for Hamilton-Crosser model
        - Spheres: n = 3
        - Cylinders: n = 6
        - Platelets: n → ∞
    aspect_ratio : float
        Length-to-diameter ratio (for anisotropic particles)
    surface_area_factor : float
        Relative surface area compared to sphere
    description : str
        Human-readable description
    """
    name: str
    shape_factor: float
    aspect_ratio: float = 1.0
    surface_area_factor: float = 1.0
    description: str = ""


# Predefined shapes library
STANDARD_SHAPES = {
    'sphere': ParticleShape(
        name='sphere',
        shape_factor=3.0,
        aspect_ratio=1.0,
        surface_area_factor=1.0,
        description='Spherical nanoparticles (isotropic)'
    ),
    
    'cylinder': ParticleShape(
        name='cylinder',
        shape_factor=6.0,
        aspect_ratio=3.0,
        surface_area_factor=1.5,
        description='Cylindrical nanorods (L/D ≈ 3)'
    ),
    
    'long_cylinder': ParticleShape(
        name='long_cylinder',
        shape_factor=6.0,
        aspect_ratio=10.0,
        surface_area_factor=3.0,
        description='Long cylindrical nanorods (L/D ≈ 10)'
    ),
    
    'platelet': ParticleShape(
        name='platelet',
        shape_factor=50.0,
        aspect_ratio=0.1,
        surface_area_factor=2.5,
        description='Disk-shaped platelets (very thin)'
    ),
    
    'brick': ParticleShape(
        name='brick',
        shape_factor=3.0,
        aspect_ratio=2.0,
        surface_area_factor=1.2,
        description='Rectangular brick-shaped particles'
    ),
    
    'blade': ParticleShape(
        name='blade',
        shape_factor=16.1,
        aspect_ratio=8.0,
        surface_area_factor=2.0,
        description='Blade-shaped particles (L/W ≈ 8)'
    ),
    
    'carbon_nanotube': ParticleShape(
        name='carbon_nanotube',
        shape_factor=6.0,
        aspect_ratio=100.0,
        surface_area_factor=10.0,
        description='Single-walled carbon nanotubes (very high aspect ratio)'
    ),
    
    'graphene': ParticleShape(
        name='graphene',
        shape_factor=100.0,
        aspect_ratio=0.001,
        surface_area_factor=5.0,
        description='Graphene nanosheets (2D structure)'
    ),
}


class CustomShapeManager:
    """
    Manager for custom nanoparticle shapes.
    
    Allows users to add, modify, and retrieve shape definitions
    for use in thermal conductivity models.
    """
    
    def __init__(self):
        """Initialize with standard shapes"""
        self.shapes = STANDARD_SHAPES.copy()
    
    def add_custom_shape(self, shape: ParticleShape) -> None:
        """
        Add a custom particle shape definition.
        
        Parameters
        ----------
        shape : ParticleShape
            Custom shape definition
            
        Examples
        --------
        >>> manager = CustomShapeManager()
        >>> custom = ParticleShape(
        ...     name='hexagonal',
        ...     shape_factor=4.5,
        ...     aspect_ratio=1.0,
        ...     surface_area_factor=1.1,
        ...     description='Hexagonal prism particles'
        ... )
        >>> manager.add_custom_shape(custom)
        """
        self.shapes[shape.name] = shape
        print(f"✓ Added custom shape: {shape.name}")
        print(f"  Shape factor: {shape.shape_factor}")
        print(f"  Aspect ratio: {shape.aspect_ratio}")
    
    def get_shape(self, name: str) -> Optional[ParticleShape]:
        """
        Retrieve shape definition by name.
        
        Parameters
        ----------
        name : str
            Shape identifier
            
        Returns
        -------
        shape : ParticleShape or None
            Shape definition if found
        """
        return self.shapes.get(name)
    
    def list_shapes(self) -> None:
        """Print all available shapes"""
        print("\n" + "="*70)
        print("AVAILABLE NANOPARTICLE SHAPES")
        print("="*70)
        print(f"{'Name':<20} {'Shape Factor':<15} {'Aspect Ratio':<15}")
        print("-"*70)
        
        for name, shape in sorted(self.shapes.items()):
            print(f"{name:<20} {shape.shape_factor:<15.1f} {shape.aspect_ratio:<15.2f}")
        
        print("-"*70)
        print(f"Total: {len(self.shapes)} shapes\n")
        
        # Detailed descriptions
        print("DESCRIPTIONS:")
        for name, shape in sorted(self.shapes.items()):
            if shape.description:
                print(f"  • {name}: {shape.description}")
    
    def calculate_shape_factor_empirical(self, aspect_ratio: float) -> float:
        """
        Calculate empirical shape factor from aspect ratio.
        
        Based on correlations from literature for ellipsoids.
        
        Parameters
        ----------
        aspect_ratio : float
            Length-to-diameter ratio (L/D)
            
        Returns
        -------
        n : float
            Shape factor for Hamilton-Crosser model
            
        Notes
        -----
        Empirical correlation:
        - Spheres (L/D = 1): n = 3
        - Prolate ellipsoids (L/D > 1): n increases with L/D
        - Oblate ellipsoids (L/D < 1): n increases as L/D → 0
        """
        if aspect_ratio < 1.0:
            # Oblate (disk-like)
            n = 3.0 + 20.0 * (1.0 - aspect_ratio)
        elif aspect_ratio > 1.0:
            # Prolate (rod-like)
            n = 3.0 + 3.0 * np.log(aspect_ratio)
        else:
            # Sphere
            n = 3.0
        
        return n
    
    def create_shape_from_aspect_ratio(self, name: str, aspect_ratio: float,
                                      description: str = "") -> ParticleShape:
        """
        Create shape definition from aspect ratio.
        
        Automatically calculates shape factor using empirical correlation.
        
        Parameters
        ----------
        name : str
            Shape identifier
        aspect_ratio : float
            Length-to-diameter ratio
        description : str, optional
            Human-readable description
            
        Returns
        -------
        shape : ParticleShape
            New shape definition
            
        Examples
        --------
        >>> manager = CustomShapeManager()
        >>> shape = manager.create_shape_from_aspect_ratio(
        ...     'my_rod',
        ...     aspect_ratio=5.0,
        ...     description='Custom nanorods with L/D=5'
        ... )
        >>> manager.add_custom_shape(shape)
        """
        shape_factor = self.calculate_shape_factor_empirical(aspect_ratio)
        
        # Estimate surface area factor
        if aspect_ratio < 1.0:
            # Oblate
            surface_area_factor = 1.0 + 1.5 * (1.0 - aspect_ratio)
        else:
            # Prolate
            surface_area_factor = 1.0 + 0.5 * (aspect_ratio - 1.0)
        
        shape = ParticleShape(
            name=name,
            shape_factor=shape_factor,
            aspect_ratio=aspect_ratio,
            surface_area_factor=surface_area_factor,
            description=description
        )
        
        return shape
    
    def get_shape_factor(self, shape_name: str) -> float:
        """
        Get shape factor for use in Hamilton-Crosser model.
        
        Parameters
        ----------
        shape_name : str
            Shape identifier
            
        Returns
        -------
        n : float
            Shape factor
            
        Raises
        ------
        ValueError
            If shape not found
        """
        shape = self.get_shape(shape_name)
        if shape is None:
            raise ValueError(f"Shape '{shape_name}' not found. Use list_shapes() to see available shapes.")
        return shape.shape_factor
    
    def export_shapes_to_file(self, filename: str = "custom_shapes.txt") -> None:
        """
        Export all shape definitions to file.
        
        Parameters
        ----------
        filename : str
            Output filename
        """
        with open(filename, 'w') as f:
            f.write("# Custom Nanoparticle Shape Definitions\n")
            f.write("# Format: name, shape_factor, aspect_ratio, surface_area_factor, description\n\n")
            
            for name, shape in sorted(self.shapes.items()):
                f.write(f"{name}, {shape.shape_factor:.2f}, {shape.aspect_ratio:.3f}, "
                       f"{shape.surface_area_factor:.3f}, {shape.description}\n")
        
        print(f"✓ Shapes exported to {filename}")
    
    def import_shapes_from_file(self, filename: str) -> None:
        """
        Import shape definitions from file.
        
        Parameters
        ----------
        filename : str
            Input filename
        """
        count = 0
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        name = parts[0]
                        shape_factor = float(parts[1])
                        aspect_ratio = float(parts[2])
                        surface_area_factor = float(parts[3])
                        description = parts[4] if len(parts) > 4 else ""
                        
                        shape = ParticleShape(
                            name=name,
                            shape_factor=shape_factor,
                            aspect_ratio=aspect_ratio,
                            surface_area_factor=surface_area_factor,
                            description=description
                        )
                        self.add_custom_shape(shape)
                        count += 1
        
        print(f"✓ Imported {count} shapes from {filename}")


# Global instance for easy access
shape_manager = CustomShapeManager()


def get_available_shapes() -> Dict[str, ParticleShape]:
    """Get dictionary of all available shapes"""
    return shape_manager.shapes.copy()


def add_shape(name: str, shape_factor: float, aspect_ratio: float = 1.0,
              surface_area_factor: float = 1.0, description: str = "") -> None:
    """
    Convenience function to add custom shape.
    
    Parameters
    ----------
    name : str
        Shape identifier
    shape_factor : float
        Empirical shape factor (n) for Hamilton-Crosser
    aspect_ratio : float
        Length-to-diameter ratio
    surface_area_factor : float
        Relative surface area vs sphere
    description : str
        Human-readable description
        
    Examples
    --------
    >>> from nanofluid_simulator.custom_shapes import add_shape
    >>> add_shape(
    ...     'my_particle',
    ...     shape_factor=4.5,
    ...     aspect_ratio=2.5,
    ...     description='My custom particle geometry'
    ... )
    """
    shape = ParticleShape(
        name=name,
        shape_factor=shape_factor,
        aspect_ratio=aspect_ratio,
        surface_area_factor=surface_area_factor,
        description=description
    )
    shape_manager.add_custom_shape(shape)


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("CUSTOM NANOPARTICLE SHAPE MANAGER")
    print("="*70)
    
    # Show standard shapes
    shape_manager.list_shapes()
    
    print("\n" + "="*70)
    print("EXAMPLE: Creating Custom Shape")
    print("="*70)
    
    # Create custom shape from aspect ratio
    custom = shape_manager.create_shape_from_aspect_ratio(
        'my_nanorod',
        aspect_ratio=15.0,
        description='Custom nanorods for my experiment'
    )
    
    print(f"\nCreated shape:")
    print(f"  Name: {custom.name}")
    print(f"  Shape factor: {custom.shape_factor:.2f}")
    print(f"  Aspect ratio: {custom.aspect_ratio:.2f}")
    print(f"  Surface area factor: {custom.surface_area_factor:.2f}")
    
    # Add to manager
    shape_manager.add_custom_shape(custom)
    
    print("\n" + "="*70)
    print("USAGE IN SIMULATOR:")
    print("="*70)
    print("""
from nanofluid_simulator.custom_shapes import add_shape, get_available_shapes
from nanofluid_simulator import NanofluidSimulator

# Add your custom shape
add_shape(
    'my_particle',
    shape_factor=5.2,
    aspect_ratio=4.0,
    description='My experimental nanoparticles'
)

# Use in simulator (requires integration)
# sim = NanofluidSimulator(particle_shape='my_particle')
""")
