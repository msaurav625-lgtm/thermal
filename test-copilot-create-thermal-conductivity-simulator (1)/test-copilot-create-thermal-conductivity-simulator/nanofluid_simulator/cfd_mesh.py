"""
CFD Mesh Generation Module

Provides structured and unstructured mesh generation for 2D CFD simulations.
Supports rectangular domains, circular pipes, and custom geometries.

Author: Nanofluid Simulator v4.0
License: MIT
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum


class BoundaryType(Enum):
    """Boundary condition types"""
    INLET = "inlet"
    OUTLET = "outlet"
    WALL = "wall"
    SYMMETRY = "symmetry"
    PERIODIC = "periodic"


@dataclass
class Cell:
    """Finite volume cell"""
    id: int
    center: np.ndarray  # (x, y)
    vertices: np.ndarray  # (n_vertices, 2)
    area: float
    faces: List[int]  # Face IDs
    neighbors: List[int]  # Neighbor cell IDs
    
    
@dataclass
class Face:
    """Cell face for flux calculation"""
    id: int
    center: np.ndarray  # (x, y)
    normal: np.ndarray  # Unit normal vector
    area: float  # Face length in 2D
    owner: int  # Owner cell ID
    neighbor: Optional[int]  # Neighbor cell ID (None for boundary)
    boundary_type: Optional[BoundaryType] = None
    vertices: Optional[np.ndarray] = None  # (2, 2) for 2D face


@dataclass
class MeshQuality:
    """Mesh quality metrics"""
    n_cells: int
    n_faces: int
    n_boundary_faces: int
    min_cell_area: float
    max_cell_area: float
    aspect_ratio_min: float
    aspect_ratio_max: float
    skewness_max: float
    orthogonality_min: float


class StructuredMesh2D:
    """
    Structured rectangular mesh generator for 2D domains.
    
    Generates uniform or non-uniform grids with boundary layer refinement.
    """
    
    def __init__(self, 
                 x_range: Tuple[float, float],
                 y_range: Tuple[float, float],
                 nx: int,
                 ny: int,
                 boundary_types: Optional[Dict[str, BoundaryType]] = None):
        """
        Initialize structured mesh.
        
        Parameters
        ----------
        x_range : tuple
            (x_min, x_max) domain extent
        y_range : tuple
            (y_min, y_max) domain extent
        nx : int
            Number of cells in x-direction
        ny : int
            Number of cells in y-direction
        boundary_types : dict, optional
            Boundary types for each face: 'left', 'right', 'top', 'bottom'
        """
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.nx = nx
        self.ny = ny
        self.n_cells = nx * ny
        
        # Default boundary types
        if boundary_types is None:
            boundary_types = {
                'left': BoundaryType.INLET,
                'right': BoundaryType.OUTLET,
                'top': BoundaryType.WALL,
                'bottom': BoundaryType.WALL
            }
        self.boundary_types = boundary_types
        
        # Storage
        self.cells: List[Cell] = []
        self.faces: List[Face] = []
        self.boundary_faces: Dict[BoundaryType, List[int]] = {}
        
        # Generate mesh
        self._generate_uniform_grid()
        
    def _generate_uniform_grid(self):
        """Generate uniform rectangular grid"""
        dx = (self.x_max - self.x_min) / self.nx
        dy = (self.y_max - self.y_min) / self.ny
        
        # Generate nodes
        x = np.linspace(self.x_min, self.x_max, self.nx + 1)
        y = np.linspace(self.y_min, self.y_max, self.ny + 1)
        
        # Generate cells
        cell_id = 0
        for j in range(self.ny):
            for i in range(self.nx):
                # Cell vertices (counter-clockwise from bottom-left)
                vertices = np.array([
                    [x[i], y[j]],
                    [x[i+1], y[j]],
                    [x[i+1], y[j+1]],
                    [x[i], y[j+1]]
                ])
                
                # Cell center
                center = np.array([x[i] + dx/2, y[j] + dy/2])
                
                # Cell area
                area = dx * dy
                
                # Create cell (faces and neighbors will be added later)
                cell = Cell(
                    id=cell_id,
                    center=center,
                    vertices=vertices,
                    area=area,
                    faces=[],
                    neighbors=[]
                )
                self.cells.append(cell)
                cell_id += 1
        
        # Generate faces
        self._generate_faces()
        
    def _generate_faces(self):
        """Generate internal and boundary faces"""
        face_id = 0
        
        # Internal vertical faces (x-direction)
        for j in range(self.ny):
            for i in range(self.nx + 1):
                if i == 0:
                    # Left boundary
                    owner = j * self.nx
                    neighbor = None
                    boundary_type = self.boundary_types['left']
                elif i == self.nx:
                    # Right boundary
                    owner = j * self.nx + (self.nx - 1)
                    neighbor = None
                    boundary_type = self.boundary_types['right']
                else:
                    # Internal face
                    owner = j * self.nx + (i - 1)
                    neighbor = j * self.nx + i
                    boundary_type = None
                
                # Face properties
                x = self.x_min + i * (self.x_max - self.x_min) / self.nx
                y_center = self.y_min + (j + 0.5) * (self.y_max - self.y_min) / self.ny
                dy = (self.y_max - self.y_min) / self.ny
                
                face = Face(
                    id=face_id,
                    center=np.array([x, y_center]),
                    normal=np.array([1.0, 0.0]) if neighbor is not None or boundary_type == self.boundary_types['left'] else np.array([-1.0, 0.0]),
                    area=dy,
                    owner=owner,
                    neighbor=neighbor,
                    boundary_type=boundary_type
                )
                
                self.faces.append(face)
                self.cells[owner].faces.append(face_id)
                if neighbor is not None:
                    self.cells[neighbor].faces.append(face_id)
                    self.cells[owner].neighbors.append(neighbor)
                    self.cells[neighbor].neighbors.append(owner)
                
                face_id += 1
        
        # Internal horizontal faces (y-direction)
        for j in range(self.ny + 1):
            for i in range(self.nx):
                if j == 0:
                    # Bottom boundary
                    owner = i
                    neighbor = None
                    boundary_type = self.boundary_types['bottom']
                elif j == self.ny:
                    # Top boundary
                    owner = (self.ny - 1) * self.nx + i
                    neighbor = None
                    boundary_type = self.boundary_types['top']
                else:
                    # Internal face
                    owner = (j - 1) * self.nx + i
                    neighbor = j * self.nx + i
                    boundary_type = None
                
                # Face properties
                x_center = self.x_min + (i + 0.5) * (self.x_max - self.x_min) / self.nx
                y = self.y_min + j * (self.y_max - self.y_min) / self.ny
                dx = (self.x_max - self.x_min) / self.nx
                
                face = Face(
                    id=face_id,
                    center=np.array([x_center, y]),
                    normal=np.array([0.0, 1.0]) if neighbor is not None or boundary_type == self.boundary_types['bottom'] else np.array([0.0, -1.0]),
                    area=dx,
                    owner=owner,
                    neighbor=neighbor,
                    boundary_type=boundary_type
                )
                
                self.faces.append(face)
                self.cells[owner].faces.append(face_id)
                if neighbor is not None:
                    self.cells[neighbor].faces.append(face_id)
                    if neighbor not in self.cells[owner].neighbors:
                        self.cells[owner].neighbors.append(neighbor)
                    if owner not in self.cells[neighbor].neighbors:
                        self.cells[neighbor].neighbors.append(owner)
                
                face_id += 1
        
        # Organize boundary faces
        for face in self.faces:
            if face.boundary_type is not None:
                if face.boundary_type not in self.boundary_faces:
                    self.boundary_faces[face.boundary_type] = []
                self.boundary_faces[face.boundary_type].append(face.id)
    
    def get_cell(self, i: int, j: int) -> Cell:
        """Get cell by (i, j) indices"""
        return self.cells[j * self.nx + i]
    
    def get_mesh_quality(self) -> MeshQuality:
        """Calculate mesh quality metrics"""
        areas = np.array([cell.area for cell in self.cells])
        
        # Aspect ratios (for structured grid, simplified)
        dx = (self.x_max - self.x_min) / self.nx
        dy = (self.y_max - self.y_min) / self.ny
        aspect_ratio = max(dx/dy, dy/dx)
        
        # Count boundary faces
        n_boundary = sum(len(faces) for faces in self.boundary_faces.values())
        
        return MeshQuality(
            n_cells=self.n_cells,
            n_faces=len(self.faces),
            n_boundary_faces=n_boundary,
            min_cell_area=float(np.min(areas)),
            max_cell_area=float(np.max(areas)),
            aspect_ratio_min=aspect_ratio,
            aspect_ratio_max=aspect_ratio,
            skewness_max=0.0,  # Rectangular cells have zero skewness
            orthogonality_min=1.0  # Rectangular cells are perfectly orthogonal
        )
    
    def refine_boundary_layer(self, 
                               boundary: str,
                               first_cell_height: float,
                               growth_ratio: float = 1.2,
                               n_layers: int = 5):
        """
        Apply boundary layer refinement near specified boundary.
        
        Parameters
        ----------
        boundary : str
            Boundary to refine ('left', 'right', 'top', 'bottom')
        first_cell_height : float
            Height of first cell adjacent to boundary
        growth_ratio : float
            Growth ratio for subsequent layers
        n_layers : int
            Number of boundary layer cells
        """
        # TODO: Implement boundary layer meshing
        # This requires non-uniform grid generation
        raise NotImplementedError("Boundary layer refinement coming in next update")


class CircularPipeMesh:
    """
    Specialized mesh for circular pipe flows.
    
    Uses cylindrical coordinates with structured grid in r-theta plane.
    """
    
    def __init__(self,
                 radius: float,
                 length: float,
                 nr: int,
                 ntheta: int,
                 nz: int):
        """
        Initialize circular pipe mesh.
        
        Parameters
        ----------
        radius : float
            Pipe radius
        length : float
            Pipe length
        nr : int
            Number of cells in radial direction
        ntheta : int
            Number of cells in circumferential direction
        nz : int
            Number of cells in axial direction
        """
        self.radius = radius
        self.length = length
        self.nr = nr
        self.ntheta = ntheta
        self.nz = nz
        
        # TODO: Implement cylindrical mesh generation
        raise NotImplementedError("Circular pipe mesh coming in next update")


def visualize_mesh(mesh: StructuredMesh2D, filename: Optional[str] = None):
    """
    Visualize mesh structure.
    
    Parameters
    ----------
    mesh : StructuredMesh2D
        Mesh to visualize
    filename : str, optional
        If provided, save figure to file
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
    except ImportError:
        print("Warning: matplotlib not available for mesh visualization")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot cells
    patches = []
    for cell in mesh.cells:
        polygon = Polygon(cell.vertices, closed=True)
        patches.append(polygon)
    
    collection = PatchCollection(patches, alpha=0.3, edgecolor='black', facecolor='lightblue')
    ax.add_collection(collection)
    
    # Plot cell centers
    centers = np.array([cell.center for cell in mesh.cells])
    ax.plot(centers[:, 0], centers[:, 1], 'ro', markersize=2, label='Cell centers')
    
    # Plot boundary faces
    colors = {'inlet': 'blue', 'outlet': 'red', 'wall': 'black', 'symmetry': 'green'}
    for boundary_type, face_ids in mesh.boundary_faces.items():
        for face_id in face_ids:
            face = mesh.faces[face_id]
            ax.plot(face.center[0], face.center[1], 'o', 
                   color=colors.get(boundary_type.value, 'gray'),
                   markersize=3)
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Structured Mesh: {mesh.nx}×{mesh.ny} = {mesh.n_cells} cells')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Mesh visualization saved to {filename}")
    else:
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("CFD MESH GENERATION MODULE - Demo")
    print("=" * 60)
    
    # Create simple rectangular mesh
    print("\n📐 Creating 10×5 rectangular mesh...")
    mesh = StructuredMesh2D(
        x_range=(0.0, 1.0),
        y_range=(0.0, 0.5),
        nx=10,
        ny=5
    )
    
    # Print mesh statistics
    quality = mesh.get_mesh_quality()
    print(f"\n📊 Mesh Statistics:")
    print(f"   Cells:          {quality.n_cells}")
    print(f"   Faces:          {quality.n_faces}")
    print(f"   Boundary faces: {quality.n_boundary_faces}")
    print(f"   Cell area:      {quality.min_cell_area:.6f} - {quality.max_cell_area:.6f} m²")
    print(f"   Aspect ratio:   {quality.aspect_ratio_max:.2f}")
    print(f"   Orthogonality:  {quality.orthogonality_min:.2f}")
    
    print(f"\n🔢 Boundary faces by type:")
    for boundary_type, face_ids in mesh.boundary_faces.items():
        print(f"   {boundary_type.value:10s}: {len(face_ids)} faces")
    
    # Example: Access specific cell
    cell_0_0 = mesh.get_cell(0, 0)
    print(f"\n🔍 Cell (0,0) details:")
    print(f"   Center:    ({cell_0_0.center[0]:.3f}, {cell_0_0.center[1]:.3f})")
    print(f"   Area:      {cell_0_0.area:.6f} m²")
    print(f"   Neighbors: {cell_0_0.neighbors}")
    print(f"   Faces:     {len(cell_0_0.faces)}")
    
    # Visualize mesh
    print("\n🎨 Generating mesh visualization...")
    visualize_mesh(mesh, filename='mesh_demo.png')
    
    print("\n✅ Mesh generation complete!")
    print("\n📝 Next steps:")
    print("   1. Implement finite volume discretization")
    print("   2. Add Navier-Stokes solver")
    print("   3. Integrate with nanofluid properties")
