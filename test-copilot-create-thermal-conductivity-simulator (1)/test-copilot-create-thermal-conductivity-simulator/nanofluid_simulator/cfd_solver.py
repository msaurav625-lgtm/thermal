"""
CFD Finite Volume Solver

Implements finite volume method for solving Navier-Stokes equations
with energy transport for nanofluid flows.

Author: Nanofluid Simulator v4.0 - CFD Module
License: MIT
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from typing import Optional, Dict, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

from .cfd_mesh import StructuredMesh2D, Face, Cell, BoundaryType
from .cfd_turbulence import (
    TurbulenceModel,
    create_turbulence_model,
    TurbulenceField
)

# Import AI-CFD integration (optional, graceful fallback)
try:
    from .ai_cfd_integration import (
        AIFlowRegimeClassifier,
        AIConvergenceMonitor,
        AISolverParameterRecommender
    )
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


class InterpolationScheme(Enum):
    """Face interpolation schemes"""
    CENTRAL = "central"  # 2nd order, may be unstable
    UPWIND = "upwind"    # 1st order, stable but diffusive
    QUICK = "quick"      # 3rd order quasi-upstream
    

class GradientScheme(Enum):
    """Gradient calculation methods"""
    GAUSS = "gauss"           # Gauss theorem (Green's theorem in 2D)
    LEAST_SQUARES = "least_squares"  # Least squares fit


@dataclass
class FlowField:
    """Store flow field variables"""
    u: np.ndarray  # x-velocity (m/s)
    v: np.ndarray  # y-velocity (m/s)
    p: np.ndarray  # Pressure (Pa)
    T: np.ndarray  # Temperature (K)
    rho: np.ndarray  # Density (kg/m³)
    mu: np.ndarray  # Dynamic viscosity (Pa·s)
    k: np.ndarray  # Thermal conductivity (W/m·K)
    
    
@dataclass
class SolverSettings:
    """CFD solver configuration"""
    max_iterations: int = 1000
    convergence_tol: float = 1e-6
    under_relaxation_u: float = 0.7
    under_relaxation_v: float = 0.7
    under_relaxation_p: float = 0.3
    under_relaxation_T: float = 0.8
    time_step: float = 0.001  # For transient
    interpolation_scheme: InterpolationScheme = InterpolationScheme.CENTRAL
    gradient_scheme: GradientScheme = GradientScheme.GAUSS
    turbulence_model: TurbulenceModel = TurbulenceModel.LAMINAR
    turbulence_intensity: float = 0.05  # 5% default
    

@dataclass
class BoundaryCondition:
    """Boundary condition specification"""
    bc_type: BoundaryType
    velocity: Optional[Tuple[float, float]] = None  # (u, v) for velocity BC
    pressure: Optional[float] = None  # Pressure for pressure BC
    temperature: Optional[float] = None  # Temperature for thermal BC
    heat_flux: Optional[float] = None  # Heat flux for Neumann BC
    

class NavierStokesSolver:
    """
    2D incompressible Navier-Stokes solver using SIMPLE algorithm.
    
    Solves:
    - Continuity: ∇·u = 0
    - Momentum: ∂u/∂t + u·∇u = -∇p/ρ + ν∇²u
    - Energy: ∂T/∂t + u·∇T = α∇²T
    """
    
    def __init__(self, 
                 mesh: StructuredMesh2D,
                 settings: Optional[SolverSettings] = None):
        """
        Initialize Navier-Stokes solver.
        
        Parameters
        ----------
        mesh : StructuredMesh2D
            Computational mesh
        settings : SolverSettings, optional
            Solver configuration
        """
        self.mesh = mesh
        self.settings = settings if settings is not None else SolverSettings()
        
        # Initialize flow field
        n_cells = mesh.n_cells
        self.field = FlowField(
            u=np.zeros(n_cells),
            v=np.zeros(n_cells),
            p=np.zeros(n_cells),
            T=np.ones(n_cells) * 300.0,  # Default 300 K
            rho=np.ones(n_cells) * 1000.0,  # Default water density
            mu=np.ones(n_cells) * 0.001,  # Default water viscosity
            k=np.ones(n_cells) * 0.6  # Default water thermal conductivity
        )
        
        # Initialize turbulence model
        self.turbulence = create_turbulence_model(
            self.settings.turbulence_model,
            mesh
        )
        if self.turbulence is not None:
            print(f"   Turbulence model: {self.settings.turbulence_model.value}")
        
        # Boundary conditions
        self.boundary_conditions: Dict[BoundaryType, BoundaryCondition] = {}
        
        # Convergence history
        self.residuals: Dict[str, list] = {
            'u': [], 'v': [], 'p': [], 'T': [], 'continuity': []
        }
        
        # AI integration (optional)
        self.ai_enabled = False
        self.ai_convergence_monitor = None
        self.ai_regime_classifier = None
        self.ai_recommender = None
        
    def set_boundary_condition(self, 
                                boundary_type: BoundaryType,
                                bc: BoundaryCondition):
        """Set boundary condition for a boundary type"""
        self.boundary_conditions[boundary_type] = bc
        
    def initialize_field(self,
                         u0: float = 0.0,
                         v0: float = 0.0,
                         p0: float = 0.0,
                         T0: float = 300.0):
        """Initialize flow field with uniform values"""
        self.field.u[:] = u0
        self.field.v[:] = v0
        self.field.p[:] = p0
        self.field.T[:] = T0
        
    def set_nanofluid_properties(self,
                                 rho: np.ndarray,
                                 mu: np.ndarray,
                                 k: np.ndarray):
        """
        Set nanofluid thermophysical properties.
        
        Parameters
        ----------
        rho : array
            Density field (kg/m³)
        mu : array
            Dynamic viscosity field (Pa·s)
        k : array
            Thermal conductivity field (W/m·K)
        """
        self.field.rho = rho.copy()
        self.field.mu = mu.copy()
        self.field.k = k.copy()
        
        # Initialize turbulence if model is active
        if self.turbulence is not None:
            u_ref = max(np.max(np.abs(self.field.u)), 0.1)
            length_scale = min(self.mesh.x_max - self.mesh.x_min,
                              self.mesh.y_max - self.mesh.y_min) * 0.1
            
            self.turbulence.initialize_from_intensity(
                turbulence_intensity=self.settings.turbulence_intensity,
                length_scale=length_scale,
                u_ref=u_ref,
                rho=rho
            )
    
    def set_fluid_properties(self,
                            viscosity: float,
                            density: float,
                            thermal_conductivity: float,
                            specific_heat: float = 4180.0):
        """
        Set uniform fluid properties across domain.
        
        Parameters
        ----------
        viscosity : float
            Dynamic viscosity (Pa·s)
        density : float
            Density (kg/m³)
        thermal_conductivity : float
            Thermal conductivity (W/m·K)
        specific_heat : float
            Specific heat capacity (J/kg·K)
        """
        n_cells = self.mesh.n_cells
        self.field.rho = np.ones(n_cells) * density
        self.field.mu = np.ones(n_cells) * viscosity
        self.field.k = np.ones(n_cells) * thermal_conductivity
        
        # Store specific heat (if field exists)
        if hasattr(self.field, 'cp'):
            self.field.cp = np.ones(n_cells) * specific_heat
        
        # Initialize turbulence if model is active
        if self.turbulence is not None:
            u_ref = max(np.max(np.abs(self.field.u)), 0.1)
            length_scale = min(self.mesh.x_max - self.mesh.x_min,
                              self.mesh.y_max - self.mesh.y_min) * 0.1
            
            self.turbulence.initialize_from_intensity(
                turbulence_intensity=self.settings.turbulence_intensity,
                length_scale=length_scale,
                u_ref=u_ref,
                rho=self.field.rho
            )
        
    def interpolate_to_face(self, 
                            face: Face,
                            phi: np.ndarray,
                            scheme: InterpolationScheme = None) -> float:
        """
        Interpolate cell-centered value to face.
        
        Parameters
        ----------
        face : Face
            Face where interpolation is needed
        phi : array
            Cell-centered field
        scheme : InterpolationScheme, optional
            Interpolation scheme
            
        Returns
        -------
        float
            Interpolated value at face
        """
        if scheme is None:
            scheme = self.settings.interpolation_scheme
            
        owner_value = phi[face.owner]
        
        # Boundary face
        if face.neighbor is None:
            return owner_value  # Use owner value (will be overridden by BC)
        
        neighbor_value = phi[face.neighbor]
        
        if scheme == InterpolationScheme.CENTRAL:
            # Linear interpolation (2nd order central differencing)
            owner_cell = self.mesh.cells[face.owner]
            neighbor_cell = self.mesh.cells[face.neighbor]
            
            # Distance from owner center to face center
            d_owner = np.linalg.norm(face.center - owner_cell.center)
            d_neighbor = np.linalg.norm(face.center - neighbor_cell.center)
            d_total = d_owner + d_neighbor
            
            # Linear interpolation weight
            w = d_neighbor / d_total
            return w * owner_value + (1 - w) * neighbor_value
            
        elif scheme == InterpolationScheme.UPWIND:
            # First order upwind (use upstream value)
            # Determine flow direction using face normal and velocity
            owner_cell = self.mesh.cells[face.owner]
            u_face = self.interpolate_to_face(face, self.field.u, InterpolationScheme.CENTRAL)
            v_face = self.interpolate_to_face(face, self.field.v, InterpolationScheme.CENTRAL)
            velocity_face = np.array([u_face, v_face])
            
            # Flux through face
            flux = np.dot(velocity_face, face.normal)
            
            if flux > 0:
                return owner_value
            else:
                return neighbor_value
                
        else:  # QUICK or other schemes
            # For now, fall back to central
            return (owner_value + neighbor_value) / 2
    
    def compute_gradient(self,
                        phi: np.ndarray,
                        scheme: GradientScheme = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cell-centered gradient of scalar field.
        
        Parameters
        ----------
        phi : array
            Cell-centered scalar field
        scheme : GradientScheme, optional
            Gradient calculation method
            
        Returns
        -------
        grad_x, grad_y : arrays
            Gradient components
        """
        if scheme is None:
            scheme = self.settings.gradient_scheme
            
        n_cells = self.mesh.n_cells
        grad_x = np.zeros(n_cells)
        grad_y = np.zeros(n_cells)
        
        if scheme == GradientScheme.GAUSS:
            # Gauss theorem: ∇φ = (1/V) Σ φ_f S_f
            for cell in self.mesh.cells:
                grad_sum = np.array([0.0, 0.0])
                
                for face_id in cell.faces:
                    face = self.mesh.faces[face_id]
                    phi_face = self.interpolate_to_face(face, phi)
                    
                    # Face area vector (outward normal)
                    S_f = face.normal * face.area
                    grad_sum += phi_face * S_f
                
                grad_x[cell.id] = grad_sum[0] / cell.area
                grad_y[cell.id] = grad_sum[1] / cell.area
                
        elif scheme == GradientScheme.LEAST_SQUARES:
            # Least squares gradient (more accurate for irregular meshes)
            for cell in self.mesh.cells:
                if len(cell.neighbors) < 2:
                    continue
                    
                # Build least squares system
                A = []
                b_x = []
                
                for neighbor_id in cell.neighbors:
                    neighbor = self.mesh.cells[neighbor_id]
                    dx = neighbor.center - cell.center
                    dphi = phi[neighbor_id] - phi[cell.id]
                    
                    A.append(dx)
                    b_x.append(dphi)
                
                A = np.array(A)
                b_x = np.array(b_x)
                
                # Solve least squares
                if len(A) >= 2:
                    grad, _, _, _ = np.linalg.lstsq(A, b_x, rcond=None)
                    grad_x[cell.id] = grad[0]
                    grad_y[cell.id] = grad[1]
        
        return grad_x, grad_y
    
    def assemble_momentum_equation(self, 
                                   direction: str) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Assemble momentum equation matrix.
        
        ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u
        
        Parameters
        ----------
        direction : str
            'u' or 'v'
            
        Returns
        -------
        A : sparse matrix
            Coefficient matrix
        b : array
            Right-hand side
        """
        n = self.mesh.n_cells
        A = sparse.lil_matrix((n, n))
        b = np.zeros(n)
        
        # Select velocity component
        vel = self.field.u if direction == 'u' else self.field.v
        
        for cell in self.mesh.cells:
            i = cell.id
            
            # Temporal term (implicit Euler): ρV/Δt
            A[i, i] += self.field.rho[i] * cell.area / self.settings.time_step
            b[i] += self.field.rho[i] * cell.area * vel[i] / self.settings.time_step
            
            # Convection and diffusion through faces
            for face_id in cell.faces:
                face = self.mesh.faces[face_id]
                
                # Skip boundary faces (handled separately)
                if face.neighbor is None:
                    continue
                
                j = face.neighbor
                
                # Interpolate properties to face
                rho_f = (self.field.rho[i] + self.field.rho[j]) / 2
                mu_f = (self.field.mu[i] + self.field.mu[j]) / 2
                
                # Add turbulent viscosity if turbulence model active
                if self.turbulence is not None:
                    mu_t_f = (self.turbulence.field.mu_t[i] + 
                             self.turbulence.field.mu_t[j]) / 2
                    mu_f += mu_t_f
                
                # Face velocity
                u_f = self.interpolate_to_face(face, self.field.u)
                v_f = self.interpolate_to_face(face, self.field.v)
                vel_f = np.array([u_f, v_f])
                
                # Mass flux through face
                mdot = rho_f * np.dot(vel_f, face.normal) * face.area
                
                # Distance between cell centers
                d = np.linalg.norm(self.mesh.cells[j].center - cell.center)
                
                # Diffusion coefficient
                gamma = mu_f * face.area / d
                
                # Convection-diffusion coefficients
                if face.owner == i:
                    # Outward face from cell i
                    A[i, i] += mdot + gamma
                    A[i, j] -= gamma
                    if mdot > 0:  # Upwind
                        A[i, i] += 0
                    else:
                        A[i, j] += mdot
                else:
                    # Inward face to cell i
                    A[i, i] += -mdot + gamma
                    A[i, j] -= gamma
                    if mdot < 0:  # Upwind
                        A[i, i] += 0
                    else:
                        A[i, j] -= mdot
            
            # Pressure gradient term
            grad_p_x, grad_p_y = self.compute_gradient(self.field.p)
            if direction == 'u':
                b[i] -= grad_p_x[i] * cell.area
            else:
                b[i] -= grad_p_y[i] * cell.area
        
        return A.tocsr(), b
    
    def solve_momentum_step(self):
        """Solve momentum equations (u and v)"""
        # Solve u-momentum
        A_u, b_u = self.assemble_momentum_equation('u')
        A_u_csr = A_u.tocsr()
        
        # Use iterative solver with maximum iterations to prevent hanging
        try:
            u_new, info = sparse.linalg.cg(A_u_csr, b_u, x0=self.field.u, maxiter=50, atol=1e-5)
            if info != 0:
                # Fallback to direct solver if iterative fails
                u_new = sparse.linalg.spsolve(A_u_csr, b_u)
        except:
            # If all else fails, keep current solution
            u_new = self.field.u.copy()
        
        # Under-relaxation
        alpha_u = self.settings.under_relaxation_u
        u_relaxed = alpha_u * u_new + (1 - alpha_u) * self.field.u
        
        # Solve v-momentum
        A_v, b_v = self.assemble_momentum_equation('v')
        A_v_csr = A_v.tocsr()
        
        try:
            v_new, info = sparse.linalg.cg(A_v_csr, b_v, x0=self.field.v, maxiter=50, atol=1e-5)
            if info != 0:
                v_new = sparse.linalg.spsolve(A_v_csr, b_v)
        except:
            v_new = self.field.v.copy()
        
        # Under-relaxation
        alpha_v = self.settings.under_relaxation_v
        v_relaxed = alpha_v * v_new + (1 - alpha_v) * self.field.v
        
        # Calculate residuals
        res_u = np.linalg.norm(u_relaxed - self.field.u) / (np.linalg.norm(u_relaxed) + 1e-10)
        res_v = np.linalg.norm(v_relaxed - self.field.v) / (np.linalg.norm(v_relaxed) + 1e-10)
        
        # Update fields
        self.field.u = u_relaxed
        self.field.v = v_relaxed
        
        return res_u, res_v
    
    def solve_pressure_correction(self):
        """Solve pressure correction equation (continuity)"""
        # This is a simplified version
        # Full SIMPLE algorithm requires more sophisticated implementation
        
        n = self.mesh.n_cells
        A = sparse.lil_matrix((n, n))
        b = np.zeros(n)
        
        for cell in self.mesh.cells:
            i = cell.id
            
            # Mass imbalance (continuity residual)
            mass_imbalance = 0.0
            
            for face_id in cell.faces:
                face = self.mesh.faces[face_id]
                
                if face.neighbor is None:
                    continue
                    
                j = face.neighbor
                
                # Face velocity
                u_f = self.interpolate_to_face(face, self.field.u)
                v_f = self.interpolate_to_face(face, self.field.v)
                rho_f = (self.field.rho[i] + self.field.rho[j]) / 2
                
                # Mass flux
                velocity_f = np.array([u_f, v_f])
                mdot = rho_f * np.dot(velocity_f, face.normal) * face.area
                
                if face.owner == i:
                    mass_imbalance += mdot
                else:
                    mass_imbalance -= mdot
                
                # Pressure correction coefficients
                d = np.linalg.norm(self.mesh.cells[j].center - cell.center)
                coeff = rho_f * face.area / d
                
                A[i, i] += coeff
                A[i, j] -= coeff
            
            b[i] = -mass_imbalance
        
        # Solve pressure correction
        p_prime = sparse.linalg.spsolve(A.tocsr(), b)
        
        # Update pressure with under-relaxation
        alpha_p = self.settings.under_relaxation_p
        self.field.p += alpha_p * p_prime
        
        # Calculate continuity residual
        res_continuity = np.linalg.norm(b) / (self.mesh.n_cells + 1e-10)
        
        return res_continuity
    
    def solve_energy_equation(self):
        """Solve energy equation for temperature field"""
        n = self.mesh.n_cells
        A = sparse.lil_matrix((n, n))
        b = np.zeros(n)
        
        for cell in self.mesh.cells:
            i = cell.id
            
            # Temporal term
            rho_cp = self.field.rho[i] * 4180  # Assume water cp for now
            A[i, i] += rho_cp * cell.area / self.settings.time_step
            b[i] += rho_cp * cell.area * self.field.T[i] / self.settings.time_step
            
            # Convection and diffusion
            for face_id in cell.faces:
                face = self.mesh.faces[face_id]
                
                if face.neighbor is None:
                    continue
                    
                j = face.neighbor
                
                # Face properties
                rho_f = (self.field.rho[i] + self.field.rho[j]) / 2
                k_f = (self.field.k[i] + self.field.k[j]) / 2
                
                u_f = self.interpolate_to_face(face, self.field.u)
                v_f = self.interpolate_to_face(face, self.field.v)
                velocity_f = np.array([u_f, v_f])
                
                # Convection term
                mdot = rho_f * np.dot(velocity_f, face.normal) * face.area
                mdot_cp = mdot * 4180
                
                # Diffusion term
                d = np.linalg.norm(self.mesh.cells[j].center - cell.center)
                gamma = k_f * face.area / d
                
                # Assembly
                A[i, i] += gamma
                A[i, j] -= gamma
                
                if face.owner == i:
                    if mdot > 0:
                        A[i, i] += mdot_cp
                    else:
                        A[i, j] += mdot_cp
                else:
                    if mdot < 0:
                        A[i, i] -= mdot_cp
                    else:
                        A[i, j] -= mdot_cp
        
        # Solve temperature
        T_new = sparse.linalg.spsolve(A.tocsr(), b)
        
        # Under-relaxation
        alpha_T = self.settings.under_relaxation_T
        self.field.T = alpha_T * T_new + (1 - alpha_T) * self.field.T
        
        res_T = np.linalg.norm(T_new - self.field.T) / (np.linalg.norm(T_new) + 1e-10)
        
        return res_T
    
    def solve(self, max_iterations: int = None, verbose: bool = True, progress_callback=None) -> bool:
        """
        Run SIMPLE algorithm to solve coupled Navier-Stokes equations.
        
        Parameters
        ----------
        max_iterations : int, optional
            Maximum iterations (overrides settings)
        verbose : bool, optional
            Print convergence information (default: True)
        progress_callback : callable, optional
            Callback function(iteration, max_iter, residual) for progress updates
            
        Returns
        -------
        bool
            True if converged, False otherwise
        """
        if max_iterations is None:
            max_iterations = self.settings.max_iterations
        
        if verbose:
            print("\n" + "="*60)
            print("CFD SOLVER - SIMPLE Algorithm")
            print("="*60)
            print(f"Mesh: {self.mesh.n_cells} cells")
            print(f"Max iterations: {max_iterations}")
            print(f"Convergence tolerance: {self.settings.convergence_tol:.1e}")
            print()
        
        for iteration in range(max_iterations):
            # Debug: print iteration start
            if verbose and iteration == 0:
                print("Starting iteration 0...")
            
            # Update progress every 5 iterations
            if progress_callback and iteration % 5 == 0:
                progress = 15 + int(70 * iteration / max_iterations)  # 15% to 85%
                progress_callback(progress)
            
            # SIMPLE algorithm steps
            
            # 1. Solve momentum equations
            try:
                res_u, res_v = self.solve_momentum_step()
            except Exception as e:
                if verbose:
                    print(f"ERROR in momentum step: {e}")
                return False
            
            # 2. Solve pressure correction (continuity)
            res_continuity = self.solve_pressure_correction()
            
            # 3. Solve energy equation
            res_T = self.solve_energy_equation()
            
            # 4. Solve turbulence equations if model active
            if self.turbulence is not None:
                self.turbulence.solve(
                    self.field.u,
                    self.field.v,
                    self.field.rho,
                    self.field.mu,
                    self.settings.time_step,
                    n_iterations=3
                )
            
            # Store residuals
            self.residuals['u'].append(res_u)
            self.residuals['v'].append(res_v)
            self.residuals['continuity'].append(res_continuity)
            self.residuals['T'].append(res_T)
            
            # AI convergence monitoring
            if self.ai_enabled and self.ai_convergence_monitor is not None:
                max_residual = max(res_u, res_v, res_continuity, res_T)
                ai_status = self.ai_convergence_monitor.update(max_residual)
                
                # Show AI warnings/recommendations
                if ai_status['status'] in ['diverging', 'oscillating', 'stalled']:
                    if verbose and iteration > 20:
                        print(f"\n🤖 AI Monitor: {ai_status['status'].upper()}")
                        for rec in ai_status['recommendations'][:3]:
                            print(f"   {rec}")
                        print()
            
            # Check convergence
            max_residual = max(res_u, res_v, res_continuity, res_T)
            
            if verbose and ((iteration + 1) % 10 == 0 or max_residual < self.settings.convergence_tol):
                print(f"Iteration {iteration+1:4d}: "
                      f"u={res_u:.2e}, v={res_v:.2e}, "
                      f"p={res_continuity:.2e}, T={res_T:.2e}")
            
            if max_residual < self.settings.convergence_tol:
                if verbose:
                    print(f"\n✅ Converged in {iteration+1} iterations!")
                return True
        
        if verbose:
            print(f"\n⚠️  Did not converge in {max_iterations} iterations")
            print(f"   Final residual: {max_residual:.2e}")
        return False
    
    def get_results(self) -> FlowField:
        """Get converged flow field"""
        return self.field
    
    def enable_ai_assistance(self, enable: bool = True):
        """
        Enable AI-powered assistance for CFD solving.
        
        Features:
        - Flow regime classification and turbulence model recommendation
        - Convergence monitoring and divergence prediction
        - Solver parameter optimization
        
        Parameters
        ----------
        enable : bool
            Enable (True) or disable (False) AI assistance
        """
        if not AI_AVAILABLE:
            print("⚠️  AI integration not available (scikit-learn not installed)")
            print("   Install with: pip install scikit-learn")
            return
        
        self.ai_enabled = enable
        
        if enable:
            self.ai_convergence_monitor = AIConvergenceMonitor()
            self.ai_regime_classifier = AIFlowRegimeClassifier()
            self.ai_recommender = AISolverParameterRecommender()
            print("✅ AI assistance enabled")
        else:
            self.ai_convergence_monitor = None
            self.ai_regime_classifier = None
            self.ai_recommender = None
            print("AI assistance disabled")
    
    def ai_classify_flow(self, velocity: float, length_scale: float) -> Dict:
        """
        Use AI to classify flow regime and recommend turbulence model.
        
        Parameters
        ----------
        velocity : float
            Characteristic velocity (m/s)
        length_scale : float
            Characteristic length (m), e.g. channel height
        
        Returns
        -------
        dict
            Classification results with recommendations
        """
        if not self.ai_enabled or self.ai_regime_classifier is None:
            print("⚠️  AI not enabled. Call enable_ai_assistance() first.")
            return {}
        
        # Calculate dimensionless numbers
        rho = np.mean(self.field.rho)
        mu = np.mean(self.field.mu)
        k = np.mean(self.field.k)
        cp = 4186.0  # Approximate for water/nanofluid
        
        Re = rho * velocity * length_scale / mu
        Pr = mu * cp / k
        
        # Domain aspect ratio
        AR = (self.mesh.x_max - self.mesh.x_min) / (self.mesh.y_max - self.mesh.y_min)
        
        # Classify
        result = self.ai_regime_classifier.predict_regime(Re, Pr, AR, length_scale)
        
        print("\n" + "="*60)
        print("AI FLOW REGIME CLASSIFICATION")
        print("="*60)
        print(f"Reynolds number: {Re:.0f}")
        print(f"Prandtl number: {Pr:.2f}")
        print(f"Aspect ratio: {AR:.1f}")
        print(f"\nPredicted regime: {result['regime'].upper()}")
        print(f"Recommended model: {result['turbulence_model']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print("\nRecommendations:")
        for rec in result['recommendations']:
            print(f"  {rec}")
        print("="*60 + "\n")
        
        return result
    
    def ai_recommend_parameters(self, velocity: float, length_scale: float) -> Dict:
        """
        Use AI to recommend optimal solver parameters.
        
        Parameters
        ----------
        velocity : float
            Characteristic velocity (m/s)
        length_scale : float
            Characteristic length (m)
        
        Returns
        -------
        dict
            Parameter recommendations
        """
        if not self.ai_enabled or self.ai_recommender is None:
            print("⚠️  AI not enabled. Call enable_ai_assistance() first.")
            return {}
        
        # Calculate dimensionless numbers
        rho = np.mean(self.field.rho)
        mu = np.mean(self.field.mu)
        k = np.mean(self.field.k)
        cp = 4186.0
        
        Re = rho * velocity * length_scale / mu
        Pr = mu * cp / k
        
        L = self.mesh.x_max - self.mesh.x_min
        H = self.mesh.y_max - self.mesh.y_min
        
        # Get recommendations
        params = self.ai_recommender.recommend_parameters(
            Re, Pr, L, H, self.settings.turbulence_model.value
        )
        
        print("\n" + "="*60)
        print("AI SOLVER PARAMETER RECOMMENDATIONS")
        print("="*60)
        print(f"Reynolds number: {Re:.0f}")
        print(f"Prandtl number: {Pr:.2f}")
        print(f"\n📐 MESH RECOMMENDATIONS:")
        print(f"  Suggested: {params['mesh']['nx']} × {params['mesh']['ny']}")
        print(f"  Current:   {self.mesh.nx} × {params['mesh']['ny']}")
        for just in params['mesh']['justification']:
            print(f"    {just}")
        
        print(f"\n⚙️  RELAXATION FACTORS:")
        print(f"  Velocity: {params['relaxation']['alpha_u']:.2f}")
        print(f"  Pressure: {params['relaxation']['alpha_p']:.2f}")
        for just in params['relaxation']['justification']:
            print(f"    {just}")
        
        print(f"\n🔄 SOLVER SETTINGS:")
        print(f"  Max iterations: {params['solver']['max_iterations']}")
        print(f"  Tolerance: {params['solver']['tolerance']:.0e}")
        for just in params['solver']['justification']:
            print(f"    {just}")
        
        if params['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warn in params['warnings']:
                print(f"  {warn}")
        
        print("="*60 + "\n")
        
        return params
    
    def ai_apply_recommendations(self, params: Dict):
        """
        Apply AI-recommended parameters to solver settings.
        
        Parameters
        ----------
        params : dict
            Parameters from ai_recommend_parameters()
        """
        if 'relaxation' in params:
            self.settings.under_relaxation_u = params['relaxation']['alpha_u']
            self.settings.under_relaxation_v = params['relaxation']['alpha_u']
            self.settings.under_relaxation_p = params['relaxation']['alpha_p']
            print(f"✅ Applied relaxation factors")
        
        if 'solver' in params:
            self.settings.max_iterations = params['solver']['max_iterations']
            self.settings.convergence_tol = params['solver']['tolerance']
            print(f"✅ Applied solver settings")


# Example usage
if __name__ == "__main__":
    print("CFD SOLVER MODULE - Demo")
    print("=" * 60)
    
    # Create simple mesh
    mesh = StructuredMesh2D(
        x_range=(0.0, 0.1),
        y_range=(0.0, 0.05),
        nx=20,
        ny=10
    )
    
    # Create solver
    solver = NavierStokesSolver(mesh)
    
    # Set boundary conditions (simplified for demo)
    bc_inlet = BoundaryCondition(
        bc_type=BoundaryType.INLET,
        velocity=(0.1, 0.0),
        temperature=300.0
    )
    solver.set_boundary_condition(BoundaryType.INLET, bc_inlet)
    
    # Initialize
    solver.initialize_field(u0=0.1, T0=300.0)
    
    print("\n✅ CFD solver module created successfully!")
    print("\n📝 Next steps:")
    print("   1. Implement boundary condition application")
    print("   2. Add turbulence models")
    print("   3. Create post-processing tools")
    print("   4. Build GUI interface")
