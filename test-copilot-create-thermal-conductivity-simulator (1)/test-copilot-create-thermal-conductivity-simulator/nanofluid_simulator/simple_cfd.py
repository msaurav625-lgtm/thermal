"""
Simple Finite Difference CFD Solver for Nanofluid Flow
Uses projection method (Chorin's algorithm) for incompressible Navier-Stokes

Much simpler than finite volume SIMPLE algorithm - actually works!
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class SimpleCFDConfig:
    """Configuration for simple CFD solver"""
    # Domain
    length: float = 0.1  # m
    height: float = 0.01  # m
    nx: int = 50
    ny: int = 50
    
    # Fluid properties
    rho: float = 998.0  # kg/m³
    mu: float = 0.001  # Pa·s
    k: float = 0.6  # W/m·K
    cp: float = 4182.0  # J/kg·K
    
    # Flow conditions
    inlet_velocity: float = 1.0  # m/s
    inlet_temperature: float = 300.0  # K
    
    # Solver settings
    max_iterations: int = 200
    dt: float = 0.0001  # Time step for pseudo-transient
    tolerance: float = 1e-3  # Relaxed tolerance
    alpha: float = 0.5  # Under-relaxation factor


class SimpleCFDSolver:
    """
    Finite difference solver for 2D incompressible Navier-Stokes
    
    Uses projection method (fractional step):
    1. Momentum prediction (explicit)
    2. Pressure Poisson equation (implicit)
    3. Velocity correction (explicit)
    4. Energy equation (explicit)
    """
    
    def __init__(self, config: SimpleCFDConfig):
        self.config = config
        
        # Grid spacing
        self.dx = config.length / (config.nx - 1)
        self.dy = config.height / (config.ny - 1)
        
        # Create mesh
        self.x = np.linspace(0, config.length, config.nx)
        self.y = np.linspace(0, config.height, config.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Initialize fields
        self.u = np.zeros((config.nx, config.ny))  # x-velocity
        self.v = np.zeros((config.nx, config.ny))  # y-velocity
        self.p = np.zeros((config.nx, config.ny))  # pressure
        self.T = np.ones((config.nx, config.ny)) * config.inlet_temperature
        
        # Temporary arrays
        self.u_star = np.zeros_like(self.u)
        self.v_star = np.zeros_like(self.v)
        
        # Residual history
        self.residuals = {'u': [], 'v': [], 'p': [], 'T': []}
    
    def apply_boundary_conditions(self):
        """Apply boundary conditions on all fields"""
        # Inlet (left): fixed velocity
        self.u[0, :] = self.config.inlet_velocity
        self.v[0, :] = 0.0
        self.T[0, :] = self.config.inlet_temperature
        
        # Outlet (right): zero gradient
        self.u[-1, :] = self.u[-2, :]
        self.v[-1, :] = self.v[-2, :]
        self.p[-1, :] = 0.0  # Reference pressure
        self.T[-1, :] = self.T[-2, :]
        
        # Walls (top/bottom): no-slip
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0
        # Wall temperature - adiabatic (zero gradient)
        self.T[:, 0] = self.T[:, 1]
        self.T[:, -1] = self.T[:, -2]
    
    def momentum_prediction(self):
        """Explicit momentum prediction (without pressure)"""
        nx, ny = self.config.nx, self.config.ny
        dt = self.config.dt
        dx, dy = self.dx, self.dy
        rho = self.config.rho
        mu = self.config.mu
        
        # Interior points only
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # U-momentum with first-order upwind (more stable)
                # Convection
                if self.u[i, j] > 0:
                    dudx = (self.u[i, j] - self.u[i-1, j]) / dx
                else:
                    dudx = (self.u[i+1, j] - self.u[i, j]) / dx
                
                if self.v[i, j] > 0:
                    dudy = (self.u[i, j] - self.u[i, j-1]) / dy
                else:
                    dudy = (self.u[i, j+1] - self.u[i, j]) / dy
                
                conv_u = self.u[i, j] * dudx + self.v[i, j] * dudy
                
                # Diffusion (central - always stable)
                d2udx2 = (self.u[i+1, j] - 2*self.u[i, j] + self.u[i-1, j]) / dx**2
                d2udy2 = (self.u[i, j+1] - 2*self.u[i, j] + self.u[i, j-1]) / dy**2
                diff_u = mu / rho * (d2udx2 + d2udy2)
                
                # Predictor with smaller factor for stability
                self.u_star[i, j] = self.u[i, j] + 0.5 * dt * (-conv_u + diff_u)
                
                # V-momentum
                if self.u[i, j] > 0:
                    dvdx = (self.v[i, j] - self.v[i-1, j]) / dx
                else:
                    dvdx = (self.v[i+1, j] - self.v[i, j]) / dx
                    
                if self.v[i, j] > 0:
                    dvdy = (self.v[i, j] - self.v[i, j-1]) / dy
                else:
                    dvdy = (self.v[i, j+1] - self.v[i, j]) / dy
                
                conv_v = self.u[i, j] * dvdx + self.v[i, j] * dvdy
                
                d2vdx2 = (self.v[i+1, j] - 2*self.v[i, j] + self.v[i-1, j]) / dx**2
                d2vdy2 = (self.v[i, j+1] - 2*self.v[i, j] + self.v[i, j-1]) / dy**2
                diff_v = mu / rho * (d2vdx2 + d2vdy2)
                
                self.v_star[i, j] = self.v[i, j] + 0.5 * dt * (-conv_v + diff_v)
    
    def pressure_poisson(self):
        """Solve pressure Poisson equation using Jacobi iteration"""
        nx, ny = self.config.nx, self.config.ny
        dt = self.config.dt
        dx, dy = self.dx, self.dy
        rho = self.config.rho
        
        p_new = self.p.copy()
        
        # Jacobi iterations (fast for our small mesh)
        for _ in range(50):
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    # RHS: divergence of predicted velocity
                    div_u = (self.u_star[i+1, j] - self.u_star[i-1, j]) / (2*dx) + \
                            (self.v_star[i, j+1] - self.v_star[i, j-1]) / (2*dy)
                    
                    # Poisson equation
                    p_new[i, j] = ((p_new[i+1, j] + p_new[i-1, j]) / dx**2 + \
                                   (p_new[i, j+1] + p_new[i, j-1]) / dy**2 - \
                                   rho / dt * div_u) / (2/dx**2 + 2/dy**2)
            
            # Boundary conditions for pressure
            p_new[0, :] = p_new[1, :]   # Inlet: zero gradient
            p_new[-1, :] = 0.0           # Outlet: reference
            p_new[:, 0] = p_new[:, 1]   # Walls: zero gradient
            p_new[:, -1] = p_new[:, -2]
        
        self.p = p_new
    
    def velocity_correction(self):
        """Correct velocity to satisfy continuity"""
        nx, ny = self.config.nx, self.config.ny
        dt = self.config.dt
        dx, dy = self.dx, self.dy
        rho = self.config.rho
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # Pressure gradient
                dpdx = (self.p[i+1, j] - self.p[i-1, j]) / (2*dx)
                dpdy = (self.p[i, j+1] - self.p[i, j-1]) / (2*dy)
                
                # Velocity correction with under-relaxation
                alpha = self.config.alpha
                u_new = self.u_star[i, j] - dt / rho * dpdx
                v_new = self.v_star[i, j] - dt / rho * dpdy
                self.u[i, j] = alpha * u_new + (1 - alpha) * self.u[i, j]
                self.v[i, j] = alpha * v_new + (1 - alpha) * self.v[i, j]
    
    def energy_equation(self):
        """Solve energy equation explicitly"""
        nx, ny = self.config.nx, self.config.ny
        dt = self.config.dt
        dx, dy = self.dx, self.dy
        rho = self.config.rho
        k = self.config.k
        cp = self.config.cp
        alpha = k / (rho * cp)  # Thermal diffusivity
        
        T_new = self.T.copy()
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # Convection
                dTdx = (self.T[i, j] - self.T[i-1, j]) / dx if self.u[i, j] > 0 else (self.T[i+1, j] - self.T[i, j]) / dx
                dTdy = (self.T[i, j] - self.T[i, j-1]) / dy if self.v[i, j] > 0 else (self.T[i, j+1] - self.T[i, j]) / dy
                conv_T = self.u[i, j] * dTdx + self.v[i, j] * dTdy
                
                # Diffusion
                d2Tdx2 = (self.T[i+1, j] - 2*self.T[i, j] + self.T[i-1, j]) / dx**2
                d2Tdy2 = (self.T[i, j+1] - 2*self.T[i, j] + self.T[i, j-1]) / dy**2
                diff_T = alpha * (d2Tdx2 + d2Tdy2)
                
                T_new[i, j] = self.T[i, j] + dt * (-conv_T + diff_T)
        
        self.T = T_new
    
    def compute_residual(self):
        """Compute residual for convergence check"""
        # Velocity divergence (continuity residual)
        div = np.zeros_like(self.u)
        for i in range(1, self.config.nx-1):
            for j in range(1, self.config.ny-1):
                div[i, j] = (self.u[i+1, j] - self.u[i-1, j]) / (2*self.dx) + \
                           (self.v[i, j+1] - self.v[i, j-1]) / (2*self.dy)
        
        return np.max(np.abs(div))
    
    def solve(self, progress_callback=None):
        """
        Run projection method solver
        
        Returns
        -------
        dict with velocity, pressure, temperature fields and convergence info
        """
        # Initialize with reasonable guess
        self.u[:, :] = self.config.inlet_velocity * 0.5
        self.apply_boundary_conditions()
        
        converged = False
        
        for iteration in range(self.config.max_iterations):
            # Update progress
            if progress_callback and iteration % 5 == 0:
                progress = 15 + int(70 * iteration / self.config.max_iterations)
                progress_callback(progress)
            
            # Projection method steps
            self.momentum_prediction()
            self.pressure_poisson()
            self.velocity_correction()
            self.energy_equation()
            self.apply_boundary_conditions()
            
            # Check convergence
            residual = self.compute_residual()
            self.residuals['u'].append(residual)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: residual = {residual:.3e}")
            
            if residual < self.config.tolerance:
                print(f"✓ Converged in {iteration} iterations!")
                converged = True
                break
        
        return {
            'converged': converged,
            'velocity_u': self.u,
            'velocity_v': self.v,
            'pressure': self.p,
            'temperature': self.T,
            'mesh': {
                'x': self.x,
                'y': self.y,
                'X': self.X,
                'Y': self.Y,
                'nx': self.config.nx,
                'ny': self.config.ny
            },
            'residuals': self.residuals,
            'metrics': self.compute_metrics()
        }
    
    def compute_metrics(self):
        """Compute flow metrics"""
        # Average quantities
        avg_u = np.mean(self.u[1:-1, 1:-1])
        max_u = np.max(self.u)
        pressure_drop = np.mean(self.p[0, :]) - np.mean(self.p[-1, :])
        avg_T = np.mean(self.T[1:-1, 1:-1])
        
        # Reynolds number
        Re = self.config.rho * avg_u * self.config.height / self.config.mu
        
        # Nusselt number (simplified)
        Nu = 4.86  # Constant for fully developed laminar flow
        h = Nu * self.config.k / self.config.height
        
        return {
            'reynolds_number': Re,
            'pressure_drop': pressure_drop,
            'avg_velocity': avg_u,
            'max_velocity': max_u,
            'avg_temperature': avg_T,
            'nusselt_number': Nu,
            'heat_transfer_coefficient': h
        }
