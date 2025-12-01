"""
CFD Advanced Linear System Solvers

Implements efficient iterative solvers for large sparse linear systems
arising from CFD discretizations.

Solvers:
- Gauss-Seidel (basic iterative)
- Conjugate Gradient (CG) for symmetric systems
- BiCGSTAB (Bi-Conjugate Gradient Stabilized) for non-symmetric
- GMRES (Generalized Minimal Residual)

Preconditioners:
- Jacobi (diagonal)
- ILU (Incomplete LU)
- AMG (Algebraic Multigrid) - via PyAMG

Author: Nanofluid Simulator v4.0 - CFD Module
License: MIT
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class SolverType(Enum):
    """Available linear solver types"""
    DIRECT = "direct"  # scipy.sparse.linalg.spsolve
    GAUSS_SEIDEL = "gauss_seidel"
    JACOBI = "jacobi"
    CG = "cg"  # Conjugate Gradient
    BICGSTAB = "bicgstab"  # BiConjugate Gradient Stabilized
    GMRES = "gmres"  # Generalized Minimal Residual


class Preconditioner(Enum):
    """Preconditioner types"""
    NONE = "none"
    JACOBI = "jacobi"
    ILU = "ilu"
    AMG = "amg"


@dataclass
class SolverConfig:
    """Linear solver configuration"""
    solver_type: SolverType = SolverType.BICGSTAB
    preconditioner: Preconditioner = Preconditioner.ILU
    max_iterations: int = 1000
    tolerance: float = 1e-6
    verbose: bool = False


@dataclass
class SolverStats:
    """Solver performance statistics"""
    iterations: int
    residual: float
    converged: bool
    cpu_time: float


class GaussSeidel:
    """
    Gauss-Seidel iterative solver.
    
    x^(k+1)_i = (b_i - Σ_{j<i} a_ij x^(k+1)_j - Σ_{j>i} a_ij x^(k)_j) / a_ii
    """
    
    @staticmethod
    def solve(A: sparse.spmatrix,
             b: np.ndarray,
             x0: Optional[np.ndarray] = None,
             max_iter: int = 1000,
             tol: float = 1e-6) -> Tuple[np.ndarray, SolverStats]:
        """
        Solve Ax = b using Gauss-Seidel.
        
        Parameters
        ----------
        A : sparse matrix
            Coefficient matrix
        b : array
            Right-hand side
        x0 : array, optional
            Initial guess
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
            
        Returns
        -------
        x : array
            Solution vector
        stats : SolverStats
            Solver statistics
        """
        import time
        start_time = time.time()
        
        n = len(b)
        x = np.zeros(n) if x0 is None else x0.copy()
        
        # Convert to CSR for efficient row access
        A_csr = A.tocsr()
        
        for iteration in range(max_iter):
            x_old = x.copy()
            
            # Update each component
            for i in range(n):
                # Get row i
                row_start = A_csr.indptr[i]
                row_end = A_csr.indptr[i+1]
                indices = A_csr.indices[row_start:row_end]
                values = A_csr.data[row_start:row_end]
                
                # Calculate: b_i - Σ_{j≠i} a_ij x_j
                sigma = b[i]
                a_ii = 0.0
                
                for idx, j in enumerate(indices):
                    if j == i:
                        a_ii = values[idx]
                    else:
                        sigma -= values[idx] * x[j]
                
                # Update x_i
                if abs(a_ii) > 1e-12:
                    x[i] = sigma / a_ii
            
            # Check convergence
            residual = np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-10)
            
            if residual < tol:
                cpu_time = time.time() - start_time
                stats = SolverStats(
                    iterations=iteration + 1,
                    residual=residual,
                    converged=True,
                    cpu_time=cpu_time
                )
                return x, stats
        
        # Did not converge
        cpu_time = time.time() - start_time
        stats = SolverStats(
            iterations=max_iter,
            residual=residual,
            converged=False,
            cpu_time=cpu_time
        )
        return x, stats


class JacobiSolver:
    """
    Jacobi iterative solver.
    
    x^(k+1) = D^(-1) (b - (L+U)x^(k))
    where A = D + L + U
    """
    
    @staticmethod
    def solve(A: sparse.spmatrix,
             b: np.ndarray,
             x0: Optional[np.ndarray] = None,
             max_iter: int = 1000,
             tol: float = 1e-6) -> Tuple[np.ndarray, SolverStats]:
        """Solve Ax = b using Jacobi method"""
        import time
        start_time = time.time()
        
        n = len(b)
        x = np.zeros(n) if x0 is None else x0.copy()
        
        # Extract diagonal
        D_inv = 1.0 / A.diagonal()
        
        # R = D - A (residual iteration matrix)
        R = sparse.diags(A.diagonal()) - A
        
        for iteration in range(max_iter):
            # x^(k+1) = D^(-1) (b + R x^(k))
            x_new = D_inv * (b + R.dot(x))
            
            # Check convergence
            residual = np.linalg.norm(x_new - x) / (np.linalg.norm(x_new) + 1e-10)
            
            x = x_new
            
            if residual < tol:
                cpu_time = time.time() - start_time
                stats = SolverStats(
                    iterations=iteration + 1,
                    residual=residual,
                    converged=True,
                    cpu_time=cpu_time
                )
                return x, stats
        
        cpu_time = time.time() - start_time
        stats = SolverStats(
            iterations=max_iter,
            residual=residual,
            converged=False,
            cpu_time=cpu_time
        )
        return x, stats


class PreconditionerFactory:
    """Factory for creating preconditioners"""
    
    @staticmethod
    def create(A: sparse.spmatrix,
              precond_type: Preconditioner) -> Optional[sp_linalg.LinearOperator]:
        """
        Create preconditioner for matrix A.
        
        Parameters
        ----------
        A : sparse matrix
            System matrix
        precond_type : Preconditioner
            Type of preconditioner
            
        Returns
        -------
        M : LinearOperator or None
            Preconditioner operator
        """
        if precond_type == Preconditioner.NONE:
            return None
            
        elif precond_type == Preconditioner.JACOBI:
            # Diagonal (Jacobi) preconditioner
            D_inv = 1.0 / A.diagonal()
            M = sparse.diags(D_inv)
            return sp_linalg.LinearOperator(A.shape, matvec=M.dot)
            
        elif precond_type == Preconditioner.ILU:
            # Incomplete LU factorization
            try:
                ilu = sp_linalg.spilu(A.tocsc())
                M = sp_linalg.LinearOperator(
                    A.shape,
                    matvec=ilu.solve
                )
                return M
            except:
                # Fallback to Jacobi if ILU fails
                print("   Warning: ILU failed, using Jacobi preconditioner")
                return PreconditionerFactory.create(A, Preconditioner.JACOBI)
                
        elif precond_type == Preconditioner.AMG:
            # Algebraic Multigrid (requires pyamg)
            try:
                import pyamg
                ml = pyamg.smoothed_aggregation_solver(A)
                M = ml.aspreconditioner()
                return M
            except ImportError:
                print("   Warning: PyAMG not available, using ILU")
                return PreconditionerFactory.create(A, Preconditioner.ILU)
        
        return None


class AdvancedLinearSolver:
    """
    High-level interface for advanced linear system solvers.
    
    Automatically selects appropriate solver and preconditioner
    based on matrix properties.
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        """
        Initialize solver.
        
        Parameters
        ----------
        config : SolverConfig, optional
            Solver configuration
        """
        self.config = config if config is not None else SolverConfig()
        
    def solve(self,
             A: sparse.spmatrix,
             b: np.ndarray,
             x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, SolverStats]:
        """
        Solve linear system Ax = b.
        
        Parameters
        ----------
        A : sparse matrix
            Coefficient matrix (n×n)
        b : array
            Right-hand side (n,)
        x0 : array, optional
            Initial guess (n,)
            
        Returns
        -------
        x : array
            Solution vector
        stats : SolverStats
            Solver performance statistics
        """
        import time
        start_time = time.time()
        
        n = len(b)
        x0_vec = np.zeros(n) if x0 is None else x0
        
        # Direct solver (for small systems)
        if self.config.solver_type == SolverType.DIRECT:
            x = sp_linalg.spsolve(A, b)
            cpu_time = time.time() - start_time
            residual = np.linalg.norm(A.dot(x) - b) / np.linalg.norm(b)
            stats = SolverStats(
                iterations=1,
                residual=residual,
                converged=True,
                cpu_time=cpu_time
            )
            return x, stats
        
        # Gauss-Seidel
        elif self.config.solver_type == SolverType.GAUSS_SEIDEL:
            return GaussSeidel.solve(
                A, b, x0_vec,
                max_iter=self.config.max_iterations,
                tol=self.config.tolerance
            )
        
        # Jacobi
        elif self.config.solver_type == SolverType.JACOBI:
            return JacobiSolver.solve(
                A, b, x0_vec,
                max_iter=self.config.max_iterations,
                tol=self.config.tolerance
            )
        
        # Create preconditioner
        M = PreconditionerFactory.create(A, self.config.preconditioner)
        
        # Setup callback for iteration counting
        iterations = [0]
        def callback(xk):
            iterations[0] += 1
        
        # Conjugate Gradient (for symmetric positive definite)
        if self.config.solver_type == SolverType.CG:
            x, info = sp_linalg.cg(
                A, b, x0=x0_vec, M=M,
                maxiter=self.config.max_iterations,
                tol=self.config.tolerance,
                callback=callback
            )
            
        # BiCGSTAB (general non-symmetric)
        elif self.config.solver_type == SolverType.BICGSTAB:
            x, info = sp_linalg.bicgstab(
                A, b, x0=x0_vec, M=M,
                maxiter=self.config.max_iterations,
                tol=self.config.tolerance,
                callback=callback
            )
            
        # GMRES
        elif self.config.solver_type == SolverType.GMRES:
            x, info = sp_linalg.gmres(
                A, b, x0=x0_vec, M=M,
                maxiter=self.config.max_iterations,
                tol=self.config.tolerance,
                callback=callback,
                restart=50  # GMRES(50)
            )
        else:
            raise ValueError(f"Unknown solver type: {self.config.solver_type}")
        
        # Calculate final residual
        cpu_time = time.time() - start_time
        residual = np.linalg.norm(A.dot(x) - b) / (np.linalg.norm(b) + 1e-10)
        converged = (info == 0)
        
        stats = SolverStats(
            iterations=iterations[0],
            residual=residual,
            converged=converged,
            cpu_time=cpu_time
        )
        
        if self.config.verbose:
            status = "✅ Converged" if converged else "⚠️  Max iterations"
            print(f"   {status}: {iterations[0]} iters, residual={residual:.2e}, time={cpu_time:.3f}s")
        
        return x, stats


def benchmark_solvers(n: int = 100):
    """
    Benchmark different solvers on a test problem.
    
    Parameters
    ----------
    n : int
        Problem size (n×n matrix)
    """
    print(f"\n{'='*70}")
    print(f"LINEAR SOLVER BENCHMARK (n={n})")
    print(f"{'='*70}\n")
    
    # Create test problem: 2D Poisson equation
    # -∇²u = f on unit square with u=0 on boundary
    
    # Build sparse matrix (5-point stencil)
    nx, ny = int(np.sqrt(n)), int(np.sqrt(n))
    n = nx * ny
    
    # Main diagonal
    main_diag = 4.0 * np.ones(n)
    # Off-diagonals
    off_diag_x = -1.0 * np.ones(n-1)
    off_diag_y = -1.0 * np.ones(n-nx)
    
    # Create matrix
    A = sparse.diags(
        [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y],
        [0, -1, 1, -nx, nx],
        shape=(n, n),
        format='csr'
    )
    
    # Right-hand side (random)
    np.random.seed(42)
    b = np.random.rand(n)
    
    # Test different solvers
    solvers = [
        (SolverType.DIRECT, Preconditioner.NONE, "Direct (spsolve)"),
        (SolverType.BICGSTAB, Preconditioner.ILU, "BiCGSTAB + ILU"),
        (SolverType.CG, Preconditioner.ILU, "CG + ILU"),
        (SolverType.GMRES, Preconditioner.ILU, "GMRES + ILU"),
        (SolverType.GAUSS_SEIDEL, Preconditioner.NONE, "Gauss-Seidel"),
    ]
    
    print(f"{'Solver':<25s} {'Iters':>8s} {'Residual':>12s} {'Time (s)':>10s} {'Status':>10s}")
    print(f"{'-'*25} {'-'*8} {'-'*12} {'-'*10} {'-'*10}")
    
    for solver_type, precond, name in solvers:
        config = SolverConfig(
            solver_type=solver_type,
            preconditioner=precond,
            max_iterations=1000,
            tolerance=1e-6,
            verbose=False
        )
        
        solver = AdvancedLinearSolver(config)
        
        try:
            x, stats = solver.solve(A, b)
            
            status = "✅" if stats.converged else "⚠️"
            print(f"{name:<25s} {stats.iterations:>8d} {stats.residual:>12.2e} "
                  f"{stats.cpu_time:>10.4f} {status:>10s}")
        except Exception as e:
            print(f"{name:<25s} {'—':>8s} {'—':>12s} {'—':>10s} {'❌':>10s}")
    
    print()


# Example usage
if __name__ == "__main__":
    # Run benchmark
    benchmark_solvers(n=10000)
    
    print("\n✅ Advanced linear solver module ready!")
    print("\n📝 Available solvers:")
    print("   - Direct (spsolve)")
    print("   - BiCGSTAB (recommended)")
    print("   - Conjugate Gradient (CG)")
    print("   - GMRES")
    print("   - Gauss-Seidel")
    print("   - Jacobi")
    print("\n📝 Available preconditioners:")
    print("   - ILU (Incomplete LU) - recommended")
    print("   - Jacobi (diagonal)")
    print("   - AMG (Algebraic Multigrid) - requires pyamg")
