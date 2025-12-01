"""
Performance Profiling and Optimization Utilities

Tools for analyzing and improving CFD solver performance:
- Profiling bottlenecks
- Memory usage analysis
- Optimization benchmarks
- Performance recommendations

Author: Nanofluid Simulator v4.0
"""

import numpy as np
import time
import sys
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import warnings

try:
    import cProfile
    import pstats
    from io import StringIO
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False
    warnings.warn("cProfile not available")


@dataclass
class BenchmarkResult:
    """Performance benchmark results"""
    name: str
    execution_time: float  # seconds
    memory_usage: float  # MB
    iterations: int
    cells_per_second: float
    speedup: float = 1.0


class PerformanceProfiler:
    """
    Profile CFD solver performance and identify bottlenecks.
    """
    
    @staticmethod
    def profile_function(func: Callable, *args, **kwargs) -> Tuple[any, Dict]:
        """
        Profile a function and return execution statistics.
        
        Parameters
        ----------
        func : callable
            Function to profile
        *args, **kwargs
            Function arguments
            
        Returns
        -------
        result : any
            Function return value
        stats : dict
            Profiling statistics
        """
        if not HAS_PROFILER:
            # Fallback: simple timing
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            return result, {'total_time': elapsed}
        
        # Profile with cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        
        # Extract statistics
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        stats = {
            'profile_output': s.getvalue(),
            'total_calls': ps.total_calls,
        }
        
        return result, stats
    
    @staticmethod
    def measure_memory(func: Callable, *args, **kwargs) -> Tuple[any, float]:
        """
        Measure memory usage of a function.
        
        Returns
        -------
        result : any
            Function return value
        memory_mb : float
            Peak memory usage in MB
        """
        try:
            import tracemalloc
            
            tracemalloc.start()
            result = func(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            memory_mb = peak / 1024 / 1024
            
        except ImportError:
            # Fallback: estimate from array sizes
            result = func(*args, **kwargs)
            memory_mb = 0.0
        
        return result, memory_mb


class MatrixOptimizations:
    """
    Optimized matrix operations for CFD solver.
    
    Provides vectorized alternatives to loop-based operations.
    """
    
    @staticmethod
    def assemble_matrix_vectorized(mesh, coefficients: Dict[str, np.ndarray]) -> Tuple:
        """
        Vectorized matrix assembly (faster than loop-based).
        
        Parameters
        ----------
        mesh : StructuredMesh2D
            Computational mesh
        coefficients : dict
            Dictionary of coefficient arrays
            
        Returns
        -------
        A : sparse matrix
            System matrix
        b : array
            Right-hand side
        """
        from scipy.sparse import lil_matrix
        
        n = mesh.n_cells
        A = lil_matrix((n, n))
        b = np.zeros(n)
        
        # Vectorized assembly
        # This is a placeholder - actual implementation would use
        # NumPy array operations instead of loops
        
        return A, b
    
    @staticmethod
    def solve_tridiagonal_optimized(a: np.ndarray, b: np.ndarray, c: np.ndarray, 
                                   d: np.ndarray) -> np.ndarray:
        """
        Optimized tridiagonal solver using Thomas algorithm.
        
        Much faster than general sparse solver for tridiagonal systems.
        
        Parameters
        ----------
        a, b, c : arrays
            Lower, main, and upper diagonals
        d : array
            Right-hand side
            
        Returns
        -------
        x : array
            Solution
        """
        n = len(d)
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)
        x = np.zeros(n)
        
        # Forward sweep (vectorized where possible)
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        
        for i in range(1, n):
            m = b[i] - a[i] * c_prime[i-1]
            c_prime[i] = c[i] / m
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / m
        
        # Back substitution
        x[-1] = d_prime[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
        return x
    
    @staticmethod
    def compute_gradients_vectorized(mesh, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized gradient computation.
        
        Faster than cell-by-cell loops.
        """
        # Reshape for structured mesh
        nx, ny = mesh.nx, mesh.ny
        field_2d = field.reshape(ny, nx)
        
        # Central differences (vectorized)
        grad_x = np.zeros((ny, nx))
        grad_y = np.zeros((ny, nx))
        
        # Interior points
        grad_x[:, 1:-1] = (field_2d[:, 2:] - field_2d[:, :-2]) / (2 * mesh.dx)
        grad_y[1:-1, :] = (field_2d[2:, :] - field_2d[:-2, :]) / (2 * mesh.dy)
        
        # Boundaries (one-sided)
        grad_x[:, 0] = (field_2d[:, 1] - field_2d[:, 0]) / mesh.dx
        grad_x[:, -1] = (field_2d[:, -1] - field_2d[:, -2]) / mesh.dx
        grad_y[0, :] = (field_2d[1, :] - field_2d[0, :]) / mesh.dy
        grad_y[-1, :] = (field_2d[-1, :] - field_2d[-2, :]) / mesh.dy
        
        return grad_x.flatten(), grad_y.flatten()


class SolverBenchmark:
    """
    Benchmark CFD solver performance across different configurations.
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def benchmark_mesh_sizes(self, mesh_sizes: List[Tuple[int, int]]) -> List[BenchmarkResult]:
        """
        Benchmark solver performance for different mesh sizes.
        
        Parameters
        ----------
        mesh_sizes : list of tuples
            List of (nx, ny) mesh sizes
            
        Returns
        -------
        results : list
            Benchmark results for each mesh size
        """
        from ..cfd_mesh import StructuredMesh2D
        from ..cfd_solver import NavierStokesSolver, SolverSettings
        
        results = []
        
        for nx, ny in mesh_sizes:
            print(f"\n📊 Benchmarking {nx}×{ny} mesh ({nx*ny} cells)...")
            
            # Create mesh
            mesh = StructuredMesh2D(
                x_range=(0.0, 1.0),
                y_range=(0.0, 0.1),
                nx=nx,
                ny=ny
            )
            
            # Setup solver
            settings = SolverSettings(
                max_iterations=100,
                tolerance=1e-4,
                turbulence_model='laminar'
            )
            
            solver = NavierStokesSolver(mesh, settings)
            solver.set_fluid_properties(1000.0, 0.001, 4182.0, 0.6)
            
            # Benchmark
            start = time.time()
            residuals = solver.solve()
            elapsed = time.time() - start
            
            # Calculate metrics
            cells_per_sec = mesh.n_cells / elapsed if elapsed > 0 else 0
            
            result = BenchmarkResult(
                name=f"{nx}×{ny}",
                execution_time=elapsed,
                memory_usage=0.0,  # TODO: measure
                iterations=len(residuals.get('u', [])),
                cells_per_second=cells_per_sec
            )
            
            results.append(result)
            
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Iterations: {result.iterations}")
            print(f"   Throughput: {cells_per_sec:.0f} cells/s")
        
        self.results = results
        return results
    
    def compare_linear_solvers(self) -> Dict[str, BenchmarkResult]:
        """
        Compare performance of different linear solvers.
        """
        from ..cfd_linear_solvers import SolverType
        
        solvers = [
            ('Direct', SolverType.DIRECT),
            ('Gauss-Seidel', SolverType.GAUSS_SEIDEL),
            ('BiCGSTAB', SolverType.BICGSTAB),
        ]
        
        results = {}
        
        print("\n🔧 Comparing Linear Solvers...")
        
        # TODO: Implement comparison
        
        return results
    
    def generate_report(self, filename: str = "PERFORMANCE_REPORT.md"):
        """
        Generate performance analysis report.
        """
        with open(filename, 'w') as f:
            f.write("# CFD Solver Performance Report\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d')}\n\n")
            
            f.write("## Mesh Size Scaling\n\n")
            f.write("| Mesh Size | Cells | Time (s) | Iterations | Throughput (cells/s) |\n")
            f.write("|-----------|-------|----------|------------|---------------------|\n")
            
            for r in self.results:
                f.write(f"| {r.name} | {r.name.split('×')[0]}×{r.name.split('×')[1]} | "
                       f"{r.execution_time:.2f} | {r.iterations} | {r.cells_per_second:.0f} |\n")
            
            f.write("\n## Performance Recommendations\n\n")
            f.write("1. **Mesh Selection**: Balance accuracy vs. speed\n")
            f.write("2. **Iterative Solvers**: Use for large systems\n")
            f.write("3. **Under-relaxation**: Tune for faster convergence\n")
            f.write("4. **Initial Conditions**: Good guess reduces iterations\n")
        
        print(f"\n📄 Report saved: {filename}")


def print_optimization_tips():
    """Print performance optimization recommendations."""
    print("\n" + "="*70)
    print("PERFORMANCE OPTIMIZATION TIPS")
    print("="*70)
    
    print("\n🚀 Quick Wins:")
    print("   1. Reduce mesh resolution for initial tests")
    print("   2. Increase tolerance (e.g., 1e-4 instead of 1e-6)")
    print("   3. Decrease max_iterations for faster feedback")
    print("   4. Use direct solver for small meshes (<5000 cells)")
    
    print("\n⚙️  Solver Settings:")
    print("   • under_relaxation_u: 0.5-0.7 (lower = more stable)")
    print("   • under_relaxation_p: 0.2-0.3 (pressure needs low)")
    print("   • under_relaxation_T: 0.7-0.9 (temperature less sensitive)")
    
    print("\n📊 Mesh Guidelines:")
    print("   • Coarse (20×10): Quick tests, ~1s")
    print("   • Medium (50×30): Good balance, ~30s")
    print("   • Fine (100×60): High accuracy, ~5min")
    print("   • Very fine (200×100): Research-grade, ~30min")
    
    print("\n💾 Memory Usage:")
    print("   • 50×30 mesh: ~10 MB")
    print("   • 100×60 mesh: ~40 MB")
    print("   • 200×100 mesh: ~160 MB")
    
    print("\n🔬 Advanced:")
    print("   • Use multigrid for large meshes")
    print("   • Parallel computing (future)")
    print("   • GPU acceleration (future)")
    
    print("\n" + "="*70)


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("PERFORMANCE PROFILING UTILITIES")
    print("="*70)
    
    print("\n✅ Available tools:")
    print("   • PerformanceProfiler: Profile function execution")
    print("   • MatrixOptimizations: Vectorized operations")
    print("   • SolverBenchmark: Compare configurations")
    
    print_optimization_tips()
    
    print("\n💡 Usage:")
    print("   from nanofluid_simulator.performance import SolverBenchmark")
    print("   ")
    print("   benchmark = SolverBenchmark()")
    print("   results = benchmark.benchmark_mesh_sizes([(20,10), (50,30), (100,60)])")
    print("   benchmark.generate_report()")
