"""
CFD Benchmarks and Verification Tools

Available benchmarks:
- lid_driven_cavity: Classic 2D cavity flow vs Ghia et al. (1982)
- mesh_independence: Grid independence study helper
"""

from .lid_driven_cavity import run_lid_driven_cavity

__all__ = ['run_lid_driven_cavity']
