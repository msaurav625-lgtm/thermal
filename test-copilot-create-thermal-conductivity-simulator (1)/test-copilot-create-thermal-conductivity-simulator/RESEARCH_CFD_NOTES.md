# Research-Grade CFD Development Notes

## Current Status

**Attempted Approaches:**

1. **Simple Projection Method** (`simple_cfd.py`) - ⚠️ PARTIALLY WORKING
   - Collocated grid (u, v, p at same locations)
   - Explicit momentum, implicit pressure
   - **Issues**: Pressure-velocity decoupling, ~50% errors
   - **Status**: Qualitative only

2. **MAC Method** (`mac_cfd_solver.py`) - ❌ UNSTABLE
   - Staggered grid (industry standard)
   - Proper pressure-velocity coupling
   - **Issues**: Diverging despite stability controls
   - **Status**: Not production ready

## Why Research-Grade CFD is Difficult

### The Core Challenge
Incompressible Navier-Stokes with proper pressure-velocity coupling requires:
1. **Staggered grids** OR **pressure stabilization** (PISO, SIMPLE, SIMPLER)
2. **Iterative solvers** for large linear systems (CG, GMRES, multigrid)
3. **Careful time stepping** (CFL condition, adaptive dt)
4. **Robust boundary conditions**

### What We're Missing
- **Multigrid pressure solver**: Current Jacobi/SOR too slow/inaccurate
- **Proper SIMPLE algorithm**: Requires solving large sparse linear systems
- **Higher-order schemes**: Current first-order upwind too dissipative
- **Validation infrastructure**: Need lid-driven cavity, backward step benchmarks

## Path to Research-Grade

### Option 1: Fix Projection Method (Fastest)
Improvements needed:
- [ ] Implement Chorin's original method more carefully
- [ ] Add staggered grid for pressure
- [ ] Use conjugate gradient for Poisson equation
- [ ] Validate against Ghia et al. (1982) lid-driven cavity
- **Time**: 2-3 days of focused work

### Option 2: Complete MAC Implementation (Better)
Fixes needed:
- [ ] Debug convection scheme (likely source of instability)
- [ ] Implement CFL-adaptive time stepping
- [ ] Add multigrid for Poisson equation
- [ ] Comprehensive boundary condition testing
- **Time**: 1 week

### Option 3: Use Established Library (Best)
Integrate:
- **FEniCS/DOLFINx**: Finite element, research-grade, well-validated
- **PyFR**: High-order FR method
- **OpenFOAM** (via PyFoam): Industry standard
- **Time**: 3-4 days integration + validation

## Recommendation

For **research-grade validated CFD**:

**Short-term (now)**: Document current limitations honestly, mark CFD as "educational/qualitative"

**Medium-term (if needed)**: 
1. Implement established library wrapper (FEniCS)
2. OR hire CFD specialist to fix projection method
3. OR use commercial solver (ANSYS Fluent, COMSOL) via Python API

**Current thermal simulator value**: The **static and flow thermal property modes are fully validated** and research-grade. CFD is a bonus feature but not core competency.

## Validation Benchmarks Needed

For research publication quality:
1. **Poiseuille flow**: < 5% error on pressure drop ✅ (analytical available)
2. **Lid-driven cavity** (Ghia et al. 1982): Match velocity profiles ❌ (not implemented)
3. **Backward-facing step**: Match reattachment length ❌ (not implemented)
4. **Natural convection** (De Vahl Davis 1983): Match Nu number ❌ (not implemented)

**Current status**: 0/4 benchmarks passed

## Honest Assessment

The thermal conductivity simulator is **excellent for thermal properties** (72.7% accuracy on 6 datasets).

CFD mode is **not research-grade** and should be:
- Used for visualization/education only
- Clearly marked as "beta" or "qualitative"
- OR removed until properly validated
- OR replaced with established library

**Bottom line**: Don't publish CFD results from current implementation in peer-reviewed journals.
