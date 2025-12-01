#!/usr/bin/env python3
"""
Run research-grade benchmarks and save results.
- Lid-driven cavity CFD benchmark versus Ghia et al. (1982)

Usage:
  python scripts/run_benchmarks.py --nx 50 --ny 50 --Re 100 --out out/benchmarks
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from nanofluid_simulator.benchmarks.lid_driven_cavity import run_lid_driven_cavity


def plot_and_save_lid_cavity(res: dict, outdir: Path) -> None:
    """Save velocity profile comparisons and CSVs."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Figure: velocity profiles vs Ghia et al.
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(res['u_center'], res['y'], label='Simulated')
    ax1.plot(res['ghia_u'], res['ghia_y'], 'o-', label='Ghia et al.')
    ax1.set_xlabel('u (centerline)')
    ax1.set_ylabel('y')
    ax1.set_title('U velocity profile')
    ax1.legend()

    ax2.plot(res['x'], res['v_center'], label='Simulated')
    ax2.plot(res['ghia_x'], res['ghia_v'], 'o-', label='Ghia et al.')
    ax2.set_xlabel('x')
    ax2.set_ylabel('v (centerline)')
    ax2.set_title('V velocity profile')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(outdir / 'lid_driven_cavity_profiles.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # CSV exports
    np.savetxt(outdir / 'u_centerline.csv', np.column_stack([res['y'], res['u_center']]),
               delimiter=',', header='y,u_center', comments='')
    np.savetxt(outdir / 'v_centerline.csv', np.column_stack([res['x'], res['v_center']]),
               delimiter=',', header='x,v_center', comments='')


def main():
    parser = argparse.ArgumentParser(description='Run CFD/thermal benchmarks and save results.')
    parser.add_argument('--nx', type=int, default=50, help='Grid points in x')
    parser.add_argument('--ny', type=int, default=50, help='Grid points in y')
    parser.add_argument('--Re', type=float, default=100.0, help='Reynolds number for lid-driven cavity')
    parser.add_argument('--out', type=str, default='out/benchmarks', help='Output directory')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots (always saved)')

    args = parser.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Run lid-driven cavity benchmark (no interactive plotting here)
    res = run_lid_driven_cavity(nx=args.nx, ny=args.ny, Re=args.Re, plot=False)

    # Save metrics
    metrics = {
        'benchmark': 'lid_driven_cavity',
        'nx': args.nx,
        'ny': args.ny,
        'Re': args.Re,
        'mae_u': float(res.get('mae_u', float('nan'))),
        'mae_v': float(res.get('mae_v', float('nan')))
    }
    (outdir / 'lid_driven_cavity_metrics.json').write_text(json.dumps(metrics, indent=2))

    # Save figures and CSVs
    plot_and_save_lid_cavity(res, outdir)

    print('✓ Lid-driven cavity complete:')
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved in: {outdir}")


if __name__ == '__main__':
    main()
