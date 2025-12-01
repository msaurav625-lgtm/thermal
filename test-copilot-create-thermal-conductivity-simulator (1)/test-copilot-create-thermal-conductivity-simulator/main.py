#!/usr/bin/env python3
"""
BKPS NFL Thermal Pro 7.0 - Main Entry Point
Dedicated to: Brijesh Kumar Pandey

Single launcher for all simulation modes with professional startup logging
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime
import platform

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from nanofluid_simulator.unified_engine import (
    BKPSNanofluidEngine, UnifiedConfig, SimulationMode,
    __version__, __release_date__, __codename__
)

# ASCII Art Banner
BANNER = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██████╗ ██╗  ██╗██████╗ ███████╗    ███╗   ██╗███████╗██╗              ║
║   ██╔══██╗██║ ██╔╝██╔══██╗██╔════╝    ████╗  ██║██╔════╝██║              ║
║   ██████╔╝█████╔╝ ██████╔╝███████╗    ██╔██╗ ██║█████╗  ██║              ║
║   ██╔══██╗██╔═██╗ ██╔═══╝ ╚════██║    ██║╚██╗██║██╔══╝  ██║              ║
║   ██████╔╝██║  ██╗██║     ███████║    ██║ ╚████║██║     ███████╗         ║
║   ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚══════╝    ╚═╝  ╚═══╝╚═╝     ╚══════╝         ║
║                                                                           ║
║                    THERMAL PRO - VERSION 7.0                              ║
║                                                                           ║
║          Professional Nanofluid Simulation & Analysis Platform            ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""


def setup_logging(level=logging.INFO, log_file=None):
    """Setup professional logging"""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers
    )


def print_startup_info():
    """Print professional startup information"""
    print(BANNER)
    print(f"  Version: {__version__}")
    print(f"  Release Date: {__release_date__}")
    print(f"  Dedicated to: Brijesh Kumar Pandey")
    print()
    print("="*80)
    print("SYSTEM INFORMATION")
    print("="*80)
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Python Version: {platform.python_version()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Processor: {platform.processor()}")
    print()
    
    # Check for acceleration backends
    print("COMPUTATIONAL BACKENDS:")
    backends = []
    
    try:
        import numpy as np
        numpy_ver = np.__version__
        backends.append(f"  ✓ NumPy {numpy_ver}")
    except ImportError:
        backends.append("  ✗ NumPy not available")
    
    try:
        import numba
        numba_ver = numba.__version__
        backends.append(f"  ✓ Numba {numba_ver} (JIT compilation)")
    except ImportError:
        backends.append("  ○ Numba not available (optional)")
    
    try:
        import torch
        torch_ver = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            backends.append(f"  ✓ PyTorch {torch_ver} (CUDA enabled)")
        else:
            backends.append(f"  ✓ PyTorch {torch_ver} (CPU only)")
    except ImportError:
        backends.append("  ○ PyTorch not available (optional)")
    
    try:
        import cupy
        cupy_ver = cupy.__version__
        backends.append(f"  ✓ CuPy {cupy_ver} (GPU acceleration)")
    except ImportError:
        backends.append("  ○ CuPy not available (optional)")
    
    for backend in backends:
        print(backend)
    
    print()
    print("="*80)
    print()


def run_cli_mode(args):
    """Run in CLI mode"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting {args.mode} simulation in CLI mode")
    
    # Load configuration if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = UnifiedConfig.load(args.config)
    else:
        # Create quick configuration
        logger.info("Creating quick-start configuration")
        engine = BKPSNanofluidEngine.quick_start(
            mode=args.mode,
            base_fluid=args.base_fluid,
            nanoparticle=args.nanoparticle,
            volume_fraction=args.volume_fraction,
            temperature=args.temperature,
            diameter=args.diameter * 1e-9  # Convert nm to m
        )
        config = engine.config
    
    # Create engine
    engine = BKPSNanofluidEngine(config)
    
    # Run simulation
    logger.info("Running simulation...")
    results = engine.run(progress_callback=lambda p: logger.info(f"Progress: {p}%"))
    
    # Print results
    print()
    print("="*80)
    print("SIMULATION RESULTS")
    print("="*80)
    
    if config.mode == SimulationMode.STATIC:
        print(f"  Base Conductivity (k_base):    {results['k_base']:.6f} W/m·K")
        print(f"  Nanofluid Conductivity (k_nf):  {results['k_static']:.6f} W/m·K")
        print(f"  Enhancement:                    {results['enhancement_k']:.2f}%")
        print(f"  Base Viscosity (μ_base):        {results['mu_base']*1000:.6f} mPa·s")
        print(f"  Nanofluid Viscosity (μ_nf):     {results['mu_nf']*1000:.6f} mPa·s")
        print(f"  Viscosity Ratio:                {results['viscosity_ratio']:.4f}")
    
    print()
    
    # Export results if requested
    if args.output:
        logger.info(f"Exporting results to {args.output}")
        engine.export_results(args.output, format=args.format)
        print(f"  Results saved to: {args.output}")
    
    print()
    print("="*80)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("="*80)


def run_gui_mode(args):
    """Run in GUI mode"""
    logger = logging.getLogger(__name__)
    logger.info("Starting GUI mode (v7.0)")
    
    try:
        from PyQt6.QtWidgets import QApplication
        from bkps_professional_gui_v7 import BKPSProfessionalGUI_V7
        
        app = QApplication(sys.argv)
        app.setApplicationName(f"{__codename__} {__version__}")
        app.setOrganizationName("BKPS")
        app.setOrganizationDomain("bkps.nfl")
        
        gui = BKPSProfessionalGUI_V7()
        
        # Load configuration if provided
        if args.config:
            logger.info(f"Loading configuration into GUI: {args.config}")
            # Would need to implement config loading in GUI
        
        gui.show()
        sys.exit(app.exec())
        
    except ImportError as e:
        logger.error(f"GUI dependencies not available: {e}")
        print("\nERROR: PyQt6 not installed. Install with:")
        print("  pip install PyQt6 matplotlib")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description=f"{__codename__} v{__version__} - Professional Nanofluid Simulation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch GUI (default)
  python main.py
  
  # Run static simulation in CLI
  python main.py --cli --mode static --nanoparticle Al2O3 --volume-fraction 0.02
  
  # Load configuration and run
  python main.py --cli --config my_config.json --output results.json
  
  # Run with custom parameters
  python main.py --cli --mode flow --base-fluid Water --temperature 320 --diameter 50
        """
    )
    
    # Mode selection
    parser.add_argument('--gui', action='store_true', help='Launch GUI (default)')
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode')
    
    # Configuration
    parser.add_argument('--config', '-c', type=str, help='Load configuration from JSON file')
    parser.add_argument('--mode', '-m', type=str, default='static',
                       choices=['static', 'flow', 'cfd', 'hybrid'],
                       help='Simulation mode (default: static)')
    
    # Quick start parameters
    parser.add_argument('--base-fluid', type=str, default='Water',
                       help='Base fluid (default: Water)')
    parser.add_argument('--nanoparticle', type=str, default='Al2O3',
                       help='Nanoparticle material (default: Al2O3)')
    parser.add_argument('--volume-fraction', type=float, default=0.02,
                       help='Volume fraction (default: 0.02)')
    parser.add_argument('--temperature', type=float, default=300,
                       help='Temperature in Kelvin (default: 300)')
    parser.add_argument('--diameter', type=float, default=30,
                       help='Particle diameter in nanometers (default: 30)')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--format', type=str, default='json',
                       choices=['json', 'csv', 'hdf5'],
                       help='Output format (default: json)')
    
    # Logging options
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--log-file', type=str, help='Log to file')
    
    # Version
    parser.add_argument('--version', '-v', action='version',
                       version=f"{__codename__} v{__version__} ({__release_date__})")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        level=getattr(logging, args.log_level),
        log_file=args.log_file
    )
    
    # Print startup info
    print_startup_info()
    
    # Determine mode
    if args.cli:
        run_cli_mode(args)
    else:
        # Default to GUI
        run_gui_mode(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFATAL ERROR: {e}")
        print("Check log file for details.")
        sys.exit(1)
