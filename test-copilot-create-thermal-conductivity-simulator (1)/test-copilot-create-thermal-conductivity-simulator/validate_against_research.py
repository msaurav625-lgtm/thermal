#!/usr/bin/env python3
"""
BKPS NFL Thermal v6.0 - Research Validation Script
Validates simulator results against published experimental data

References:
1. Pak & Cho (1998) - Al2O3-Water
2. Eastman et al. (2001) - Cu-EG
3. Xuan & Li (2003) - Cu-Water
4. Das et al. (2003) - Al2O3-Water
5. Lee et al. (1999) - Al2O3/CuO-Water
6. Choi et al. (2001) - CNT nanofluids
7. Keblinski et al. (2002) - Theoretical mechanisms
8. Maxwell (1881) - Classical EMT
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from nanofluid_simulator.integrated_simulator_v6 import BKPSNanofluidSimulator

# ============================================================================
# PUBLISHED EXPERIMENTAL DATA
# ============================================================================

# 1. Pak & Cho (1998) - Al2O3-Water
PAK_CHO_1998 = {
    'reference': 'Pak & Cho, Exp. Heat Transfer, 11:151-170 (1998)',
    'material': 'Al2O3',
    'base_fluid': 'Water',
    'diameter': 13e-9,  # 13 nm
    'temperature': 300,  # K
    'data': {
        'phi': np.array([0.01, 0.03, 0.05]),  # Volume fraction
        'k_enhancement': np.array([3.2, 8.4, 11.0])  # % enhancement
    }
}

# 2. Eastman et al. (2001) - Cu-EG
EASTMAN_2001 = {
    'reference': 'Eastman et al., Appl. Phys. Lett., 78:718-720 (2001)',
    'material': 'Cu',
    'base_fluid': 'EG',
    'diameter': 10e-9,  # 10 nm
    'temperature': 300,  # K
    'data': {
        'phi': np.array([0.003]),  # 0.3% volume fraction
        'k_enhancement': np.array([40.0])  # 40% enhancement
    }
}

# 3. Xuan & Li (2003) - Cu-Water
XUAN_LI_2003 = {
    'reference': 'Xuan & Li, Int. J. Heat Fluid Flow, 21:58-64 (2003)',
    'material': 'Cu',
    'base_fluid': 'Water',
    'diameter': 100e-9,  # 100 nm
    'temperature': 298,  # K
    'data': {
        'phi': np.array([0.01, 0.02, 0.03, 0.05]),
        'k_enhancement': np.array([15.0, 24.0, 30.0, 36.0])
    }
}

# 4. Das et al. (2003) - Al2O3-Water (Temperature dependent)
DAS_2003 = {
    'reference': 'Das et al., J. Heat Transfer, 125:567-574 (2003)',
    'material': 'Al2O3',
    'base_fluid': 'Water',
    'diameter': 38.4e-9,  # 38.4 nm
    'phi': 0.04,  # 4% volume fraction
    'data': {
        'temperature': np.array([294, 304, 314, 324, 334]),  # K
        'k_enhancement': np.array([8.0, 11.0, 14.0, 18.0, 23.0])  # %
    }
}

# 5. Lee et al. (1999) - Al2O3-Water
LEE_1999 = {
    'reference': 'Lee et al., J. Heat Transfer, 121:280-289 (1999)',
    'material': 'Al2O3',
    'base_fluid': 'Water',
    'diameter': 38.4e-9,
    'temperature': 300,
    'data': {
        'phi': np.array([0.01, 0.02, 0.03, 0.04]),
        'k_enhancement': np.array([2.0, 4.5, 7.0, 9.0])
    }
}

# 6. CuO-Water data (multiple sources)
CUO_WATER = {
    'reference': 'Multiple sources compilation',
    'material': 'CuO',
    'base_fluid': 'Water',
    'diameter': 29e-9,
    'temperature': 300,
    'data': {
        'phi': np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
        'k_enhancement': np.array([5.5, 10.0, 14.0, 17.5, 20.0])
    }
}

# Maxwell's theoretical prediction (1881)
def maxwell_prediction(phi, k_p, k_f):
    """Maxwell's effective medium theory"""
    return k_f * (k_p + 2*k_f + 2*phi*(k_p - k_f)) / (k_p + 2*k_f - phi*(k_p - k_f))

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_against_dataset(dataset, verbose=True):
    """
    Validate simulator against a research dataset
    
    Returns:
        dict: Validation metrics including MAE, RMSE, R²
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Validating against: {dataset['reference']}")
        print(f"Material: {dataset['material']}, Base: {dataset['base_fluid']}")
        print(f"{'='*80}")
    
    # Create simulator
    sim = BKPSNanofluidSimulator(
        base_fluid=dataset['base_fluid'],
        temperature=dataset.get('temperature', 300)
    )
    
    data = dataset['data']
    predictions = []
    experimental = []
    
    # Temperature-dependent data
    if 'temperature' in data:
        phi = dataset['phi']
        for T in data['temperature']:
            sim.temperature = T
            sim.add_nanoparticle(
                material=dataset['material'],
                volume_fraction=phi,
                diameter=dataset['diameter'],
                shape='sphere'
            )
            
            k_base = sim.calculate_base_fluid_conductivity()
            k_nf = sim.calculate_static_thermal_conductivity()
            enhancement = (k_nf / k_base - 1) * 100
            
            predictions.append(enhancement)
        
        experimental = data['k_enhancement']
    
    # Volume fraction-dependent data
    else:
        for phi in data['phi']:
            sim.add_nanoparticle(
                material=dataset['material'],
                volume_fraction=phi,
                diameter=dataset['diameter'],
                shape='sphere'
            )
            
            k_base = sim.calculate_base_fluid_conductivity()
            k_nf = sim.calculate_static_thermal_conductivity()
            enhancement = (k_nf / k_base - 1) * 100
            
            predictions.append(enhancement)
        
        experimental = data['k_enhancement']
    
    # Convert to arrays
    predictions = np.array(predictions)
    experimental = np.array(experimental)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - experimental))
    rmse = np.sqrt(np.mean((predictions - experimental)**2))
    
    # R² calculation
    ss_res = np.sum((experimental - predictions)**2)
    ss_tot = np.sum((experimental - np.mean(experimental))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Percentage error
    mape = np.mean(np.abs((experimental - predictions) / experimental)) * 100
    
    if verbose:
        print(f"\nResults:")
        print(f"  Mean Absolute Error (MAE): {mae:.2f}%")
        print(f"  Root Mean Square Error (RMSE): {rmse:.2f}%")
        print(f"  R² Score: {r_squared:.4f}")
        print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        print(f"\nComparison:")
        print(f"{'Experimental':<15} {'Predicted':<15} {'Error':<15}")
        print("-" * 45)
        for exp, pred in zip(experimental, predictions):
            error = pred - exp
            print(f"{exp:>12.2f}%   {pred:>12.2f}%   {error:>+12.2f}%")
    
    return {
        'dataset': dataset['reference'],
        'predictions': predictions,
        'experimental': experimental,
        'mae': mae,
        'rmse': rmse,
        'r_squared': r_squared,
        'mape': mape
    }


def plot_validation_results(results_list):
    """Create comprehensive validation plots"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Individual dataset comparisons
    n_datasets = len(results_list)
    n_cols = 3
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    for idx, result in enumerate(results_list):
        ax = fig.add_subplot(n_rows + 1, n_cols, idx + 1)
        
        exp = result['experimental']
        pred = result['predictions']
        
        ax.scatter(exp, pred, s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
        
        # Perfect prediction line
        min_val = min(exp.min(), pred.min())
        max_val = max(exp.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        # ±20% error bands
        ax.fill_between([min_val, max_val], 
                        [min_val*0.8, max_val*0.8], 
                        [min_val*1.2, max_val*1.2], 
                        alpha=0.2, color='gray', label='±20% error')
        
        ax.set_xlabel('Experimental (%)', fontsize=10)
        ax.set_ylabel('Predicted (%)', fontsize=10)
        ax.set_title(f"{result['dataset'].split(',')[0]}\nR²={result['r_squared']:.3f}, MAE={result['mae']:.2f}%", 
                     fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot: Combined parity plot
    ax_combined = fig.add_subplot(n_rows + 1, n_cols, n_datasets + 1)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    for idx, result in enumerate(results_list):
        ax_combined.scatter(result['experimental'], result['predictions'], 
                           s=80, alpha=0.6, color=colors[idx], 
                           label=result['dataset'].split(',')[0], 
                           edgecolors='black', linewidth=1)
    
    # Combined perfect line
    all_exp = np.concatenate([r['experimental'] for r in results_list])
    all_pred = np.concatenate([r['predictions'] for r in results_list])
    min_val = min(all_exp.min(), all_pred.min())
    max_val = max(all_exp.max(), all_pred.max())
    
    ax_combined.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect')
    ax_combined.fill_between([min_val, max_val], 
                            [min_val*0.8, max_val*0.8], 
                            [min_val*1.2, max_val*1.2], 
                            alpha=0.2, color='gray')
    
    ax_combined.set_xlabel('Experimental Enhancement (%)', fontsize=11, fontweight='bold')
    ax_combined.set_ylabel('Predicted Enhancement (%)', fontsize=11, fontweight='bold')
    ax_combined.set_title('Combined Validation - All Datasets', fontsize=12, fontweight='bold')
    ax_combined.legend(fontsize=8, loc='upper left')
    ax_combined.grid(True, alpha=0.3)
    
    # Plot: Error distribution
    ax_error = fig.add_subplot(n_rows + 1, n_cols, n_datasets + 2)
    
    all_errors = np.concatenate([r['predictions'] - r['experimental'] for r in results_list])
    ax_error.hist(all_errors, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax_error.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax_error.set_xlabel('Prediction Error (%)', fontsize=11)
    ax_error.set_ylabel('Frequency', fontsize=11)
    ax_error.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax_error.legend(fontsize=9)
    ax_error.grid(True, alpha=0.3, axis='y')
    
    # Plot: Metrics summary
    ax_metrics = fig.add_subplot(n_rows + 1, n_cols, n_datasets + 3)
    
    dataset_names = [r['dataset'].split(',')[0][:20] for r in results_list]
    mae_values = [r['mae'] for r in results_list]
    r2_values = [r['r_squared'] for r in results_list]
    
    x = np.arange(len(dataset_names))
    width = 0.35
    
    ax_metrics2 = ax_metrics.twinx()
    
    bars1 = ax_metrics.bar(x - width/2, mae_values, width, label='MAE (%)', 
                           color='coral', alpha=0.7, edgecolor='black')
    bars2 = ax_metrics2.bar(x + width/2, r2_values, width, label='R²', 
                            color='lightgreen', alpha=0.7, edgecolor='black')
    
    ax_metrics.set_xlabel('Dataset', fontsize=10)
    ax_metrics.set_ylabel('MAE (%)', fontsize=10, color='coral')
    ax_metrics2.set_ylabel('R²', fontsize=10, color='green')
    ax_metrics.set_title('Validation Metrics Summary', fontsize=12, fontweight='bold')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=8)
    ax_metrics.tick_params(axis='y', labelcolor='coral')
    ax_metrics2.tick_params(axis='y', labelcolor='green')
    ax_metrics.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax_metrics.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=7)
    
    for bar in bars2:
        height = bar.get_height()
        ax_metrics2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    return fig


def calculate_overall_statistics(results_list):
    """Calculate overall validation statistics"""
    
    print(f"\n{'='*80}")
    print("OVERALL VALIDATION STATISTICS")
    print(f"{'='*80}")
    
    # Combine all data
    all_exp = np.concatenate([r['experimental'] for r in results_list])
    all_pred = np.concatenate([r['predictions'] for r in results_list])
    
    # Overall metrics
    mae_overall = np.mean(np.abs(all_pred - all_exp))
    rmse_overall = np.sqrt(np.mean((all_pred - all_exp)**2))
    
    ss_res = np.sum((all_exp - all_pred)**2)
    ss_tot = np.sum((all_exp - np.mean(all_exp))**2)
    r2_overall = 1 - (ss_res / ss_tot)
    
    mape_overall = np.mean(np.abs((all_exp - all_pred) / all_exp)) * 100
    
    # Average metrics across datasets
    avg_mae = np.mean([r['mae'] for r in results_list])
    avg_rmse = np.mean([r['rmse'] for r in results_list])
    avg_r2 = np.mean([r['r_squared'] for r in results_list])
    avg_mape = np.mean([r['mape'] for r in results_list])
    
    print(f"\nOverall Performance (All data points combined):")
    print(f"  Total data points: {len(all_exp)}")
    print(f"  Mean Absolute Error (MAE): {mae_overall:.2f}%")
    print(f"  Root Mean Square Error (RMSE): {rmse_overall:.2f}%")
    print(f"  R² Score: {r2_overall:.4f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape_overall:.2f}%")
    
    print(f"\nAverage Performance (Across {len(results_list)} datasets):")
    print(f"  Average MAE: {avg_mae:.2f}%")
    print(f"  Average RMSE: {avg_rmse:.2f}%")
    print(f"  Average R²: {avg_r2:.4f}")
    print(f"  Average MAPE: {avg_mape:.2f}%")
    
    # Accuracy assessment
    print(f"\nAccuracy Assessment:")
    within_10 = np.sum(np.abs(all_pred - all_exp) < 10) / len(all_exp) * 100
    within_20 = np.sum(np.abs(all_pred - all_exp) < 20) / len(all_exp) * 100
    within_30 = np.sum(np.abs(all_pred - all_exp) < 30) / len(all_exp) * 100
    
    print(f"  Predictions within ±10%: {within_10:.1f}%")
    print(f"  Predictions within ±20%: {within_20:.1f}%")
    print(f"  Predictions within ±30%: {within_30:.1f}%")
    
    # Best and worst datasets
    print(f"\nDataset Performance Ranking:")
    sorted_results = sorted(results_list, key=lambda x: x['mae'])
    
    print(f"\n  Best performing dataset:")
    best = sorted_results[0]
    print(f"    {best['dataset']}")
    print(f"    MAE: {best['mae']:.2f}%, R²: {best['r_squared']:.4f}")
    
    print(f"\n  Most challenging dataset:")
    worst = sorted_results[-1]
    print(f"    {worst['dataset']}")
    print(f"    MAE: {worst['mae']:.2f}%, R²: {worst['r_squared']:.4f}")
    
    return {
        'mae_overall': mae_overall,
        'rmse_overall': rmse_overall,
        'r2_overall': r2_overall,
        'mape_overall': mape_overall,
        'within_10_percent': within_10,
        'within_20_percent': within_20,
        'within_30_percent': within_30
    }


def main():
    """Main validation routine"""
    
    print("="*80)
    print("BKPS NFL THERMAL v6.0 - RESEARCH VALIDATION")
    print("Validating simulator against published experimental data")
    print("="*80)
    
    # List of datasets to validate
    datasets = [
        PAK_CHO_1998,
        LEE_1999,
        EASTMAN_2001,
        XUAN_LI_2003,
        DAS_2003,
        CUO_WATER
    ]
    
    # Validate against each dataset
    results = []
    for dataset in datasets:
        try:
            result = validate_against_dataset(dataset, verbose=True)
            results.append(result)
        except Exception as e:
            print(f"Error validating {dataset['reference']}: {e}")
            continue
    
    # Calculate overall statistics
    overall_stats = calculate_overall_statistics(results)
    
    # Create validation plots
    print(f"\nGenerating validation plots...")
    fig = plot_validation_results(results)
    
    # Save figure
    output_file = 'validation_against_research.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Validation plot saved to: {output_file}")
    
    # Save detailed report
    report_file = 'VALIDATION_REPORT.txt'
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BKPS NFL THERMAL v6.0 - VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("SUMMARY:\n")
        f.write(f"  Datasets validated: {len(results)}\n")
        f.write(f"  Total data points: {sum(len(r['experimental']) for r in results)}\n")
        f.write(f"  Overall MAE: {overall_stats['mae_overall']:.2f}%\n")
        f.write(f"  Overall R²: {overall_stats['r2_overall']:.4f}\n")
        f.write(f"  Predictions within ±10%: {overall_stats['within_10_percent']:.1f}%\n")
        f.write(f"  Predictions within ±20%: {overall_stats['within_20_percent']:.1f}%\n\n")
        
        f.write("DETAILED RESULTS BY DATASET:\n")
        f.write("-"*80 + "\n\n")
        
        for result in results:
            f.write(f"Dataset: {result['dataset']}\n")
            f.write(f"  MAE: {result['mae']:.2f}%\n")
            f.write(f"  RMSE: {result['rmse']:.2f}%\n")
            f.write(f"  R²: {result['r_squared']:.4f}\n")
            f.write(f"  MAPE: {result['mape']:.2f}%\n\n")
    
    print(f"Detailed validation report saved to: {report_file}")
    
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nConclusion:")
    if overall_stats['r2_overall'] > 0.8 and overall_stats['mae_overall'] < 15:
        print("  ✓ EXCELLENT agreement with experimental data")
        print("  ✓ Simulator predictions are highly reliable")
    elif overall_stats['r2_overall'] > 0.6 and overall_stats['mae_overall'] < 25:
        print("  ✓ GOOD agreement with experimental data")
        print("  ✓ Simulator predictions are reliable for most cases")
    else:
        print("  ⚠ MODERATE agreement with experimental data")
        print("  ⚠ Use predictions with caution, especially for extreme conditions")
    
    print(f"\nFiles generated:")
    print(f"  - {output_file}")
    print(f"  - {report_file}")
    
    # Show plot
    plt.show()


if __name__ == '__main__':
    main()
