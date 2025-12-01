#!/usr/bin/env python3
"""
BKPS NFL Thermal Pro 7.0 - Validation Center Module
Dedicated to: Brijesh Kumar Pandey

Provides validation against published experimental datasets
for GUI and standalone use.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class ValidationDataset:
    """Container for experimental validation data"""
    name: str
    reference: str
    material: str
    base_fluid: str
    diameter: float  # meters
    temperature: float  # K (single value or reference)
    phi_values: np.ndarray  # Volume fractions
    k_enhancement_exp: np.ndarray  # Experimental % enhancement
    temp_dependent: bool = False
    temp_values: np.ndarray = None  # For temperature-dependent data


class ValidationCenter:
    """
    Validation against published experimental datasets
    
    Datasets:
    1. Pak & Cho (1998) - Al2O3/Water
    2. Lee et al. (1999) - Al2O3/Water
    3. Eastman et al. (2001) - Cu/EG
    4. Xuan & Li (2000) - Cu/Water
    5. Das et al. (2003) - Al2O3/Water (temperature-dependent)
    6. CuO/Water compilation
    """
    
    def __init__(self):
        self.datasets = self._load_datasets()
    
    def _load_datasets(self) -> Dict[str, ValidationDataset]:
        """Load all experimental datasets"""
        datasets = {}
        
        # 1. Pak & Cho (1998) - Al2O3-Water
        datasets['pak_cho_1998'] = ValidationDataset(
            name="Pak & Cho (1998) - Al2O3/Water",
            reference="Pak & Cho, Exp. Heat Transfer, 11:151-170 (1998)",
            material="Al2O3",
            base_fluid="Water",
            diameter=13e-9,  # 13 nm
            temperature=300,
            phi_values=np.array([0.01, 0.03, 0.05]),
            k_enhancement_exp=np.array([3.2, 8.4, 11.0])
        )
        
        # 2. Lee et al. (1999) - Al2O3-Water
        datasets['lee_1999'] = ValidationDataset(
            name="Lee et al. (1999) - Al2O3/Water",
            reference="Lee et al., J. Heat Transfer, 121:280-289 (1999)",
            material="Al2O3",
            base_fluid="Water",
            diameter=38.4e-9,  # 38.4 nm
            temperature=300,
            phi_values=np.array([0.01, 0.02, 0.03, 0.04]),
            k_enhancement_exp=np.array([2.0, 4.5, 7.0, 9.0])
        )
        
        # 3. Eastman et al. (2001) - Cu-EG
        datasets['eastman_2001'] = ValidationDataset(
            name="Eastman et al. (2001) - Cu/EG",
            reference="Eastman et al., Appl. Phys. Lett., 78:718-720 (2001)",
            material="Cu",
            base_fluid="EG",
            diameter=10e-9,  # 10 nm
            temperature=300,
            phi_values=np.array([0.003]),
            k_enhancement_exp=np.array([40.0])
        )
        
        # 4. Xuan & Li (2000) - Cu-Water
        datasets['xuan_li_2000'] = ValidationDataset(
            name="Xuan & Li (2000) - Cu/Water",
            reference="Xuan & Li, Int. J. Heat Fluid Flow, 21:58-64 (2000)",
            material="Cu",
            base_fluid="Water",
            diameter=100e-9,  # 100 nm
            temperature=298,
            phi_values=np.array([0.01, 0.02, 0.03, 0.05]),
            k_enhancement_exp=np.array([15.0, 24.0, 30.0, 36.0])
        )
        
        # 5. Das et al. (2003) - Al2O3-Water (temperature-dependent)
        datasets['das_2003'] = ValidationDataset(
            name="Das et al. (2003) - Al2O3/Water (T-dependent)",
            reference="Das et al., J. Heat Transfer, 125:567-574 (2003)",
            material="Al2O3",
            base_fluid="Water",
            diameter=38.4e-9,  # 38.4 nm
            temperature=314,  # Reference temperature
            phi_values=np.array([0.04]),  # Fixed phi
            k_enhancement_exp=np.array([8.0, 11.0, 14.0, 18.0, 23.0]),
            temp_dependent=True,
            temp_values=np.array([294, 304, 314, 324, 334])
        )
        
        # 6. CuO-Water compilation
        datasets['cuo_water'] = ValidationDataset(
            name="CuO/Water (Multiple Sources)",
            reference="Multiple sources compilation",
            material="CuO",
            base_fluid="Water",
            diameter=29e-9,  # 29 nm
            temperature=300,
            phi_values=np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
            k_enhancement_exp=np.array([3.5, 7.0, 11.0, 14.0, 17.5])
        )
        
        return datasets
    
    def get_dataset_names(self) -> List[str]:
        """Get list of available dataset names"""
        return [ds.name for ds in self.datasets.values()]
    
    def get_dataset(self, key: str) -> ValidationDataset:
        """Get specific dataset by key"""
        return self.datasets.get(key)
    
    def validate_simulation(self, 
                          engine: Any,
                          dataset_key: str) -> Dict[str, Any]:
        """
        Run validation against a specific dataset
        
        Args:
            engine: BKPSNanofluidEngine instance
            dataset_key: Key identifying the dataset
            
        Returns:
            Validation results with metrics
        """
        dataset = self.datasets[dataset_key]
        
        results = {
            'dataset': dataset,
            'phi_values': [],
            'k_enhancement_sim': [],
            'k_enhancement_exp': [],
            'errors': [],
            'relative_errors': []
        }
        
        # Run simulations
        if dataset.temp_dependent:
            # Temperature sweep
            for temp, k_enh_exp in zip(dataset.temp_values, dataset.k_enhancement_exp):
                # Update config
                engine.config.base_fluid.temperature = temp
                engine.config.nanoparticles[0].material = dataset.material
                engine.config.nanoparticles[0].volume_fraction = dataset.phi_values[0]
                engine.config.nanoparticles[0].diameter = dataset.diameter
                
                # Run
                try:
                    res = engine.run()
                    if 'static' in res:
                        k_enh_sim = res['static']['enhancement_k']
                    else:
                        k_enh_sim = res['enhancement_k']
                    
                    results['phi_values'].append(temp)  # Using temp as x-axis
                    results['k_enhancement_sim'].append(k_enh_sim)
                    results['k_enhancement_exp'].append(k_enh_exp)
                    
                    error = k_enh_sim - k_enh_exp
                    rel_error = (error / k_enh_exp) * 100 if k_enh_exp != 0 else 0
                    
                    results['errors'].append(error)
                    results['relative_errors'].append(rel_error)
                    
                except Exception as e:
                    print(f"Simulation failed at T={temp} K: {e}")
        else:
            # Volume fraction sweep
            for phi, k_enh_exp in zip(dataset.phi_values, dataset.k_enhancement_exp):
                # Update config
                engine.config.base_fluid.name = dataset.base_fluid
                engine.config.base_fluid.temperature = dataset.temperature
                engine.config.nanoparticles[0].material = dataset.material
                engine.config.nanoparticles[0].volume_fraction = phi
                engine.config.nanoparticles[0].diameter = dataset.diameter
                
                # Run
                try:
                    res = engine.run()
                    if 'static' in res:
                        k_enh_sim = res['static']['enhancement_k']
                    else:
                        k_enh_sim = res['enhancement_k']
                    
                    results['phi_values'].append(phi)
                    results['k_enhancement_sim'].append(k_enh_sim)
                    results['k_enhancement_exp'].append(k_enh_exp)
                    
                    error = k_enh_sim - k_enh_exp
                    rel_error = (error / k_enh_exp) * 100 if k_enh_exp != 0 else 0
                    
                    results['errors'].append(error)
                    results['relative_errors'].append(rel_error)
                    
                except Exception as e:
                    print(f"Simulation failed at φ={phi}: {e}")
        
        # Calculate metrics
        if results['errors']:
            results['mae'] = np.mean(np.abs(results['errors']))
            results['mape'] = np.mean(np.abs(results['relative_errors']))
            results['rmse'] = np.sqrt(np.mean(np.array(results['errors'])**2))
            results['max_error'] = np.max(np.abs(results['errors']))
            
            # Accuracy categories
            within_10 = sum(1 for e in results['relative_errors'] if abs(e) <= 10)
            within_20 = sum(1 for e in results['relative_errors'] if abs(e) <= 20)
            total = len(results['relative_errors'])
            
            results['accuracy_10'] = (within_10 / total) * 100 if total > 0 else 0
            results['accuracy_20'] = (within_20 / total) * 100 if total > 0 else 0
            
            # Overall rating
            if results['mape'] < 10:
                results['rating'] = "EXCELLENT"
                results['badge'] = "✓ PASS"
                results['color'] = "green"
            elif results['mape'] < 20:
                results['rating'] = "GOOD"
                results['badge'] = "✓ PASS"
                results['color'] = "yellow"
            else:
                results['rating'] = "FAIR"
                results['badge'] = "✗ REVIEW"
                results['color'] = "red"
        
        return results
    
    def validate_all(self, engine: Any) -> Dict[str, Dict[str, Any]]:
        """
        Validate against all datasets
        
        Args:
            engine: BKPSNanofluidEngine instance
            
        Returns:
            Dictionary of validation results for each dataset
        """
        all_results = {}
        
        for key in self.datasets.keys():
            print(f"Validating: {self.datasets[key].name}")
            all_results[key] = self.validate_simulation(engine, key)
        
        # Overall statistics
        all_mapes = [r['mape'] for r in all_results.values() if 'mape' in r]
        if all_mapes:
            all_results['overall'] = {
                'mean_mape': np.mean(all_mapes),
                'datasets_validated': len(all_mapes)
            }
        
        return all_results


def get_validation_summary(results: Dict[str, Any]) -> str:
    """
    Generate text summary of validation results
    
    Args:
        results: Validation results from validate_simulation()
        
    Returns:
        Formatted text summary
    """
    summary = []
    summary.append("═══ VALIDATION RESULTS ═══\n")
    summary.append(f"Dataset: {results['dataset'].name}")
    summary.append(f"Reference: {results['dataset'].reference}")
    summary.append(f"Material: {results['dataset'].material}")
    summary.append(f"Base Fluid: {results['dataset'].base_fluid}\n")
    
    if 'mape' in results:
        summary.append(f"Mean Absolute Percentage Error: {results['mape']:.2f}%")
        summary.append(f"Mean Absolute Error: {results['mae']:.4f}%")
        summary.append(f"RMSE: {results['rmse']:.4f}%")
        summary.append(f"Max Error: {results['max_error']:.4f}%\n")
        
        summary.append(f"Within ±10%: {results['accuracy_10']:.1f}%")
        summary.append(f"Within ±20%: {results['accuracy_20']:.1f}%\n")
        
        summary.append(f"Rating: {results['rating']}")
        summary.append(f"Badge: {results['badge']}")
    
    return "\n".join(summary)


if __name__ == '__main__':
    # Test validation center
    from nanofluid_simulator import BKPSNanofluidEngine
    
    print("Testing Validation Center...")
    
    center = ValidationCenter()
    print(f"✓ Loaded {len(center.datasets)} datasets")
    
    # Create engine
    engine = BKPSNanofluidEngine.quick_start(
        mode="static",
        nanoparticle="Al2O3",
        volume_fraction=0.02
    )
    
    # Validate against one dataset
    results = center.validate_simulation(engine, 'pak_cho_1998')
    
    print("\n" + get_validation_summary(results))
    
    print("\n✅ Validation center test complete!")
