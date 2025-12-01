"""
AI-CFD Integration Module
==========================

This module provides AI-powered enhancements for CFD simulations:
1. Flow regime classification (auto-select turbulence model)
2. Convergence monitoring and prediction
3. Solver parameter recommendations

Uses lightweight ML models (scikit-learn) - no GPU required.
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional

# Try to import scikit-learn, fall back to rule-based if not available
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Using rule-based AI fallback.")


class AIFlowRegimeClassifier:
    """
    AI-powered flow regime classifier.
    
    Predicts flow regime (laminar, transitional, turbulent) and recommends
    appropriate turbulence model based on Reynolds number, Prandtl number,
    and geometry parameters.
    
    Uses RandomForest when scikit-learn available, falls back to expert rules.
    """
    
    def __init__(self):
        """Initialize the classifier with pre-trained knowledge."""
        self.trained = False
        self.scaler = None
        self.model = None
        
        # Flow regime thresholds (expert knowledge)
        self.re_laminar_max = 2300
        self.re_transitional_min = 2300
        self.re_transitional_max = 4000
        self.re_turbulent_min = 4000
        
        # Initialize with synthetic training data
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model with synthetic training data based on fluid mechanics theory."""
        if not SKLEARN_AVAILABLE:
            self.trained = True
            return
        
        # Generate synthetic training data based on established correlations
        np.random.seed(42)
        n_samples = 1000
        
        # Features: [Re, Pr, aspect_ratio, hydraulic_diameter]
        X = []
        y = []  # 0=laminar, 1=transitional, 2=turbulent_ke, 3=turbulent_kw
        
        # Laminar flow samples (Re < 2300)
        for _ in range(250):
            Re = np.random.uniform(10, 2300)
            Pr = np.random.uniform(0.7, 15)
            AR = np.random.uniform(5, 100)
            Dh = np.random.uniform(0.0001, 0.1)
            X.append([Re, Pr, AR, Dh])
            y.append(0)  # laminar
        
        # Transitional flow samples (2300 < Re < 4000)
        for _ in range(150):
            Re = np.random.uniform(2300, 4000)
            Pr = np.random.uniform(0.7, 15)
            AR = np.random.uniform(5, 100)
            Dh = np.random.uniform(0.0001, 0.1)
            X.append([Re, Pr, AR, Dh])
            y.append(1)  # transitional
        
        # Turbulent k-epsilon samples (4000 < Re < 50000, general cases)
        for _ in range(300):
            Re = np.random.uniform(4000, 50000)
            Pr = np.random.uniform(0.7, 15)
            AR = np.random.uniform(10, 100)
            Dh = np.random.uniform(0.001, 0.1)
            X.append([Re, Pr, AR, Dh])
            y.append(2)  # k-epsilon
        
        # Turbulent k-omega SST samples (4000 < Re, near-wall flows)
        for _ in range(300):
            Re = np.random.uniform(4000, 100000)
            Pr = np.random.uniform(0.7, 15)
            AR = np.random.uniform(5, 50)  # Lower AR = near-wall effects
            Dh = np.random.uniform(0.0001, 0.01)  # Smaller channels
            X.append([Re, Pr, AR, Dh])
            y.append(3)  # k-omega SST
        
        X = np.array(X)
        y = np.array(y)
        
        # Train the model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        self.trained = True
    
    def predict_regime(
        self,
        reynolds_number: float,
        prandtl_number: float,
        aspect_ratio: float = 20.0,
        hydraulic_diameter: float = 0.01
    ) -> Dict[str, any]:
        """
        Predict flow regime and recommend turbulence model.
        
        Parameters
        ----------
        reynolds_number : float
            Reynolds number (Re = ρVDh/μ)
        prandtl_number : float
            Prandtl number (Pr = μcp/k)
        aspect_ratio : float, optional
            Channel aspect ratio (L/H), default=20
        hydraulic_diameter : float, optional
            Hydraulic diameter in meters, default=0.01
        
        Returns
        -------
        dict
            Dictionary containing:
            - regime: str ('laminar', 'transitional', 'turbulent')
            - turbulence_model: str ('none', 'k-epsilon', 'k-omega-sst')
            - confidence: float (0-1)
            - recommendations: list of str
        """
        # Rule-based fallback
        if not SKLEARN_AVAILABLE or not self.trained:
            return self._rule_based_prediction(
                reynolds_number, prandtl_number, aspect_ratio, hydraulic_diameter
            )
        
        # ML-based prediction
        X = np.array([[reynolds_number, prandtl_number, aspect_ratio, hydraulic_diameter]])
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probability
        pred = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        confidence = float(proba[pred])
        
        # Map prediction to regime
        regime_map = {
            0: ('laminar', 'none'),
            1: ('transitional', 'k-epsilon'),
            2: ('turbulent', 'k-epsilon'),
            3: ('turbulent', 'k-omega-sst')
        }
        
        regime, turb_model = regime_map[pred]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            reynolds_number, regime, turb_model, aspect_ratio, hydraulic_diameter
        )
        
        return {
            'regime': regime,
            'turbulence_model': turb_model,
            'confidence': confidence,
            'reynolds_number': reynolds_number,
            'prandtl_number': prandtl_number,
            'recommendations': recommendations
        }
    
    def _rule_based_prediction(
        self, Re: float, Pr: float, AR: float, Dh: float
    ) -> Dict[str, any]:
        """Fallback rule-based prediction when ML unavailable."""
        if Re < self.re_laminar_max:
            regime = 'laminar'
            turb_model = 'none'
            confidence = 0.95
        elif Re < self.re_transitional_max:
            regime = 'transitional'
            turb_model = 'k-epsilon'
            confidence = 0.70
        else:
            regime = 'turbulent'
            # Choose model based on geometry
            if Dh < 0.001 or AR < 30:
                turb_model = 'k-omega-sst'  # Better for near-wall
                confidence = 0.85
            else:
                turb_model = 'k-epsilon'  # General purpose
                confidence = 0.90
        
        recommendations = self._generate_recommendations(Re, regime, turb_model, AR, Dh)
        
        return {
            'regime': regime,
            'turbulence_model': turb_model,
            'confidence': confidence,
            'reynolds_number': Re,
            'prandtl_number': Pr,
            'recommendations': recommendations
        }
    
    def _generate_recommendations(
        self, Re: float, regime: str, turb_model: str, AR: float, Dh: float
    ) -> List[str]:
        """Generate expert recommendations based on flow conditions."""
        recommendations = []
        
        if regime == 'laminar':
            recommendations.append("✓ Laminar flow: expect parabolic velocity profile")
            recommendations.append("✓ No turbulence model needed")
            if Re > 1500:
                recommendations.append("⚠ Approaching transition, flow may be unstable")
        
        elif regime == 'transitional':
            recommendations.append("⚠ Transitional flow: results may be sensitive to setup")
            recommendations.append("⚠ Consider running both laminar and turbulent cases")
            recommendations.append(f"✓ Recommended model: {turb_model}")
            recommendations.append("⚠ Increase mesh resolution near walls")
        
        elif regime == 'turbulent':
            recommendations.append(f"✓ Turbulent flow: using {turb_model} model")
            if turb_model == 'k-omega-sst':
                recommendations.append("✓ k-ω SST excellent for near-wall flows")
                recommendations.append("✓ Good for microchannels and boundary layers")
            else:
                recommendations.append("✓ k-ε model suitable for general turbulent flows")
            
            if Dh < 0.001:
                recommendations.append("⚠ Microchannel detected: ensure fine mesh (y+ < 1)")
            
            if Re > 50000:
                recommendations.append("⚠ High Reynolds number: ensure sufficient iterations")
        
        # Mesh recommendations
        if AR > 50:
            recommendations.append("⚠ High aspect ratio: may need longer domain for development")
        
        return recommendations


class AIConvergenceMonitor:
    """
    AI-powered convergence monitoring and prediction.
    
    Tracks residual history, predicts divergence, and suggests parameter adjustments.
    """
    
    def __init__(self, window_size: int = 50):
        """
        Initialize convergence monitor.
        
        Parameters
        ----------
        window_size : int
            Number of iterations to track for trend analysis
        """
        self.window_size = window_size
        self.residual_history = []
        self.iteration_count = 0
        self.divergence_predicted = False
        self.convergence_predicted = False
    
    def update(self, residual: float) -> Dict[str, any]:
        """
        Update monitor with new residual value.
        
        Parameters
        ----------
        residual : float
            Current residual value
        
        Returns
        -------
        dict
            Status dictionary with predictions and recommendations
        """
        self.residual_history.append(residual)
        self.iteration_count += 1
        
        # Keep only recent history
        if len(self.residual_history) > self.window_size:
            self.residual_history.pop(0)
        
        # Analyze convergence
        status = self._analyze_convergence()
        
        return status
    
    def _analyze_convergence(self) -> Dict[str, any]:
        """Analyze residual history and predict convergence behavior."""
        if len(self.residual_history) < 10:
            return {
                'status': 'initializing',
                'converging': None,
                'diverging': False,
                'oscillating': False,
                'recommendations': ['Collecting data...'],
                'confidence': 0.0
            }
        
        residuals = np.array(self.residual_history[-self.window_size:])
        
        # Check for divergence (exponential growth)
        if len(residuals) >= 20:
            recent = residuals[-10:]
            older = residuals[-20:-10]
            ratio = np.mean(recent) / np.mean(older)
            
            if ratio > 2.0:  # Doubling
                self.divergence_predicted = True
                return {
                    'status': 'diverging',
                    'converging': False,
                    'diverging': True,
                    'oscillating': False,
                    'recommendations': [
                        '⚠ DIVERGENCE DETECTED!',
                        '→ Reduce relaxation factors by 50%',
                        '→ Check boundary conditions',
                        '→ Reduce time step if unsteady'
                    ],
                    'confidence': 0.95,
                    'growth_rate': ratio
                }
        
        # Check for oscillations
        if len(residuals) >= 20:
            # Look for periodic behavior
            diffs = np.diff(residuals[-20:])
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            
            if sign_changes > 10:  # Many direction changes
                return {
                    'status': 'oscillating',
                    'converging': False,
                    'diverging': False,
                    'oscillating': True,
                    'recommendations': [
                        '⚠ Oscillations detected',
                        '→ Reduce relaxation factors (try 0.3-0.5)',
                        '→ Increase under-relaxation',
                        '→ Check for physical instabilities'
                    ],
                    'confidence': 0.80
                }
        
        # Check for convergence (exponential decay)
        if len(residuals) >= 20:
            # Fit exponential decay
            x = np.arange(len(residuals))
            log_res = np.log(residuals + 1e-16)
            
            # Linear fit to log(residual) = mx + b
            slope = np.polyfit(x, log_res, 1)[0]
            
            if slope < -0.05:  # Decaying
                # Estimate remaining iterations
                current_res = residuals[-1]
                target_res = 1e-6
                
                if current_res > target_res:
                    iters_remaining = int(np.log(target_res / current_res) / slope)
                else:
                    iters_remaining = 0
                
                self.convergence_predicted = True
                return {
                    'status': 'converging',
                    'converging': True,
                    'diverging': False,
                    'oscillating': False,
                    'decay_rate': float(slope),
                    'estimated_iterations_remaining': max(0, iters_remaining),
                    'recommendations': [
                        f'✓ Converging steadily (rate: {slope:.4f})',
                        f'✓ Estimated {max(0, iters_remaining)} iterations remaining'
                    ],
                    'confidence': 0.85
                }
            elif slope < 0:  # Slow decay
                return {
                    'status': 'slow_convergence',
                    'converging': True,
                    'diverging': False,
                    'oscillating': False,
                    'decay_rate': float(slope),
                    'recommendations': [
                        '⚠ Slow convergence detected',
                        '→ Consider increasing relaxation factors',
                        '→ Check mesh quality',
                        '→ May need more iterations'
                    ],
                    'confidence': 0.70
                }
        
        # Stalled convergence
        if len(residuals) >= 30:
            recent_std = np.std(residuals[-20:])
            recent_mean = np.mean(residuals[-20:])
            
            if recent_std / recent_mean < 0.1:  # Flat
                return {
                    'status': 'stalled',
                    'converging': False,
                    'diverging': False,
                    'oscillating': False,
                    'recommendations': [
                        '⚠ Convergence stalled',
                        '→ May have reached best possible solution',
                        '→ Check if residual is acceptable',
                        '→ Consider adjusting tolerance'
                    ],
                    'confidence': 0.75
                }
        
        # Default: still progressing
        return {
            'status': 'progressing',
            'converging': None,
            'diverging': False,
            'oscillating': False,
            'recommendations': ['Solving... monitor for trends'],
            'confidence': 0.50
        }
    
    def reset(self):
        """Reset the monitor for a new solve."""
        self.residual_history = []
        self.iteration_count = 0
        self.divergence_predicted = False
        self.convergence_predicted = False


class AISolverParameterRecommender:
    """
    AI-powered solver parameter recommendations.
    
    Suggests optimal mesh size, relaxation factors, and solver settings
    based on problem characteristics.
    """
    
    def recommend_parameters(
        self,
        reynolds_number: float,
        prandtl_number: float,
        domain_length: float,
        domain_height: float,
        turbulence_model: str = 'none'
    ) -> Dict[str, any]:
        """
        Recommend optimal solver parameters.
        
        Parameters
        ----------
        reynolds_number : float
            Reynolds number
        prandtl_number : float
            Prandtl number
        domain_length : float
            Domain length (m)
        domain_height : float
            Domain height (m)
        turbulence_model : str
            Turbulence model ('none', 'k-epsilon', 'k-omega-sst')
        
        Returns
        -------
        dict
            Recommended parameters and justifications
        """
        aspect_ratio = domain_length / domain_height
        
        # Mesh recommendations
        mesh_params = self._recommend_mesh(
            reynolds_number, aspect_ratio, domain_length, domain_height, turbulence_model
        )
        
        # Relaxation factor recommendations
        relaxation_params = self._recommend_relaxation(reynolds_number, turbulence_model)
        
        # Solver settings
        solver_params = self._recommend_solver_settings(reynolds_number, turbulence_model)
        
        return {
            'mesh': mesh_params,
            'relaxation': relaxation_params,
            'solver': solver_params,
            'warnings': self._generate_warnings(reynolds_number, aspect_ratio, domain_height)
        }
    
    def _recommend_mesh(self, Re, AR, L, H, turb_model):
        """Recommend mesh parameters."""
        # Base mesh on Reynolds number and geometry
        if Re < 100:
            nx_base = 30
            ny_base = 20
        elif Re < 1000:
            nx_base = 50
            ny_base = 30
        elif Re < 10000:
            nx_base = 80
            ny_base = 50
        else:
            nx_base = 120
            ny_base = 70
        
        # Adjust for aspect ratio
        nx = int(nx_base * min(2.0, AR / 20))
        ny = ny_base
        
        # Turbulence models need finer mesh
        if turb_model in ['k-epsilon', 'k-omega-sst']:
            nx = int(nx * 1.3)
            ny = int(ny * 1.5)
        
        # k-omega SST needs very fine near-wall
        if turb_model == 'k-omega-sst':
            ny = int(ny * 1.3)
        
        return {
            'nx': nx,
            'ny': ny,
            'justification': [
                f'Re={Re:.0f} → Base mesh: {nx_base}×{ny_base}',
                f'Aspect ratio {AR:.1f} → Adjusted nx to {nx}',
                f'Turbulence model: {turb_model} → Final: {nx}×{ny}'
            ]
        }
    
    def _recommend_relaxation(self, Re, turb_model):
        """Recommend relaxation factors."""
        if Re < 100:
            alpha_u = 0.7
            alpha_p = 0.3
        elif Re < 1000:
            alpha_u = 0.5
            alpha_p = 0.2
        elif Re < 10000:
            alpha_u = 0.4
            alpha_p = 0.2
        else:
            alpha_u = 0.3
            alpha_p = 0.2
        
        # Turbulence needs more relaxation
        if turb_model in ['k-epsilon', 'k-omega-sst']:
            alpha_u *= 0.8
            alpha_p *= 0.9
        
        return {
            'alpha_u': alpha_u,
            'alpha_p': alpha_p,
            'justification': [
                f'Re={Re:.0f} → Velocity relax: {alpha_u}',
                f'Re={Re:.0f} → Pressure relax: {alpha_p}',
                'Higher Re needs more relaxation for stability'
            ]
        }
    
    def _recommend_solver_settings(self, Re, turb_model):
        """Recommend solver settings."""
        if Re < 1000:
            max_iters = 1000
            tolerance = 1e-6
        elif Re < 10000:
            max_iters = 2000
            tolerance = 1e-5
        else:
            max_iters = 3000
            tolerance = 1e-5
        
        if turb_model in ['k-epsilon', 'k-omega-sst']:
            max_iters = int(max_iters * 1.5)
        
        return {
            'max_iterations': max_iters,
            'tolerance': tolerance,
            'justification': [
                f'Re={Re:.0f} → Max iterations: {max_iters}',
                f'Tolerance: {tolerance:.0e}',
                'Higher Re and turbulence need more iterations'
            ]
        }
    
    def _generate_warnings(self, Re, AR, H):
        """Generate warnings about potential issues."""
        warnings = []
        
        if Re > 100000:
            warnings.append('⚠ Very high Re: solution may be challenging')
        
        if AR > 100:
            warnings.append('⚠ Very high aspect ratio: long development length')
        
        if H < 0.0001:
            warnings.append('⚠ Microchannel: special attention to near-wall effects')
        
        if Re > 2000 and Re < 4000:
            warnings.append('⚠ Transitional flow: results may be sensitive')
        
        return warnings


# Convenience functions
def classify_flow_regime(Re: float, Pr: float, AR: float = 20.0, Dh: float = 0.01) -> Dict:
    """Quick function to classify flow regime."""
    classifier = AIFlowRegimeClassifier()
    return classifier.predict_regime(Re, Pr, AR, Dh)


def recommend_solver_parameters(
    Re: float, Pr: float, L: float, H: float, turb_model: str = 'none'
) -> Dict:
    """Quick function to get solver parameter recommendations."""
    recommender = AISolverParameterRecommender()
    return recommender.recommend_parameters(Re, Pr, L, H, turb_model)
