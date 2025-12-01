"""
Uncertainty Quantification (UQ) utilities for BKPS NFL Thermal Pro

Features:
- Monte Carlo sampling for input uncertainty propagation
- Confidence intervals (68/95/99%) and summary stats
- Tornado chart data preparation
- Optional SALib integration (Sobol, Morris) passthrough
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple, List, Optional

try:
    from SALib.sample import saltelli, morris as morris_sampler
    from SALib.analyze import sobol as sobol_analyze, morris as morris_analyze
    HAS_SALIB = True
except Exception:
    HAS_SALIB = False


@dataclass
class Distribution:
    name: str  # 'normal' | 'uniform' | 'lognormal'
    params: Tuple[float, float]  # (mu, sigma) or (low, high)


def sample_distribution(dist: Distribution, n: int) -> np.ndarray:
    if dist.name == 'normal':
        mu, sigma = dist.params
        return np.random.normal(mu, sigma, size=n)
    if dist.name == 'uniform':
        low, high = dist.params
        return np.random.uniform(low, high, size=n)
    if dist.name == 'lognormal':
        mu, sigma = dist.params
        return np.random.lognormal(mu, sigma, size=n)
    raise ValueError(f"Unsupported distribution: {dist.name}")


def monte_carlo(func: Callable[..., float],
                inputs: Dict[str, Distribution],
                n_samples: int = 5000,
                random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Run Monte Carlo propagation.
    Returns dict with samples, mean, std, and 95% CI.
    """
    if random_state is not None:
        np.random.seed(random_state)
    names = list(inputs.keys())
    samples = {k: sample_distribution(inputs[k], n_samples) for k in names}
    outputs = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        kwargs = {k: samples[k][i] for k in names}
        outputs[i] = func(**kwargs)
    mean = float(np.mean(outputs))
    std = float(np.std(outputs, ddof=1))
    p2_5, p97_5 = np.percentile(outputs, [2.5, 97.5])
    return {
        'inputs': samples,
        'outputs': outputs,
        'mean': mean,
        'std': std,
        'ci95': (float(p2_5), float(p97_5))
    }


def summarize_ci(arr: np.ndarray, levels: Tuple[float, ...] = (68, 95, 99)) -> Dict[str, Tuple[float,float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for L in levels:
        alpha = (100 - L) / 2
        lo, hi = np.percentile(arr, [alpha, 100 - alpha])
        out[f"ci{int(L)}"] = (float(lo), float(hi))
    return out


def tornado_from_mc(inputs: Dict[str, np.ndarray], outputs: np.ndarray) -> List[Tuple[str, float]]:
    """Compute simple correlation-based importance ranking for tornado charts."""
    ranks: List[Tuple[str, float]] = []
    y = outputs
    y_std = np.std(y)
    if y_std == 0:
        return [(k, 0.0) for k in inputs]
    for k, x in inputs.items():
        x_std = np.std(x)
        if x_std == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        ranks.append((k, abs(corr)))
    ranks.sort(key=lambda t: t[1], reverse=True)
    return ranks


# Optional SALib wrappers

def sobol_sensitivity(problem: Dict[str, Any],
                      model: Callable[[np.ndarray], np.ndarray],
                      n_samples: int = 1000,
                      calc_second_order: bool = False) -> Dict[str, Any]:
    if not HAS_SALIB:
        raise ImportError("SALib not installed. pip install SALib")
    param_values = saltelli.sample(problem, n_samples, calc_second_order=calc_second_order)
    Y = model(param_values)
    return sobol_analyze.analyze(problem, Y, calc_second_order=calc_second_order)


def morris_screening(problem: Dict[str, Any],
                     model: Callable[[np.ndarray], np.ndarray],
                     n_levels: int = 4,
                     n_trajectories: int = 10) -> Dict[str, Any]:
    if not HAS_SALIB:
        raise ImportError("SALib not installed. pip install SALib")
    param_values = morris_sampler.sample(problem, N=n_trajectories, num_levels=n_levels)
    Y = model(param_values)
    return morris_analyze.analyze(problem, param_values, Y)
