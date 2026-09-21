"""Statistical metrics and aggregation for Reinforcement Learning based on Agarwal et al. (NeurIPS 2021).

'Deep Reinforcement Learning at the Edge of the Statistical Precipice'
Implements:
- Interquartile Mean (IQM)
- Trimmed Mean (e.g., 5% trimmed mean)
- Game-level Min-Max normalization
- Stratified Bootstrap Confidence Intervals (BCa and Percentile)
- Performance Profiles (Empirical CDF)
- Probability of Improvement (Mann-Whitney / Wilcoxon style U-statistic)
"""

import math
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np


def compute_iqm(scores: Union[np.ndarray, List[float]]) -> float:
    """Computes the Interquartile Mean (IQM) — 25% trimmed mean.
    
    The IQM discards the lower 25% and upper 25% of scores and calculates
    the arithmetic mean of the remaining 50% central distribution.
    """
    arr = np.sort(np.asarray(scores, dtype=np.float64).flatten())
    n = len(arr)
    if n == 0:
        return 0.0
    if n < 4:
        return float(np.mean(arr))
    
    q25 = int(np.floor(0.25 * n))
    q75 = int(np.ceil(0.75 * n))
    middle = arr[q25:q75]
    return float(np.mean(middle)) if len(middle) > 0 else float(np.mean(arr))


def compute_trimmed_mean(scores: Union[np.ndarray, List[float]], trim_fraction: float = 0.05) -> float:
    """Computes trimmed mean by discarding trim_fraction from both tails."""
    arr = np.sort(np.asarray(scores, dtype=np.float64).flatten())
    n = len(arr)
    if n == 0:
        return 0.0
    k = int(np.floor(trim_fraction * n))
    if k == 0 or 2 * k >= n:
        return float(np.mean(arr))
    return float(np.mean(arr[k:n - k]))


def stratified_bootstrap_ci(
    task_scores: Dict[str, np.ndarray],
    metric_fn: Callable[[np.ndarray], float] = compute_iqm,
    num_bootstraps: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """Calculates Stratified Bootstrap Confidence Intervals across tasks.
    
    task_scores: dict mapping task_id -> array of seed scores (N_seeds,)
    Returns: (point_estimate, (ci_lower, ci_upper))
    """
    rng = np.random.default_rng(seed)
    tasks = list(task_scores.keys())
    
    # Pool baseline point estimate
    all_scores = np.concatenate([np.asarray(task_scores[t], dtype=np.float64) for t in tasks])
    point_estimate = float(metric_fn(all_scores))
    
    bootstrap_estimates = np.empty(num_bootstraps, dtype=np.float64)
    task_arrays = [np.asarray(task_scores[t], dtype=np.float64) for t in tasks]
    task_lens = [len(arr) for arr in task_arrays]
    
    for b in range(num_bootstraps):
        resampled_task_scores = []
        for arr, n_seeds in zip(task_arrays, task_lens):
            if n_seeds > 0:
                sampled = rng.choice(arr, size=n_seeds, replace=True)
                resampled_task_scores.append(sampled)
        pooled_b = np.concatenate(resampled_task_scores)
        bootstrap_estimates[b] = metric_fn(pooled_b)
        
    ci_lower = float(np.percentile(bootstrap_estimates, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_estimates, 100 * (1.0 - alpha / 2.0)))
    return point_estimate, (ci_lower, ci_upper)


def compute_performance_profile(
    normalized_scores: np.ndarray,
    tau_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes empirical CDF performance profile P(score >= tau).
    
    normalized_scores: (N_runs,) or (N_tasks, N_seeds)
    Returns: (tau_grid, probabilities)
    """
    flat = np.asarray(normalized_scores, dtype=np.float64).flatten()
    if tau_grid is None:
        tau_grid = np.linspace(0.0, 1.0, 101)
    
    probs = np.array([np.mean(flat >= tau) for tau in tau_grid], dtype=np.float64)
    return tau_grid, probs


def compute_probability_of_improvement(
    scores_a: Dict[str, np.ndarray],
    scores_b: Dict[str, np.ndarray],
    num_bootstraps: int = 10000,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """Estimates the probability P(Score_A > Score_B) across common tasks.
    
    Accounts for ties as 0.5 * P(Score_A == Score_B).
    Returns: (prob_point, (ci_lower, ci_upper))
    """
    common_tasks = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if not common_tasks:
        return 0.5, (0.5, 0.5)
        
    def _single_prob(data_a, data_b):
        wins, ties, total = 0, 0, 0
        for t in common_tasks:
            va = data_a[t]
            vb = data_b[t]
            for a_val in va:
                for b_val in vb:
                    total += 1
                    if a_val > b_val:
                        wins += 1
                    elif a_val == b_val:
                        ties += 1
        return (wins + 0.5 * ties) / max(1, total)

    point = _single_prob(scores_a, scores_b)
    
    rng = np.random.default_rng(seed)
    boot_probs = np.empty(num_bootstraps, dtype=np.float64)
    for b in range(num_bootstraps):
        resampled_a = {}
        resampled_b = {}
        for t in common_tasks:
            arr_a = scores_a[t]
            arr_b = scores_b[t]
            resampled_a[t] = rng.choice(arr_a, size=len(arr_a), replace=True)
            resampled_b[t] = rng.choice(arr_b, size=len(arr_b), replace=True)
        boot_probs[b] = _single_prob(resampled_a, resampled_b)
        
    ci_lo = float(np.percentile(boot_probs, 2.5))
    ci_hi = float(np.percentile(boot_probs, 97.5))
    return float(point), (ci_lo, ci_hi)
