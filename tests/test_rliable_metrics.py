"""Tests for the RLiable aggregation helpers (rliable_metrics.py).

These lock in the semantics that were wrong before 23/09/2026: the aggregate metric
must be computed per game and then averaged across games, with bootstrap resampling
stratified inside each game.
"""

import numpy as np

from rliable_metrics import (
    NUM_BOOTSTRAPS,
    compute_iqm,
    compute_mean,
    compute_performance_profile,
    compute_performance_profile_ci,
    compute_probability_of_improvement,
    compute_trimmed_mean,
    stratified_bootstrap_ci,
)


def test_iqm_discards_quartiles():
    assert compute_iqm([0, 1, 2, 3, 4, 5, 6, 7]) == np.mean([2, 3, 4, 5])


def test_trimmed_mean_is_five_percent_by_default():
    scores = list(range(100))
    assert compute_trimmed_mean(scores) == np.mean(range(5, 95))


def test_aggregate_weights_games_equally_regardless_of_seed_count():
    # A 20-seed game must not outvote a 2-seed game: each game contributes once.
    small = np.array([0.0, 0.0])
    large = np.array([1.0] * 20)
    est, (lo, hi) = stratified_bootstrap_ci({"a": small, "b": large}, metric_fn=compute_mean)
    assert est == 0.5, est
    assert lo <= est <= hi


def test_bootstrap_ci_brackets_point_estimate_for_spread_data():
    rng = np.random.default_rng(0)
    tasks = {g: rng.normal(0.6, 0.2, size=5) for g in ["bossfight", "starpilot", "dodgeball"]}
    est, (lo, hi) = stratified_bootstrap_ci(tasks, metric_fn=compute_iqm, num_bootstraps=500)
    per_game = float(np.mean([compute_iqm(v) for v in tasks.values()]))
    assert abs(est - per_game) < 1e-9
    assert lo < est < hi
    assert (hi - lo) < 1.0


def test_empty_tasks_do_not_crash():
    est, (lo, hi) = stratified_bootstrap_ci({"a": np.array([])}, metric_fn=compute_iqm)
    assert (est, lo, hi) == (0.0, 0.0, 0.0)


def test_performance_profile_matches_brute_force():
    scores = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    taus, probs = compute_performance_profile(scores, np.array([0.0, 0.5, 0.6, 1.0]))
    brute = [np.mean(scores >= t) for t in taus]
    assert np.allclose(probs, brute)


def test_profile_ci_bands_contain_point_estimate():
    tasks = {g: np.linspace(0, 1, 5) for g in ["a", "b"]}
    taus = np.array([0.0, 0.25, 0.5, 1.0, 1.5])
    _, point, lo, hi = compute_performance_profile_ci(tasks, taus, num_bootstraps=200)
    assert np.all(lo <= point + 1e-12)
    assert np.all(point <= hi + 1e-12)
    assert point[0] == 1.0          # every run clears tau = 0
    assert point[-1] == 0.0         # no run clears tau above the maximum score
    assert np.all(np.diff(point) <= 1e-12)  # a profile is non-increasing in tau


def test_probability_of_improvement_is_antisymmetric():
    a = {"g": np.array([5.0, 6.0, 7.0])}
    b = {"g": np.array([1.0, 2.0, 3.0])}
    p_ab, _ = compute_probability_of_improvement(a, b, num_bootstraps=100)
    p_ba, _ = compute_probability_of_improvement(b, a, num_bootstraps=100)
    assert p_ab == 1.0 and p_ba == 0.0


def test_bootstrap_count_is_single_source_of_truth():
    assert NUM_BOOTSTRAPS == 10000
