"""Sanity checks for stats.py with known values (no GPU).

mean_ci([1..5]) = 3 ± t4*std/sqrt5; cohen_d = -3; triangle auc = 1.0.
Usage: python -m jax_port.tests.test_stats
"""

import math


def test_stats():
    from jax_port.stats import auc_norm, cohen_d, mean_ci, rank_cells
    m, s, n, (lo, hi) = mean_ci([1, 2, 3, 4, 5])
    assert m == 3.0 and n == 5
    assert abs(s - math.sqrt(2.5)) < 1e-9
    assert abs((hi - lo) / 2 - 2.776 * math.sqrt(2.5) / math.sqrt(5)) < 1e-9
    assert abs(cohen_d([1, 2, 3], [4, 5, 6]) + 3.0) < 1e-9
    assert abs(auc_norm([{"steps": 0, "ret20": 0.0},
                         {"steps": 100, "ret20": 2.0}], 100) - 1.0) < 1e-9
    # CI overlap: separated cells must report overlap=False, near-identical ones True.
    # The formula used before 23/09/2026 reported overlap=False for both.
    separated = rank_cells({"a": [10.0, 10.1, 9.9, 10.05, 10.02],
                            "b": [0.0, 0.1, -0.1, 0.05, 0.02]})
    assert separated["top1_vs_top2"]["overlap"] is False, separated["top1_vs_top2"]
    indistinguishable = rank_cells({"a": [1.0, 1.2, 0.8, 1.1, 0.9],
                                    "b": [1.05, 0.9, 1.1, 1.0, 0.95]})
    assert indistinguishable["top1_vs_top2"]["overlap"] is True, \
        indistinguishable["top1_vs_top2"]
    r = rank_cells({"a": [1.0, 1.1, 0.9, 1.0, 1.0],
                    "b": [0.0, 0.1, -0.1, 0.0, 0.05]})
    assert r["ranking"][0]["cell"] == "a"
    assert r["top1_vs_top2"]["cohen_d"] > 5  # obvious separation
    return True


if __name__ == "__main__":
    test_stats()
    print("STATS_OK")
