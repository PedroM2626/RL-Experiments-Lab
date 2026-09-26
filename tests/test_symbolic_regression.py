"""Tests for the symbolic-regression / system-identification study.

Two things are pinned here. The first is the property that makes the study valid at all: the
feature matrix must be built only from the protocol, never from the measured return - if a
return ever leaked into X, R^2 would be an artefact rather than a result. The second is the
gplearn/scikit-learn interface, which broke in practice when scikit-learn 1.7 removed the
helper gplearn 0.4.2 calls: a tiny real fit in CI fails loudly on that drift instead of
failing in the middle of a study run.
"""

import argparse
import copy
import math

import numpy as np
import pytest

import symbolic_regression as SR


def _rows():
    return [{"suite": "main", "cfg": "classic", "game": "bossfight", "seed": 42,
             "budget": 100000.0, "ret": 0.5, "family": "cnn_classic", "hard": 0.0,
             "memory": 0.0}]


def test_load_rows_covers_every_mapped_cfg_and_drops_budgetless_cells():
    rows = SR.load_rows()
    assert len(rows) > 800
    assert all({"suite", "cfg", "game", "seed", "budget", "ret", "family"} <= set(r)
               for r in rows)
    assert all(r["budget"] and r["budget"] > 0 for r in rows), "hrl cells must not enter"
    assert "hrl" not in {r["suite"] for r in rows}


def test_unmapped_cfg_raises_instead_of_being_silently_dropped(monkeypatch):
    monkeypatch.setattr(SR, "FAMILY_OF", {k: v for k, v in SR.FAMILY_OF.items()
                                          if k != "transformer_xl"})
    with pytest.raises(KeyError, match="transformer_xl"):
        SR.load_rows()


def test_design_matrix_never_reads_the_measured_return():
    rows = _rows() * 4
    for i, r in enumerate(rows):
        r["ret"] = float(i)
    X_before, y, _g = SR.design(rows)
    for r in rows:
        r["ret"] += 1000.0
    X_after, y_after, _g = SR.design(rows)
    np.testing.assert_array_equal(X_before, X_after)
    assert list(y_after) == [r["ret"] for r in rows]
    assert X_before.shape[1] == 3            # log10N, hard, memory


def test_game_dummies_only_add_columns():
    rows = _rows() + [{**_rows()[0], "game": "starpilot"}]
    X_small, _y, _g = SR.design(rows)
    X_full, _y2, _g2 = SR.design(rows, game_dummies=True)
    assert X_full.shape[1] > X_small.shape[1]
    np.testing.assert_array_equal(X_full[:, :3], X_small)


def test_expression_renames_known_indices_and_marks_unknown_ones():
    class _P:
        _program = "add(mul(X0, X1), sub(X2, 0.5))"

        def __bool__(self):
            return True

    assert SR.expression(_P(), ["N", "hard"]) == "add(mul(N, hard), sub(X2?, 0.5))"


def test_to_latex_handles_the_protected_functions():
    tex = SR.to_latex("add(mul(log10N, 0.5), bpow(hard, 2))", ["log10N", "hard"])
    assert tex and "log10N" in tex


def test_wilson_ci_is_bounded_and_widens_for_fewer_trials():
    lo, hi = SR.wilson_ci(50, 100)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    narrow = SR.wilson_ci(500, 1000)
    wide = SR.wilson_ci(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])
    assert SR.wilson_ci(0, 0) is None


def test_scores_recovers_a_perfect_fit_and_a_null_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert SR.scores(y, y)["r2"] == pytest.approx(1.0)
    null = SR.scores(y, np.full_like(y, y.mean()))
    assert null["r2"] == pytest.approx(0.0)
    assert null["mae"] == pytest.approx(1.0)     # |1,1,1,1| deviations about the mean 2.5


def test_flat_law_is_reported_as_not_identifiable():
    rows = []
    for game in ("coinrun",):
        for b in (100000.0, 250000.0, 500000.0):
            for seed in (42, 43, 44, 45, 46):
                rows.append({**_rows()[0], "game": game, "budget": b, "seed": seed,
                             "ret": 1.0})          # perfectly flat in budget
    out = {}
    SR.p3_inversion(rows, out)
    v = out["P3_budget_inversion"]["coinrun"]
    assert v.get("identifiable") is False or "skipped" in v


def test_paired_bootstrap_is_zero_when_a_is_identical_to_b():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert SR.paired_bootstrap(y, y, y) == 0.0


def test_a_real_gplearn_fit_runs_against_the_pinned_sklearn():
    """The search must actually execute: API drift between gplearn and sklearn is the
    failure mode that would otherwise only appear mid-study."""
    args = argparse.Namespace(pop=60, gens=3, parsimony=0.005, seed=0)
    X = np.array([[math.log10(1e5) + i * 0.01, 0.0, float(i % 2)] for i in range(60)])
    y = 2.0 * X[:, 0] - X[:, 2]
    m = SR.sr_model(args)
    m.fit(X, y)
    assert m._program.length_ > 0
    assert np.allclose(m.predict(X[:5]), m.predict(X[:5]))
    expr = SR.expression(m, ["log10N", "hard", "memory"])
    assert "X[" not in expr and "X0" not in expr
