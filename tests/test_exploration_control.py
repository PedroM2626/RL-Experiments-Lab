"""Tests for the beta=0 control comparison in jax_port/exploration_control_report.py.

The port published 40 exploration cells where ICM/RND/NGU differ from their PPO control, with
nothing recorded about whether the intrinsic reward ever reached the policy. The control design
only answers that question if the bookkeeping is honoured: a "control" cell that actually
injected a bonus is just another arm, and a cell from before the bookkeeping proves nothing.
Both of those must be rejected rather than averaged in, which is what these cases check.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_port.exploration_control_report import build, control_cells

GAMES = ("maze", "heist")


def _ctrl(tmp_path):
    d = os.path.join(str(tmp_path), "cells")
    os.makedirs(d, exist_ok=True)
    return d


def _stats(kind, effective=0.0, bonus_mean=0.05, beta=0.0):
    return {"kind": kind, "beta": beta, "steps": 512, "bonus_cells": 4096,
            "bonus_mean": bonus_mean, "bonus_max": bonus_mean * 3,
            "effective_bonus": effective}


def _write_control(dirpath, game, cfg, value, stats="ok", seed=42):
    st = {"ok": _stats(cfg),
          "injected": _stats(cfg, effective=1e-4),
          "missing": None,
          "no_bonus": _stats(cfg, bonus_mean=0.0)}[stats]
    cell = {"game": game, "explore": cfg, "timesteps": 100000,
            "eval_unseen": {"mean": value}}
    if st is not None:
        cell["explore_stats"] = st
    with open(os.path.join(dirpath, f"{cfg}__{game}__seed{seed}__100k.json"), "w",
              encoding="utf-8") as f:
        json.dump(cell, f)


def _published(path, values):
    """values: {(game, cfg): [eval_unseen per seed]}"""
    cells = [{"game": g, "cfg": c, "eval_unseen": v, "seed": 42 + i,
              "timesteps": 100000, "curve_points": 3, "gen_gap": 0.0}
             for (g, c), vals in values.items() for i, v in enumerate(vals)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"exploration": cells}, f)
    return path


def _seeds(vals):
    return list(vals)


def test_effect_larger_than_the_drift_floor_is_attributable(tmp_path):
    ctrl = _ctrl(tmp_path)
    for seed, value in enumerate((2.4, 2.6, 2.5, 2.5, 2.5)):
        _write_control(ctrl, "maze", "rnd", value, seed=42 + seed)
    pub = _published(os.path.join(os.path.dirname(ctrl), "pub.json"), {
        ("maze", "ppo"): _seeds((2.0, 2.2, 2.1, 2.1, 2.1)),        # mean 2.1
        ("maze", "rnd"): _seeds((6.0, 6.2, 6.1, 6.1, 6.1)),        # mean 6.1, effect +4.0
    })
    rep = build(ctrl, pub)
    g = rep["groups"]["maze_rnd"]
    assert g["drift_floor"] == pytest.approx(0.4, abs=1e-6)
    assert g["published_effect"] == pytest.approx(4.0, abs=1e-6)
    assert g["attributable_to_mechanism"] is True
    assert g["effect_over_drift"] > 1


def test_effect_within_the_drift_floor_is_not_attributable(tmp_path):
    ctrl = _ctrl(tmp_path)
    for seed, value in enumerate((2.5, 2.3, 2.4, 2.4, 2.4)):
        _write_control(ctrl, "maze", "rnd", value, seed=42 + seed)   # control mean 2.4, drift 0.3
    pub = _published(os.path.join(os.path.dirname(ctrl), "pub.json"), {
        ("maze", "ppo"): _seeds((2.1, 2.1, 2.1, 2.1, 2.1)),
        ("maze", "rnd"): _seeds((2.3, 2.3, 2.3, 2.3, 2.3)),           # effect 0.2 < drift
    })
    g = build(ctrl, os.path.join(os.path.dirname(ctrl), "pub.json"))["groups"]["maze_rnd"]
    assert g["attributable_to_mechanism"] is False


def test_cells_that_injected_or_lack_bookkeeping_are_rejected(tmp_path):
    ctrl = str(tmp_path / "cells")
    os.makedirs(ctrl)
    _write_control(ctrl, "maze", "rnd", 2.0, stats="injected", seed=42)
    _write_control(ctrl, "maze", "rnd", 2.0, stats="missing", seed=43)
    _write_control(ctrl, "maze", "rnd", 2.0, stats="no_bonus", seed=44)
    _write_control(ctrl, "maze", "rnd", 2.0, stats="ok", seed=45)
    kept, _stats, bad = control_cells(ctrl)
    assert kept == {("maze", "rnd"): [2.0]}, "only the real control cell survives"
    assert len(bad) == 3
    reasons = " ".join(bad)
    assert "not usable as a control" in reasons
    assert "not a control arm" in reasons
    assert "computed nothing" in reasons


def test_groups_needing_all_three_sources_are_skipped_not_guessed(tmp_path):
    ctrl = _ctrl(tmp_path)
    _write_control(ctrl, "heist", "ngu", 1.0, seed=42)
    pub = _published(os.path.join(os.path.dirname(ctrl), "pub.json"), {("heist", "ppo"): [1.0]})
    rep = build(ctrl, os.path.join(os.path.dirname(ctrl), "pub.json"))
    assert "heist_ngu" not in rep["groups"]
    assert rep["groups"] == {}


def test_provenance_states_that_no_control_injected_anything(tmp_path):
    ctrl = _ctrl(tmp_path)
    for seed in range(5):
        _write_control(ctrl, "maze", "rnd", 2.0 + seed * 0.1, seed=42 + seed)
    pub = _published(os.path.join(os.path.dirname(ctrl), "pub.json"), {
        ("maze", "ppo"): [2.0] * 5, ("maze", "rnd"): [2.1] * 5})
    prov = build(ctrl, os.path.join(os.path.dirname(ctrl), "pub.json"))["_provenance"]
    assert prov["every_control_cell_injected_nothing"] is True
    assert prov["bonus_mean_by_kind"] == {"rnd": 0.05}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
