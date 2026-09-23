"""Tests for the scorecard's per-seed loading and learning-curve attribution.

The AUC column of README section 3.9 is computed from tensorboard directories, and the
original code paired a curve with a config by creation order (the k-th PPO_k directory was
the k-th expected cell). That holds only for one uninterrupted sequential sweep: a resumed
or split run shifts every pairing and the table would attribute curves to the wrong
architecture without any error. These tests pin the name-based attribution instead.
"""

import json
import os
import sys

import pytest

from scorecard_analysis import load_per_seed, tb_run_cells

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["maze_ppo", "maze_icm", "maze_rnd", "maze_ngu"]


def _cell(seed, value=2.5):
    return {"seed": seed, "mean_reward": value, "std_reward": 0.1}


def test_named_runs_are_attributed_by_their_directory_name(tmp_path):
    for name in ["maze_ppo_seed42", "heist_rnd_seed43", "maze_icm_seed44_2", "maze_icm_seed44"]:
        (tmp_path / name).mkdir()
    runs, notes = tb_run_cells(str(tmp_path), ["maze_ppo", "maze_icm"])
    keys = [os.path.basename(d) for _k, d in runs]
    assert keys == ["maze_icm_seed44_2", "maze_ppo_seed42"], keys
    assert any("not attributable" in n for n in notes), notes


def test_unnamed_runs_fall_back_to_positional_order(tmp_path):
    for i in (1, 2):
        (tmp_path / f"PPO_{i}").mkdir()
    runs, notes = tb_run_cells(str(tmp_path), ORDER)
    assert [k for k, _d in runs] == ORDER
    assert any("positional" in n for n in notes), notes


def test_missing_directory_yields_no_runs(tmp_path):
    runs, _notes = tb_run_cells(str(tmp_path / "absent"), ORDER)
    assert runs == []


def test_split_sweep_dirs_merge_and_protocol_metadata_is_not_a_config(tmp_path, capsys):
    root = tmp_path / "logs"
    for run, contents in [
        ("maze_heist_maze_a", {"maze_ppo": [_cell(42)], "_protocol": {"timesteps": 100000}}),
        ("maze_heist_maze_b", {"maze_rnd": [_cell(42), _cell(43, 3.5)]}),
        ("maze_heist_maze_c", {"maze_rnd": [_cell(44)]}),
    ]:
        d = root / "logs_maze_heist" / run
        d.mkdir(parents=True)
        (d / "comparison_results.json").write_text(json.dumps(contents), encoding="utf-8")
    data, sources = load_per_seed(BASE, str(root))
    captured = capsys.readouterr().out
    assert "no new_archs run JSONs" in captured
    assert "_protocol" not in data
    assert data["maze_ppo"] == [2.5]
    assert data["maze_rnd"] == [2.5, 3.5, 2.5]
    assert sources["maze_ppo"].startswith("maze_heist:")
    # the frozen per-seed records stand in for configs the logs no longer carry
    assert "coinrun_classic" in data
    assert sources["coinrun_classic"] == "results/legacy_records.json"


def test_double_measured_cell_warns_and_keeps_the_later_value(tmp_path, capsys):
    root = tmp_path / "logs"
    for run, value in [("run_a", 1.0), ("run_b", 9.0)]:
        d = root / "logs_maze_heist" / f"maze_heist_{run}"
        d.mkdir(parents=True)
        (d / "comparison_results.json").write_text(
            json.dumps({"maze_icm": [_cell(42, value)]}), encoding="utf-8")
    data, _sources = load_per_seed(BASE, str(root))
    assert data["maze_icm"] == [9.0]
    assert "measured twice" in capsys.readouterr().out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
