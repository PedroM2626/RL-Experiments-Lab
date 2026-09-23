"""Guards the integrity checks of merge_exploration_remeasure.py.

The maze/heist exploration benchmark was published with 'ppo == rnd == ngu' ties that turned
out to be one PPO run trained three times through wrappers whose network geometry was wrong.
The only reason that was detectable after the fact is that the numbers were identical; a
re-measurement needs to refuse to publish an arm that did not actually explore, and these are
the checks that do it.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from merge_exploration_remeasure import load_runs, verify_bonus_fired

SEEDS = (42, 43, 44, 45, 46)
CONFIGS = [f"{g}_{a}" for g in ("maze", "heist") for a in ("ppo", "icm", "rnd", "ngu")]


def _cell(seed, value=2.0, bonus=True, arm="ppo"):
    """One measured cell; plain PPO has no wrapper, so it carries no bonus bookkeeping."""
    if arm == "ppo" or not bonus:
        return {"seed": seed, "mean_reward": value, "bonus": None}
    return {"seed": seed, "mean_reward": value,
            "bonus": {"steps": 100000, "bonus_applied": 100000, "intrinsic_mean": 0.004}}


def _write_run(root, name, payload, mtime=1000.0):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "comparison_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.utime(path, (mtime, mtime))
    return path


def _full_grid(bonus=True):
    grid = {}
    for k in CONFIGS:
        arm = k.split("_", 1)[1]
        grid[k] = [_cell(s, 2.0 + s % 3, bonus=bonus, arm=arm) for s in SEEDS]
    grid["_protocol"] = {"timesteps": 100000}
    return grid


def test_complete_grid_is_loaded_and_protocol_is_not_a_config(tmp_path):
    _write_run(str(tmp_path), "maze_heist_maze_ppo_icm_a", _full_grid())
    cells, sources, conflicts = load_runs(str(tmp_path))
    assert set(cells) == set(CONFIGS)
    assert all(len(cells[k]) == 5 for k in CONFIGS)
    assert "_protocol" not in sources
    assert conflicts == []


def test_incomplete_grid_is_refused(tmp_path, capsys):
    grid = _full_grid()
    grid["maze_rnd"] = grid["maze_rnd"][:4]  # seed 46 never finished
    _write_run(str(tmp_path), "maze_heist_maze_rnd_ngu_a", grid)
    with pytest.raises(SystemExit) as exc:
        load_runs(str(tmp_path))
    assert exc.value.code == 1
    assert "maze_rnd seed 46" in capsys.readouterr().out


def test_failed_cell_is_reported_and_refused(tmp_path, capsys):
    grid = _full_grid()
    grid["heist_ngu"][2] = {"seed": 44, "mean_reward": None, "error": "CUDA out of memory"}
    _write_run(str(tmp_path), "maze_heist_heist_rnd_ngu_a", grid)
    with pytest.raises(SystemExit):
        load_runs(str(tmp_path))
    assert "CUDA out of memory" in capsys.readouterr().out
    # the same data may still be inspected, just not published
    cells, _sources, conflicts = load_runs(str(tmp_path), require_complete=False)
    assert 44 not in cells["heist_ngu"]
    assert any("failed run" in c for c in conflicts)


def test_cell_measured_twice_keeps_the_newer_file_and_says_so(tmp_path, capsys):
    first, second = _full_grid(), _full_grid()
    first["maze_icm"] = [_cell(s, 1.0, arm="icm") for s in SEEDS]
    second["maze_icm"] = [_cell(s, 9.0, arm="icm") for s in SEEDS]
    _write_run(str(tmp_path), "maze_heist_maze_ppo_icm_old", first, mtime=1000.0)
    _write_run(str(tmp_path), "maze_heist_maze_ppo_icm_new", second, mtime=2000.0)
    cells, _sources, conflicts = load_runs(str(tmp_path))
    assert [cells["maze_icm"][s]["mean_reward"] for s in SEEDS] == [9.0] * 5
    assert any("measured twice" in c for c in conflicts), conflicts


def test_arm_without_bonus_bookkeeping_is_refused(tmp_path):
    cells, _sources, _conflicts = load_runs(_write_run_dir(tmp_path))
    with pytest.raises(SystemExit) as exc:
        verify_bonus_fired(cells)
    assert exc.value.code == 1


def _write_run_dir(tmp_path):
    root = os.path.join(str(tmp_path), "logs")
    _write_run(root, "maze_heist_all_no_bonus", _full_grid(bonus=False))
    return root


def test_rare_bonus_is_refused_but_plain_ppo_is_exempt():
    cells = {"maze_ppo": {42: {"mean_reward": 2.0, "bonus": None}},
             "maze_rnd": {42: {"mean_reward": 2.0,
                               "bonus": {"steps": 100000, "bonus_applied": 1000}}}}
    with pytest.raises(SystemExit):
        verify_bonus_fired(cells)


def test_fully_verified_grid_passes():
    cells = {"maze_ppo": {42: {"mean_reward": 2.0}},
             "maze_rnd": {42: {"mean_reward": 2.0,
                               "bonus": {"steps": 100000, "bonus_applied": 99000}}}}
    verify_bonus_fired(cells)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
