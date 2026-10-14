"""Gate README 3.6.4 (the dense 0-50k probe and the n_envs intervention) on its artifacts.

Section 3.6.3 left one surviving candidate and section 3.6.4 tested it. Every number the
section publishes is recomputed here from the committed JSON, in the same spirit as
test_mechanism_timeline: the dense baseline must be the published run (bitwise, at the
shared grid points), the dose-response must be in the curves (1 stream dies, 4 dips and
revives, 16 never dips), the staircase arm must have exactly as many segments as it had
gradient updates, and the coinrun reference must pass through the fragile window the
section says it passes through. The Watcher's cadence is also unit-tested without procgen:
probing on the vec-env step count was the bug this experiment could not afford, because
every arm would have measured a different grid and the comparison would have silently
compared steps that do not line up.
"""
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from diagnose_policy_saturation import Watcher  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")

ARMS = {  # tag -> (artifact, n_envs, n_steps)
    "1x256": ("policy_saturation_dense_nenv1_nsteps256.json", 1, 256),
    "4x64": ("policy_saturation_dense_nenv4_nsteps64.json", 4, 64),
    "16x16": ("policy_saturation_dense_nenv16_nsteps16.json", 16, 16),
    "16x256": ("policy_saturation_dense_nenv16_nsteps256.json", 16, 256),
}
COINRUN = "coinrun_dense_curve.json"
GRID = list(range(0, 50001, 500))
DEAD, ALIVE = 5e-5, 1e-4


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def arms():
    return {tag: _load(fn) for tag, (fn, _, _) in ARMS.items()}


@pytest.fixture(scope="module")
def coinrun():
    return _load(COINRUN)


@pytest.fixture(scope="module")
def readme():
    with open(README, encoding="utf-8") as f:
        return f.read()


def _tv(curve, step):
    return next(r["cross_frame_tv_mean"] for r in curve if r["steps"] == step)


def _first_dead(curve):
    return next((r["steps"] for r in curve if r["cross_frame_tv_mean"] < DEAD), None)


def _alive(curve):
    return sum(1 for r in curve if r["cross_frame_tv_mean"] > ALIVE)


def _l1_50(curve):
    return next((r["steps"] for r in curve if r["tanh_saturated_by_layer"][0] > 0.5), None)


def _l1_max(curve):
    return max(r["tanh_saturated_by_layer"][0] for r in curve)


# ----------------------------------------------------------- the instrument's own grid

def test_every_arm_measured_the_same_environment_step_grid(arms, coinrun):
    for tag, d in arms.items():
        assert [r["steps"] for r in d["curve"]] == GRID, f"{tag}: probe grid"
        p = d["_provenance"]
        _, n_envs, n_steps = ARMS[tag]
        assert (p["n_envs"], p["n_steps"]) == (n_envs, n_steps), f"{tag}: provenance"
        assert p["per_update_batch"] == n_envs * n_steps
        assert p["probe_every"] == 500 and p["timesteps"] == 50000
        assert (p["game"], p["extractor"], p["seed"], p["probe_frames"]) == ("maze", "repo", 42, 20)
        for r in d["curve"]:
            assert 0 <= r["env_steps"] - r["steps"] < n_envs, f"{tag}: probe drift at {r['steps']}"
    assert [r["steps"] for r in coinrun["curve"]] == GRID
    assert coinrun["_provenance"]["game"] == "coinrun"


def test_parallel_envs_get_distinct_level_streams():
    """16 envs on one seed would be 16 copies of the same trajectory and the
    intervention would be empty. The factory must offset the seed by rank, and the
    model must consume n_steps from the command line, not the published constant."""
    with open(os.path.join(ROOT, "diagnose_policy_saturation.py"), encoding="utf-8") as f:
        src = f.read()
    assert "seed=args.seed + rank" in src, "envs no longer get per-rank seeds"
    assert "range(args.n_envs)" in src, "the vec env no longer has n_envs members"
    assert "n_steps=args.n_steps" in src, "n_steps drifted off the argument"


class _FakeModel:
    num_timesteps = 0


class _RecordingWatcher(Watcher):
    """Measures nothing; records the nominal grid point and the actual count."""

    def __init__(self, every, timesteps):
        super().__init__(None, every, timesteps)
        self.fired = []

    def _probe(self, at):
        self.fired.append((at, self.model.num_timesteps))


def _run_watcher(n_envs, every, timesteps):
    w = _RecordingWatcher(every, timesteps)
    w.model = _FakeModel()
    while w.model.num_timesteps < timesteps:
        w.model.num_timesteps += n_envs
        w._on_step()
    return w


def test_watcher_fires_on_environment_steps_not_vec_steps():
    """A 16-env rollout advances 16 environment steps per callback call. The old
    n_calls cadence would have spaced this arm's probes 16x apart and silently moved
    every arm onto its own grid."""
    w = _run_watcher(n_envs=16, every=500, timesteps=50000)
    w.flush()
    assert [at for at, _ in w.fired] == GRID[1:], "the probe grid is not the shared one"
    assert all(0 <= actual - at < 16 for at, actual in w.fired)


def test_watcher_survives_jumps_larger_than_the_cadence():
    """n_envs above the cadence crosses several grid points inside one callback call;
    every nominal point must still be probed and recorded against the honest count."""
    w = _run_watcher(n_envs=1024, every=500, timesteps=8192)
    w.flush()
    assert [at for at, _ in w.fired] == [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000,
                                        4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
    assert all(0 <= actual - at < 1024 for at, actual in w.fired)


def test_flush_refuses_to_ship_an_incomplete_grid():
    w = _RecordingWatcher(every=500, timesteps=50000)
    w.model = _FakeModel()
    w.model.num_timesteps = 600
    w._on_step()
    assert w.fired == [(500, 600)]
    with pytest.raises(RuntimeError, match="incomplete"):
        w.flush()


# ----------------------------------------------------- claim 1: the baseline is published

def test_dense_baseline_is_bitwise_the_published_run(arms):
    """The 1x256 arm is the same trajectory as the 100k timeline of 3.6.3, sampled
    4x finer. Any mismatch would mean the dense run answered a different question."""
    pub = _load("policy_saturation_perlayer_repo.json")["curve"]
    dense = {r["steps"]: r for r in arms["1x256"]["curve"]}
    shared = [r for r in pub if r["steps"] <= 50000]
    assert len(shared) == 26
    for r in shared:
        a = dense[r["steps"]]
        for k, v in r.items():
            if k != "steps":
                assert a[k] == v, f"step {r['steps']} field {k}: {v} != {a[k]}"


def test_coinrun_dense_is_bitwise_the_published_run(coinrun):
    pub = _load("coinrun_responsive_curve.json")["curve"]
    dense = {r["steps"]: r for r in coinrun["curve"]}
    shared = [r for r in pub if r["steps"] <= 50000]
    assert len(shared) == 11
    for r in shared:
        a = dense[r["steps"]]
        for k, v in r.items():
            if k != "steps":
                assert a[k] == v, f"step {r['steps']} field {k}: {v} != {a[k]}"


def test_all_four_maze_arms_share_one_initialization(arms):
    """Same seed, same architecture: the step-0 row must be identical to the last
    digit across arms, or the arms differ by more than n_envs."""
    rows = [d["curve"][0] for d in arms.values()]
    for r in rows[1:]:
        assert r == rows[0]


# ------------------------------------------------------- claims 1-2: the dose-response

def test_one_stream_dies_at_9500_and_the_dense_grid_found_21_alive(arms):
    a = arms["1x256"]["curve"]
    assert _first_dead(a) == 9500, "the dense grid says death is at 9500, not 10k"
    assert _alive(a) == 21, "the coarse grid said 5 of 51; dense says 21 of 101"
    # the early window is continuous, then sags, then one last excursion
    assert all(r["cross_frame_tv_mean"] > ALIVE for r in a if r["steps"] <= 5500)
    assert _tv(a, 6000) < ALIVE and _tv(a, 6500) < ALIVE
    assert _tv(a, 8000) == pytest.approx(0.00374, abs=1e-6) and _tv(a, 8500) > ALIVE
    assert _l1_50(a) == 4500, "dense moves the 2k-grid's 6000 crossing to 4500"


def test_four_streams_dip_into_death_and_come_back(arms):
    b = arms["4x64"]["curve"]
    assert _first_dead(b) == 12000
    block = [r["cross_frame_tv_mean"] for r in b if 14500 <= r["steps"] <= 18000]
    assert len(block) == 8 and all(v < DEAD for v in block), "the dip must be the real dead level"
    assert max(block) <= 2.6e-5
    assert _tv(b, 18500) > ALIVE, "the revival starts at 18.5k"
    assert _tv(b, 25500) == pytest.approx(0.109794, abs=1e-6), "27x the published arm's peak"
    assert _tv(b, 8000) == pytest.approx(0.001158, abs=1e-6)
    tail = [r["cross_frame_tv_mean"] for r in b if r["steps"] >= 24000]
    assert min(tail) == pytest.approx(0.002786, abs=1e-6), "never below 0.0025 after 24k"
    assert _alive(b) == 89


def test_sixteen_streams_never_dip(arms):
    c = arms["16x16"]["curve"]
    assert _first_dead(c) is None
    assert min(r["cross_frame_tv_mean"] for r in c) == pytest.approx(0.0003, abs=1e-9)
    assert _alive(c) == 101
    assert _l1_50(c) is None and _l1_max(c) == pytest.approx(0.2547, abs=1e-4)
    d = arms["16x256"]["curve"]
    assert _first_dead(d) is None and _alive(d) == 101
    assert _l1_max(d) == 0.0, "the first Tanh never saturates one unit in the whole run"


def test_alive_counts_form_a_monotone_dose_response(arms):
    assert (_alive(arms["1x256"]["curve"]), _alive(arms["4x64"]["curve"]),
            _alive(arms["16x16"]["curve"])) == (21, 89, 101)


# --------------------------------------------------- claim 3: shape, not size or updates

def test_the_staircase_arm_has_one_segment_per_gradient_update(arms):
    """16x256 fills a 4096-step buffer per update, so the weights only move at
    multiples of 4096 and the dense probes must form exactly 13 constant segments
    (the initialization plus 12 updates = floor(50000/4096))."""
    d = arms["16x256"]["curve"]
    segs = []
    for r in d:
        if not segs or r["cross_frame_tv_mean"] != segs[-1][2]:
            segs.append([r["steps"], r["steps"], r["cross_frame_tv_mean"]])
        else:
            segs[-1][1] = r["steps"]
    assert len(segs) == 13
    assert 50000 // 4096 == len(segs) - 1, "updates and segments disagree"
    assert segs[0][2] == pytest.approx(0.000356, abs=1e-6), "segment 0 is the initialization"
    posts = [s[2] for s in segs[1:]]
    assert min(posts) == pytest.approx(0.005792, abs=1e-6), "no post-update regression below 0.0058"
    assert posts == [0.005792, 0.01112, 0.015083, 0.026368, 0.021379, 0.015755, 0.028278,
                     0.03648, 0.040264, 0.039061, 0.035473, 0.041931]
    assert posts[-1] == max(posts), "the run ends at its most responsive update"


def test_the_widest_arm_is_the_only_one_with_two_argmax_across_games(arms):
    for tag, d in arms.items():
        cg = d["cross_game"]
        want = 2 if tag == "16x256" else 1
        assert cg["distinct_argmax_across_games"] == want, f"{tag}: {cg}"
    assert arms["16x256"]["cross_game"]["max_pairwise_tv"] == pytest.approx(0.088322, abs=1e-6)
    assert arms["1x256"]["cross_game"]["max_pairwise_tv"] == pytest.approx(0.000432, abs=1e-6)
    assert arms["16x256"]["curve"][-1]["v_std"] == pytest.approx(1.5311, abs=1e-5)


# ---------------------------------------------------- claim 4: saturation is a symptom

def test_saturation_reverses_when_the_response_revives(arms):
    b = arms["4x64"]["curve"]
    at17, at24 = next(r for r in b if r["steps"] == 17500), next(r for r in b if r["steps"] == 24500)
    assert at17["tanh_saturated_by_layer"][0] == pytest.approx(0.9961, abs=1e-4)
    assert at24["tanh_saturated_by_layer"][0] == pytest.approx(0.0586, abs=1e-4)
    assert at24["cross_frame_tv_mean"] == pytest.approx(0.0576, abs=1e-6), "response back at 0.058"


# ------------------------------------------------------- claim 5: the coinrun window

def test_coinrun_sags_where_maze_dies_and_ignites_at_9000(coinrun, arms):
    c = coinrun["curve"]
    sag = [r["steps"] for r in c if r["cross_frame_tv_mean"] <= ALIVE]
    assert sag == [5000, 6000], "the fragile window is 5k-6k, where maze is dying"
    assert next(r["steps"] for r in c if r["cross_frame_tv_mean"] >= 5e-3) == 9000
    after = [r["cross_frame_tv_mean"] for r in c if r["steps"] > 9000]
    assert min(after) == pytest.approx(0.000472, abs=1e-6), "takeoff is never given back"
    assert _alive(c) == 99 and _first_dead(c) is None
    assert _l1_50(c) == 3000 and max(r["tanh_saturated_by_layer"][0] for r in c) == pytest.approx(0.7094, abs=1e-4)
    # the fork resolves inside the same 500 steps: maze dead at 9500, coinrun ignited at 9000
    assert _first_dead(arms["1x256"]["curve"]) == 9500


def test_coinrun_dense_endpoint(coinrun):
    last = coinrun["curve"][-1]
    assert last["cross_frame_tv_mean"] == pytest.approx(0.084866, abs=1e-6)
    assert (last["entropy_mean_nats"], last["v_std"]) == (pytest.approx(0.971, abs=1e-3),
                                                         pytest.approx(0.74167, abs=1e-5))
    assert coinrun["cross_game"]["max_pairwise_tv"] == pytest.approx(0.410747, abs=1e-6)


# ----------------------------------------------------- the published table and prose

def _rows(text, header_start):
    i = text.index(header_start)
    out = []
    for line in text[i:].splitlines():
        if not line.startswith("|"):
            if out:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        out.append(cells)
    return out


def test_section_table_cells_match_the_artifacts(readme, arms, coinrun):
    rows = _rows(readme, "### 3.6.4.")
    assert len(rows) == 6, f"header + 5 runs, got {len(rows)}"
    key = {"maze, 1×256": "1x256", "maze, 4×64": "4x64", "maze, 16×16": "16x16",
           "maze, 16×256": "16x256", "coinrun, 1×256": "coinrun"}
    for cells in rows[1:]:
        name = cells[0].replace("`", "")
        tag = next(v for k, v in key.items() if name.startswith(k))
        d = coinrun if tag == "coinrun" else arms[tag]
        curve = d["curve"]
        l1_cell, dead_cell, alive_cell, cross_cell, argmax_cell, vstd_cell = cells[1:]
        # first-Tanh column
        got = _l1_50(curve)
        if got is None:
            if "peak" in l1_cell:
                assert l1_cell == f"never (peak {100 * _l1_max(curve):.1f}%)", name
            else:
                assert l1_cell == "never (0.0%)" and _l1_max(curve) == 0.0, name
        elif "then back down" in l1_cell:
            over = [r["steps"] for r in curve if r["tanh_saturated_by_layer"][0] > 0.5]
            assert l1_cell == f"{over[0]}–{over[-1]}, then back down", name
        else:
            assert l1_cell == str(got), name
        # dead column
        got = _first_dead(curve)
        if got is None:
            mn = min(r["cross_frame_tv_mean"] for r in curve)
            if "min" in dead_cell:
                if "$" in dead_cell:
                    assert mn == pytest.approx(6.7e-05, abs=1e-9), name
                    assert dead_cell == "never (min $6.7\\times10^{-5}$)", name
                else:
                    assert dead_cell == f"never (min {mn:.4f})" and mn == 0.0003, name
            else:
                assert dead_cell == "never", name
        elif "back at" in dead_cell:
            assert dead_cell == f"{got}, back at 18.5k", name
            assert _tv(curve, 18500) > ALIVE, name
        else:
            assert dead_cell == str(got), name
        assert alive_cell == str(_alive(curve)), name
        assert cross_cell == f"{d['cross_game']['max_pairwise_tv']:.6f}", name
        assert argmax_cell == str(d["cross_game"]["distinct_argmax_across_games"]), name
        assert vstd_cell == f"{curve[-1]['v_std']:.5f}", name


def test_section_prose_carries_the_measured_numbers(readme):
    for s in (
        "zero mismatches in every field at all 26 shared grid points",
        "**21 of 101** probes are alive, not 5 of 51",
        "death at **9500**",
        "peaks at **`0.109794` at 25500**",
        "**not one probe below `0.0003`**",
        "cuts the gradient updates from 195 to 12",
        "first `maze`-trained policy in this study to emit two distinct argmax actions across the five games",
        "**99.6% saturated at 17500**",
        "saturation has collapsed to **5.9%**",
        "sags below the alive threshold at **5000 and 6000**",
        "takes off at **9000** (`0.007985`)",
        "`maze` dead at 9500, `coinrun` ignited at 9000",
        "6.2 GPU-minutes",
        "never loses the response (**101/101** dense probes alive",
        "dies at step 9500",
    ):
        assert s in readme, f"prose lost: {s!r}"


def test_summary_and_roadmap_carry_the_intervention(readme):
    # section 0, conclusion 2
    i = readme.index("§3.6.4 names it: **the single-stream rollout**")
    assert "**21/101** for the published single-stream arm" in readme[i:i + 700]
    # section 6, item 9: closed 14/10 after the replication, the levers and the grid
    # (the 12/10 status it carried while the item was open is superseded)
    assert "✅ Closed (14/10/2026): cause identified and replicated, levers split, grid re-run" in readme
    assert "**Then the intervention**" in readme and "**Then the (iii) grid**" in readme
    # 3.6.3's claim 5 points forward to the section that decided it
    j = readme.index("The next measurement is therefore")
    assert "that measurement ran" in readme[j:j + 700]


def test_the_five_training_loops_sum_to_the_published_gpu_minutes(arms, coinrun):
    total = sum(d["_provenance"]["training_seconds"] for d in arms.values())
    total += coinrun["_provenance"]["training_seconds"]
    assert round(total / 60, 1) == 6.2
