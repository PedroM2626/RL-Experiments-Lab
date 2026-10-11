"""Gate README 3.6.3 (the collapse timeline) and the dashboard clip against their artifacts.

3.6.2 excluded feature scale from endpoint numbers; 3.6.3 makes four stronger claims that all
rest on time-resolved curves -- which threshold was crossed first, by how many steps, what the
frame-stack intervention multiplied, and that the published clip shows a policy that actually
reads its screen. Each is recomputed here from the committed JSON, and the clip is checked at
pixel level rather than by its filename: a recording of a frozen panel would otherwise pass
every other test in this file.
"""
import json
import os
import re

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")
CURVES = {
    "repo": "policy_saturation_perlayer_repo.json",
    "layernorm": "policy_saturation_perlayer_layernorm.json",
    "maze_fs3": "frame_stack_maze.json",
    "starpilot_fs3": "frame_stack_starpilot.json",
    "coinrun": "coinrun_responsive_curve.json",
}


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def curves():
    return {k: _load(v) for k, v in CURVES.items()}


@pytest.fixture(scope="module")
def readme():
    with open(README, encoding="utf-8") as f:
        return f.read()


def _rows(text, header_start):
    i = text.index(header_start)
    out = []
    for line in text[i:].splitlines():
        if not line.startswith("|"):
            if out:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: ") or cells[0].startswith("run"):
            continue
        out.append(cells)
    return out


def _first_crossing(curve, pred):
    return next((r["steps"] for r in curve if pred(r)), None)


def _feat_over_10(c):
    return _first_crossing(c, lambda r: r["feat_max"] > 10)


def _l1_over_50(c):
    return _first_crossing(c, lambda r: r["tanh_saturated_by_layer"][0] > 0.5)


def _tv_dead(c):
    return _first_crossing(c, lambda r: r["cross_frame_tv_mean"] < 5e-5)


# ------------------------------------------------------------------ the instrument itself

def test_per_layer_field_is_two_layers_and_agrees_with_the_published_mean(curves):
    """The first version indexed buckets by network position, leaving an empty bucket per Linear
    and writing NaN into the curve. That is what this catches."""
    for tag, d in curves.items():
        for r in d["curve"]:
            layers = r["tanh_saturated_by_layer"]
            assert len(layers) == 2, f"{tag}: expected two Tanh layers, got {layers}"
            assert all(0.0 <= x <= 1.0 for x in layers), f"{tag}: non-fraction {layers}"
            assert abs(r["tanh_saturated_fraction"] - sum(layers) / 2) < 1e-3, f"{tag}: mean drift"


def test_every_curve_is_a_real_timeline(curves):
    for tag, d in curves.items():
        steps = [r["steps"] for r in d["curve"]]
        assert steps[0] == 0 and steps[-1] == 100000, f"{tag}: {steps[:2]} ... {steps[-1]}"
        assert steps == sorted(steps) and len(set(steps)) == len(steps), f"{tag}: steps"


# ------------------------------------------------------------------ claim 1: ordering

def test_layernorm_collapses_long_before_it_saturates(curves):
    ln, repo = curves["layernorm"]["curve"], curves["repo"]["curve"]
    assert _feat_over_10(ln) is None, "the clamped arm was supposed to never exceed 10"
    assert max(r["feat_max"] for r in ln) == pytest.approx(5.88, abs=0.01)
    collapse, saturates = _tv_dead(ln), _l1_over_50(ln)
    assert (collapse, saturates) == (18000, 92000)
    assert saturates - collapse == 74000, "the README quotes the gap in steps"
    # ... while the published arm has the opposite ordering, which is what made the story plausible
    assert (_feat_over_10(repo), _l1_over_50(repo), _tv_dead(repo)) == (10000, 6000, 10000)


def test_timeline_table_cells_match_the_curves(readme, curves):
    rows = _rows(readme, "### 3.6.3. When the Collapse Happens")
    assert len(rows) == 5, f"expected one row per run, got {len(rows)}"
    key = {"maze, published extractor": "repo",
           "maze, LayerNorm counterfactual": "layernorm",
           "maze, frame_stack=3": "maze_fs3",
           "starpilot, frame_stack=3": "starpilot_fs3",
           "coinrun, published extractor (the responsive reference)": "coinrun"}
    for cells in rows:
        name, feat, l1, tv, cross, vstd = [c.replace("`", "").strip() for c in cells]
        assert name in key, f"row for {name!r} is not one of the five runs the section measures"
        d = curves[key[name]]
        c = d["curve"]
        for value, pred, metric in ((feat, _feat_over_10, "feat_max"),
                                    (l1, _l1_over_50, "l1"),
                                    (tv, _tv_dead, "tv")):
            got = pred(c)
            if value.startswith("never"):
                assert got is None, f"{name}: {value!r} but the curve crosses at {got}"
                continue
            stated = int(re.match(r"(\d+)k", value).group(1)) * 1000
            assert got == stated, f"{name}: says {value}, curve says {got}"
            if "back down" in value:
                # a transient crossing: the run goes over the threshold and comes out of it,
                # which is the whole content of claim 2, so it has to be checked as such
                tail = [r for r in c if r["steps"] > got]
                if metric == "feat_max":
                    assert min(r["feat_max"] for r in tail) < 10, "not transient"
                else:
                    assert min(r["tanh_saturated_by_layer"][0] for r in tail) < 0.5, "not transient"
        assert cross.rstrip("k") == f"{d['cross_game']['max_pairwise_tv']:.6f}", f"{name}: cross TV"
        assert vstd == f"{c[-1]['v_std']:.5f}", f"{name}: final V(s) std"


# ------------------------------------------------------------------ claim 2: the early window

def test_both_arms_start_with_a_signal_and_only_one_keeps_it(curves, readme):
    """Claim 2 is a count and five numbers, so it is checkable rather than rhetorical."""
    maze = curves["repo"]["curve"]
    alive = [(r["steps"], r["cross_frame_tv_mean"]) for r in maze if r["cross_frame_tv_mean"] > 1e-4]
    assert len(alive) == 5 and len(maze) == 51, f"{len(alive)} of {len(maze)} probes"
    assert [s for s, _ in alive] == [0, 2000, 4000, 8000, 44000]
    assert [round(v, 5) for _, v in alive] == [0.00036, 0.00041, 0.00050, 0.00374, 0.00400]
    peak8 = next(r for r in maze if r["steps"] == 8000)
    assert peak8["distinct_argmax_actions"] == 2
    late = next(r for r in maze if r["steps"] == 44000)
    assert late["v_std"] == pytest.approx(0.12017, abs=1e-5)
    # and after the last excursion the arm is silent to the end
    tail = [r for r in maze if r["steps"] > 44000]
    assert len(tail) == 28
    assert max(r["cross_frame_tv_mean"] for r in tail) == pytest.approx(0.000002, abs=1e-9)
    assert sum(1 for r in tail if r["cross_frame_tv_mean"] == 0.0) == 25
    assert all(r["v_std"] == 0.0 for r in tail[-5:]), "V(s) std is not the published 0.00000"
    # coinrun goes the other way: deeper than maze at 5k, two orders above it by 65k
    coin = {r["steps"]: r for r in curves["coinrun"]["curve"]}
    assert coin[5000]["cross_frame_tv_mean"] == pytest.approx(0.00009, abs=1e-5)
    assert coin[5000]["tanh_saturated_by_layer"][0] == pytest.approx(0.609, abs=1e-3)
    assert coin[5000]["v_std"] == pytest.approx(0.0030, abs=1e-4)
    for step, want in ((10000, 0.00485), (20000, 0.01100), (65000, 0.26280)):
        assert coin[step]["cross_frame_tv_mean"] == pytest.approx(want, abs=1e-5), step
    assert max(v for _, v in alive) < 0.005 < coin[65000]["cross_frame_tv_mean"]
    assert _tv_dead(list(coin.values())) is None, "coinrun must never fall below the dead threshold"
    assert "**5 of the 51 probes**" in readme and "steps 0, 2k, 4k, 8k and 44k" in readme


# ------------------------------------------------------------------ claim 3: the intervention

def test_frame_stack_multipliers_against_the_single_frame_baselines(curves, readme):
    maze_fs1 = curves["repo"]["cross_game"]["max_pairwise_tv"]
    maze = curves["maze_fs3"]["cross_game"]["max_pairwise_tv"]
    starpilot = curves["starpilot_fs3"]["cross_game"]
    starpilot_fs1 = _load("suite_validity_starpilot.json")["arms"]["classic"]["max_pairwise_tv"]
    assert maze / maze_fs1 == pytest.approx(3.5, abs=0.05)
    assert starpilot["max_pairwise_tv"] / starpilot_fs1 == pytest.approx(15, abs=0.5)
    # ... and neither crosses the responsive reference, nor varies its argmax
    coinrun = _load("suite_validity_coinrun_control.json")["arms"]["classic"]["max_pairwise_tv"]
    for x in (maze, starpilot["max_pairwise_tv"]):
        assert x < coinrun / 40
        assert x != 0.0, "a zero here would mean the intervention was never measured"
    assert starpilot["distinct_argmax_across_games"] == 1
    assert curves["maze_fs3"]["cross_game"]["distinct_argmax_across_games"] == 1
    assert "`0.000221 → 0.000774`" in readme and "`0.001149 → 0.017082`" in readme


# ------------------------------------------------------------------ claim 4: worse than random

def test_the_blind_policy_is_below_the_random_baseline_on_pace():
    rb = _load("random_baselines.json")["maze"]
    trained = _load("exploration_remeasure.json")["statistics_100eps"]["maze_ppo"]
    solved = sum(1 for e in rb["episodes"] if e >= 10)
    assert (rb["mean"], solved, rb["n_episodes"]) == (4.4, 22, 50)
    assert trained["mean"] == pytest.approx(2.58, abs=0.005)
    assert solved / rb["n_episodes"] == pytest.approx(0.44, abs=0.005)
    assert trained["mean"] / 10.0 == pytest.approx(0.258, abs=0.005)


def test_random_and_trained_numbers_are_published_as_readme_prose(readme):
    assert "`22/50` episodes, mean `4.40`" in readme
    assert "`2.58` over 5 seeds" in readme
    assert "**44%**" in readme and "**26%**" in readme


# ------------------------------------------------------------------ the clip itself

def test_dashboard_clip_exists_and_its_panels_actually_move():
    import imageio.v3 as iio
    path = os.path.join(RESULTS, "agent_dashboard.gif")
    assert os.path.exists(path), "README embeds results/agent_dashboard.gif, which is missing"
    frames = iio.imread(path, index=None)
    assert len(frames) >= 20, f"only {len(frames)} frames -- too short to show a policy acting"
    a = frames.astype(np.int16)
    # the right half of the frame is the panel column; if the policy were state-independent this
    # is exactly the region that would sit still, which is how the defect was first noticed
    panels = a[:, 400:, :, :3]
    deltas = np.abs(np.diff(panels, axis=0)).mean(axis=(1, 2, 3))
    moving = float((deltas > 0.5).mean())
    assert moving > 0.5, f"only {moving:.0%} of consecutive frames change the panels"
    assert 0.0 < os.path.getsize(path) / 1e6 < 10.0, "GitHub rejects large assets"


def test_dashboard_caption_carries_the_audit_number_it_claims(readme):
    """The README says the clip is of the arm measured as responsive; the caption is on the
    image, so the number it prints has to be the number in the artifact."""
    row = _load("suite_validity_coinrun_control.json")["arms"]["classic"]
    assert row["max_pairwise_tv"] == pytest.approx(0.739958, abs=5e-7)
    assert f"cross-game TV `{row['max_pairwise_tv']:.6f}`" in readme
    assert "return `10.00`" in readme and "`V(s)` rising to `9.66`" in readme
