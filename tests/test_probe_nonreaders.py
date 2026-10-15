"""Gate README 3.6.9 (the non-reader probes) on their nine artifacts.

The question: did the grid's non-readers never condition, or condition and lose it?
The answer splits by mechanism -- vit/bossfight never ignites, mlp_vector/dodgeball dies
absorbing-ly at 8500 -- and the other four cells read within-game at a third to a tenth
of classic's depth. Every number is recomputed from the probe curves, and the probe's
classic/starpilot run is checked against the 3.6.6 pilot cell it reproduces field by
field: two scripts, one trajectory.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")
GRID = list(range(0, 100001, 500))
ARCHS = ("classic", "vit", "mlp", "impoola")


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def probes():
    return {(a, g): _load(f"probe_nonreaders_{a}_{g}.json")
            for a in ARCHS for g in ("bossfight", "dodgeball", "starpilot")}


@pytest.fixture(scope="module")
def readme():
    with open(README, encoding="utf-8") as f:
        return f.read()


def _curve(probes, a, g):
    return probes[(a, g)]["curve"]


def _alive(c):
    return sum(1 for r in c if r["cross_frame_tv_mean"] > 1e-4)


def _first_dead(c):
    return next((r["steps"] for r in c if r["cross_frame_tv_mean"] < 5e-5), None)


def test_all_twelve_probes_are_complete_grid_runs(probes):
    for (a, g), d in probes.items():
        assert [r["steps"] for r in d["curve"]] == GRID, f"{a}/{g}"
        p = d["_provenance"]
        assert (p["arch"], p["n_envs"], p["n_steps"], p["seed"], p["timesteps"]) == \
            (a, 16, 256, 42, 100000)
        assert d["cross_game"] is not None and p["probe_every"] == 500


def test_vit_on_bossfight_never_ignites(probes):
    c = _curve(probes, "vit", "bossfight")
    assert c[0]["cross_frame_tv_mean"] == 0.000792 and c[0]["distinct_argmax_actions"] == 2
    assert c[-1]["cross_frame_tv_mean"] == 2e-06
    assert _first_dead(c) == 16500 and _alive(c) == 25, "all alive probes must be early"
    assert max(r["cross_frame_tv_mean"] for r in c) == c[0]["cross_frame_tv_mean"], \
        "the peak is the initialization -- no update ever raised the response"
    assert probes[("vit", "bossfight")]["cross_game"]["max_pairwise_tv"] == 2.6e-05


def test_mlp_vector_on_dodgeball_dies_absorbingly(probes):
    c = _curve(probes, "mlp", "dodgeball")
    assert c[0]["cross_frame_tv_mean"] == 0.000744
    assert _first_dead(c) == 8500 and _alive(c) == 17
    assert c[-1]["cross_frame_tv_mean"] == 0.0 and c[-1]["v_std"] == 0.0
    assert c[-1]["tanh_saturated_by_layer"][0] == 1.0, "the head ends fully saturated"
    # born saturated and it still starts alive: saturation is structural, not causal
    assert c[0]["tanh_saturated_by_layer"][0] == 0.9789


def test_mlp_heads_are_saturated_from_birth(probes):
    for g in ("bossfight", "starpilot", "dodgeball"):
        assert _curve(probes, "mlp", g)[0]["tanh_saturated_by_layer"][0] > 0.9
    assert _curve(probes, "mlp", "bossfight")[0]["distinct_argmax_actions"] == 5


def test_the_other_four_cells_read_within_game(probes):
    weak = {("vit", "dodgeball"), ("vit", "starpilot"),
            ("mlp", "bossfight"), ("mlp", "starpilot")}
    finals = {("vit", "dodgeball"): 0.015897, ("vit", "starpilot"): 0.024421,
              ("mlp", "bossfight"): 0.060599, ("mlp", "starpilot"): 0.042439}
    for (a, g) in weak:
        c = _curve(probes, a, g)
        assert _alive(c) == 201 and _first_dead(c) is None, f"{a}/{g} dipped"
        assert c[-1]["cross_frame_tv_mean"] == finals[(a, g)]
        assert c[-1]["cross_frame_tv_mean"] > 15 * c[0]["cross_frame_tv_mean"], \
            f"{a}/{g} did not grow from birth"


def test_classic_reads_everywhere_and_deepest(probes):
    finals = {g: _curve(probes, "classic", g)[-1]["cross_frame_tv_mean"]
              for g in ("bossfight", "dodgeball", "starpilot")}
    assert finals == {"bossfight": 0.184762, "dodgeball": 0.053882, "starpilot": 0.138053}
    for g in finals:
        c = _curve(probes, "classic", g)
        assert _alive(c) == 201
        assert max(r["tanh_saturated_by_layer"][0] for r in c) <= 0.1445
    # the graded separation: readers are 3-10x deeper within-game
    assert finals["bossfight"] / _curve(probes, "mlp", "bossfight")[-1]["cross_frame_tv_mean"] \
        == pytest.approx(3.05, abs=0.05)
    assert finals["dodgeball"] / _curve(probes, "vit", "dodgeball")[-1]["cross_frame_tv_mean"] \
        == pytest.approx(3.4, abs=0.1)
    # and the game contributes as much as the front end: classic/dodgeball is BELOW
    # mlp/bossfight within-game -- why the cross-game measure separates architectures
    assert finals["dodgeball"] < _curve(probes, "mlp", "bossfight")[-1]["cross_frame_tv_mean"]


def test_the_probe_reproduces_the_pilot_cell_field_by_field(probes):
    row = probes[("classic", "starpilot")]["cross_game"]
    pilot = _load("suite_validity_starpilot_nenv16.json")["arms"]["classic"]
    for k in ("max_pairwise_tv", "distinct_argmax_across_games",
              "first_tanh_saturated_fraction", "feat_max"):
        assert row[k] == pilot[k], f"probe vs pilot drifted on {k}"
    assert (row["max_pairwise_tv"], row["distinct_argmax_across_games"],
            row["first_tanh_saturated_fraction"], row["feat_max"]) == \
        (0.361022, 3, 0.1688, 11.358)


def test_impoola_is_the_weak_reader_family_with_one_inert_cell(probes):
    # dodgeball and starpilot: born small, grows 120x / 33x, never dips
    for g, final, peak in (("dodgeball", 0.017502, 0.021441), ("starpilot", 0.020158, 0.066320)):
        c = _curve(probes, "impoola", g)
        assert _alive(c) == 201 and _first_dead(c) is None
        assert c[-1]["cross_frame_tv_mean"] == final
        assert max(r["cross_frame_tv_mean"] for r in c) == peak
    # bossfight: born flat, one early excursion, neither ignites nor dies
    c = _curve(probes, "impoola", "bossfight")
    assert c[0]["cross_frame_tv_mean"] == 0.000397 and c[-1]["cross_frame_tv_mean"] == 0.000325
    assert max(r["cross_frame_tv_mean"] for r in c) == 0.008743
    assert _alive(c) == 201 and c[-1]["tanh_saturated_by_layer"][0] == 0.75


def test_the_impoola_probe_reproduces_the_pilot_cell(probes):
    row = probes[("impoola", "starpilot")]["cross_game"]
    pilot = _load("suite_validity_starpilot_nenv16.json")["arms"]["impoola"]
    for k in ("max_pairwise_tv", "distinct_argmax_across_games",
              "first_tanh_saturated_fraction", "feat_max"):
        assert row[k] == pilot[k], f"probe vs pilot drifted on {k}"
    assert (row["max_pairwise_tv"], row["distinct_argmax_across_games"],
            row["first_tanh_saturated_fraction"], row["feat_max"]) == \
        (0.026367, 1, 0.0063, 2.35)


def test_section_prose_carries_the_measured_numbers(readme):
    for s in (
        "Twelve curves, **~23 GPU-minutes**",
        "**never ignites**: born near-dead, decays from step 0",
        "**dies absorbing-ly at 8500**",
        "`25/201` probes alive, all early",
        "first dead probe at `8500`, `17/201` alive",
        "**grows 15–30× over training**",
        "`0.0159`–`0.0606` against `classic`'s `0.0539`–`0.1848`",
        "`91.33%`–`97.89%` of its first `Tanh` is already saturated",
        "identical to the §3.6.6 pilot's `classic` cell on all four fields",
        "Two scripts, one trajectory",
        "three distinct failure shapes",
        "Inertness without collapse",
        "grows **120× and 33×**",
        "the third cross-script reproduction of this section",
        "a third to a tenth of `classic`'s depth",
        "first `Tanh` never above `14.5%`",
    ):
        assert s in readme, f"prose lost: {s!r}"


def test_the_3_6_7_note_points_here(readme):
    i = readme.index("**Follow-up (14/10/2026):** the dense probe ran on all three non-readers")
    assert "§3.6.9" in readme[i:i + 500] and "§3.6.10" in readme[i:i + 500]
