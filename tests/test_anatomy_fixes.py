"""Gate README 3.6.11 (the tested fixes and the five-seed anatomy), the figures of the
dense-probe arc, and the section 0.1 response scorecard, on their artifacts.

The anatomy's two predicted fixes were tested and both failed informatively; the
anatomy itself went to five seeds and came back robust where it matters, with one
honest revision (impoola's "inert" cell was seed 42's). These tests recompute every
published number -- including the 20% escape rate of vit/dodgeball and the 5/5
head-side death of mlp_vector -- and check the five figures are embedded and mapped.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from report_figures import PROVENANCE, readme_figures  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")
FIGURES = ("dense_probe_timelines.png", "dose_response_reliability.png",
           "response_cliff.png", "budget_response.png", "anatomy_matrix.png")


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def readme():
    with open(README, encoding="utf-8") as f:
        return f.read()


# ----------------------------------------------------------- the LayerNorm counterfactual

def test_layernorm_at_the_vit_output_repeats_the_death_to_the_step():
    c = _load("probe_vitlayernorm_bossfight.json")["curve"]
    tv = [r["cross_frame_tv_mean"] for r in c]
    assert next(r["steps"] for r in c if r["cross_frame_tv_mean"] < 5e-5) == 16500, \
        "vit_ln must die at the same step as bare vit (16500)"
    assert c[-1]["cross_frame_tv_mean"] == 0.000001
    assert sum(1 for v in tv if v > 1e-4) == 33
    d = _load("probe_vitlayernorm_dodgeball.json")["curve"]
    tvd = [r["cross_frame_tv_mean"] for r in d]
    assert max(tvd) == 0.007601 and d[tvd.index(max(tvd))]["steps"] == 29000
    assert next(r["steps"] for r in d if r["cross_frame_tv_mean"] < 5e-5) == 41000, \
        "the counterfactual still falls off the knife's edge"
    assert d[-1]["cross_frame_tv_mean"] == 0.000013
    assert sum(1 for v in tvd if v > 1e-4) == 74


def test_layernorm_cannot_create_a_range_that_is_not_there():
    for g, frange, fdims, act in (("bossfight", 0.247, 0.002, 9e-06),
                                  ("dodgeball", 0.1168, 0.0, 2e-06)):
        w = _load(f"feature_vs_actor_vit_ln_{g}.json")["within_game"]
        assert w["features"]["feat_range_over_scale_mean"] == frange, g
        assert w["features"]["feat_dims_range_over_scale"] == fdims, g
        assert w["action_tv"] == act, g
        assert w["features"]["feat_scale"] > 0.7, g


# ------------------------------------------------------------------- the head levers

def test_entropy_pressure_delays_but_does_not_save():
    d = _load("probe_headlevers_mlp_dodgeball_entcoef001.json")
    c = d["curve"]
    tv = [r["cross_frame_tv_mean"] for r in c]
    assert next(r["steps"] for r in c if r["cross_frame_tv_mean"] < 5e-5) == 16500, \
        "ent_coef stretches the fragile window from 8500 to 16500"
    assert c[-1]["cross_frame_tv_mean"] == 0.0
    assert c[-1]["tanh_saturated_by_layer"][0] == 1.0 and c[-1]["v_std"] == 0.0
    assert sum(1 for v in tv if v > 1e-4) == 17


def test_four_times_the_updates_change_nothing():
    d = _load("probe_headlevers_mlp_dodgeball_nsteps64.json")
    c = d["curve"]
    tv = [r["cross_frame_tv_mean"] for r in c]
    assert next(r["steps"] for r in c if r["cross_frame_tv_mean"] < 5e-5) == 8500, \
        "n_steps=64 dies at the bare run's exact step"
    assert c[-1]["cross_frame_tv_mean"] == 0.0
    assert max(tv) == 0.002845 and c[tv.index(max(tv))]["steps"] == 1500
    assert sum(1 for v in tv if v > 1e-4) == 29


# --------------------------------------------------------------- the anatomy at five seeds

def _cells(arch, game):
    rows = []
    for s in (42, 43, 44, 45, 46):
        fn = f"feature_vs_actor_{arch}_{game}.json" if s == 42 else \
             f"feature_vs_actor_{arch}_{game}_seed{s}.json"
        rows.append(_load(fn)["within_game"])
    return rows


def test_vit_bossfight_is_frontally_collapsed_on_every_seed():
    for w in _cells("vit", "bossfight"):
        assert w["features"]["feat_range_over_scale_mean"] <= 0.1793
        assert w["features"]["feat_dims_range_over_scale"] == 0.0
        assert w["action_tv"] <= 2.5e-05


def test_vit_dodgeball_collapses_four_times_in_five():
    rows = _cells("vit", "dodgeball")
    collapsed = [w for w in rows if w["features"]["feat_dims_range_over_scale"] == 0.0]
    assert len(collapsed) == 4
    escapes = [w for w in rows if w["features"]["feat_dims_range_over_scale"] > 0.3]
    assert len(escapes) == 1
    assert escapes[0]["features"]["feat_range_over_scale_mean"] == 1.03
    assert escapes[0]["features"]["feat_dims_range_over_scale"] == 0.3789
    assert escapes[0]["action_tv"] == 0.000160


def test_mlp_dodgeball_is_head_side_dead_with_live_features_every_seed():
    for w in _cells("mlp", "dodgeball"):
        assert w["action_tv"] == 0.0, "a seed where the head survived would revise the claim"
        assert 0.8 <= w["features"]["feat_range_over_scale_mean"] <= 0.89


def test_impoolas_inert_cell_was_its_seed():
    rows = _cells("impoola", "bossfight")
    acts = sorted(w["action_tv"] for w in rows)
    assert acts[0] == 0.000143 and acts[-1] == 0.065539
    assert sum(1 for a in acts if a > 1e-4) >= 3, "most seeds read weakly"
    assert all(w["features"]["feat_range_over_scale_mean"] > 0.6 for w in rows)


# ------------------------------------------------------------------------ the figures

def test_the_five_arc_figures_exist_are_embedded_and_mapped(readme):
    embedded = readme_figures()
    for name in FIGURES:
        path = os.path.join(RESULTS, name)
        assert os.path.exists(path), f"{name} missing"
        assert os.path.getsize(path) > 10_000, f"{name} is suspiciously small"
        assert name in PROVENANCE, f"{name} has no provenance entry"
        assert f"results/{name}" in embedded, f"{name} is not embedded in the README"
    for name in FIGURES:
        script, produced, source = PROVENANCE[name]
        assert os.path.exists(os.path.join(ROOT, script))
        assert produced.endswith(name)


# ------------------------------------------------------------------- the 0.1 scorecard

def test_the_response_scorecard_matches_the_grid(readme):
    grid = _load("suite_validity_nenv16_grid.json")
    resp = grid["responsiveness"]
    i = readme.index("### 0.1. The Response Scorecard")
    rows = []
    for line in readme[i:].splitlines():
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    data = [r for r in rows if r[0] != "architecture"]
    assert len(data) == 14, f"14 rows (16 arms, two same-program pairs merged), got {len(data)}"
    for cells in data:
        names = [n.strip().strip("`") for n in cells[0].split("≡")]
        got, tv = cells[1].replace("*", ""), float(cells[2].replace("*", ""))
        want_got, want_tv = None, None
        for n in names:
            r = resp[n]
            want_got = f"{r['cells_tv_over_0p1']}/{r['cells']}"
            want_tv = r["tv_mean"]
        assert got == want_got, cells[0]
        assert tv == want_tv, cells[0]
    # the cliff the caption names
    cells_alive = sorted({resp[a]["cells_tv_over_0p1"] for a in resp})
    assert cells_alive == [1, 3, 9, 10, 11, 12, 13, 15]


def test_section_prose_carries_the_measured_numbers(readme):
    for s in (
        "**~35 GPU-minutes**",
        "first dead probe at `16500`",
        "peak `0.007601` at `29000`, dead at `41000`",
        "from `0.154` to **`0.247`**/`0.1168`",
        "normalizes an almost-invariant vector, and an almost-invariant vector is what comes out",
        "stretches the fragile window from `8500` to **`16500`**",
        "first dead probe at `8500`, exactly the bare run's",
        "raw `uint8`",
        "**4/5 collapse, 1 escape**",
        "a 20% escape rate",
        "range `1.03`, `37.9%` of dims",
        "TV exactly `0.000000`, five times",
        "`0.000143–0.065539`",
        "was a single-seed reading",
        "The study's response scorecard (§0.1) is unchanged by all of this",
    ):
        assert s in readme, f"prose lost: {s!r}"


def test_the_3_6_10_note_and_the_revised_3_6_9_row_point_forward(readme):
    i = readme.index("**Follow-up (14/10/2026):** both predicted fixes were tested")
    assert "the collapse is created upstream of its output" in readme[i:i + 400]
    j = readme.index("a seed-42 reading")
    assert "0.000143–0.065539" in readme[j:j + 120]
