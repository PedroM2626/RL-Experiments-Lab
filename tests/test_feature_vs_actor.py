"""Gate README 3.6.10 (front end or head) on its seven artifacts.

The endpoint of a blind cell -- a frame-invariant action distribution -- has two
possible organs. The diagnostic separates them: feature response (per-dim cross-frame
range over the run's own activation scale) against action response on the same frames.
vit's blindness is frontal (features frame-invariant, zero dims past scale); mlp's and
impoola's are in the head (features vary, actions dead); classic is the reference on
both axes. And the vit/dodgeball knife's edge is gated as such: this run is
feature-flat and action-dead where the 3.6.9 probe run of the same seed read.
"""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")
CELLS = {  # (arch, game): locus the section assigns
    ("classic", "bossfight"): "both live",
    ("classic", "dodgeball"): "both live",
    ("vit", "bossfight"): "front end",
    ("vit", "dodgeball"): "front end",
    ("impoola", "bossfight"): "head",
    ("mlp", "bossfight"): "head",
    ("mlp", "dodgeball"): "head",
}


def _load(arch, game):
    with open(os.path.join(RESULTS, f"feature_vs_actor_{arch}_{game}.json"),
              encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cells():
    return {k: _load(*k) for k in CELLS}


@pytest.fixture(scope="module")
def readme():
    with open(README, encoding="utf-8") as f:
        return f.read()


def test_all_seven_diagnostic_cells_are_the_grid_recipe(cells):
    for (a, g), d in cells.items():
        p = d["_provenance"]
        assert (p["arch"], p["game"], p["timesteps"], p["seed"],
                p["n_envs"], p["n_steps"]) == (a, g, 100000, 42, 16, 256), (a, g)
        for half in ("within_game", "cross_game"):
            assert d[half]["action_tv"] is not None
            assert d[half]["features"]["feat_dims"] in (256, 512)


def test_vits_blindness_is_frontal(cells):
    for g in ("bossfight", "dodgeball"):
        w = cells[("vit", g)]["within_game"]
        f = w["features"]
        assert f["feat_range_over_scale_mean"] < 0.155, f"vit/{g}: features vary after all"
        assert f["feat_dims_range_over_scale"] == 0.0, \
            f"vit/{g}: a dim ranges past scale -- not a total range collapse"
        assert w["action_tv"] <= 7e-06
        assert f["feat_scale"] > 0.38, "the collapse is range, not magnitude"


def test_the_readers_reference_is_alive_on_both_axes(cells):
    for g in ("bossfight", "dodgeball"):
        w = cells[("classic", g)]["within_game"]
        assert w["features"]["feat_range_over_scale_mean"] >= 1.3
        assert w["features"]["feat_dims_range_over_scale"] >= 0.47
        assert w["action_tv"] >= 0.057


def test_mlp_and_impoola_blindnesses_are_in_the_head(cells):
    # mlp/dodgeball: the absorbing death left features varying and actions dead
    m = cells[("mlp", "dodgeball")]["within_game"]
    assert m["action_tv"] == 0.0
    assert m["features"]["feat_range_over_scale_mean"] == 0.8417
    assert m["features"]["feat_dims_range_over_scale"] == 0.0117
    assert cells[("mlp", "dodgeball")]["cross_game"]["action_tv"] == 0.009228
    # impoola/bossfight: inert cell, features at half the reader's depth, actions flat
    i = cells[("impoola", "bossfight")]["within_game"]
    assert i["action_tv"] == 0.000452
    assert i["features"]["feat_range_over_scale_mean"] == 0.7798
    assert i["features"]["feat_dims_range_over_scale"] == 0.2637
    # mlp/bossfight reads weakly through a live front end
    assert cells[("mlp", "bossfight")]["within_game"]["action_tv"] == 0.037994


def test_the_knife_edge_is_recorded_between_the_two_runs(cells):
    """This run's vit/dodgeball is action-dead at the front end; the 3.6.9 probe run of
    the same seed and recipe read at 0.015897. The disagreement is the finding -- and
    it is gated from both artifacts, not just asserted in prose."""
    with open(os.path.join(RESULTS, "probe_nonreaders_vit_dodgeball.json"),
              encoding="utf-8") as f:
        probe = json.load(f)
    assert cells[("vit", "dodgeball")]["within_game"]["action_tv"] < 1e-05
    assert probe["curve"][-1]["cross_frame_tv_mean"] == 0.015897


def test_section_table_cells_match_the_artifacts(readme, cells):
    i = readme.index("### 3.6.10.")
    out = []
    for line in readme[i:].splitlines():
        if not line.startswith("|"):
            if out:
                break
            continue
        c = [x.strip() for x in line.strip("|").split("|")]
        if set("".join(c)) <= set("-: "):
            continue
        out.append(c)
    data = [r for r in out if len(r) == 7 and r[0] != "cell"]
    assert len(data) == 7, f"7 data rows expected, got {len(data)}"
    for row in data:
        name = row[0].replace("`", "").replace("_vector", "")
        arch = "mlp" if name.startswith("mlp") else name.split(" ")[0]
        game = name.split(" ")[-1]
        d = cells[(arch, game)]
        w, c = d["within_game"], d["cross_game"]
        assert float(row[1]) == w["action_tv"], row[0]
        assert float(row[2].replace("**", "")) == w["features"]["feat_range_over_scale_mean"]
        assert float(row[3].replace("%", "").replace("**", "")) == \
            round(100 * w["features"]["feat_dims_range_over_scale"], 1)
        assert float(row[4]) == c["action_tv"], row[0]
        assert float(row[5].replace("**", "")) == c["features"]["feat_range_over_scale_mean"]


def test_section_prose_carries_the_measured_numbers(readme):
    for s in (
        "**~13 GPU-minutes.**",
        "**front end**: features are frame-invariant",
        "`vit`'s blindness is frontal",
        "not one of the 512 dimensions",
        "it is a *range* collapse",
        "**head**: features vary, actions flat",
        "**head**: features vary, actions dead",
        "The knife's edge, seen from inside",
        "read at `0.015897`",
        "features go frame-invariant first",
        "Two anatomies, one endpoint",
        "the §3.6.3 LayerNorm counterfactual, applied to the ViT",
    ):
        assert s in readme, f"prose lost: {s!r}"
