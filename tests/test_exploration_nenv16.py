"""Gate the responsive-recipe curiosity experiment (README 3.6.1 follow-up, item 8).

Roadmap item 8 spent a month as "measured, not interpretable": the scaled-bonus arms
beat the control, but every policy behind that table was state-independent. This
artifact is the re-ask at the responsive recipe -- 40 cells, fresh control, response
row on every cell -- and these are the claims the test pins: the maze null, the heist
deltas that do not survive correction, the recipe lift that dominates both, and the
response inversion that gives the whole thing its interpretation.
"""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(BASE, "results", "exploration_nenv16.json")


def _load():
    with open(ART, encoding="utf-8") as f:
        return json.load(f)


def test_forty_cells_five_seeds_zero_errors():
    d = _load()
    assert d["_provenance"]["cells"] == 40
    assert d["_provenance"]["seeds_present"] == [42, 43, 44, 45, 46]
    for config, rows in d["per_seed"].items():
        assert len(rows["cells"]) == 5, config
        assert all(c["mean_reward_100eps"] is not None for c in rows["cells"]), config
        # the response guard ran on every cell -- that is what makes this table
        # interpretable where the 1-env one was not
        assert all(c.get("response") for c in rows["cells"]), config


def test_the_scaled_bonus_definition_held_at_the_new_recipe():
    """Every bonus cell injected O(1e-3) per step (beta=0.01 against a 0/1 extrinsic
    reward) -- the divisive normalization scaled with the recipe, not silently off."""
    d = _load()
    for config, s in d["injected_scale"].items():
        assert 1e-4 <= s["injected_per_step_mean"] <= 1e-1, config
        assert s["n_envs_per_wrapper"] == [16], config


def test_maze_is_a_null_everywhere():
    d = _load()
    for arm in ("icm", "rnd", "ngu"):
        t = d["vs_control"][f"maze_{arm}"]
        assert t["welch_p"] > 0.25, arm  # no maze comparison approaches significance
    # and the fresh control towers over the published 1-env one
    assert d["statistics_100eps"]["maze_ppo"]["mean"] == 4.20


def test_heist_deltas_are_directional_but_do_not_survive_correction():
    d = _load()
    for arm in ("icm", "rnd", "ngu"):
        t = d["vs_control"][f"heist_{arm}"]
        assert t["delta_mean"] > 0, arm
        assert t["welch_p"] > 0.0083, arm  # Bonferroni over six comparisons
    # the strongest cell sits exactly at the uncorrected boundary
    assert d["vs_control"]["heist_ngu"]["welch_p"] == 0.05066
    assert d["vs_control"]["heist_ngu"]["hedges_g"] == 1.366


def test_the_response_column_inverts_the_heist_story():
    """The control is the arm that READS on heist (4/5) while the bonus arms that
    outscore it read less (2/5, 2/5, 0/5) -- the interpretation the 1-env table
    could not have given, now carried by the committed data."""
    d = _load()
    resp = d["responsiveness"]
    assert resp["heist_ppo"]["responsive"] == 4
    assert resp["heist_icm"]["responsive"] == 2
    assert resp["heist_rnd"]["responsive"] == 2
    assert resp["heist_ngu"]["responsive"] == 0
    assert resp["heist_ppo"]["tv_mean"] == 0.225675
    # on maze every arm sits in the within-game-reader band, at the bar's edge
    for arm in ("ppo", "icm", "rnd", "ngu"):
        assert resp[f"maze_{arm}"]["responsive"] <= 2, arm
        assert resp[f"maze_{arm}"]["tv_mean"] < 0.11, arm


def test_readme_carries_the_followup_and_the_closing_answer():
    with open(os.path.join(BASE, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    sec = readme.split("### 3.6.1")[1].split("### 3.6.2")[0]
    assert "the question re-asked under the responsive recipe" in sec
    assert "| 4.20 | [3.76, 4.64] |" in sec
    assert "0.051" in sec
    assert "does not survive the recipe fix as a visual-curiosity effect" in sec
