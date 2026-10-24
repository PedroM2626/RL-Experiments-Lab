"""Gate the best-anchored canonical normalization (README 3.14, item 4).

The classic-anchored canonical normalization excludes dodgeball because the
reference architecture does not beat the random policy there. The best-measured
anchor added on 24/10/2026 extends the SAME Agarwal formula to all three suite
games by using the dataset's own best policy as the ceiling -- and leaves
maze/heist excluded with the honest reason: no measured arm beats random there.
"""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(BASE, "results", "rliable_scorecard.json")


def _load():
    with open(ART, encoding="utf-8") as f:
        return json.load(f)


def test_best_anchor_covers_all_three_suite_games():
    d = _load()
    assert set(d["best_anchored_games"]) == {"bossfight", "starpilot", "dodgeball"}
    assert d["best_anchor"]["dodgeball"]["arch"] == "resnet18"
    assert d["best_anchor"]["dodgeball"]["mean"] > 0.960  # beats the random anchor


def test_maze_and_heist_stay_excluded_for_the_sharp_reason():
    d = _load()
    for g, best in (("maze", "ngu mean 2.760"), ("heist", "rnd mean 1.040")):
        why = d["best_anchored_exclusions"][g]
        assert "no measured architecture beats the random anchor" in why, why
        assert best in why, why


def test_classic_anchor_block_is_unchanged_beside_it():
    """The published canonical numbers keep their 2-game scope and their values."""
    d = _load()
    assert set(d["canonical_games"]) == {"bossfight", "starpilot"}
    assert d["canonical_architectures"]["resnet18"]["canonical_iqm"] == 1.6934


def test_every_suite_arch_has_a_best_anchored_row_over_three_games():
    d = _load()
    assert len(d["best_anchored_architectures"]) == len(d["architectures"])
    for a, row in d["best_anchored_architectures"].items():
        assert row["games"] == ["bossfight", "starpilot", "dodgeball"], a
        lo, hi = row["best_anchored_ci95"]
        assert lo <= row["best_anchored_iqm"] <= hi, a
