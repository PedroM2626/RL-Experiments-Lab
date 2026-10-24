"""Gate the mlp_scaled grid artifact (README 3.6.12 follow-up, section 0.1 row).

mlp_scaled is the 3.6.12 input-scale lever run as a full audit arm: the same 256-D
vector observation as the published mlp_vector, scaled to [0,1] float32 by the wrapper
(procgen_wrapper.ProcgenVectorWrapper(scale=True)). The rescue was first measured with
dense probes on bossfight/dodgeball; this grid extends it to all three suite games at
the audit's own protocol, so these are the claims the test pins: the scaled arm reads
on more cells than the raw arm, the birth saturation is gone, and the dodgeball
caveat (a within-game reader that does not generalize across games) is carried by
the committed data rather than by prose.
"""
import json
import os

import pytest

from summarize_mlpscaled_grid import fold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(BASE, "results", "suite_validity_nenv16_mlp_scaled.json")


def _load():
    with open(ART, encoding="utf-8") as f:
        return json.load(f)


def test_artifact_matches_a_refold():
    """The committed artifact is exactly what the merged per-game files fold into."""
    with open(ART, encoding="utf-8") as f:
        committed = json.load(f)
    assert committed == fold()


def test_artifact_complete_15_cells_no_errors():
    d = _load()
    for game in ("starpilot", "bossfight", "dodgeball"):
        pg = d["games"][game]
        assert len(pg["returns"]) == 5, game
        assert pg["error_seeds"] == [], game
    assert d["responsiveness"]["cells"] == 15
    assert d["responsiveness"]["error_cells"] == 0


def test_rescue_beats_the_raw_arm_on_the_response_measure():
    d = _load()
    cmp_ = d["vs_raw_mlp_vector"]
    assert cmp_["raw_responsive_cells"] == 3      # the published raw-uint8 arm
    assert cmp_["scaled_responsive_cells"] == 7
    assert cmp_["scaled_tv_mean"] == pytest.approx(0.135081, abs=1e-6)
    assert cmp_["raw_tv_mean"] == pytest.approx(0.072704, abs=1e-6)
    assert cmp_["scaled_tv_mean"] > cmp_["raw_tv_mean"]


def test_birth_saturation_is_gone():
    """The raw arm's first Tanh is 91-98% saturated at initialization; the scaled arm's
    is born under 30% on every measured cell (3.6.12's mechanism, at grid scale)."""
    d = _load()
    for game, pg in d["games"].items():
        for s in pg["saturated_firsts"]:
            assert s < 0.30, (game, s)


def test_dodgeball_reads_within_game_but_not_across_games():
    """The caveat the probes could not see: on dodgeball the scaled arm keeps the
    audit's cross-game TV below 0.1 on all five seeds -- the 3.6.10 within-game
    reader pattern -- while bossfight reads deeply on 4/5."""
    d = _load()
    assert all(t < 0.1 for t in d["games"]["dodgeball"]["tvs"])
    assert sum(1 for t in d["games"]["bossfight"]["tvs"] if t >= 0.1) == 4


def test_readme_scorecarries_the_row_and_the_note():
    with open(os.path.join(BASE, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    assert "| `mlp_scaled` † | 7/15 | 0.135081 |" in readme
    # the note is inside section 0.1 and explains what the arm is
    sec = readme.split("### 0.1")[1].split("## 1.")[0]
    assert "input-scale rescue" in sec
    assert "ProcgenVectorWrapper(scale=True)" in sec
