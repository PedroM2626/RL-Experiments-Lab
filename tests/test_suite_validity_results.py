"""Gate the 16-arm validity audit published in README 3.6.2 against its two artifacts.

The audit is the evidence behind the study's strongest negative claim (the 16-architecture
ranking is unverified as an architecture result), so every number quoted from it is recomputed
here from results/suite_validity_starpilot.json and results/suite_validity_coinrun_control.json,
including the prose aggregates (how many arms are inert, the extremes of feature scale, the
GPU hours) and the "these two arms are the same program" assertion, which is checked against
the extractors themselves rather than against the table.

The README is parsed, not assumed: a cell that drifts from the artifact fails the suite.
"""
import json
import os
import re

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")


def _load(name):
    with open(os.path.join(ROOT, "results", name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def audit():
    return _load("suite_validity_starpilot.json")


@pytest.fixture(scope="module")
def control():
    return _load("suite_validity_coinrun_control.json")


@pytest.fixture(scope="module")
def readme():
    with open(README, encoding="utf-8") as f:
        return f.read()


def _table_rows(text, header):
    """Cells of the markdown table that follows `header`, as lists of stripped strings."""
    i = text.index(header)
    rows = []
    for line in text[i:].splitlines():
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: ") or cells[0].startswith(("arm", "game")):
            continue
        rows.append(cells)
    return rows


def _code(cells):
    return [c.strip("`") for c in cells]


def _fmt(value, spec):
    """The cell as the README prints it -- compared as a string, so a rounded digit and a wrong
    digit fail the same way, and the gate cannot be satisfied by a lucky tolerance."""
    return spec % value


# ---------------------------------------------------------------- the 16-arm table

def test_audit_table_cells_match_the_artifact(readme, audit):
    rows = _table_rows(readme, "**All sixteen arms, at the published budget")
    assert len(rows) == 16, f"the audit table has {len(rows)} rows, expected all 16 arms"
    seen = set()
    for name, family, tv, ret, sat, feat in (_code(r) for r in rows):
        assert name in audit["arms"], f"{name} is in the README but not in the artifact"
        a = audit["arms"][name]
        seen.add(name)
        assert family == a["family"], f"{name}: family {family} != {a['family']}"
        assert tv == _fmt(a["max_pairwise_tv"], "%.6f"), f"{name}: TV"
        assert ret == _fmt(a["return_10eps"], "%.2f"), f"{name}: return"
        assert sat == _fmt(100 * a["first_tanh_saturated_fraction"], "%.1f") + "%", f"{name}: saturation"
        assert feat == _fmt(a["feat_max"], "%.2f"), f"{name}: feature max"
    assert seen == set(audit["arms"]), f"arms missing from the table: {sorted(set(audit['arms']) - seen)}"


def test_audit_table_is_ordered_by_response_and_a_stale_cell_is_caught(readme, audit):
    """Positive control for the gate above: the check must fail on a typed digit, not just on absence."""
    rows = _table_rows(readme, "**All sixteen arms, at the published budget")
    tvs = [float(_code(r)[2]) for r in rows]
    assert tvs == sorted(tvs, reverse=True), "the table claims to be ordered by response"
    tampered = readme.replace("| `spatial` | `procgen_cnn` | 0.000000 |",
                              "| `spatial` | `procgen_cnn` | 0.250000 |")
    assert tampered != readme, "the tamper did not find the cell it meant to change"
    with pytest.raises(AssertionError, match="spatial: TV"):
        test_audit_table_cells_match_the_artifact(tampered, audit)


def test_every_audited_arm_is_state_independent(audit, control):
    for name, a in audit["arms"].items():
        assert a["distinct_argmax_across_games"] == 1, f"{name}: this arm DOES vary its argmax"
    # and the detector is not broken: coinrun/classic must respond
    assert control["arms"]["classic"]["distinct_argmax_across_games"] == 1
    assert control["arms"]["classic"]["max_pairwise_tv"] > 0.5, "the positive control stopped firing"


def test_prose_aggregates_of_the_sixteen_arms(readme, audit):
    A = audit["arms"]
    n = len(A)
    below = sum(1 for a in A.values() if a["max_pairwise_tv"] < 5e-3)
    sat = sum(1 for a in A.values() if a["first_tanh_saturated_fraction"] >= 0.99)
    hi = max(A.values(), key=lambda a: a["max_pairwise_tv"])
    steps = n * audit["_provenance"]["timesteps"]
    hours = sum(a["seconds"] for a in A.values()) / 3600.0
    assert (n, below, sat) == (16, 14, 12)
    assert hi["max_pairwise_tv"] == pytest.approx(0.101104), "the largest response moved"
    quoted = [f"`{n}` cells", f"`{steps / 1e6:.1f}M`", f"`{hours:.2f} h`",
              f"`{below}/{n}`", f"`{sat}/{n}`", f"`{hi['max_pairwise_tv']:.3f}`"]
    for lit in quoted:
        assert lit in readme, f"prose does not carry the artifact's value: {lit}"


def test_extremes_of_feature_scale_quoted_in_the_prose(readme, audit):
    A = audit["arms"]
    lo = min(A.values(), key=lambda a: a["feat_max"])
    hi = max(A.values(), key=lambda a: a["feat_max"])
    assert lo["feat_max"] == pytest.approx(3.69, abs=0.01) and hi["feat_max"] > 1500
    assert f"`vit`, `{_fmt(lo['feat_max'], '%.2f')}`" in readme
    assert f"`impala`, `{_fmt(hi['feat_max'], '%.2f')}`" in readme
    assert lo["max_pairwise_tv"] < 5e-3 and hi["max_pairwise_tv"] == 0.0, \
        "the prose says both extremes are inert"


# ---------------------------------------------------------------- the coinrun control

def test_control_table_cells_match_the_control_artifact(readme, control):
    rows = _table_rows(readme, "**The measurement has a positive control.")
    assert len(rows) == 3, f"control table has {len(rows)} rows"
    for name, tv, ret, sat, feat in (_code(r) for r in rows):
        c = control["arms"][name]
        assert tv == _fmt(c["max_pairwise_tv"], "%.6f"), f"{name}: TV"
        assert ret == _fmt(c["return_10eps"], "%.2f"), f"{name}: return"
        assert sat == _fmt(100 * c["first_tanh_saturated_fraction"], "%.1f") + "%", f"{name}: saturation"
        assert feat == _fmt(c["feat_max"], "%.2f"), f"{name}: feature max"
    # the point of the control: it reproduces the 0.740 already published in the scope table
    assert control["arms"]["classic"]["max_pairwise_tv"] == pytest.approx(0.739958, abs=5e-7)
    assert re.search(r"\|\s*`coinrun`\s*\|\s*\*\*0\.740\*\*", readme), "scope table's coinrun row"


# ---------------------------------------------------------------- the duplicate arms

def test_the_two_duplicate_pairs_are_one_program(audit):
    A = audit["arms"]
    for x, y in (("ae", "recon"), ("contrastive", "c_aug_noise")):
        kx = {k: v for k, v in A[x].items() if k not in ("seconds", "family")}
        ky = {k: v for k, v in A[y].items() if k not in ("seconds", "family")}
        assert kx == ky, f"{x}/{y} no longer produce identical rows"


def test_ae_and_recon_are_the_same_network():
    """The README says 14 distinct front ends, not 16 -- that claim is about the code."""
    import gymnasium as gymn
    from models.world_model_extractors import AEExtractor, ReconExtractor

    space = gymn.spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)
    torch.manual_seed(0)
    ae = AEExtractor(space, features_dim=512)
    torch.manual_seed(0)
    rc = ReconExtractor(space, features_dim=512)
    assert sum(p.numel() for p in ae.parameters()) == sum(p.numel() for p in rc.parameters())
    renamed = {k.replace("fc.", "fc_enc."): v for k, v in ae.state_dict().items()}
    assert set(renamed) == set(rc.state_dict()), f"{set(renamed) ^ set(rc.state_dict())}"
    for k, v in rc.state_dict().items():
        assert torch.equal(renamed[k], v), f"{k}: weights differ"
    x = torch.zeros(2, 3, 64, 64)
    ae.eval(), rc.eval()
    with torch.no_grad():
        assert torch.equal(ae(x), rc(x)), "forward outputs differ"


# ------------------------------------------------------- the three-game fold (item 9 scope)

GAMES = ("starpilot", "bossfight", "dodgeball")


def _arms(game):
    return _load(f"suite_validity_{game}.json")["arms"]


def _measured(arms):
    return {k: v for k, v in arms.items() if "error" not in v}


def test_per_game_table_matches_the_three_artifacts(readme):
    rows = _table_rows(readme, "**The comparison's other two games, same recipe")
    assert len(rows) == 3, f"expected one row per game, got {len(rows)}"
    for cells in rows:
        game, measured, below, sat, largest, hours = _code(cells)
        arms, ok = _arms(game), _measured(_arms(game))
        assert measured == f"{len(ok)}/{len(arms)}", f"{game}: measured"
        n = len(ok)
        assert below == f"{sum(1 for v in ok.values() if v['max_pairwise_tv'] < 5e-3)}/{n}"
        assert sat == f"{sum(1 for v in ok.values() if v['first_tanh_saturated_fraction'] >= 0.99)}/{n}"
        # the cell reads `0.287505` (`c_aug_noise` -- possibly with a note about its twin arm
        m = re.match(r"`([0-9.]+)` \(`([a-z_]+)`", cells[4].strip())
        assert m, f"{game}: largest-TV cell is not in the published form: {cells[4]!r}"
        tv, arm = m.groups()
        top = max(ok.values(), key=lambda v: v["max_pairwise_tv"])
        assert tv == _fmt(top["max_pairwise_tv"], "%.6f"), f"{game}: largest TV"
        assert arms[arm]["max_pairwise_tv"] == top["max_pairwise_tv"], f"{game}: {arm} is not the top"
        assert hours == _fmt(sum(v["seconds"] for v in ok.values()) / 3600, "%.2f"), f"{game}: hours"


def test_all_forty_eight_cells_and_forty_seven_measured_arms(readme):
    tot = sum(len(_arms(g)) for g in GAMES)
    ok = sum(len(_measured(_arms(g))) for g in GAMES)
    assert (tot, ok) == (48, 47)
    assert all(v["distinct_argmax_across_games"] == 1
               for g in GAMES for v in _measured(_arms(g)).values())
    for lit in (f"`{tot}` cells", f"`{tot * 100000 / 1e6:.1f}M` steps",
                f"`{sum(v['seconds'] for g in GAMES for v in _measured(_arms(g)).values()) / 3600:.2f} h`",
                f"**All {ok} measured arms emit one argmax"):
        assert lit in readme, f"not published: {lit}"


def test_dodgeball_dissociates_saturation_from_collapse(readme):
    ok = _measured(_arms("dodgeball"))
    inert = sum(1 for v in ok.values() if v["max_pairwise_tv"] < 5e-3)
    sat = sum(1 for v in ok.values() if v["first_tanh_saturated_fraction"] >= 0.99)
    assert (inert, sat) == (15, 5) and inert > sat
    assert f"`dodgeball` `{inert}/{len(ok)}` arms are inert while only `{sat}/{len(ok)}`" in readme


# ------------------------------------------------------- the vae failure and its seed probe

def test_vae_bossfight_is_recorded_as_a_failure_not_a_zero(readme):
    v = _arms("bossfight")["vae"]
    assert "error" in v and "max_pairwise_tv" not in v, "the errored cell must not look like a 0.0"
    assert "nan" in v["error"].lower()
    assert v["steps_logged"] < 100000, "the audit records how far the run got"
    assert _fmt(v["steps_logged"], "`%d`") in readme, "the published step count is not the artifact's"
    assert "`100000`" in readme


def test_vae_seed_probe_cells_match_their_artifact(readme):
    d = _load("suite_validity_bossfight_vae_seeds.json")
    assert set(d["arms"]) == {"vae@43", "vae@44"}
    published = {"vae@43": ("0.000013", "65.04", "0.40"), "vae@44": ("0.037041", "40.47", "0.10")}
    for key, (tv, feat, ret) in published.items():
        a = d["arms"][key]
        assert _fmt(a["max_pairwise_tv"], "%.6f") == tv, f"{key}: TV {a['max_pairwise_tv']}"
        assert _fmt(a["feat_max"], "%.2f") == feat, f"{key}: feature max"
        assert _fmt(a["return_10eps"], "%.2f") == ret, f"{key}: return"
        assert tv in readme and feat in readme, f"{key}: {tv}/{feat} not published"
    # the arm is well-behaved on starpilot, which is what the README contrasts against
    assert _fmt(_arms("starpilot")["vae"]["feat_max"], "%.2f") == "233.98"
    assert "`233.98`" in readme


def test_vae_dodgeball_feature_explosion_is_the_published_number(readme):
    v = _arms("dodgeball")["vae"]
    assert v["feat_max"] > 1e13
    assert _fmt(v["feat_max"], "%.1f") in readme, "the exact peak is not published"
    assert "7.2e13" in readme
