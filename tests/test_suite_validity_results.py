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
        if set("".join(cells)) <= set("-: ") or cells[0].startswith("arm"):
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
    assert f"(`vit`, max `{lo['feat_max']:.2f}`" in readme
    assert f"(`impala`, max `{hi['feat_max']:.2f}`" in readme
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
