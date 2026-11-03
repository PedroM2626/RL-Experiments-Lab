"""Gate README 15.5 (the port grid under the response measure) and 21.9 (section 21 re-asked
with the response as the target) on their committed artifacts."""
import json
import os
import sys

import pytest

from docs_text import open_docs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "jax_port"))


def _load(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def docs():
    with open_docs() as f:
        return f.read()


@pytest.fixture(scope="module")
def port():
    return _load("jax_port", "port_response_grid.json")


@pytest.fixture(scope="module")
def ident():
    return _load("results", "response_identification.json")


def test_port_fold_is_current(port):
    import summarize_port_response as spr
    rows = _load("jax_port", "port_response_cells.json")["cells"]
    assert spr.fold(rows) == port, "stale: rerun jax_port/summarize_port_response.py"


def test_port_grid_is_complete_and_reads(port):
    rows = _load("jax_port", "port_response_cells.json")["cells"]
    assert len(rows) == 240 and port["_provenance"]["cells"] == 240
    assert {(r["cfg"], r["game"], r["seed"]) for r in rows} == {
        (c, g, s) for c in port["responsiveness"] for g in ("bossfight", "starpilot", "dodgeball")
        for s in (42, 43, 44, 45, 46)}
    r = port["responsiveness"]
    assert (sum(x["cells_tv_over_0p1"] for x in r.values()), sum(x["cells"] for x in r.values())) == (208, 240)
    assert [c for c, x in r.items() if x["cells_tv_over_0p1"] == 15] == ["ae", "aug_noise", "mlp", "recon",
                                                                         "spatial", "vae"]
    assert (r["vit"]["cells_tv_over_0p1"], r["resnet18"]["cells_tv_over_0p1"]) == (12, 8)
    v = port["vs_archived"]
    assert (v["mean_return_rerun"], v["mean_return_archived"], v["mean_abs_cell_delta"]) == (
        1.213042, 1.219292, 0.192583)


def test_port_prose(docs):
    for s in ("**`208/240`** cells clear TV `0.1` (TV mean `0.162441`", "`vit` reads on `12/15`",
              "Mean return `1.213042` against the archived `1.219292`", "**ρ = `0.30`** (p = `0.25`)",
              "**240/240 cells, 0 errors**"):
        assert s in docs, f"15.5 lost: {s!r}"


def test_identification_numbers(ident):
    p = ident["P2_identification"]
    acc = {(g, k): round(x["pooled"]["k_1_j_1"]["accuracy"], 3) for g, v in p.items() for k, x in v.items()}
    assert acc[("sb3_16x256", "architecture|return")] == 0.084
    assert acc[("sb3_16x256", "architecture|response")] == 0.176
    assert acc[("sb3_16x256", "family|response")] == 0.22
    assert acc[("port_64x128", "architecture|response")] == 0.12
    # the response beats the return on the SB3 grid, outside each other's intervals
    ret = p["sb3_16x256"]["architecture|return"]["pooled"]["k_1_j_1"]["ci95_wilson"]
    resp = p["sb3_16x256"]["architecture|response"]["pooled"]["k_1_j_1"]["ci95_wilson"]
    assert ret[1] < resp[0]
    assert ret[0] < p["sb3_16x256"]["architecture|return"]["mean_chance"] < ret[1]


def test_budget_law(ident):
    law = ident["P1_P3_budget_law"]
    for arm in ("classic", "cbam"):
        assert all(x["excludes_zero"] and x["slope_per_decade"] > 0 for x in law[arm]["response"].values())
    ret_excl = [(a, g) for a in law for g, x in law[a]["return"].items() if x["excludes_zero"]]
    assert ret_excl == [("cbam", "bossfight")]
    s = ident["P1_symbolic"]
    assert (s["response"]["linear_game_only"]["r2"], s["response"]["linear_per_game_slope"]["r2"]) == (0.2467, 0.7375)
    assert (s["return"]["linear_game_only"]["r2"], s["return"]["linear_per_game_slope"]["r2"]) == (0.5596, 0.526)
    assert law["cbam"]["response"]["starpilot"]["budget_for_tv_0p5"] == 240000
    assert law["classic"]["response"]["dodgeball"]["budget_for_tv_0p5"] == 1593000


def test_published_defect_is_recorded(ident):
    d = ident["P2_published_defect"]
    assert d["starpilot|main"]["published_profile_was_config"]["cnn_deep"] == "resnet18"
    assert d["_pooled"]["published_artifact"]["k_1_j_1"] == 0.133
    assert d["_pooled"]["corrected"]["k_1_j_1"] == 0.1195
    fixed = _load("results", "symbolic_regression.json")["P2_hidden_family_identification"]
    assert d["_pooled"]["fixed_artifact"] == {k: v["nearest_centroid_accuracy"]
                                              for k, v in sorted(fixed["pooled_across_strata"].items())}


def test_identification_prose(docs):
    for s in ("**0.176** [0.151, 0.203]", "raises it to **`0.738`**", "`0.560` and `0.526`",
              "the first full run returned the constant `0.336`"):
        assert s in docs, f"21.9 lost: {s!r}"
