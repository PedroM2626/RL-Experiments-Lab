"""Gate REPORT.md -- the distilled narrative of the dense-probe arc -- on the same
artifacts the README sections cite.

REPORT.md carries no number of its own: every figure it prints is one the README
already publishes and the section tests already gate. This file re-derives the
report's numbers directly from the committed JSONs, so the two documents cannot
drift apart without failing CI.
"""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def report():
    with open(os.path.join(ROOT, "REPORT.md"), encoding="utf-8") as f:
        return f.read()


def test_the_finding_section_matches_the_audit_artifacts(report):
    ck = _load("checkpoint_cross_game_response.json")
    row = _load("suite_validity_starpilot.json")["arms"]
    assert "69/70" in report and "4.8e-4" in report and "47/48" in report
    # the audit artifact carries the 47/48 the report quotes
    cells = {g: _load(f"suite_validity_{g}.json")["arms"] for g in
             ("starpilot", "bossfight", "dodgeball")}
    measured = sum(1 for g in cells for r in cells[g].values() if "error" not in r)
    one_argmax = sum(1 for g in cells for r in cells[g].values()
                     if r.get("distinct_argmax_across_games") == 1)
    assert (measured, one_argmax) == (47, 47)


def test_the_cause_section_matches_the_dense_artifacts(report):
    m = _load("policy_saturation_dense_multiseed.json")
    assert "`101/101` probes alive" in report
    assert m["by_arm"]["16x256"]["end_responsive"] == 5
    assert m["by_arm"]["1x256"]["end_responsive"] == 0
    for s in ("9.5k", "9k–36k", "4/5", "5/5", "5/5"):
        assert s in report, f"cause section lost: {s!r}"
    # levers at five seeds: repeat 1/5, entcoef 0/5, nsteps1024 5/5
    dead = lambda lever: sum(1 for r in m["levers"][lever]["rows"]
                            if r["final_tv"] <= 1e-4)
    assert dead("repeat4") == 4 and dead("entcoef001") == 5 and dead("nsteps1024") == 0


def test_the_threshold_section_matches_the_cliff(report):
    m = _load("policy_saturation_dense_multiseed.json")
    assert "`0, 0, 0, 3, 3, 4, 5`" in report
    assert "15 of 15" in report and "20 of 25" in report and "24/25" in report
    counts = [m["by_arm"][a]["end_responsive"] for a in
              ("1x256", "2x128", "3x64", "4x64", "8x32", "16x16", "16x256")]
    assert counts == [0, 0, 0, 3, 3, 4, 5]


def test_the_grid_section_matches_the_nenv16_artifacts(report):
    g = _load("suite_validity_nenv16_grid.json")
    rank = dict((a, v) for a, v in g["global_mean_ranking"])
    resp = g["responsiveness"]
    assert "`1.907`" in report and "`1.707`" in report
    assert (rank["vit"], rank["impoola"]) == (1.907, 1.707)
    assert "`1/15`" in report and "`3/15`" in report
    assert resp["vit"]["cells_tv_over_0p1"] == 1 and resp["impoola"]["cells_tv_over_0p1"] == 3
    assert "12th–14th" in report
    order = [a for a, _ in g["global_mean_ranking"]]
    assert sorted(order.index(a) + 1 for a in ("spatial", "resnet18", "classic")) == [12, 13, 14]


def test_the_budget_section_matches_the_paired_sweep(report):
    b = _load("suite_validity_nenv16_budget250k.json")["arms"]["vit"]["starpilot"]
    assert "paired delta of `−4.2`" in report
    assert b["paired_deltas"][0] == -4.2
    assert "20/20" in report
    for arm in ("classic", "cbam"):
        for game in ("starpilot", "dodgeball"):
            d = _load("suite_validity_nenv16_budget250k.json")["arms"][arm][game]
            assert d["tv_cells_250k_over_0p1"] == 5


def test_the_anatomy_section_matches_the_diagnostic(report):
    for s in ("`0.154`", "`1.31–3.10`", "zero", "8500", "a third to a tenth"):
        assert s in report, f"anatomy section lost: {s!r}"
    v = _load("feature_vs_actor_vit_bossfight.json")["within_game"]
    assert v["features"]["feat_range_over_scale_mean"] == 0.154
    assert v["features"]["feat_dims_range_over_scale"] == 0.0
    m = _load("feature_vs_actor_mlp_dodgeball.json")["within_game"]
    assert m["action_tv"] == 0.0 and m["features"]["feat_dims_range_over_scale"] > 0


def test_the_bug_section_matches_the_rescue(report):
    for s in ("`91–98%`", "5/5", "`0.379`", "`0.275`", "3/3", "0/3", "`0.053`"):
        assert s in report, f"bug section lost: {s!r}"
    runs = []
    for s in (42, 43, 44, 45, 46):
        fn = "probe_mlpscaled_dodgeball.json" if s == 42 else f"probe_mlpscaled_dodgeball_seed{s}.json"
        c = _load(fn)["curve"]
        assert all(r["cross_frame_tv_mean"] >= 5e-5 for r in c), f"s{s} died"
    best = _load("probe_mlpscaled_bossfight_seed44.json")["cross_game"]["max_pairwise_tv"]
    assert best == 0.379173, "the run the report credits with exceeding classic"


def test_the_honesty_and_answer_sections_carry_the_established_facts(report):
    for s in (
        "~56 GPU-hours",
        "every claim recomputed from committed artifacts by CI on every push",
        "4/5",
        "was its seed's",
        "`15/15` readers",
        "one division by 255",
        "starves each update of coverage",
        "the response measure does",
        "`100k`, `250k` and `500k`",
        "`0.254 → 0.538 → 0.682`",
        "reads deeply and scores `1.14`",
        "reads just as deeply and scores `0.02`",
        "reads nothing at all and scores `0.64`",
    ):
        assert s in report, f"lost: {s!r}"
