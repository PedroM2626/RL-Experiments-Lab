"""The published JAX-grid analysis must be derivable from data that is in git.

The raw results_grade/ cell JSONs are git-ignored, so jax_port/cells_summary.json is the
committed record of them. This test rebuilds the rankings from that record and compares
against the committed jax_port/analysis_full.json — if either drifts, the aggregate study
in README sections 15-17 is no longer backed by anything.
"""

import json
import os

from jax_port.analyze_grade import SummaryLoader, build_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(*rel):
    with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
        return json.load(f)


def test_summary_covers_every_suite_in_the_analysis():
    summary = load("jax_port", "cells_summary.json")
    analysis = load("jax_port", "analysis_full.json")
    derived = {"meta", "temporal_delta", "gen_gap", "marl"}
    assert set(summary) - {"marl"} == set(analysis) - derived
    assert summary["marl"], "no MARL cells recorded"


def test_analysis_is_reproducible_from_the_committed_summary():
    loader = SummaryLoader(os.path.join(ROOT, "jax_port", "cells_summary.json"))
    rep, _ = build_report(loader)
    published = {k: v for k, v in load("jax_port", "analysis_full.json").items() if k != "meta"}
    rebuilt = {k: v for k, v in rep.items() if k != "meta"}
    assert json.dumps(rebuilt, sort_keys=True) == json.dumps(published, sort_keys=True), (
        "analysis_full.json is stale with respect to cells_summary.json — "
        "re-run: python -m jax_port.analyze_grade --from_summary jax_port/cells_summary.json")


def test_marl_cells_are_all_recorded_with_a_winrate():
    rows = load("jax_port", "cells_summary.json")["marl"]
    assert rows
    assert all("winrate" in r and "map" in r and "algo" in r for r in rows)
