"""Gate README 3.6.5 (the five-seed replication and the mechanism levers) on its artifacts.

Section 3.6.4's dose-response was one seed per arm; this file checks what five seeds say
about it, and what the three levers did. Every number is recomputed from the per-run
files, not trusted from the merged artifact -- and the merged artifact itself is verified
against a fresh fold, so a per-run file edited without re-running the summarizer fails
here rather than in front of a reader.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")
GRID = list(range(0, 50001, 500))
DEAD, ALIVE = 5e-5, 1e-4
MERGED = "policy_saturation_dense_multiseed.json"

sys.path.insert(0, ROOT)
from summarize_dense_multiseed import by_arm, collect  # noqa: E402


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def merged():
    return _load(MERGED)


@pytest.fixture(scope="module")
def readme():
    with open(README, encoding="utf-8") as f:
        return f.read()


# ----------------------------------------------------------------- the fold is current

def test_the_merged_artifact_matches_the_per_run_files(merged):
    arms, levers = collect()
    for arm, rows in arms.items():
        assert by_arm(rows) == merged["by_arm"][arm], f"{arm}: merged artifact is stale"
        for a, b in zip(rows, merged["runs"][arm]):
            assert a == b, f"{arm}: run rows drifted"
    for lever, d in levers.items():
        for a, b in zip(d["rows"], merged["levers"][lever]["rows"]):
            assert a == b, f"{lever}: run rows drifted"


def test_all_thirty_one_runs_are_complete_and_on_the_shared_grid(merged):
    """25 dose-response runs (5 arms x 5 seeds, the seed-42 five of them the committed
    3.6.4 artifacts) plus 6 lever runs -- the 26 NEW ones are what 3.6.5 added."""
    files = [r["file"] for rows in merged["runs"].values() for r in rows]
    files += [r["file"] for d in merged["levers"].values() for r in d["rows"]]
    assert len(files) == len(set(files)) == 31
    for fn in files:
        d = _load(fn)
        assert [r["steps"] for r in d["curve"]] == GRID, f"{fn}: grid"
        assert len(d["curve"]) == 101 and d["cross_game"] is not None
        p = d["_provenance"]
        assert p["timesteps"] == 50000 and p["probe_every"] == 500


# ----------------------------------------------------------------- the replication table

def test_the_gradient_is_monotone_in_reliability(merged):
    counts = {arm: merged["by_arm"][arm]["end_responsive"] for arm in
              ("1x256", "4x64", "16x16", "16x256")}
    assert counts == {"1x256": 0, "4x64": 3, "16x16": 4, "16x256": 5}
    assert merged["by_arm"]["coinrun"]["end_responsive"] == 5


def test_the_single_stream_always_ends_blind(merged):
    rows = merged["runs"]["1x256"]
    assert [r["final_tv"] for r in rows] == [0.0, 1e-06, 0.0, 7e-06, 1e-05]
    assert all(r["final_v_std"] <= 2e-05 for r in rows), "V(s) std must die with the policy"
    assert all(r["cross_game_tv"] <= 0.00109 for r in rows)
    assert sorted(r["first_dead_at"] for r in rows) == [9000, 9500, 16500, 31500, 36000]
    assert sorted(r["alive_probes"] for r in rows) == [21, 38, 44, 70, 74]


def test_the_critic_can_outlive_the_policy(merged):
    """4x64 seed 45 ends with V(s) std 0.236 and cross-frame TV 2e-6: the value head
    still tracks the rollout while the actor no longer reads the screen. That kills
    'the rollout contains nothing learnable' as the collapse's explanation."""
    row = next(r for r in merged["runs"]["4x64"] if r["seed"] == 45)
    assert row["final_tv"] == 2e-06 and row["final_v_std"] == 0.23554


def test_only_the_staircase_arm_replicates_perfectly(merged):
    rows = merged["runs"]["16x256"]
    assert [r["first_dead_at"] for r in rows] == [None] * 5, "one dead probe would break 5/5"
    assert [r["alive_probes"] for r in rows] == [101] * 5
    assert all(r["final_v_std"] > 1.0 for r in rows)
    assert all(r["l1_max"] <= 0.008 for r in rows), "the first Tanh never saturates here"
    assert merged["by_arm"]["16x256"]["two_argmax_seeds"] == [42, 44]


def test_the_wide_rollout_reaches_coinrun_grade_on_its_best_seed(merged):
    row = next(r for r in merged["runs"]["16x16"] if r["seed"] == 45)
    assert row["cross_game_tv"] == pytest.approx(0.664942, abs=1e-6)
    assert row["cross_game_argmax"] == 2
    assert row["final_tv"] == pytest.approx(0.085055, abs=1e-6)
    control = _load("suite_validity_coinrun_control.json")["arms"]["classic"]["max_pairwise_tv"]
    assert row["cross_game_tv"] / control == pytest.approx(0.90, abs=0.005), "0.90x the control"
    # and the exception that keeps the arm out of the perfect class
    late = next(r for r in merged["runs"]["16x16"] if r["seed"] == 43)
    assert late["first_dead_at"] == 34500 and late["final_tv"] == 0.0


def test_coinruns_property_is_recovery_not_immunity(merged):
    rows = merged["runs"]["coinrun"]
    assert sum(1 for r in rows if r["first_dead_at"] is None) == 1, "seed 42 was the lucky one"
    assert all(r["final_tv"] > ALIVE for r in rows), "every seed ends responsive"
    finals = sorted(r["final_tv"] for r in rows)
    assert finals[0] == pytest.approx(0.011613, abs=1e-6) and finals[-1] == pytest.approx(0.084866, abs=1e-6)
    assert all(0.2 < r["final_v_std"] < 0.8 for r in rows)


# ----------------------------------------------------------------- the mechanism levers

def test_the_input_change_and_entropy_levers_fail_on_both_seeds(merged):
    for lever, deads in (("repeat4", [17000, 29500]), ("entcoef001", [10500, 18000])):
        rows = merged["levers"][lever]["rows"]
        assert [r["first_dead_at"] for r in rows] == deads
        assert all(r["final_tv"] <= 2.2e-05 for r in rows)
        assert all(r["final_v_std"] <= 1e-4 for r in rows)
    # the entropy bonus did its own job: the marginal stayed entropic while the policy
    # stopped conditioning on the observation
    assert [r["final_entropy"] for r in merged["levers"]["entcoef001"]["rows"]] == [2.2751, 2.3627]
    # and repeat4's seed 43 is the non-absorbing revival, ending at exact zero
    s43 = merged["levers"]["repeat4"]["rows"][1]
    assert s43["peak_tv"] == pytest.approx(0.149056, abs=1e-6) and s43["final_tv"] == 0.0


def test_the_coverage_lever_survives_on_both_seeds(merged):
    rows = merged["levers"]["nsteps1024"]["rows"]
    assert all(r["first_dead_at"] is None and r["alive_probes"] == 101 for r in rows)
    assert [r["final_tv"] for r in rows] == [0.005821, 0.019111]
    assert all(r["final_v_std"] > 0.6 for r in rows)


def test_the_coverage_rule_holds_across_all_twenty_six_runs(merged):
    """The computable form of the section's closing claim: every configuration at the
    published coverage (256 transitions from one stream, with or without repeat and
    entropy pressure) ends dead on every seed; responsive endpoints appear only in the
    widened configurations -- 14 of 17."""
    at_published = (list(merged["runs"]["1x256"])
                    + merged["levers"]["repeat4"]["rows"]
                    + merged["levers"]["entcoef001"]["rows"])
    assert len(at_published) == 9
    assert all(r["final_tv"] <= ALIVE for r in at_published), "an at-coverage run survived"
    widened = (list(merged["runs"]["4x64"]) + list(merged["runs"]["16x16"])
               + list(merged["runs"]["16x256"]) + merged["levers"]["nsteps1024"]["rows"])
    assert len(widened) == 17
    assert sum(1 for r in widened if r["final_tv"] > ALIVE) == 14


# ----------------------------------------------------------------- the published prose

def _rows(text, header_start):
    i = text.index(header_start)
    out = []
    for line in text[i:].splitlines():
        if not line.startswith("|"):
            if out:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        out.append(cells)
    return out


def test_replication_table_cells_match_the_artifacts(readme, merged):
    rows = _rows(readme, "### 3.6.5.")
    assert len(rows) == 6, f"header + 5 runs, got {len(rows)}"
    key = {"maze`, `1×256`": "1x256", "maze`, `4×64`": "4x64", "maze`, `16×16`": "16x16",
           "maze`, `16×256`": "16x256", "`coinrun`, `1×256`": "coinrun"}
    n_found = 0
    for cells in rows:
        tag = next((v for k, v in key.items() if k in cells[0]), None)
        if tag is None or len(cells) != 4:
            continue
        n_found += 1
        agg = merged["by_arm"][tag]
        assert cells[1].replace("**", "") == f"{agg['end_responsive']}/{agg['seeds']}", cells[0]
        dead = ["never" if x is None else str(x) for x in agg["first_dead_at"]]
        assert cells[2] == ", ".join(dead), f"{cells[0]}: dead column says {cells[2]}"
        alive = ", ".join(str(x) for x in agg["alive_probes"])
        assert cells[3] == alive, f"{cells[0]}: alive column says {cells[3]}"
    assert n_found == 5, f"the replication table must have 5 data rows, saw {n_found}"


def test_lever_table_cells_match_the_artifacts(readme, merged):
    rows = _rows(readme, "levers split exactly along")
    assert len(rows) == 4, f"header + 3 levers, got {len(rows)}"
    key = {"`action_repeat=4`": "repeat4", "`ent_coef=0.01`": "entcoef001",
           "`n_steps=1024`": "nsteps1024"}
    for cells in rows[1:]:
        tag = next(v for k, v in key.items() if k in cells[0])
        for cell, row in zip(cells[2:], merged["levers"][tag]["rows"]):
            dead, final = row["first_dead_at"], row["final_tv"]
            if dead is None:
                assert cell == f"alive, 101/101, ends TV `{final:.6f}`", cells[0]
            else:
                assert f"dead at {dead}" in cell, f"{cells[0]}: {cell}"
                assert f"ends TV `{final:g}`" in cell or "ends TV" not in cell, cells[0]


def test_section_prose_carries_the_measured_numbers(readme):
    for s in (
        "the first dead probe lands anywhere in **9k–36k** (median 16.5k)",
        "V(s) std `0.23554`",
        "**`16×256`: 5/5 seeds, not one dead probe in 505",
        "cross-game TV **`0.664942` with 2 argmax across games**",
        "dips below the dead threshold somewhere in 9.5k–24.5k",
        "dead at 29500, revives to `0.149`, ends TV `0`",
        "final entropy `2.2751`/`2.3627`",
        "2 of 5 baseline seeds are already dead by their 35th–37th update",
        "never survive it (9/9 seeds dead)",
        "survive in 14 of 17 seeds**",
        "47.3 GPU-minutes",
        "**53.6 GPU-minutes**",
        "`2×128` (~2 levels, 256 transitions) against `8×32` (~8 levels, 256 transitions)",
        "**`16×256`, the only arm that replicated 5/5**",
    ):
        assert s in readme, f"prose lost: {s!r}"


def test_the_3_6_4_note_and_section_0_point_at_the_replication(readme):
    i = readme.index("§3.6.5 replicates this across five seeds")
    assert "`0/5` vs `5/5` ending responsive" in readme[i:i + 400]
    j = readme.index("**Follow-up (12/10/2026):** five seeds per arm ran")
    assert "the death clock moves (9k–36k across seeds)" in readme[j:j + 500]
    assert "§3.6.5." in readme[j:j + 700]
