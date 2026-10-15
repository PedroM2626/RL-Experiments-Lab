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


def test_all_fifty_five_runs_are_complete_and_on_the_shared_grid(merged):
    """40 dose-response runs (8 arms x 5 seeds -- the seed-42 five of the first four
    arms are the committed 3.6.4 artifacts) plus 15 lever runs (3 levers x 5 seeds,
    completed from n=2 by the evening pass)."""
    files = [r["file"] for rows in merged["runs"].values() for r in rows]
    files += [r["file"] for d in merged["levers"].values() for r in d["rows"]]
    assert len(files) == len(set(files)) == 55
    for fn in files:
        d = _load(fn)
        assert [r["steps"] for r in d["curve"]] == GRID, f"{fn}: grid"
        assert len(d["curve"]) == 101 and d["cross_game"] is not None
        p = d["_provenance"]
        assert p["timesteps"] == 50000 and p["probe_every"] == 500


# ----------------------------------------------------------------- the replication table

def test_the_curve_is_monotone_with_a_cliff_between_three_and_four_levels(merged):
    counts = {arm: merged["by_arm"][arm]["end_responsive"] for arm in
              ("1x256", "2x128", "3x64", "4x64", "8x32", "16x16", "16x256")}
    assert counts == {"1x256": 0, "2x128": 0, "3x64": 0, "4x64": 3, "8x32": 3,
                      "16x16": 4, "16x256": 5}
    assert merged["by_arm"]["coinrun"]["end_responsive"] == 5


def test_the_three_level_arm_is_the_cliffs_middle(merged):
    """3x64 is the cell that moved the cliff: ~3 levels per update still ends dead on
    every seed, so the jump in survival probability happens between 3 and 4 levels."""
    rows = merged["runs"]["3x64"]
    assert [r["first_dead_at"] for r in rows] == [14000, 15000, 22500, 30000, 20000]
    assert all(r["final_tv"] <= ALIVE for r in rows)
    assert sorted(r["alive_probes"] for r in rows) == [28, 29, 42, 46, 58]


def test_the_two_level_arm_is_indistinguishable_from_the_published_one(merged):
    """2x128 doubles the update's distinct levels and changes nothing: this is the
    cell that makes the levels factor a threshold rather than a slope."""
    rows = merged["runs"]["2x128"]
    assert [r["first_dead_at"] for r in rows] == [10000, 13000, 8000, 17000, 18500]
    assert all(r["final_tv"] <= 6.5e-05 for r in rows)
    assert sorted(r["alive_probes"] for r in rows) == [17, 47, 47, 48, 51]


def test_the_eight_level_arm_matches_the_four_level_rate(merged):
    rows = merged["runs"]["8x32"]
    deads = sorted((r["first_dead_at"] for r in rows), key=lambda x: (x is None, x))
    assert deads == [16000, 42000, None, None, None]
    best = max(rows, key=lambda r: r["final_tv"])
    assert best["seed"] == 46 and best["final_tv"] == pytest.approx(0.156991, abs=1e-6)
    assert sorted(r["alive_probes"] for r in rows) == [76, 83, 100, 101, 101]


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

def test_the_input_change_and_entropy_levers_fail_at_five_seeds(merged):
    # repeat4: 4/5 dead, one named survivor
    rows = merged["levers"]["repeat4"]["rows"]
    assert [r["first_dead_at"] for r in rows] == [17000, 29500, None, 10500, 15000]
    survivor = next(r for r in rows if r["first_dead_at"] is None)
    assert (survivor["seed"], survivor["final_tv"], survivor["cross_game_tv"]) == \
        (44, 0.002981, 0.013026), "the one lucky seed, named in the section"
    for r in rows:
        if r["first_dead_at"] is not None:
            assert r["final_tv"] <= 3.6e-05
    # entcoef: 0/5, and the bonus still did its own job -- the marginal stayed
    # entropic while every endpoint went blind (seed 45's V(s) std 0.00342 is the
    # same critic-outlives-policy dissociation claim 1 measures on 4x64)
    rows = merged["levers"]["entcoef001"]["rows"]
    assert [r["first_dead_at"] for r in rows] == [10500, 18000, 8000, 13000, 6500]
    assert all(r["final_tv"] <= 2.2e-05 for r in rows)
    ents = sorted(r["final_entropy"] for r in rows)
    assert (ents[0], ents[-1]) == (1.905, 2.5283), "held at the near-uniform marginal"


def test_the_coverage_lever_survives_on_all_five_seeds(merged):
    rows = merged["levers"]["nsteps1024"]["rows"]
    assert all(r["first_dead_at"] is None and r["alive_probes"] == 101 for r in rows)
    assert [r["final_tv"] for r in rows] == [0.005821, 0.019111, 0.004369, 0.018883, 0.009897]
    assert max(r["cross_game_tv"] for r in rows) == 0.046536


def test_the_coverage_cliff_rule_holds_across_the_whole_grid(merged):
    """The computable form of claim 5's closing count, at five seeds everywhere:
    below the cliff -- 1 to ~3 distinct levels at the published 256 transitions, with
    or without repeat and entropy pressure -- 24 of 25 seeds die (the one survivor is
    the repeat4 seed the section names); at or above it, 20 of 25 survive."""
    below = (list(merged["runs"]["1x256"]) + list(merged["runs"]["2x128"])
             + list(merged["runs"]["3x64"])
             + merged["levers"]["repeat4"]["rows"]
             + merged["levers"]["entcoef001"]["rows"])
    assert len(below) == 25
    alive_below = [r for r in below if r["final_tv"] > ALIVE]
    assert len(alive_below) == 1 and alive_below[0]["seed"] == 44
    at_or_above = (list(merged["runs"]["4x64"]) + list(merged["runs"]["8x32"])
                   + list(merged["runs"]["16x16"]) + list(merged["runs"]["16x256"])
                   + merged["levers"]["nsteps1024"]["rows"])
    assert len(at_or_above) == 25
    assert sum(1 for r in at_or_above if r["final_tv"] > ALIVE) == 20


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
    assert len(rows) == 9, f"header + 8 runs, got {len(rows)}"
    key = {"maze`, `1×256`": "1x256", "maze`, `2×128`": "2x128", "maze`, `3×64`": "3x64",
           "maze`, `4×64`": "4x64", "maze`, `8×32`": "8x32", "maze`, `16×16`": "16x16",
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
    assert n_found == 8, f"the replication table must have 8 data rows, saw {n_found}"


def test_lever_table_cells_match_the_artifacts(readme, merged):
    rows = _rows(readme, "levers split exactly along")
    assert len(rows) == 4, f"header + 3 levers, got {len(rows)}"
    key = {"`action_repeat=4`": "repeat4", "`ent_coef=0.01`": "entcoef001",
           "`n_steps=1024`": "nsteps1024"}
    for cells in rows[1:]:
        tag = next(v for k, v in key.items() if k in cells[0])
        rs = merged["levers"][tag]["rows"]
        assert cells[2].replace("**", "") == f"{sum(1 for r in rs if r['final_tv'] > ALIVE)}/5"
        dead = ["never" if r["first_dead_at"] is None else str(r["first_dead_at"]) for r in rs]
        assert cells[3] == ", ".join(dead), f"{cells[0]}: {cells[3]}"
        got = [float(x) for x in cells[4].split(", ")]
        assert got == [r["final_tv"] for r in rs], f"{cells[0]}: endpoints {cells[4]}"


def test_section_prose_carries_the_measured_numbers(readme):
    for s in (
        "the first dead probe lands anywhere in **9k–36k** (median 16.5k)",
        "V(s) std `0.23554`",
        "**`16×256`: 5/5 seeds, not one dead probe in 505",
        "cross-game TV **`0.664942` with 2 argmax across games**",
        "dips below the dead threshold somewhere in 9.5k–24.5k",
        "2 of 5 baseline seeds are already dead by their 35th–37th update",
        "the levels factor is a cliff, and it sits between 3 and 4",
        "**it ends dead on 0/5 seeds** (deaths `14000–30000`, alive counts `28–58`)",
        "The survival probability is 0 across fifteen 1–3-level seeds",
        "final cross-frame TV **`0.156991`**, seed 46",
        "24 of 25 seeds die**",
        "the single exception being `repeat4` seed 44",
        "20 of 25 seeds survive**",
        "cross-game TVs up to `0.046536`",
        "final entropy `1.905`–`2.5283`",
        "one lucky seed out of five is what \"mostly excluded\" looks like",
        "The account is now quantitative twice over",
        "47.3 GPU-minutes",
        "**15.0**",
        "**76.8**",
        "**145.3 GPU-minutes**",
        "the cliff sits between three and four",
        "**`16×256`, the only arm that replicated 5/5**",
    ):
        assert s in readme, f"prose lost: {s!r}"


def test_the_roadmap_row_carries_the_five_seed_verdicts(readme):
    i = readme.index("**Then the (iv) dose + the levers at five seeds**")
    row = readme[i:readme.index("\n", i)]
    for s in ("cliff **between 3 and 4**", "`3×64` dies 0/5", "24 of 25 seeds die",
              "20 of 25 survive", "`action_repeat=4` dies 4/5 (one surviving seed, named in the section)",
              "`ent_coef=0.01` dies 5/5", "`n_steps=1024` survives 5/5"):
        assert s in row, f"row lost: {s!r}"


def test_the_3_6_4_note_and_section_0_point_at_the_replication(readme):
    i = readme.index("§3.6.5 replicates this across five seeds")
    assert "`0/5` vs `5/5` ending responsive" in readme[i:i + 400]
    j = readme.index("**Follow-up (12/10/2026):** five seeds per arm ran")
    assert "the death clock moves (9k–36k across seeds)" in readme[j:j + 500]
    assert "§3.6.5." in readme[j:j + 700]
