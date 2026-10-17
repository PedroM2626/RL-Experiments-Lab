"""Gate README 3.6.13 (the 500k test) on its artifacts.

Sixty cells at 5x the published budget, paired with the 100k grid and the 250k sweep.
The claims: the response measure deepens monotonically for the readers (30/30 cells
at 500k) and not for the non-readers; bossfight leaves zero carrying the study's
sharpest reading-score dissociation (cbam 1.14 reading deeply, classic 0.02 reading
just as deeply, vit 0.64 reading nothing); and return stays seed noise at every
budget tested. Recomputed from the fold, whose freshness is itself gated.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from summarize_budget500k import fold  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")
ARMS = ("classic", "cbam", "vit", "mlp_vector")
GAMES = ("starpilot", "dodgeball", "bossfight")


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def budget():
    return _load("suite_validity_nenv16_budget500k.json")


@pytest.fixture(scope="module")
def readme():
    with open(README, encoding="utf-8") as f:
        return f.read()


def test_the_500k_artifact_matches_a_fresh_fold(budget):
    assert fold() == budget, "stale: rerun summarize_budget500k.py"


def test_all_sixty_cells_measured_at_500k():
    for g in GAMES:
        rows = _load(f"suite_validity_nenv16_{g}_500k.json")["arms"]
        assert len(rows) == 80 - 60 or True  # the audit fuses 20 arm@seed rows per game
        assert len(rows) == 20, f"{g}: {len(rows)} rows"
        for arm in ARMS:
            for seed in (42, 43, 44, 45, 46):
                assert f"{arm}@{seed}" in rows
                assert "error" not in rows[f"{arm}@{seed}"], f"{g}/{arm}@{seed}"


def test_every_reader_cell_clears_01_at_500k(budget):
    for arm in ("classic", "cbam"):
        for g in GAMES:
            cell = budget["arms"][arm][g]
            assert cell["tv_cells_500k_over_0p1"] == 5, f"{arm}/{g}"
            assert cell["tv_mean_500k"] > cell["tv_mean_250k"] if "tv_mean_250k" in cell \
                else cell["tv_mean_500k"] > cell["tv_mean_100k"], f"{arm}/{g} did not deepen"
    assert budget["arms"]["cbam"]["starpilot"]["tv_mean_500k"] == 0.681832
    chain = (budget["arms"]["cbam"]["starpilot"]["tv_mean_100k"],
             _load("suite_validity_nenv16_budget250k.json")["arms"]["cbam"]["starpilot"]["tv_mean_250k"],
             budget["arms"]["cbam"]["starpilot"]["tv_mean_500k"])
    assert chain == (0.253881, 0.538095, 0.681832)


def test_the_non_readers_do_not_wake_at_500k(budget):
    v = budget["arms"]["vit"]
    assert v["bossfight"]["tv_mean_500k"] == 0.0, "vit's frontal collapse is budget-stable"
    assert v["bossfight"]["tv_cells_500k_over_0p1"] == 0
    assert v["dodgeball"]["tv_mean_500k"] == 1e-06
    m = budget["arms"]["mlp_vector"]["dodgeball"]
    assert (m["tv_mean_100k"], m["tv_mean_500k"]) == (0.049736, 8e-05)


def test_bossfight_leaves_zero_with_the_sharpest_dissociation(budget):
    cb, cl, vi = (budget["arms"][a]["bossfight"] for a in ("cbam", "classic", "vit"))
    assert cb["mean_500k"] == 1.14 and cb["tv_mean_500k"] == 0.439787
    assert cl["mean_500k"] == 0.02 and cl["tv_mean_500k"] == 0.441471
    assert vi["mean_500k"] == 0.64 and vi["tv_mean_500k"] == 0.0
    # the mlp arm climbs too, reading weakly
    assert budget["arms"]["mlp_vector"]["bossfight"]["mean_500k"] == 0.9
    assert budget["arms"]["mlp_vector"]["bossfight"]["tv_cells_500k_over_0p1"] == 3


def test_return_stays_seed_noise_at_500k(budget):
    deltas = [budget["arms"][a][g]["delta_vs_100k"] for a in ARMS for g in GAMES]
    assert max(deltas, key=abs) == 0.84, "the only real learner is cbam on bossfight"
    signs = [d > 0 for d in deltas]
    assert sum(signs) == 6 and sum(not s for s in signs) == 6, "mixed signs, no separation"
    assert budget["arms"]["classic"]["starpilot"]["delta_vs_100k"] == -0.12
    assert budget["arms"]["cbam"]["dodgeball"]["delta_vs_100k"] == -0.48


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


def test_section_table_cells_match_the_fold(readme, budget):
    rows = _rows(readme, "### 3.6.13.")
    data = [r for r in rows if len(r) == 6 and r[0] != "arm"]
    assert len(data) == 12, f"12 data rows expected, got {len(data)}"
    for cells in data:
        arm, game = cells[0].replace("`", ""), cells[1]
        c = budget["arms"][arm][game]
        means = cells[2].replace("**", "")
        assert means == f"{c['mean_100k']:.2f} → {c['mean_500k']:.2f}", f"{arm}/{game}"
        delta = cells[3].replace("−", "-").replace("**", "")
        want = f"{c['delta_vs_100k']:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00")
        assert delta == want, f"{arm}/{game}: {delta} vs {want}"
        tv = cells[4].replace("**", "")
        assert tv == f"{c['tv_mean_100k']:.6f} → {c['tv_mean_500k']:.6f}", f"{arm}/{game}: {tv}"
        assert cells[5] == f"{c['tv_cells_500k_over_0p1']}/5"


def test_section_prose_carries_the_measured_numbers(readme):
    for s in (
        "**23.26 summed GPU-hours, 3h of wall clock at 6 workers**",
        "**30 of 30** reader cells",
        "`0.253881 → 0.538095 → 0.681832`",
        "exactly `0.000000`",
        "dies in practice (`0.049736 → 0.000080`)",
        "TV `0.439787`, `5/5` cells",
        "scores **`0.02`**",
        "a total non-reader at `0.64`",
        "no larger budget within this study's reach was left untested",
        "it is *budget-informative*",
        "the budget axis is now measured at `100k`, `250k` and `500k`",
        "at `500k` it is simply what the data does",
    ):
        assert s in readme, f"prose lost: {s!r}"


def test_the_roadmap_row_and_followup_note_carry_the_verdict(readme):
    i = readme.index("| 12 | **The last budget questions")
    row = readme[i:readme.index("\n", i)]
    for s in ("30/30", "`0.253881 → 0.538095 → 0.681832`", "`1.14`", "`0.02`", "`0.64`",
              "23.26 summed GPU-hours", "Return never tracks the front end at `100k`, `250k` or `500k`"):
        assert s in row, f"row 12 lost: {s!r}"
    j = readme.index("**Follow-up (14/10/2026):** both named gaps closed")
    assert "the arm that reads it most deeply scores `0.02`" in readme[j:j + 600]
