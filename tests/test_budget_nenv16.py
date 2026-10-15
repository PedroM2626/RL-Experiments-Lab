"""Gate README 3.6.8 (the budget test) on its artifact.

40 cells at 250k, paired seed-by-seed with their own 100k rows from the 3.6.7 grid --
same recipe, same seeds, only the budget changed. Every published number is recomputed
from the folded artifact (and the fold is re-derived, so a merged file edited without
re-folding fails here). The structural claims: the 100k champion was one seed, the
readers' returns move by noise, and the response measure is the quantity that deepens.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from summarize_budget_nenv16 import fold  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")
ARMS = ("classic", "cbam", "vit", "mlp_vector")
GAMES = ("starpilot", "dodgeball")


@pytest.fixture(scope="module")
def budget():
    with open(os.path.join(RESULTS, "suite_validity_nenv16_budget250k.json"),
              encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def readme():
    with open(README, encoding="utf-8") as f:
        return f.read()


def test_the_budget_artifact_matches_a_fresh_fold(budget):
    assert fold() == budget, "stale: rerun summarize_budget_nenv16.py"


def test_all_forty_cells_measured_at_250k(budget):
    for arm in ARMS:
        for game in GAMES:
            d = budget["arms"][arm][game]
            assert len(d["returns_250k"]) == 5 and len(d["returns_100k"]) == 5
            assert len(d["paired_deltas"]) == 5
            assert budget["arms"][arm]["reads"] == (arm in ("classic", "cbam"))


def test_the_100k_champion_was_one_seed(budget):
    v = budget["arms"]["vit"]["starpilot"]
    assert v["returns_100k"][0] == 6.4 and v["returns_250k"][0] == 2.2
    assert v["paired_deltas"][0] == -4.2
    assert max(abs(x) for x in v["paired_deltas"][1:]) <= 1.2, "only the seed delta is big"
    assert v["delta"] == -0.54
    assert budget["arms"]["vit"]["dodgeball"]["delta"] == -0.68


def test_reader_returns_move_by_noise(budget):
    for arm in ("classic", "cbam"):
        for game in GAMES:
            d = budget["arms"][arm][game]
            assert abs(d["delta"]) <= 0.24, f"{arm}/{game}: a reader grew/shrank too much"
    assert budget["arms"]["classic"]["starpilot"]["paired_deltas"] == [-0.9, -0.2, 0.3, 1.4, -0.7]
    assert budget["arms"]["classic"]["dodgeball"]["delta"] == 0.24


def test_every_reader_cell_stays_responsive_and_deepens(budget):
    for arm in ("classic", "cbam"):
        for game in GAMES:
            d = budget["arms"][arm][game]
            assert (d["tv_cells_100k_over_0p1"], d["tv_cells_250k_over_0p1"]) == (5, 5)
            assert d["tv_mean_250k"] > d["tv_mean_100k"], f"{arm}/{game} did not deepen"
    assert budget["arms"]["cbam"]["starpilot"]["tv_mean_250k"] == 0.538095
    assert budget["arms"]["classic"]["starpilot"]["tv_mean_250k"] == 0.45106


def test_the_non_readers_do_not_wake(budget):
    d = budget["arms"]["vit"]["dodgeball"]
    assert (d["tv_mean_100k"], d["tv_mean_250k"]) == (6e-05, 0.0)
    assert (d["tv_cells_100k_over_0p1"], d["tv_cells_250k_over_0p1"]) == (0, 0)
    # mlp_vector flattens further on dodgeball: more budget, more blind
    m = budget["arms"]["mlp_vector"]["dodgeball"]
    assert (m["tv_mean_100k"], m["tv_mean_250k"]) == (0.049736, 0.002116)
    # and the partial exception: it stirs on starpilot, to a quarter of classic's depth
    s = budget["arms"]["mlp_vector"]["starpilot"]
    assert s["tv_mean_250k"] == 0.120391 and s["tv_cells_250k_over_0p1"] == 3


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
    rows = _rows(readme, "### 3.6.8.")
    data = [r for r in rows if len(r) == 6 and r[0] != "arm"]
    assert len(data) == 8, f"8 data rows expected, got {len(data)}"
    for cells in data:
        arm, game = cells[0].replace("`", ""), cells[1]
        d = budget["arms"][arm][game]
        means = cells[2].replace("**", "")
        assert means == f"{d['mean_100k']:.2f} → {d['mean_250k']:.2f}", f"{arm}/{game}: {means}"
        delta = cells[3].replace("−", "-").replace("**", "")
        want = f"{d['delta']:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00")
        assert delta == want, f"{arm}/{game}: {delta} vs {want}"
        tv = cells[4].replace("**", "")
        assert tv == f"{d['tv_mean_100k']:.6f} → {d['tv_mean_250k']:.6f}", f"{arm}/{game}: {tv}"
        assert cells[5] == f"{d['tv_cells_100k_over_0p1']} → {d['tv_cells_250k_over_0p1']}"


def test_section_prose_carries_the_measured_numbers(readme):
    for s in (
        "40 cells, 67 min of wall clock at 4 workers",
        "4.17 h of summed cell time",
        "a paired delta of **`−4.2`**",
        "`[-0.9, -0.2, +0.3, +1.4, -0.7]`",
        "**20/20**",
        "`0.253881 → 0.538095`",
        "`0.280602 → 0.451060`",
        "(`0.049736 → 0.002116`)",
        "wakes to `0.120391` with `3` cells",
        "not a `100k` artifact",
        "regresses the `100k` seed-luck champion",
    ):
        assert s in readme, f"prose lost: {s!r}"


def test_the_roadmap_row_and_313_note_carry_the_verdict(readme):
    i = readme.index("| 10 | **Does a bigger budget repair the return ranking")
    row = readme[i:readme.index("\n", i)]
    assert "✅ No — return stays seed noise; the response measure deepens instead" in row
    j = readme.index("**Validity note (14/10/2026):** these runs trained on the single-stream recipe")
    assert "deepens* the response measure instead" in readme[j:j + 600]
