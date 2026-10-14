"""Gate README 3.6.7 (the full grid re-run under the fixed recipe) on its artifacts.

240 cells trained through the producers' own train_one at n_envs=16 x n_steps=256,
fused per game and folded into results/suite_validity_nenv16_grid.json. These tests
recompute every published number from the fold (and the fold itself is re-derived from
the merged files, so a shard edited without re-merging fails here), pin the ranking's
structural claims -- the archived top-3's fall, the non-reading arms holding the top,
the response cliff -- and check the merge refuses to fuse shards across games.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audit_suite_validity import merge  # noqa: E402
from summarize_grid_nenv16 import fold  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")
ARMS = ("classic", "vit", "impala", "spatial", "cbam", "impoola", "vae", "lstm_attention",
        "c_aug_crop", "ae", "recon", "resnet18", "mlp_vector", "contrastive",
        "c_aug_noise", "c_aug_color")
GAMES = ("starpilot", "bossfight", "dodgeball")


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def grid():
    return _load("suite_validity_nenv16_grid.json")


@pytest.fixture(scope="module")
def merged():
    return {g: _load(f"suite_validity_nenv16_{g}.json") for g in GAMES}


@pytest.fixture(scope="module")
def readme():
    with open(README, encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------- the fold is current

def test_the_grid_artifact_matches_a_fresh_fold(grid):
    assert fold() == grid, "the grid artifact is stale: rerun summarize_grid_nenv16.py"


def test_all_240_cells_are_present(merged):
    for g in GAMES:
        rows = merged[g]["arms"]
        assert len(rows) == 80, f"{g}: {len(rows)} rows"
        for arm in ARMS:
            for seed in (42, 43, 44, 45, 46):
                assert f"{arm}@{seed}" in rows, f"{g}: {arm}@{seed} missing"
        p = merged[g]["_provenance"]
        assert (p["game"], p["timesteps"], p["n_envs"], p["n_steps"]) == (g, 100000, 16, 256)


def test_the_one_error_cell_is_the_known_nan(merged, grid):
    e = grid["error_cells"]
    assert e == {"bossfight": {"vae": [42]}}
    row = merged["bossfight"]["arms"]["vae@42"]
    assert "ValueError" in row["error"] and row["steps_logged"] == 5888
    # the other four seeds completed: bossfight/vae has 4 measured cells, and vae has
    # 14 responsive-cells overall (15 minus the NaN)
    assert len(grid["games"]["bossfight"]["vae"]["tvs"]) == 4
    assert grid["games"]["bossfight"]["vae"]["error_seeds"] == [42]
    assert grid["responsiveness"]["vae"]["cells"] == 14


# ----------------------------------------------------------------- the ranking's structure

def test_the_ranking_order_is_as_published(grid):
    order = [a for a, _ in grid["global_mean_ranking"]]
    assert order[:4] == ["vit", "impoola", "vae", "impala"]
    assert [a for a in order if a in ("classic",)] == ["classic"]
    means = dict((a, m) for a, m in grid["global_mean_ranking"])
    assert (means["vit"], means["impoola"], means["mlp_vector"]) == (1.907, 1.707, 1.433)


def test_the_archived_top_three_falls_to_twelfth_thirteenth_fourteenth(grid, readme):
    order = [a for a, _ in grid["global_mean_ranking"]]
    # 1-indexed positions the section quotes
    assert [order.index(a) + 1 for a in ("spatial", "resnet18", "classic")] == [13, 14, 12]
    assert "13th, 14th and 12th" in readme


def test_the_non_reading_arms_hold_the_top_of_the_return_table(grid):
    resp = grid["responsiveness"]
    order = [a for a, _ in grid["global_mean_ranking"]]
    # the three least-responsive arms are 1st, 2nd and 7th on return
    assert resp["vit"]["cells_tv_over_0p1"] == 1
    assert resp["impoola"]["cells_tv_over_0p1"] == 3
    assert resp["mlp_vector"]["cells_tv_over_0p1"] == 3
    assert [order.index(a) + 1 for a in ("vit", "impoola", "mlp_vector")] == [1, 2, 7]
    # the four arms responsive on every measured cell
    fully = [a for a in ARMS if resp[a]["cells_tv_over_0p1"] == resp[a]["cells"]]
    assert sorted(fully) == ["c_aug_color", "cbam", "classic", "spatial"]
    assert sorted(order.index(a) + 1 for a in fully) == [5, 12, 13, 15]


def test_the_response_cliff_is_where_the_section_says(grid):
    resp = grid["responsiveness"]
    counts = {a: resp[a]["cells_tv_over_0p1"] for a in ARMS}
    assert counts["ae"] == counts["recon"] == counts["c_aug_noise"] == counts["contrastive"] == 13
    assert counts["lstm_attention"] == 12 and counts["resnet18"] == 11
    assert counts["c_aug_crop"] == 10 and counts["impala"] == 9
    assert counts["vae"] == 9  # of 14 -- the NaN cell is not a responsive cell
    # vit reads nothing on bossfight and dodgeball: per-game mean TV 0.000
    for g in ("bossfight", "dodgeball"):
        tvs = grid["games"][g]["vit"]["tvs"]
        assert sum(tvs) / len(tvs) < 5e-3, f"vit responds on {g}?!"


def test_mlp_vectors_leadership_is_gone_in_both_senses(grid):
    order = [a for a, _ in grid["global_mean_ranking"]]
    assert order.index("mlp_vector") + 1 == 7
    r = grid["responsiveness"]["mlp_vector"]
    assert r["tv_mean"] == 0.072704
    sp = grid["games"]["starpilot"]
    assert sp["classic"]["mean"] > sp["mlp_vector"]["mean"]
    assert sp["mlp_vector"]["mean"] == 2.36 and sp["classic"]["mean"] == 3.06


def test_the_seed_noise_the_section_warns_about_is_real(grid):
    sp = grid["games"]["starpilot"]
    assert sp["vit"]["returns"] == [6.4, 1.7, 2.2, 1.5, 2.7]
    assert sp["vit"]["std"] == 1.799
    assert sp["contrastive"]["returns"] == sp["c_aug_noise"]["returns"] == [1.5, 6.3, 1.7, 2.1, 3.0]
    bf = [grid["games"]["bossfight"][a]["mean"] for a in ARMS if grid["games"]["bossfight"][a]["mean"] is not None]
    assert max(bf) <= 0.78 and min(bf) >= 0.0


# ------------------------------------------------------------ the pairs and reproducibility

def test_the_same_program_pairs_are_identical_across_all_45_cells(merged):
    for g in GAMES:
        for a, b in (("ae", "recon"), ("contrastive", "c_aug_noise")):
            ra, rb = merged[g]["arms"][f"{a}@42"], merged[g]["arms"][f"{b}@42"]
            for seed in (43, 44, 45, 46):
                assert merged[g]["arms"][f"{a}@{seed}"]["return_10eps"] == \
                       merged[g]["arms"][f"{b}@{seed}"]["return_10eps"]
                assert merged[g]["arms"][f"{a}@{seed}"]["max_pairwise_tv"] == \
                       merged[g]["arms"][f"{b}@{seed}"]["max_pairwise_tv"]
            assert ra["return_10eps"] == rb["return_10eps"]
            assert ra["max_pairwise_tv"] == rb["max_pairwise_tv"]


def test_the_merge_refuses_shards_across_games(tmp_path):
    """--key-by-seed may vary the seed only; a shard of another game must fail the fuse
    rather than land silently in the wrong table."""
    def shard(path, game, seed, arm="classic"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"_provenance": {"game": game, "timesteps": 100000, "seed": seed,
                                       "n_envs": 16, "n_steps": 256},
                       "arms": {arm: {"return_10eps": 1.0}}}, f)
    shard(tmp_path / "sh-starpilot-s42.json", "starpilot", 42)
    shard(tmp_path / "sh-starpilot-s43.json", "starpilot", 43)
    shard(tmp_path / "sh-bossfight-s42.json", "bossfight", 42)  # the wrong game
    with pytest.raises(AssertionError, match="share game/budget/recipe"):
        merge(str(tmp_path / "sh-*.json"), str(tmp_path / "out.json"), key_by_seed=True)


# --------------------------------------------------------------------- the published table

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


def test_section_table_cells_match_the_fold(readme, grid):
    rows = _rows(readme, "### 3.6.7.")
    data = [r for r in rows if len(r) == 7 and r[0] != "arm"]
    assert len(data) == 16, f"16 data rows expected, got {len(data)}"
    means = dict((a, m) for a, m in grid["global_mean_ranking"])
    for cells in data:
        a = cells[0].replace("`", "")
        assert float(cells[2].replace("*", "")) == means[a], f"{a}: global mean"
        for col, g in ((3, "starpilot"), (4, "bossfight"), (5, "dodgeball")):
            want = grid["games"][g][a]["mean"]
            assert float(cells[col]) == want, f"{a}/{g}"
        r = grid["responsiveness"][a]
        want = f"{r['cells_tv_over_0p1']}/{r['cells']}"
        assert cells[6].replace("*", "") == want, f"{a}: responsive column"


def test_section_prose_carries_the_measured_numbers(readme):
    for s in (
        "**3h22m of wall clock**",
        "`23.1 h`",
        "went `NaN` at step `5888`",
        "7 cells died mid-run and re-trained cleanly at 6 workers",
        "concurrent processes must not share a `--out`",
        "**1/15**",
        "13th, 14th and 12th",
        "`[6.4, 1.7, 2.2, 1.5, 2.7]`, std `1.8`",
        "one `6.3` seed, std `1.77`",
        "hold return spots **1, 2 and 7**",
        "mean TV `0.000` on both) and `0.074` on `starpilot`",
        "spots 5, 12, 13 and 15",
        "identical rows across **all 45 cells of the three grids**",
        "7th on return (`1.433`, from `1.25@100` archived)",
        "mean TV `0.072704`",
        "`0.059`/`0.110`/`0.050`",
        "`classic` `3.06` and `impoola` `2.94` clear it (`2.36`)",
        "the rankings could not have been repaired by re-running them",
        "the response table, not the return table",
    ):
        assert s in readme, f"prose lost: {s!r}"


def test_the_summary_and_37_box_carry_the_verdict(readme):
    i = readme.index("§3.6.7 closes the question")
    assert "the named consistent leader, falls to 7th and third-least responsive" in readme[i:i + 500]
    j = readme.index("**Validity (11/10/2026; resolved 14/10/2026):**")
    assert "the response table in §3.6.7 is the architecture result this table never was" in \
        readme[j:j + 1000]
    k = readme.index("| 9 | **Why do the SB3 policies ignore the observation")
    row = readme[k:readme.index("\n", k)]
    assert "✅ Closed (14/10/2026)" in row
    assert "return is not the architecture measure; the response table is" in row
