"""Gate README 3.6.6 (the (iii) pilot: 16 architectures under the fixed recipe) on its artifact.

The pilot pairs cell-for-cell with the 3.6.2 starpilot audit -- same seed, same budget,
same evaluation -- with the rollout recipe as the only difference. These tests recompute
every published number from the merged artifact and that old audit, and pin the two
integrity facts the pilot's claims lean on: the published path of the producers' train_one
is untouched (the 3.6.2 audits must stay byte-reproducible), and the two same-program pairs
still return identical rows under the new recipe.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import compare_new_archs as CNA  # noqa: E402
import compare_suite as CS  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pilot():
    return _load("suite_validity_starpilot_nenv16.json")


@pytest.fixture(scope="module")
def old():
    return _load("suite_validity_starpilot.json")


@pytest.fixture(scope="module")
def readme():
    with open(README, encoding="utf-8") as f:
        return f.read()


ARMS = ("classic", "vit", "impala", "spatial", "cbam", "impoola", "vae", "lstm_attention",
        "c_aug_crop", "ae", "recon", "resnet18", "mlp_vector", "contrastive",
        "c_aug_noise", "c_aug_color")


def test_the_artifact_is_one_pilot_not_a_mix(pilot):
    p = pilot["_provenance"]
    assert (p["game"], p["seed"], p["timesteps"], p["n_envs"], p["n_steps"]) == \
        ("starpilot", 42, 100000, 16, 256)
    assert set(pilot["arms"]) == set(ARMS)
    assert all("error" not in r for r in pilot["arms"].values()), "a pilot cell failed"
    # the merge stamp now carries the recipe, so a 1x256 shard cannot fuse with these
    assert len(p["shards_merged"]) == 16


def test_the_published_train_one_path_is_untouched():
    """The producers' train_one must default to the published recipe -- the 3.6.2 audits
    were produced by that path and their reproducibility is a published claim."""
    import inspect
    for fn in (CS.train_one, CNA.train_one):
        sig = inspect.signature(fn)
        assert sig.parameters["n_envs"].default == 1, fn
        assert sig.parameters["n_steps"].default == 256, fn


def test_fourteen_arms_move_up_and_one_moves_down(pilot, old):
    news, olds = pilot["arms"], old["arms"]
    # nobody sits at the 3.6.2 inert floor anymore (14/16 were below it at n_envs=1)
    assert all(r["max_pairwise_tv"] > 5e-3 for r in news.values())
    assert sum(1 for r in olds.values() if r["max_pairwise_tv"] < 5e-3) == 14
    # 13/16 cross 0.1
    assert sum(1 for r in news.values() if r["max_pairwise_tv"] >= 0.1) == 13
    below = {a: r["max_pairwise_tv"] for a, r in news.items() if r["max_pairwise_tv"] < 0.1}
    assert below == {"mlp_vector": 0.053308, "impoola": 0.026367, "vit": 0.083030}
    # the one arm whose response FELL is the published leader, and it keeps the
    # collapse's signature
    assert news["mlp_vector"]["max_pairwise_tv"] < olds["mlp_vector"]["max_pairwise_tv"]
    m = news["mlp_vector"]
    assert (m["distinct_argmax_across_games"],
            m["first_tanh_saturated_fraction"]) == (1, pytest.approx(0.9906, abs=1e-4))
    assert m["feat_max"] == 254.0


def test_the_old_ordering_inverts(pilot, old):
    news, olds = pilot["arms"], old["arms"]
    top_old = ("contrastive", "c_aug_noise", "lstm_attention", "c_aug_color")
    bot_old = ("vae", "classic", "vit", "cbam")
    mean = lambda names, rows: sum(rows[a]["return_10eps"] for a in names) / len(names)
    assert mean(top_old, olds) == pytest.approx(2.775, abs=0.005)
    assert mean(bot_old, olds) == pytest.approx(0.50)
    assert mean(top_old, news) == pytest.approx(1.70, abs=0.005)
    assert mean(bot_old, news) == pytest.approx(4.575, abs=0.005)
    assert olds["classic"]["return_10eps"] == 0.5 and news["classic"]["return_10eps"] == 5.2
    assert olds["vit"]["return_10eps"] == 0.5 and news["vit"]["return_10eps"] == 6.4
    # returns separate: spread 1.00-6.40
    rets = sorted(r["return_10eps"] for r in news.values())
    assert (rets[0], rets[-1]) == (1.0, 6.4)


def test_high_return_is_still_not_evidence_of_a_visual_policy(pilot):
    v = pilot["arms"]["vit"]
    assert v["return_10eps"] == max(r["return_10eps"] for r in pilot["arms"].values())
    assert (v["max_pairwise_tv"], v["distinct_argmax_across_games"]) == (0.083030, 1)
    imp = pilot["arms"]["impoola"]
    assert (imp["max_pairwise_tv"], imp["distinct_argmax_across_games"],
            imp["first_tanh_saturated_fraction"]) == (0.026367, 1, pytest.approx(0.0063, abs=1e-4))


def test_the_same_program_pairs_stay_identical(pilot):
    rows = pilot["arms"]
    for a, b in (("ae", "recon"), ("contrastive", "c_aug_noise")):
        # 'family' names the config list the arm came from, not a measurement; the
        # measurement fields must agree to the digit
        ra = {k: v for k, v in rows[a].items() if k not in ("family", "seconds")}
        rb = {k: v for k, v in rows[b].items() if k not in ("family", "seconds")}
        assert rows[a]["seconds"] != rows[b]["seconds"], f"{a}/{b}: same process is suspicious"
        assert ra == rb, f"{a}/{b} drifted apart under the new recipe"
    assert rows["ae"]["max_pairwise_tv"] == 0.385176
    assert rows["contrastive"]["max_pairwise_tv"] == 0.462142


def test_the_cost_claims_are_the_measured_ones(pilot):
    total = sum(r["seconds"] for r in pilot["arms"].values())
    assert round(total / 60, 1) == 53.7, "the 53.7 cell-minutes the README quotes"
    # full grid: 240 cells at the pilot's measured average rate
    hours = 240 * (total / 16) / 3600
    assert round(hours, 1) == 13.4, "the 13.4 GPU-hours the README revises item (iii) to"


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


def test_section_table_cells_match_the_artifacts(readme, pilot, old):
    rows = _rows(readme, "### 3.6.6.")
    data = [r for r in rows if len(r) == 7 and r[0] != "arm"]
    assert len(data) == 16, f"16 data rows expected, got {len(data)}"
    for cells in data:
        a = cells[0].replace("`", "")
        n, o = pilot["arms"][a], old["arms"][a]
        assert float(cells[1].replace("*", "")) == o["return_10eps"], a
        assert float(cells[2].replace("*", "")) == n["return_10eps"], a
        assert cells[3] == f"{o['max_pairwise_tv']:.6f}", a
        assert cells[4] == f"{n['max_pairwise_tv']:.6f}", a
        assert cells[5] == str(n["distinct_argmax_across_games"]), a
        assert cells[6] == f"{100 * n['first_tanh_saturated_fraction']:.1f}%", a


def test_section_prose_carries_the_measured_numbers(readme):
    for s in (
        "53.7 cell-minutes summed",
        "**~13.4 GPU-hours**, not the ~40 previously estimated",
        "**`mlp_vector` drops from `0.101104` to `0.053308` and keeps the collapse's exact signature**",
        "99.1% of the first `Tanh` saturated",
        "feature max `254.00`",
        "average **`4.58`**",
        "`classic` goes `0.50 → 5.20`, `vit` `0.50 → 6.40`, `impala` `1.60 → 4.40`",
        "the pilot's best return (`6.40`) at TV `0.083030` and **one argmax across five games**",
        "`3.50` at TV `0.026367`, one argmax, with the first `Tanh` at 0.6%",
        "(`2.40`, TV `0.385176`, 11.6%)",
        "(`1.50`, TV `0.462142`, 6.2%)",
        "returns spread `1.00–6.40`",
        "roughly an overnight wall-clock at 3 workers",
    ):
        assert s in readme, f"prose lost: {s!r}"


def test_the_roadmap_and_summary_carry_the_pilot(readme):
    i = readme.index("§3.6.6's pilot")
    assert "the one arm the fix does not rescue" in readme[i:i + 620]
    j = readme.index("**Then the (iii) pilot**")
    assert "revises the full 240-cell grid to **~13.4 GPU-hours**, not ~40" in readme[j:j + 900]
    k = readme.index("**Follow-up (14/10/2026):** the (iii) pilot ran")
    assert "the decision it feeds is now cheaper" in readme[k:k + 300]
