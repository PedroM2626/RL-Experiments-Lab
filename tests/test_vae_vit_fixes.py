"""Gate README 3.6.14 (the vae NaN) and 3.6.15 (the ViT token pipeline) on their artifacts."""
import importlib
import json
import math
import os

import pytest

from docs_text import open_docs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def docs():
    with open_docs() as f:
        return f.read()


def test_folds_are_current():
    for mod, name in (("summarize_vae_clamped", "vae_nan_fix.json"), ("summarize_vit_tokens", "vit_token_pipeline.json")):
        assert importlib.import_module(mod).fold() == _load(name), f"stale: rerun {mod}.py"


def test_the_nan_is_a_float32_overflow_of_the_variance():
    c = _load("vae_nan_bossfight_s42.json")["curve"]
    healthy = [r["logvar_max"] for r in c if r["steps"] <= 81920]
    assert (round(min(healthy), 2), round(max(healthy), 2)) == (3.63, 12.25)
    jump = next(r for r in c if r["steps"] == 86016)
    assert round(jump["logvar_max"], 1) == 124.2
    peak = max(r["logvar_max"] for r in c if isinstance(r["logvar_max"], float))
    assert peak > 2 * math.log(3.4028235e38), "the peak must pass exp's float32 overflow point"
    assert round(peak, 2) == 177.48
    assert _load("vae_nan_bossfight_s42.json")["error"].startswith("Expected parameter logits")


def test_the_clamp_fixes_the_cell_and_the_grid():
    fix = _load("vae_nan_fix.json")
    d = fix["diagnosis_bossfight_seed42"]["clamped"]
    assert (d["steps_reached"], d["error"], d["logvar_max_overall"], d["nonfinite_rows"]) == (102400, None, 10.0, 0)
    r = fix["responsiveness"]
    assert (r["cells"], r["cells_tv_over_0p1"], r["tv_mean"], r["error_cells"]) == (15, 8, 0.113657, [])
    pub = fix["published_vae_responsiveness"]
    assert (pub["cells_tv_over_0p1"], pub["cells"], pub["tv_mean"]) == (9, 14, 0.136637)
    assert max(fix["games"]["bossfight"]["feat_max"]) == pytest.approx(804.701, abs=1e-3)


def test_the_vae_prose_carries_the_numbers(docs):
    for s in ("`3.63` and `12.25` through `81920` steps", "jumps to `124.2`", "**`177.48`** at `98304`",
              "`2·ln(FLT_MAX) = 177.45`", "`0/15` cells diverge", "**`8/15`** cells clear TV `0.1` (published `9/14`)",
              "TV mean `0.113657` (published `0.136637`)", "features at `804.7`"):
        assert s in docs, f"3.6.14 lost: {s!r}"


def test_the_token_pipeline_table_matches_the_probes():
    v = _load("vit_token_pipeline.json")
    s = v["probe_summary"]
    expect = {"vit": (1, 0), "vit_ln": (3, 0), "vit_maxpool": (3, 2), "vit_tokens": (3, 3),
              "vit_patch8": (2, 1), "vit_prenorm": (2, 0), "vit_cls": (2, 0)}
    assert {k: (x["runs"], x["survive"]) for k, x in s.items()} == expect
    assert all(x["cross_game_readers"] == 0 for x in s.values())
    best = {a: max(r["within_game_max"] for r in rows.values()) for a, rows in v["probes"].items()}
    assert best == {"vit": 0.000792, "vit_ln": 0.006816, "vit_maxpool": 0.053488, "vit_tokens": 0.059735,
                    "vit_patch8": 0.002895, "vit_prenorm": 0.0056, "vit_cls": 0.003558}
    assert v["reference_classic_bossfight"]["within_game_max"] == 0.184762


def test_the_token_pipeline_grid(docs):
    v = _load("vit_token_pipeline.json")
    g = v["grid_responsiveness"]
    assert (g["cells"], g["cells_tv_over_0p1"], g["tv_mean"]) == (15, 9, 0.144445)
    per = {game: sum(t >= 0.1 for t in d["tvs"]) for game, d in v["grid"].items()}
    assert per == {"starpilot": 5, "bossfight": 0, "dodgeball": 4}
    assert (v["published_vit_responsiveness"]["cells_tv_over_0p1"], v["published_vit_responsiveness"]["tv_mean"]) == (1, 0.02479)
    for s in ("reads on **`9/15`** cells, TV mean **`0.144445`**", "`starpilot` **`5/5`** (published `1/5`)",
              "`dodgeball` **`4/5`** (published `0/5`", "`bossfight` `0.78 → 0.30`"):
        assert s in docs, f"3.6.15 lost: {s!r}"
