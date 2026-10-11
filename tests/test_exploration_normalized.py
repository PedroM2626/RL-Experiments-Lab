"""Gate the published scaled-bonus results against the cells they were computed from.

The summarizer that writes results/exploration_normalized.json crashed on its first run (the
reused control rows were built without a seed key), so "the file exists" is not evidence it is
coherent. This re-derives every published number from the per-seed cells in the same file and
from the frozen artifacts it quotes, and fails on any drift:

  * mean / std / Student-t CI / n of both protocols, recomputed from per_seed;
  * the reused ppo control present at both protocols and marked reused_from;
  * injected-magnitude bookkeeping only on bonus arms, at the arm's declared beta;
  * the eval100 file it folded in has 30 models and no errored entries;
  * the vs_unnormalized column quotes the frozen 24/09 values verbatim.

Everything read here is committed, so this runs in CI without the git-ignored log dirs.
"""
import json
import math
import os
import statistics

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 8: 2.365, 10: 2.262}


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def published():
    return _load("exploration_normalized.json")


def test_every_published_statistic_recomputes_from_its_own_cells(published):
    for cfg, blk in published["per_seed"].items():
        vals = [c["mean_reward_10eps"] for c in blk["cells"]]
        assert len(set(c["seed"] for c in blk["cells"])) == len(vals), f"{cfg}: duplicate seeds"
        s = published["statistics_10eps"][cfg]
        mean = sum(vals) / len(vals)
        assert s["n"] == len(vals), f"{cfg}: n={s['n']} but {len(vals)} cells"
        assert abs(s["mean"] - round(mean, 3)) < 1e-9, f"{cfg}: mean {s['mean']} != {mean}"
        if len(vals) > 1:
            half = T975[len(vals)] * statistics.stdev(vals) / math.sqrt(len(vals))
            assert abs(s["ci95"][0] - round(mean - half, 3)) < 2e-3, f"{cfg}: CI low {s['ci95']}"
            assert abs(s["ci95"][1] - round(mean + half, 3)) < 2e-3, f"{cfg}: CI high {s['ci95']}"


def test_reused_control_is_present_and_declared(published):
    prov = published["_provenance"]["control"]
    assert prov["source"].startswith("results/exploration_remeasure.json")
    for game in ("maze", "heist"):
        ctl = published["statistics_100eps"][f"{game}_ppo"]
        assert ctl.get("reused_from"), f"{game}_ppo 100-ep row does not say it was reused"
        frozen = _load("exploration_remeasure.json")["statistics_100eps"][f"{game}_ppo"]
        assert ctl["mean"] == frozen["mean"], f"{game}_ppo drifted from the frozen control"


def test_bonus_arms_carry_bookkeeping_and_the_control_does_not(published):
    for cfg, s in published["injected_scale"].items():
        assert not cfg.endswith("_ppo"), f"{cfg}: the control has no bonus to report"
        assert s["normalized"] == [True], f"{cfg}: a cell without normalize=True leaked in"
        assert s["beta"] == [0.01], f"{cfg}: beta drifted from the published arm definition"
        assert s["n_cells"] == 5, f"{cfg}: {s['n_cells']} cells, expected 5 seeds"
        assert 0 < s["injected_per_step_mean"] < s["injected_per_step_max"] <= 0.01 * 5 + 1e-9, \
            f"{cfg}: injected magnitude outside (0, beta*clip]"


def test_the_folded_in_evaluation_is_complete(published):
    eval100 = _load("eval100_normalized.json")
    assert not [k for k, v in eval100.items() if "error" in v], "errored models folded in"
    expected = {f"{c}_seed{s}" for c, blk in published["per_seed"].items()
                if not c.endswith("_ppo") for s in [x["seed"] for x in blk["cells"]]}
    assert set(eval100) == expected, "eval100 and the summary disagree about which models exist"


def test_unnormalized_comparison_quotes_the_frozen_file_verbatim(published):
    frozen = _load("exploration_remeasure.json")["statistics_10eps"]
    for cfg, v in published["vs_unnormalized"].items():
        assert v["unnormalized_mean_10eps"] == frozen[cfg]["mean"], f"{cfg}: mean drifted"
        assert v["unnormalized_ci95"] == frozen[cfg]["ci95"], f"{cfg}: CI drifted"
