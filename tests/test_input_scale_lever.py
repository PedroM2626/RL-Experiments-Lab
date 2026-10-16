"""Gate README 3.6.12 (the two named levers, pulled) on their artifacts.

The input-scale lever rescues mlp_vector -- birth saturation gone, absorbing death
gone, a real reader on bossfight -- and the pooling lever does not rescue vit: the
range opens 2x and the actions stay dead. These tests recompute every published
number, check the scale-invariance sanity of the feature measure (obs vs obs/255
must give the same normalized range), and pin the one-line closes of the arc.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def readme():
    with open(README, encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------ the input-scale rescue

def test_obscaling_removes_the_birth_saturation_and_the_death():
    d = _load("probe_mlpscaled_dodgeball.json")
    c = d["curve"]
    tv = [r["cross_frame_tv_mean"] for r in c]
    # the bare arm was born 97.89% saturated and died at 8500 with 17/201 alive
    assert c[0]["tanh_saturated_by_layer"][0] == 0.0, "the birth saturation must be gone"
    assert all(r["cross_frame_tv_mean"] >= 5e-5 for r in c), "the absorbing death must be gone"
    assert sum(1 for v in tv if v > 1e-4) == 201
    assert c[-1]["cross_frame_tv_mean"] == 0.001409
    assert max(tv) == 0.009944 and c[tv.index(max(tv))]["steps"] == 61500
    assert c[-1]["v_std"] == 0.21573, "the critic tracks again (bare arm ended at 0.0)"


def test_the_scaled_arm_is_a_reader_on_bossfight():
    d = _load("probe_mlpscaled_bossfight.json")
    c = d["curve"]
    tv = [r["cross_frame_tv_mean"] for r in c]
    assert sum(1 for v in tv if v > 1e-4) == 201
    assert c[-1]["cross_frame_tv_mean"] == 0.075026
    assert max(tv) == 0.103878 and c[tv.index(max(tv))]["steps"] == 57500
    assert d["cross_game"]["max_pairwise_tv"] == 0.23054, \
        "within reach of classic's 0.275, from the bare arm's 0.106"


def test_the_feature_measure_is_scale_invariant():
    """frange is normalized by the run's own activation scale, so obs and obs/255 must
    give the same relative range -- the sanity check that the rescue is the head's,
    not the measure's."""
    raw = _load("feature_vs_actor_mlp_dodgeball.json")["within_game"]["features"]
    scaled = _load("feature_vs_actor_mlp_scaled_dodgeball.json")["within_game"]["features"]
    assert raw["feat_range_over_scale_mean"] == scaled["feat_range_over_scale_mean"] == 0.8417
    assert scaled["feat_scale"] == pytest.approx(0.4776, abs=0.001), "obs/255, ~120 -> ~0.47"
    assert _load("feature_vs_actor_mlp_scaled_dodgeball.json")["within_game"]["action_tv"] \
        == 0.002246, "the scaled head uses its features (bare: exactly 0.0)"


# --------------------------------------------------------------- the pooling refutation

def test_max_pooling_opens_the_range_but_not_the_actions():
    p = _load("probe_vitmaxpool_bossfight.json")["curve"]
    tv = [r["cross_frame_tv_mean"] for r in p]
    assert all(r["cross_frame_tv_mean"] >= 5e-5 for r in p), \
        "maxpool never dead-flattens the way bare vit does"
    assert p[-1]["cross_frame_tv_mean"] == 0.004347 and max(tv) == 0.004846
    assert p[tv.index(max(tv))]["steps"] == 37000
    assert sum(1 for v in tv if v > 1e-4) == 185, "alive, but in the basement"
    assert _load("probe_vitmaxpool_bossfight.json")["cross_game"]["max_pairwise_tv"] == 0.000138
    f = _load("feature_vs_actor_vit_maxpool_bossfight.json")["within_game"]
    assert f["features"]["feat_range_over_scale_mean"] == 0.347, \
        "the lever does open the range, from bare vit's 0.154"
    assert f["features"]["feat_dims_range_over_scale"] == 0.0898
    assert f["action_tv"] == 1.1e-05, "and the head still has nothing to use"


# --------------------------------------------------------------- the five-seed rescue

def _scaled_runs(game):
    rows = []
    for s in (42, 43, 44, 45, 46):
        fn = f"probe_mlpscaled_{game}.json" if s == 42 else f"probe_mlpscaled_{game}_seed{s}.json"
        rows.append(_load(fn))
    return rows


def test_the_rescue_holds_on_every_seed_dodgeball():
    finals = []
    for s, d in enumerate(_scaled_runs("dodgeball"), start=42):
        c = d["curve"]
        # the bare arm was born 97.89% saturated and died at 8500 with 17/201 alive
        assert c[0]["tanh_saturated_by_layer"][0] == 0.0, f"s{s}: birth saturation back"
        assert all(r["cross_frame_tv_mean"] >= 5e-5 for r in c), f"s{s}: the death returned"
        assert sum(1 for r in c if r["cross_frame_tv_mean"] > 1e-4) == 201
        assert c[-1]["v_std"] > 0.05, f"s{s}: the critic is tracking again"
        finals.append(c[-1]["cross_frame_tv_mean"])
    assert finals == [0.001409, 0.001767, 0.004998, 0.005170, 0.009165]


def test_the_rescue_makes_a_reader_on_every_seed_bossfight():
    finals, crosses, argmax2 = [], [], []
    for s, d in enumerate(_scaled_runs("bossfight"), start=42):
        c = d["curve"]
        assert sum(1 for r in c if r["cross_frame_tv_mean"] > 1e-4) == 201
        finals.append(c[-1]["cross_frame_tv_mean"])
        crosses.append(d["cross_game"]["max_pairwise_tv"])
        if d["cross_game"]["distinct_argmax_across_games"] > 1:
            argmax2.append(s)
    assert finals == [0.075026, 0.060187, 0.10757, 0.121983, 0.152112]
    assert crosses == [0.23054, 0.215641, 0.379173, 0.099723, 0.312291]
    assert argmax2 == [43, 44], "two seeds emit 2 argmax across the five games"
    assert max(crosses) > 0.275, "seed 44 exceeds classic's own bossfight cross-game TV"


def test_the_vit_refutations_hold_at_three_seeds():
    ln = [_load("probe_vitlayernorm_bossfight.json")] + \
         [_load(f"probe_vit_ln_bossfight_seed{s}.json") for s in (43, 44)]
    deads = [next((r["steps"] for r in d["curve"] if r["cross_frame_tv_mean"] < 5e-5), None)
             for d in ln]
    assert deads == [16500, 25000, 12500], "LayerNorm at the output: 3/3 dead"
    mp = [_load("probe_vitmaxpool_bossfight.json")] + \
         [_load(f"probe_vit_maxpool_bossfight_seed{s}.json") for s in (43, 44)]
    mp_deads = [next((r["steps"] for r in d["curve"] if r["cross_frame_tv_mean"] < 5e-5), None)
                for d in mp]
    assert mp_deads == [None, 29000, 29000], "max-pool: 0/3 ignite, 2/3 collapse"
    # the near-escape the section names: seed 43 climbs higher than any vit variant
    # on this game, and collapses anyway
    s43 = [r["cross_frame_tv_mean"] for r in mp[1]["curve"]]
    assert max(s43) == 0.053488
    assert mp[2]["curve"][-1]["cross_frame_tv_mean"] == 3e-06

def test_the_two_lever_extractors_build_and_do_what_they_say():
    import gymnasium as gymn
    import numpy as np
    import torch

    from diagnose_policy_saturation import ScaledFlattenExtractor, ViTMaxPoolExtractor
    vec = gymn.spaces.Box(low=0, high=255, shape=(256,), dtype=np.uint8)
    e = ScaledFlattenExtractor(vec)
    out = e(torch.randint(0, 255, (2, 256), dtype=torch.uint8))
    assert out.shape == (2, 256) and out.max() <= 1.0, "obs/255, nothing more"
    img = gymn.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
    m = ViTMaxPoolExtractor(img, 512)
    o = torch.randint(0, 255, (1, 3, 64, 64), dtype=torch.uint8)
    assert m(o).shape == (1, 512)


# ------------------------------------------------------------------------- the prose

def test_section_prose_carries_the_measured_numbers(readme):
    for s in (
        "Seventeen runs across two waves",
        "**~29 GPU-minutes**",
        "first `Tanh` at **0.0%** at step 0 on all five seeds, against `91–98%` for the raw arm",
        "**never drops below the dead threshold on any of five seeds**",
        "finals `0.001409`–`0.009165`",
        "seed 44 **exceeds `classic`'s `0.275`**",
        "cross-game TVs `0.099723`–`0.379173`",
        "raw `uint8` → a `Linear` that saturates its `Tanh` at birth",
        "the feature range opens from `0.154` to **`0.347`**",
        "`9.0%` of the 512 dims now range past scale",
        "one never quite dies nor lives (final `0.004347`, alive in the basement)",
        "The seed-42 timeline peaks at `0.004846`",
        "the mean-pool is *part* of the site",
        "one of which was a bug",
        "3/3 dead; pooling, 0/3 igniting",
        "the seed-44 run collapsing hardest, `58/201` alive",
        "seed 43 climbs to `0.053488` at mid-run",
        "**3/3** dead; first dead probe `12500`–`25000`",
        "that contrast is now replicated, not sampled",
    ):
        assert s in readme, f"prose lost: {s!r}"


def test_the_3_6_11_note_points_here_and_the_scorecard_note_exists(readme):
    i = readme.index("**Follow-up (14/10/2026):** both named levers were pulled.")
    assert "rescues** `mlp_vector`" in readme[i:i + 300]
    assert "§3.6.12" in readme[i:i + 600]
    j = readme.index("### 0.1. The Response Scorecard")
    seg = readme[j:readme.index("###", j + 10)]
    assert "The cliff between 13/15 and 3/15 is the architecture result" in seg
