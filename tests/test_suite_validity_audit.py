"""The validity audit must cover the published arms, not a copy of them.

audit_suite_validity.py imports the config lists from compare_suite and compare_new_archs
instead of restating them, so the only way it can drift is if a producer changes shape. These
tests pin that: the arm set, the families, and the fact that the audit calls the producers'
own train_one rather than building its own PPO.
"""
import ast
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import compare_new_archs as CNA  # noqa: E402
import compare_suite as CS  # noqa: E402
from audit_suite_validity import ARMS, BY_NAME, EXTRA_ARMS, PUBLISHED_ARMS, merge  # noqa: E402

PUBLISHED = (
    [k for k, _cls, _kw in CS.WORLD_CONFIGS]
    + [k for k, _cls, _kw, _vec in CS.PROCGEN_CONFIGS]
    + [k for k, _cls, _kw in CS.AUGMENT_CONFIGS]
    + [k for k, _cls, _kw in CNA.NEW_ARCH_CONFIGS])


def test_audit_covers_exactly_the_sixteen_published_arms():
    """Since 24/10/2026 the registry also carries opt-in arms added after the audit
    (mlp_scaled, the 3.6.12 input-scale lever; recon_bottleneck and
    c_aug_noise_drop, the distinct variants of the same-program pairs). The DEFAULT
    audit must still cover exactly the published sixteen, and every extra arm must
    be declared in EXTRA_ARMS so the two sets cannot drift apart."""
    assert len(PUBLISHED_ARMS) == 16, (
        f"the headline comparison is 16 architectures, found {len(PUBLISHED_ARMS)}")
    assert set(PUBLISHED_ARMS) == set(PUBLISHED), (
        f"audit-only: {sorted(set(PUBLISHED_ARMS) - set(PUBLISHED))}; "
        f"published-only: {sorted(set(PUBLISHED) - set(PUBLISHED_ARMS))}")
    assert set(BY_NAME) == set(PUBLISHED) | EXTRA_ARMS
    assert len(BY_NAME) == len(set(BY_NAME)), "duplicate arm name"
    assert EXTRA_ARMS == {"mlp_scaled", "recon_bottleneck", "c_aug_noise_drop"}


def test_audit_reuses_the_producers_and_does_not_rebuild_ppo():
    """The point of importing train_one is that hyperparameters cannot silently differ.

    An audit that trains with its own PPO call would answer a different question than the one
    the README asks, and the drift would be invisible in the results file.
    """
    with open(os.path.join(ROOT, "audit_suite_validity.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    src = ast.unparse(tree)
    assert "PPO(" not in src, "the audit builds its own PPO; it must call compare_*.train_one"
    assert "CS.train_one" in src and "CNA.train_one" in src
    for name in ("WORLD_CONFIGS", "PROCGEN_CONFIGS", "AUGMENT_CONFIGS"):
        assert hasattr(CS, name), f"compare_suite.{name} disappeared from module scope"
    assert hasattr(CNA, "NEW_ARCH_CONFIGS")


def test_every_arm_names_an_importable_extractor_or_the_vector_case():
    for name, family, cls, kwargs, vector, vector_scale in ARMS:
        assert family in {"world_model", "procgen_cnn", "augment", "new_arch"}, name
        assert isinstance(vector_scale, bool), name  # 6th field since 24/10/2026
        if vector:
            assert cls is None and kwargs == {}, f"{name}: vector arm should pass no extractor"
        else:
            assert callable(cls), f"{name}: extractor class is not importable"
            assert kwargs.get("features_dim") == 512, f"{name}: study uses FC 512 across extractors"


def _shard(tmp_path, name, arms, game="starpilot"):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"_provenance": {"game": game}, "arms": arms}), encoding="utf-8")
    return str(p)


def test_merge_fuses_shards_without_losing_arms(tmp_path):
    a = _shard(tmp_path, "s1", {"classic": {"max_pairwise_tv": 0.1}, "vit": {"max_pairwise_tv": 0.2}})
    b = _shard(tmp_path, "s2", {"vae": {"max_pairwise_tv": 0.3}})
    out = str(tmp_path / "all.json")
    data = merge(os.path.join(str(tmp_path), "s*.json"), out)
    assert set(data["arms"]) == {"classic", "vit", "vae"}
    assert os.path.exists(out) and os.path.exists(a) and os.path.exists(b), "shards must survive"


def test_merge_refuses_an_arm_measured_twice_with_different_values(tmp_path):
    _shard(tmp_path, "s1", {"classic": {"max_pairwise_tv": 0.1}})
    _shard(tmp_path, "s2", {"classic": {"max_pairwise_tv": 0.9}})
    with pytest.raises(AssertionError, match="measured twice"):
        merge(os.path.join(str(tmp_path), "s*.json"), str(tmp_path / "all.json"))
    # identical re-measurements are the same measurement, not a conflict
    _shard(tmp_path, "s3", {"classic": {"max_pairwise_tv": 0.1}})
    assert merge(os.path.join(str(tmp_path), "s[13].json"), str(tmp_path / "all.json"))


def test_merge_refuses_shards_from_different_games(tmp_path):
    """A starpilot row and a dodgeball row of one arm would otherwise overwrite each other."""
    _shard(tmp_path, "s1", {"classic": {"max_pairwise_tv": 0.1}}, game="starpilot")
    _shard(tmp_path, "s2", {"vae": {"max_pairwise_tv": 0.2}}, game="dodgeball")
    with pytest.raises(AssertionError, match="different games or budgets"):
        merge(os.path.join(str(tmp_path), "s*.json"), str(tmp_path / "all.json"))


def test_key_by_seed_keeps_a_seed_sweep_as_three_rows(tmp_path):
    for seed in (42, 43):
        p = tmp_path / f"p{seed}.json"
        p.write_text(json.dumps({"_provenance": {"game": "bossfight", "timesteps": 100000,
                                                 "seed": seed},
                                 "arms": {"vae": {"max_pairwise_tv": seed / 1e6}}}),
                     encoding="utf-8")
    data = merge(os.path.join(str(tmp_path), "p*.json"), str(tmp_path / "all.json"),
                 key_by_seed=True)
    assert set(data["arms"]) == {"vae@42", "vae@43"}
    assert data["_provenance"]["rows"] == "arm@seed, one shard per seed"
    # and without the flag the same files are a conflict, not a merge
    with pytest.raises(AssertionError, match="different games or budgets|measured twice"):
        merge(os.path.join(str(tmp_path), "p*.json"), str(tmp_path / "all2.json"))
