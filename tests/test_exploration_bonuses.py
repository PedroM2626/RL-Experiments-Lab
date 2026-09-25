"""Tests for the intrinsic-reward wrappers in models/bonuses.py.

The exploration benchmark reported ppo == rnd == ngu (README section 3.6), which is what
a silently-swallowed wrapper error looks like. These tests pin the behaviour that makes
such a regression impossible: the bonus is always added, and a malformed observation
raises instead of degrading to plain PPO.
"""

import gymnasium as gymn
import numpy as np
import pytest
import torch

from models.bonuses import ICMWrapper, NGUWrapper, RNDWrapper, to_chw_float


class DummyEnv(gymn.Env):
    """Minimal Procgen-shaped env: uint8 CHW observations, constant extrinsic reward."""

    def __init__(self, shape=(3, 64, 64)):
        super().__init__()
        self.shape = shape
        self.observation_space = gymn.spaces.Box(0, 255, shape, dtype=np.uint8)
        self.action_space = gymn.spaces.Discrete(15)
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return np.zeros(self.shape, np.uint8), {}

    def step(self, action):
        self.steps += 1
        obs = (np.arange(int(np.prod(self.shape)), dtype=np.uint8).reshape(self.shape) + self.steps)
        obs = obs.reshape(self.shape)
        return obs, 1.0, self.steps >= 5, False, {}


def _run(wrapper_cls, steps=12, shape=(3, 64, 64)):
    env = wrapper_cls(DummyEnv(shape=shape))
    env.reset()
    rewards = []
    for _ in range(steps):
        _, r, term, trunc, _ = env.step(3)
        rewards.append(r)
        if term or trunc:
            env.reset()
    return env, rewards


@pytest.mark.parametrize("cls", [ICMWrapper, RNDWrapper, NGUWrapper])
def test_bonus_is_applied_on_every_step(cls):
    env, rewards = _run(cls)
    st = env.stats()
    assert st["bonus_applied"] == st["steps"] == len(rewards)
    assert st["intrinsic_mean"] > 0.0
    # extrinsic reward is 1.0; anything above it proves the bonus reached the returned reward
    assert all(r > 1.0 for r in rewards), rewards


@pytest.mark.parametrize("cls", [ICMWrapper, RNDWrapper, NGUWrapper])
def test_hwc_observations_are_accepted(cls):
    env, rewards = _run(cls, steps=6, shape=(64, 64, 3))
    assert env.stats()["bonus_applied"] == 6
    assert all(r > 1.0 for r in rewards)


@pytest.mark.parametrize("cls", [RNDWrapper, NGUWrapper])
def test_malformed_observation_raises_instead_of_silently_skipping(cls):
    env = cls(DummyEnv(shape=(8, 8, 3)))
    env.reset()
    with pytest.raises(ValueError, match="expected"):
        env.step(1)


def test_icm_requires_reset_first():
    env = ICMWrapper(DummyEnv())
    with pytest.raises(RuntimeError, match="before reset"):
        env.step(1)


def test_to_chw_float_normalizes_and_shapes():
    t = to_chw_float(np.full((3, 64, 64), 255, np.uint8))
    assert t.shape == (1, 3, 64, 64)
    assert float(t.max()) == pytest.approx(1.0)


def test_rnd_target_stays_frozen_while_predictor_trains():
    env = RNDWrapper(DummyEnv())
    before = [p.detach().clone() for p in env.target.parameters()]
    env.reset()
    for _ in range(6):
        env.step(1)
    assert all(torch.equal(b, a) for b, a in zip(before, env.target.parameters()))
    assert any(p.grad is not None for p in env.predictor.parameters())


def test_icm_inverse_model_receives_no_gradient():
    """Documented quirk: the inverse model is built and optimized but absent from the loss."""
    env = ICMWrapper(DummyEnv())
    env.reset()
    for _ in range(6):
        env.step(1)
    assert all(p.grad is None for p in env.inverse_model.parameters())
    assert any(p.grad is not None for p in env.forward_model.parameters())


def test_ngu_fills_episodic_memory():
    env = NGUWrapper(DummyEnv())
    env.reset()
    for _ in range(30):
        env.step(1)
    assert len(env.memory) == 30


def test_stats_reachable_through_the_sb3_vec_env():
    """compare_maze_heist.train_one() reads the wrapper back through DummyVecEnv + Monitor;
    if that attribute path breaks, the bonus assertion silently stops protecting anything."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec = DummyVecEnv([lambda: Monitor(RNDWrapper(DummyEnv()))])
    vec.reset()
    for _ in range(3):
        vec.step(np.array([1]))
    st = vec.env_method("stats")[0]
    assert st["bonus_applied"] == 3, st
    vec.close()


def test_beta_scales_the_bonus():
    """With the same seed the two wrappers see identical networks and observations, so the
    added reward must be exactly beta * intrinsic."""
    rewards = {}
    for beta in (0.01, 0.1):
        torch.manual_seed(0)
        env = RNDWrapper(DummyEnv(), beta=beta)
        env.reset()
        r = [env.step(1)[1] for _ in range(5)]
        rewards[beta] = r
    scale_low = [x - 1.0 for x in rewards[0.01]]
    scale_high = [x - 1.0 for x in rewards[0.1]]
    assert all(v > 0 for v in scale_low)
    ratios = [h / lo for lo, h in zip(scale_low, scale_high)]
    assert all(abs(ratio - 10.0) < 1e-3 for ratio in ratios), ratios


@pytest.mark.parametrize("cls", [ICMWrapper, RNDWrapper, NGUWrapper])
def test_bonus_normalization_scales_and_clips(cls):
    env = cls(DummyEnv(), beta=1.0, normalize=True, clip=2.0)
    env.reset()
    rewards = []
    for _ in range(25):
        _, r, term, trunc, _ = env.step(1)
        rewards.append(r)
        if term or trunc:
            env.reset()
    st = env.stats()
    assert st["normalized"] is True
    # Base reward is 1.0; intrinsic bonus clipped to <= 2.0 with beta=1.0 means reward <= 3.0 + eps
    assert all(1.0 <= r <= 3.0001 for r in rewards)
    assert any(r > 1.0 for r in rewards)
