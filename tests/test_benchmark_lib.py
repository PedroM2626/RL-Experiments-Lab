"""Unit tests for the unified benchmark_lib evaluation infrastructure."""
import pytest
import numpy as np
from benchmark_lib import make_eval_env, evaluate_model_protocol


def test_make_eval_env_visual():
    env = make_eval_env("coinrun", num_levels=0, seed=42, vector=False)
    obs = env.reset()
    assert obs.shape == (1, 3, 64, 64)
    env.close()


def test_make_eval_env_vector():
    env = make_eval_env("coinrun", num_levels=0, seed=42, vector=True)
    obs = env.reset()
    assert obs.shape == (1, 256)
    env.close()


class DummyPolicy:
    def __init__(self, observation_space_shape=(3, 64, 64), n_actions=15):
        import gymnasium as gymn
        self.observation_space = gymn.spaces.Box(0, 255, shape=observation_space_shape, dtype=np.uint8)
        self.action_space = gymn.spaces.Discrete(n_actions)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        n_envs = observation.shape[0] if len(observation.shape) > len(self.observation_space.shape) else 1
        actions = np.zeros(n_envs, dtype=int)
        return actions, state


def test_evaluate_model_protocol_keys():
    dummy = DummyPolicy()
    res = evaluate_model_protocol(dummy, "coinrun", seed=42, n_stoch=2, n_det=2, n_train=2)
    assert set(res.keys()) == {"stoch_unseen", "det_unseen", "stoch_train", "gen_gap"}
    assert isinstance(res["stoch_unseen"], float)
    assert isinstance(res["det_unseen"], float)
    assert isinstance(res["stoch_train"], float)
    assert isinstance(res["gen_gap"], float)
    assert round(res["stoch_train"] - res["stoch_unseen"], 3) == res["gen_gap"]
