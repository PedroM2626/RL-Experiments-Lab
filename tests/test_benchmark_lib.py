"""Unit tests for the unified benchmark_lib evaluation infrastructure.

Tests both visual (CHW 3x64x64) and vector (256D) evaluation environment factories
and the canonical 3-tier evaluation protocol. If native procgen C++ binary is not
installed (e.g. standard Linux CI runners), headless mock environments conforming to
the ProcgenGymWrapper/ProcgenVectorWrapper specifications are used seamlessly.
"""
import gymnasium as gymn
import numpy as np
import pytest

from benchmark_lib import evaluate_model_protocol, make_eval_env

try:
    import procgen  # noqa: F401
    HAS_PROCGEN = True
except (ImportError, ModuleNotFoundError):
    HAS_PROCGEN = False


class MockProcgenEnv(gymn.Env):
    """Headless mock procgen environment for CI environments lacking native C++ procgen."""

    def __init__(self, vector: bool = False):
        super().__init__()
        self.vector = vector
        if vector:
            self.observation_space = gymn.spaces.Box(0, 255, shape=(256,), dtype=np.uint8)
        else:
            self.observation_space = gymn.spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8)
        self.action_space = gymn.spaces.Discrete(15)
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self._step_count += 1
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        reward = 1.0
        terminated = self._step_count >= 2
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


@pytest.fixture(autouse=True)
def mock_procgen_if_absent(monkeypatch):
    """If native procgen binary is missing (e.g. standard CI runner), mock make_procgen_env."""
    if not HAS_PROCGEN:
        def _mock_make_procgen_env(
            game: str = "coinrun",
            num_levels: int = 200,
            distribution_mode: str = "easy",
            seed: int = 0,
            frame_stack: int = 1,
            vector: bool = False,
        ):
            return MockProcgenEnv(vector=vector)

        monkeypatch.setattr("benchmark_lib.make_procgen_env", _mock_make_procgen_env)


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
