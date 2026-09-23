"""End-to-end wiring of the custom extractors through stable_baselines3.

The benchmarks hand these classes to SB3 as `features_extractor_class`; what matters
in production is not that forward() works standalone but that the policy is built on
top of them and produces actions. That is what this covers.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from models.combined_extractors import ImpalaCNNExtractor, ResNet18Extractor
from models.sb3_extractors import AttentionCNNExtractor, ClassicCNNExtractor
from models.world_model_extractors import AEExtractor, VAEExtractor

EXTRACTORS = [
    (ClassicCNNExtractor, {}, "classic"),
    (AttentionCNNExtractor, {"use_cbam": True}, "cbam"),
    (ImpalaCNNExtractor, {}, "impala"),
    (ResNet18Extractor, {}, "resnet18"),
    (AEExtractor, {}, "wm_ae"),
    (VAEExtractor, {}, "wm_vae"),
]


class PixelProcgenLike(gym.Env):
    """Procgen-shaped observation space without the native procgen dependency.

    The study wraps Procgen as CHW (procgen_wrapper.ProcgenGymWrapper); HWC is the
    layout SB3's own preprocessing produces from a channels-last space, and both must
    reach the extractor as channels-first.
    """

    def __init__(self, chw=True):
        super().__init__()
        self.observation_space = spaces.Box(0, 255, (3, 64, 64) if chw else (64, 64, 3),
                                            dtype=np.uint8)
        self.action_space = spaces.Discrete(15)
        self.chw = chw

    def reset(self, **kwargs):
        return np.zeros(self.observation_space.shape, np.uint8), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, np.uint8), 0.0, False, False, {}


@pytest.mark.parametrize("cls,kwargs,name", EXTRACTORS, ids=[e[2] for e in EXTRACTORS])
@pytest.mark.parametrize("chw", [True, False], ids=["chw-space", "hwc-space"])
def test_ppo_builds_and_acts_with_extractor(cls, kwargs, name, chw):
    env = DummyVecEnv([lambda: PixelProcgenLike(chw=chw)])
    model = PPO("CnnPolicy", env, verbose=0, device="cpu", n_steps=8, batch_size=8,
                policy_kwargs={"features_extractor_class": cls,
                               "features_extractor_kwargs": dict(features_dim=512, **kwargs)})
    assert model.policy.features_dim == 512, f"{name}: policy was not built on the extractor"
    obs = np.zeros((1, *env.observation_space.shape), np.uint8)
    action, _ = model.predict(obs, deterministic=True)
    assert env.action_space.contains(int(action[0])), action
    features = model.policy.features_extractor(
        torch.from_numpy(obs).permute(0, 3, 1, 2) if not chw else torch.from_numpy(obs))
    assert features.shape == (1, 512) and torch.isfinite(features).all()
