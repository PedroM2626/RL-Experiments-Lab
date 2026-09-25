"""Tests for the world-model / auxiliary representation extractors.

They pin the two defects found in the 23/09/2026 audit: AEExtractor.dream() read a second,
parallel encoder that forward() never used, and every decoder stack reconstructed 60x60
images from 64x64 inputs.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

from models.world_model_extractors import (
    AEExtractor,
    ContrastiveExtractor,
    ReconExtractor,
    VAEExtractor,
)

WM_CLASSES = [VAEExtractor, AEExtractor, ReconExtractor, ContrastiveExtractor]


@pytest.fixture(params=WM_CLASSES, ids=lambda c: c.__name__)
def extractor(request):
    space = gym.spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)
    return request.param(space, features_dim=512)


@pytest.fixture
def obs_batch():
    return torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8).float()


def test_forward_returns_features_dim(extractor, obs_batch):
    feats = extractor(obs_batch)
    assert feats.shape == (4, 512)
    assert torch.isfinite(feats).all()


@pytest.mark.parametrize("cls", [VAEExtractor, AEExtractor, ReconExtractor],
                         ids=lambda c: c.__name__)
def test_dream_reconstructs_the_input_resolution(cls):
    space = gym.spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)
    model = cls(space, features_dim=512)
    out = model.dream(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 3, 64, 64), f"{cls.__name__} decoded to {tuple(out.shape)}"
    assert float(out.detach().min()) >= 0.0 and float(out.detach().max()) <= 1.0


@pytest.mark.parametrize("cls", [VAEExtractor, AEExtractor, ReconExtractor])
def test_dream_and_forward_share_one_encoder(cls):
    """A dream that disagrees with forward() is reconstructing from weights the policy never uses."""
    space = gym.spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)
    model = cls(space, features_dim=512)
    state_keys = {k for k in model.state_dict() if k.startswith(("encoder.", "conv1.", "conv2.", "conv3.", "cnn."))}
    assert any(k.startswith("encoder.") for k in state_keys), state_keys
    assert not any(k.startswith(("conv1.", "cnn.")) for k in state_keys), \
        f"{cls.__name__} still carries a second encoder: {sorted(state_keys)}"


def test_dream_changes_when_encoder_is_trained():
    """The decoder must be downstream of the encoder the policy actually trains."""
    space = gym.spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)
    torch.manual_seed(0)
    model = AEExtractor(space, features_dim=512)
    obs = torch.rand(1, 3, 64, 64)
    before = model.dream(obs).detach().clone()
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)
    assert not torch.allclose(before, model.dream(obs))


def test_gradients_reach_the_shared_encoder(extractor, obs_batch):
    extractor(obs_batch).sum().backward()
    encoder_params = list(extractor.encoder.parameters()) if hasattr(extractor, "encoder") else []
    if encoder_params:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder_params)


def test_contrastive_augment_hook_is_used_only_in_training_mode():
    space = gym.spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)
    torch.manual_seed(0)
    model = ContrastiveExtractor(space, features_dim=512)
    obs = torch.rand(1, 3, 64, 64)

    calls = []

    class Spy(ContrastiveExtractor):
        def augment(self, observations):
            calls.append(observations.shape)
            return super().augment(observations)

    spy = Spy(space, features_dim=512)
    spy.train()
    spy.augment_p = 1.0
    spy(obs)
    assert calls, "augment() hook was never consulted"

    model.eval()
    torch.manual_seed(1)
    a, b = model(obs), model(obs)
    assert torch.equal(a, b), "evaluation-mode features must be deterministic"


def test_hwc_observation_layout_is_supported():
    space = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
    model = AEExtractor(space, features_dim=512)
    obs = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8).float()
    assert model(obs).shape == (2, 512)
    # dream() always returns channels-first, regardless of the input layout
    assert model.dream(obs).shape == (2, 3, 64, 64)
