"""Unit tests for the SB3 feature extractors in models/.

Covers forward-pass shape (B, features_dim) for every extractor, HWC and CHW
observation layouts, non-contiguous inputs, the configured features_dim, and
backward autograd reaching the encoder weights.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

from models.combined_extractors import (
    ImpalaCNNExtractor,
    ImpoolaCNNExtractor,
    ResNet18Extractor,
    LSTMAttentionExtractor,
    ViTExtractor
)
from models.sb3_extractors import ClassicCNNExtractor, AttentionCNNExtractor
from models.world_model_extractors import (
    VAEExtractor,
    AEExtractor,
    ReconExtractor,
    ContrastiveExtractor
)

# Every extractor the benchmarks pass to SB3, with its constructor kwargs.
EXTRACTORS = [
    (ClassicCNNExtractor, {}),
    (AttentionCNNExtractor, {"use_cbam": True}),
    (AttentionCNNExtractor, {"use_cbam": False}),
    (ImpalaCNNExtractor, {}),
    (ImpoolaCNNExtractor, {}),
    (ResNet18Extractor, {}),
    (LSTMAttentionExtractor, {}),
    (ViTExtractor, {}),
    (VAEExtractor, {}),
    (AEExtractor, {}),
    (ReconExtractor, {}),
    (ContrastiveExtractor, {}),
]
IDS = [f"{cls.__name__}{kw or ''}" for cls, kw in EXTRACTORS]
CHW_SUBSET = [(ClassicCNNExtractor, {}), (AttentionCNNExtractor, {"use_cbam": True}),
              (ImpalaCNNExtractor, {}), (ImpoolaCNNExtractor, {})]


@pytest.fixture
def procgen_obs_space():
    """Procgen HWC format: (64, 64, 3) uint8"""
    return gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def chw_obs_space():
    """CHW format: (3, 64, 64) float32"""
    return gym.spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32)


@pytest.mark.parametrize("extractor_cls,kwargs", EXTRACTORS, ids=IDS)
def test_extractor_forward_backward_hwc(procgen_obs_space, extractor_cls, kwargs):
    """Each extractor outputs (B, 512) and supports autograd backwards for HWC."""
    batch_size = 4
    extractor = extractor_cls(procgen_obs_space, features_dim=512, **kwargs)

    obs = torch.randint(0, 256, (batch_size, 64, 64, 3), dtype=torch.uint8)

    out = extractor(obs)
    assert out.shape == (batch_size, 512), f"Expected shape ({batch_size}, 512), got {out.shape}"
    assert torch.isfinite(out).all(), f"NaN/Inf in feature outputs of {extractor_cls.__name__}"

    out.sum().backward()
    assert any(p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
               for p in extractor.parameters()), \
        f"No non-zero gradients found for {extractor_cls.__name__}"


@pytest.mark.parametrize("extractor_cls,kwargs", CHW_SUBSET)
def test_extractor_chw_format(chw_obs_space, extractor_cls, kwargs):
    """Support for standard CHW observation spaces."""
    batch_size = 2
    extractor = extractor_cls(chw_obs_space, features_dim=512, **kwargs)
    obs = torch.rand(batch_size, 3, 64, 64, dtype=torch.float32)
    assert extractor(obs).shape == (batch_size, 512)


@pytest.mark.parametrize("extractor_cls,kwargs", CHW_SUBSET)
def test_non_contiguous_input_matches_contiguous(chw_obs_space, extractor_cls, kwargs):
    """A permuted (non-contiguous) view must give the same features as a copy.

    SB3 hands pixel observations over as CHW views of HWC buffers, so an extractor
    that silently mis-reads strides would corrupt every pixel benchmark.
    """
    base = torch.rand(2, 64, 64, 3)
    non_contiguous = base.permute(0, 3, 1, 2)          # CHW view of an HWC buffer
    contiguous = non_contiguous.contiguous()           # same values, own storage
    assert not non_contiguous.is_contiguous()
    assert contiguous.is_contiguous()

    torch.manual_seed(0)
    a = extractor_cls(chw_obs_space, features_dim=512, **kwargs)
    torch.manual_seed(0)
    b = extractor_cls(chw_obs_space, features_dim=512, **kwargs)
    a.eval(); b.eval()
    with torch.no_grad():
        assert torch.allclose(a(contiguous), b(non_contiguous), atol=1e-5)


def test_features_dim_is_honoured(procgen_obs_space):
    """features_dim must reach the output width; SB3 sizes the policy head from it."""
    for cls, kw in EXTRACTORS:
        extractor = cls(procgen_obs_space, features_dim=64, **kw)
        obs = torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8)
        assert extractor(obs).shape == (1, 64), f"{cls.__name__} ignored features_dim"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
