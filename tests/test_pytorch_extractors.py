"""
Unit tests for PyTorch Feature Extractors
Validates forward-pass shapes (B, features_dim), multi-format observations (HWC vs CHW),
and backward autograd differentiability (non-zero parameter gradients).
"""
try:
    import pytest
except ImportError:
    class DummyPytest:
        def fixture(self, f): return f
        def mark(self): pass
        class mark:
            @staticmethod
            def parametrize(*args, **kwargs):
                return lambda f: f
    pytest = DummyPytest()
import torch
import gymnasium as gym
import numpy as np

from models.sb3_extractors import ClassicCNNExtractor, AttentionCNNExtractor
from models.combined_extractors import (
    ImpalaCNNExtractor,
    ImpoolaCNNExtractor,
    ResNet18Extractor,
    LSTMAttentionExtractor,
    ViTExtractor
)
from models.world_model_extractors import (
    VAEExtractor,
    AEExtractor,
    ReconExtractor,
    ContrastiveExtractor
)


@pytest.fixture
def procgen_obs_space():
    """Procgen HWC format: (64, 64, 3) uint8"""
    return gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def chw_obs_space():
    """CHW format: (3, 64, 64) float32"""
    return gym.spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32)


@pytest.mark.parametrize("extractor_cls,kwargs", [
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
])
def test_extractor_forward_backward_hwc(procgen_obs_space, extractor_cls, kwargs):
    """Verifies that each extractor outputs (B, 512) and supports autograd backwards for HWC."""
    batch_size = 4
    extractor = extractor_cls(procgen_obs_space, features_dim=512, **kwargs)
    
    # Random batch of Procgen frames (uint8 0..255)
    obs = torch.randint(0, 256, (batch_size, 64, 64, 3), dtype=torch.uint8)
    
    out = extractor(obs)
    assert out.shape == (batch_size, 512), f"Expected shape ({batch_size}, 512), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN detected in feature outputs"
    
    # Backward pass verification
    loss = out.sum()
    loss.backward()
    
    # Check that encoder weights have gradients
    has_grad = False
    for p in extractor.parameters():
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, f"No non-zero gradients found for {extractor_cls.__name__}"


@pytest.mark.parametrize("extractor_cls,kwargs", [
    (ClassicCNNExtractor, {}),
    (AttentionCNNExtractor, {"use_cbam": True}),
    (ImpalaCNNExtractor, {}),
    (ImpoolaCNNExtractor, {}),
])
def test_extractor_chw_format(chw_obs_space, extractor_cls, kwargs):
    """Verifies support for standard CHW observation spaces."""
    batch_size = 2
    extractor = extractor_cls(chw_obs_space, features_dim=512, **kwargs)
    
    obs = torch.rand(batch_size, 3, 64, 64, dtype=torch.float32)
    out = extractor(obs)
    assert out.shape == (batch_size, 512)


if __name__ == "__main__":
    print("Running PyTorch Extractor unit tests...")
    proc_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    chw_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32)
    
    test_cases = [
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
    
    for cls, kw in test_cases:
        test_extractor_forward_backward_hwc(proc_space, cls, kw)
        print(f"  [PASS] {cls.__name__} (HWC, kwargs={kw})")
        
    for cls, kw in [(ClassicCNNExtractor, {}), (AttentionCNNExtractor, {"use_cbam": True}), (ImpalaCNNExtractor, {}), (ImpoolaCNNExtractor, {})]:
        test_extractor_chw_format(chw_space, cls, kw)
        print(f"  [PASS] {cls.__name__} (CHW, kwargs={kw})")
        
    print("\nALL PYTORCH EXTRACTOR TESTS PASSED!")
