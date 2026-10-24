"""Gate the same-program pairs and their distinct opt-in variants (README 3.6.2).

The audit measured that the published sixteen are fourteen distinct programs: ae and
recon build identical modules in the same order, and c_aug_noise subclasses
ContrastiveExtractor with 'pass'. These tests pin both facts bitwise -- so the count
cannot drift silently -- and verify the two opt-in variants added on 24/10/2026 are
genuinely distinct programs available to future grids (recon_bottleneck,
c_aug_noise_drop).
"""
import numpy as np
import torch

from compare_augment_contrastive import ContrastiveNoise, ContrastivePixelDrop
from models.world_model_extractors import (AEExtractor, ContrastiveExtractor,
                                           ReconBottleneckExtractor, ReconExtractor)

OBS_HWC = (64, 64, 3)


def _space():
    import gym.spaces as spaces
    return spaces.Box(low=0, high=255, shape=OBS_HWC, dtype=np.uint8)


def _same_params(m1, m2):
    p1 = [t.detach().clone() for t in m1.state_dict().values()]
    p2 = [t.detach().clone() for t in m2.state_dict().values()]
    if len(p1) != len(p2):
        return False
    return all(a.shape == b.shape and torch.equal(a, b) for a, b in zip(p1, p2))


def _features(ext):
    ext.eval()
    with torch.no_grad():
        return ext(torch.zeros(1, *OBS_HWC))


def test_ae_and_recon_are_the_same_program():
    """3.6.2's finding, now a gate: identical modules, identical output, one program."""
    torch.manual_seed(0)
    ae = AEExtractor(_space(), features_dim=512)
    torch.manual_seed(0)
    recon = ReconExtractor(_space(), features_dim=512)
    assert _same_params(ae, recon)
    assert torch.equal(_features(ae), _features(recon))


def test_contrastive_and_c_aug_noise_are_the_same_program():
    torch.manual_seed(0)
    c = ContrastiveExtractor(_space(), features_dim=512)
    torch.manual_seed(0)
    cn = ContrastiveNoise(_space(), features_dim=512)
    assert _same_params(c, cn)
    assert torch.equal(_features(c), _features(cn))


def test_recon_bottleneck_is_a_distinct_program():
    torch.manual_seed(0)
    ae = AEExtractor(_space(), features_dim=512)
    torch.manual_seed(0)
    rb = ReconBottleneckExtractor(_space(), features_dim=512, latent_dim=128)
    assert not _same_params(ae, rb)
    n_ae = sum(p.numel() for p in ae.parameters())
    n_rb = sum(p.numel() for p in rb.parameters())
    # the bottleneck adds its own projection: n_flat -> 128 -> 512
    assert n_rb != n_ae
    # a distinct encoder must produce a distinct feature vector for the same input
    assert not torch.equal(_features(ae), _features(rb))


def test_c_aug_noise_drop_is_a_distinct_program():
    torch.manual_seed(0)
    c = ContrastiveExtractor(_space(), features_dim=512)
    torch.manual_seed(0)
    cd = ContrastivePixelDrop(_space(), features_dim=512)
    # same backbone/head weights by construction, but the train-time perturbation
    # differs: pixel dropout is not expressible as the base arm's 0.01 jitter
    torch.manual_seed(7)
    a1 = c.augment(torch.ones(2, 3, 64, 64))
    torch.manual_seed(7)
    a2 = cd.augment(torch.ones(2, 3, 64, 64))
    assert not torch.equal(a1, a2)
    # dropout zeroes pixels outright: some values must be exactly 0.0
    assert (a2 == 0.0).any()
    assert (a1 > 0.9).all()  # the 0.01 jitter barely moves a ones-image


def test_audit_registry_keeps_the_published_sixteen_and_the_opt_ins():
    import audit_suite_validity as A
    assert len(A.PUBLISHED_ARMS) == 16
    assert A.PUBLISHED_ARMS == [
        "vae", "ae", "recon", "contrastive",
        "classic", "cbam", "spatial", "mlp_vector",
        "c_aug_crop", "c_aug_color", "c_aug_noise",
        "impala", "impoola", "lstm_attention", "vit", "resnet18"]
    for name in ("mlp_scaled", "recon_bottleneck", "c_aug_noise_drop"):
        assert name in A.BY_NAME, name
    # the published arms carry vector_scale=False (archived semantics); mlp_scaled is
    # the only scaled one
    assert A.BY_NAME["mlp_scaled"][5] is True
    assert all(A.BY_NAME[a][5] is False for a in A.PUBLISHED_ARMS)
