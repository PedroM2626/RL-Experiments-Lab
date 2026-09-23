"""Observation-layout handling shared by the SB3 feature extractors.

SB3's ``BasePolicy.preprocess_obs`` converts uint8 images to float and permutes a
channels-last observation space to channels-first *before* a features extractor sees
the batch. Every extractor here used to decide on the permute from the observation
*space* (``self.is_hwc``), so an HWC environment was permuted twice and died in the
first convolution with "expected input[B,64,64,3] to have 3 channels" — despite the
class docstrings advertising HWC support. The study never hit this because
``procgen_wrapper.ProcgenGymWrapper`` yields CHW.

Keying on the tensor that actually arrives makes both layouts work and keeps the CHW
path bit-identical to what produced the published numbers.
"""

import torch

CHANNELS = (1, 3, 4)


def to_chw(observations: torch.Tensor) -> torch.Tensor:
    """Channels-first view of a (B, C, H, W) or (B, H, W, C) batch."""
    if observations.dim() != 4:
        return observations
    channels_last = observations.shape[-1] in CHANNELS and observations.shape[1] not in CHANNELS
    if channels_last:
        observations = observations.permute(0, 3, 1, 2)
    return observations.contiguous()
