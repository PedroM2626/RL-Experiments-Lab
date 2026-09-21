"""Actor-critic NatureCNN in Flax (faithful reproduction of ``ClassicCNNExtractor`` from the study).

Study (``models/sb3_extractors.py:8``): Conv 32 8x8 s4 -> 64 4x4 s2 ->
64 3x3 s1 -> Flatten -> FC 512 (~600k params), CHW uint8 input.
Here: identical topology, NHWC float32 input in [0, 1] (Flax/JAX convention;
mathematically identical, avoids transposition in the hot path) +
policy head (logits) + scalar value head.
"""

import flax.linen as nn
import jax.numpy as jnp


class NatureActorCritic(nn.Module):
    n_actions: int = 15
    fc_dim: int = 512

    @nn.compact
    def __call__(self, x):
        # x: (B,64,64,3) float32 [0,1]
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.fc_dim)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.n_actions)(x)
        value = nn.Dense(1)(x).squeeze(-1)
        return logits, value


def preprocess(obs):
    """uint8 NHWC -> float32 NHWC [0,1]."""
    return obs.astype(jnp.float32) / 255.0


class ActorCritic(nn.Module):
    """Generic heads over any backbone in the zoo (512D -> logits+value).

    ``stochastic=True`` only for backbones that sample per forward pass
    (VAE): ``key`` is required and z is resampled on each forward pass,
    matching the study.
    """
    backbone: nn.Module
    n_actions: int = 15
    stochastic: bool = False

    @nn.compact
    def __call__(self, x, key=None):
        if self.stochastic:
            feat = self.backbone(x, key)
        else:
            feat = self.backbone(x)
        logits = nn.Dense(self.n_actions)(feat)
        value = nn.Dense(1)(feat).squeeze(-1)
        return logits, value


class ActorCriticXL(nn.Module):
    """Heads over memory-augmented backbone (transformer_xl).

    backbone(x, mem) -> (feat, new_mem). The only architecture with state carry across steps.
    """
    backbone: nn.Module
    n_actions: int = 15

    @nn.compact
    def __call__(self, x, mem):
        feat, new_mem = self.backbone(x, mem)
        logits = nn.Dense(self.n_actions)(feat)
        value = nn.Dense(1)(feat).squeeze(-1)
        return logits, value, new_mem
