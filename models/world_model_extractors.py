"""Self-Supervised Auxiliary Visual Representation Learning Extractors.

Theoretical Demarcation:
------------------------
In the literature of reinforcement learning and computer vision (e.g., Gelada et al., 2019;
Stooke et al., 2021; Yarats et al., 2021):
- Model-Free RL with Auxiliary Pretext Tasks: The modules below (VAE, AE, Recon, Contrastive)
  serve as visual representation feature extractors (BaseFeaturesExtractor) trained under
  auxiliary reconstruction or contrastive losses alongside model-free PPO. They do NOT learn
  environmental transition dynamics p(s_{t+1}|s_t, a_t) to perform latent imagination rollouts.
- Model-Based Imagination RL (World Models): Architectures such as Dreamer (Hafner et al.,
  2020/2023) explicitly train a recurrent state space transition model (RSSM) and optimize
  the actor-critic entirely through imagined trajectories in latent space. In this repository,
  true imagination-based model-based RL is implemented in `jax_port/train_dreamer.py`.
- The decoders (`dream()`) provided here define reconstruction architectures. However, in the
  root PyTorch / SB3 study (`compare_world_models.py`), standard SB3 PPO `model.learn()` backpropagates
  gradients exclusively from the policy and value heads through `forward()`. No auxiliary reconstruction
  loss (MSE/BCE) or KL divergence was backpropagated during training in SB3; hence the decoders remained
  at random initialization weights. The root SB3 benchmark thus measured the inductive effect of a
  stochastic/dimensional bottleneck in the encoder rather than active auxiliary representation learning.
  Active auxiliary reconstruction and imagination rollouts are fully implemented in `jax_port/train_dreamer.py`.
- Audit (23/09/2026): `dream()` and `forward()` now share one encoder and one preprocessing path.
  `AEExtractor` previously carried a second, parallel encoder (`conv1..conv3`) that `forward()` never
  used, so its "dreams" were decoded from an untrained branch; and the decoder stack reconstructed
  60x60 images from 64x64 inputs, which the visualization hid behind a resize. Both are fixed here,
  which also means the parameter-initialization order differs from the run that produced the published
  section 3.2/3.11 numbers — those checkpoints are gone, so the figures cannot be regenerated bit-exactly.
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _to_chw_float(observations: torch.Tensor, is_hwc: bool) -> torch.Tensor:
    """Scale to [0, 1] and move channels first. Shared by forward() and dream() so the
    two paths can never disagree about how an observation is preprocessed."""
    if observations.dtype == torch.uint8:
        observations = observations.float() / 255.0
    elif observations.max() > 1.5:
        observations = observations / 255.0
    if is_hwc and observations.dim() == 4 and observations.shape[-1] in [1, 3, 4]:
        observations = observations.permute(0, 3, 1, 2).contiguous()
    return observations


class _NatureCNNDecoder(nn.Module):
    """Mirror of the Nature-CNN encoder used by every extractor in this module.

    Encoder geometry on a 64x64 input: 64 -(k8,s4)-> 15 -(k4,s2)-> 6 -(k3,s1)-> 4.
    The decoder inverts that exactly; the output_padding on the middle stage is what
    makes 6 -> 15 possible (a plain transposed convolution yields 14, and the previous
    stack reconstructed 60x60 images from 64x64 inputs).
    """

    def __init__(self, bottleneck_channels: int, latent_dim: int, spatial_shape, n_input: int):
        super().__init__()
        self._shape = tuple(int(s) for s in spatial_shape)
        self.fc_dec = nn.Linear(latent_dim, int(torch.prod(torch.tensor(self._shape))))
        self.deconv1 = nn.ConvTranspose2d(bottleneck_channels, bottleneck_channels, 3, stride=1)
        self.deconv2 = nn.ConvTranspose2d(bottleneck_channels, 32, 4, stride=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, n_input, 8, stride=4)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(latent).view(-1, *self._shape)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        return torch.sigmoid(self.deconv3(h))


class _ConvEncoder(nn.Module):
    """Nature-CNN stem shared by forward() and dream(); returns (batch, n_flat)."""

    def __init__(self, n_input: int):
        super().__init__()
        self.conv1 = nn.Conv2d(n_input, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return F.relu(self.conv3(x))


class _ExtractorBase(BaseFeaturesExtractor):
    """Observation-layout detection and the shared encoder/dummy-probe used by all four heads."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int):
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) == 3 and observation_space.shape[2] in [1, 3, 4]:
            self.n_input = int(observation_space.shape[2])
            self.is_hwc = True
            dummy = torch.zeros(1, 64, 64, self.n_input).permute(0, 3, 1, 2).contiguous()
        else:
            self.n_input = int(observation_space.shape[0])
            self.is_hwc = False
            dummy = torch.zeros(1, *observation_space.shape)
        self.encoder = _ConvEncoder(self.n_input)
        with torch.no_grad():
            probe = self.encoder(dummy)
            self._shape = probe.shape[1:]
            self.n_flat = int(probe.reshape(1, -1).shape[1])

    def prepare(self, observations: torch.Tensor) -> torch.Tensor:
        return _to_chw_float(observations, self.is_hwc)

    def augment(self, observations: torch.Tensor) -> torch.Tensor:
        """Train-time perturbation hook, applied from forward() with probability augment_p.

        Subclasses override this instead of forward() so preprocessing, encoding and the
        policy-facing feature layout cannot drift away from each other.
        """
        return observations

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Feature map of the shared encoder, flattened per sample."""
        x = self.encoder(self.prepare(observations))
        return x.reshape(x.size(0), -1)


class VAEExtractor(_ExtractorBase):
    """Variational bottleneck representation extractor.

    Encodes observations o_t into a latent distribution q(z|o_t) = N(mu, diag(exp(logvar))).
    Features are sampled via the reparameterization trick and projected to features_dim.
    Note that no reconstruction or KL term enters the PPO objective (module docstring), so
    `dream()` shows what the *architecture* can decode, not what training taught it.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, latent_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(self.n_flat, latent_dim)
        self.fc_logvar = nn.Linear(self.n_flat, latent_dim)
        self.fc_out = nn.Sequential(nn.Linear(latent_dim, features_dim), nn.ReLU())
        self.decoder = _NatureCNNDecoder(64, latent_dim, self._shape, self.n_input)

    def _posterior(self, observations: torch.Tensor):
        flat = self.encode(observations)
        return self.fc_mu(flat), self.fc_logvar(flat)

    def dream(self, observations: torch.Tensor) -> torch.Tensor:
        """Decodes observations: enc(o) -> sample(z) -> dec(z) to inspect reconstruction fidelity."""
        mu, logvar = self._posterior(observations)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        mu, logvar = self._posterior(observations)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick
        return self.fc_out(z)


class AEExtractor(_ExtractorBase):
    """Deterministic autoencoder bottleneck representation extractor.

    Encodes observations deterministically: z = enc(o_t); features are passed to the
    policy/value heads, and the decoder reconstructs from the same latent the policy reads.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.fc = nn.Sequential(nn.Linear(self.n_flat, features_dim), nn.ReLU())
        self.decoder = _NatureCNNDecoder(64, features_dim, self._shape, self.n_input)

    def dream(self, observations: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fc(self.encode(observations)))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encode(observations))


class ReconExtractor(_ExtractorBase):
    """Auxiliary Reconstruction Feature Extractor.

    Explicit encoder-decoder topology where features are formed by an autoencoding bottleneck.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.fc_enc = nn.Sequential(nn.Linear(self.n_flat, features_dim), nn.ReLU())
        self.decoder = _NatureCNNDecoder(64, features_dim, self._shape, self.n_input)

    def dream(self, observations: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fc_enc(self.encode(observations)))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.fc_enc(self.encode(observations))


class ContrastiveExtractor(_ExtractorBase):
    """Noise-perturbed CNN representation extractor (the benchmark's `contrastive` arm).

    The distinguishing mechanism of this arm is a Gaussian perturbation of the input
    during training (`self.training`), which acts as a mild stochastic-augmentation
    regularizer. Subclasses swap the perturbation by overriding `augment()`.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.fc = nn.Sequential(nn.Linear(self.n_flat, features_dim), nn.ReLU())
        self.augment_p = 0.5

    def augment(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.clamp(observations + torch.randn_like(observations) * 0.01, 0.0, 1.0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self.prepare(observations)
        if self.training and torch.rand(1).item() < self.augment_p:
            observations = self.augment(observations)
        x = self.encoder(observations).reshape(observations.size(0), -1)
        return self.fc(x)
