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
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class VAEExtractor(BaseFeaturesExtractor):
    """Variational Autoencoder Auxiliary Representation Extractor.
    
    Encodes observations o_t into a latent distribution q(z|o_t) = N(mu, diag(exp(logvar))).
    Features are sampled via the reparameterization trick and projected to features_dim.
    Operates as an auxiliary visual representation head for model-free PPO.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, latent_dim: int = 128):
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) == 3 and observation_space.shape[2] in [1, 3, 4]:
            n_input = int(observation_space.shape[2])
            self.is_hwc = True
        else:
            n_input = int(observation_space.shape[0])
            self.is_hwc = False
        self.n_input = n_input
        self.conv1 = nn.Conv2d(n_input, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        with torch.no_grad():
            dummy = torch.zeros(1, 64, 64, n_input).permute(0, 3, 1, 2).contiguous() if self.is_hwc else torch.zeros(1, *observation_space.shape)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            self._shape = x.shape[1:]
            n_flat = x.reshape(1, -1).shape[1]
        self.fc_mu = nn.Linear(n_flat, latent_dim)
        self.fc_logvar = nn.Linear(n_flat, latent_dim)
        self.fc_out = nn.Sequential(nn.Linear(latent_dim, features_dim), nn.ReLU())
        self.latent_dim = latent_dim
        # Deconvolutional reconstruction decoder for qualitative visualization
        self.fc_dec = nn.Linear(latent_dim, int(torch.prod(torch.tensor(self._shape))))
        self.deconv1 = nn.ConvTranspose2d(64, 64, 3, stride=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, n_input, 8, stride=4)

    def dream(self, observations: torch.Tensor) -> torch.Tensor:
        """Decodes observations: enc(o) -> sample(z) -> dec(z) to inspect reconstruction fidelity."""
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        elif observations.max() > 1.5:
            observations = observations / 255.0
        if self.is_hwc and observations.dim() == 4 and observations.shape[-1] in [1, 3, 4]:
            observations = observations.permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.conv1(observations))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        xflat = x.reshape(x.size(0), -1)
        mu = self.fc_mu(xflat)
        logvar = self.fc_logvar(xflat)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        h = self.fc_dec(z).view(-1, *self._shape)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = torch.sigmoid(self.deconv3(h))
        return h

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        elif observations.max() > 1.5:
            observations = observations / 255.0
        if self.is_hwc and observations.dim() == 4 and observations.shape[-1] in [1, 3, 4]:
            observations = observations.permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.conv1(observations))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick
        return self.fc_out(z)


class AEExtractor(BaseFeaturesExtractor):
    """Deterministic Autoencoder Auxiliary Representation Extractor.
    
    Encodes observations deterministically: z = enc(o_t).
    Features are passed to policy/value heads; decoder provides visual reconstruction.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) == 3 and observation_space.shape[2] in [1, 3, 4]:
            n_input = int(observation_space.shape[2])
            self.is_hwc = True
        else:
            n_input = int(observation_space.shape[0])
            self.is_hwc = False
        self.n_input = n_input
        self.conv1 = nn.Conv2d(n_input, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        with torch.no_grad():
            dummy = torch.zeros(1, 64, 64, n_input).permute(0, 3, 1, 2).contiguous() if self.is_hwc else torch.zeros(1, *observation_space.shape)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            self._shape = x.shape[1:]
            n_flat = x.reshape(1, -1).shape[1]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten()
        )
        self.fc = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())
        self.fc_dec = nn.Linear(features_dim, int(torch.prod(torch.tensor(self._shape))))
        self.deconv1 = nn.ConvTranspose2d(64, 64, 3, stride=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, n_input, 8, stride=4)

    def dream(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        elif observations.max() > 1.5:
            observations = observations / 255.0
        if self.is_hwc and observations.dim() == 4 and observations.shape[-1] in [1, 3, 4]:
            observations = observations.permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.conv1(observations))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        z = self.fc(x.reshape(x.size(0), -1))
        h = self.fc_dec(z).view(-1, *self._shape)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = torch.sigmoid(self.deconv3(h))
        return h

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        elif observations.max() > 1.5:
            observations = observations / 255.0
        if self.is_hwc and observations.dim() == 4 and observations.shape[-1] in [1, 3, 4]:
            observations = observations.permute(0, 3, 1, 2).contiguous()
        return self.fc(self.cnn(observations))


class ReconExtractor(BaseFeaturesExtractor):
    """Auxiliary Reconstruction Feature Extractor.
    
    Explicit encoder-decoder topology where features are formed by an autoencoding bottleneck.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) == 3 and observation_space.shape[2] in [1, 3, 4]:
            n_input = int(observation_space.shape[2])
            self.is_hwc = True
        else:
            n_input = int(observation_space.shape[0])
            self.is_hwc = False
        self.n_input = n_input
        self.conv1 = nn.Conv2d(n_input, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        with torch.no_grad():
            dummy = torch.zeros(1, 64, 64, n_input).permute(0, 3, 1, 2).contiguous() if self.is_hwc else torch.zeros(1, *observation_space.shape)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            self._shape = x.shape[1:]
            n_flat = x.reshape(1, -1).shape[1]
        self.fc_enc = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())
        self.fc_dec = nn.Linear(features_dim, int(torch.prod(torch.tensor(self._shape))))
        self.deconv1 = nn.ConvTranspose2d(64, 64, 3, stride=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, n_input, 8, stride=4)

    def dream(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        elif observations.max() > 1.5:
            observations = observations / 255.0
        if self.is_hwc and observations.dim() == 4 and observations.shape[-1] in [1, 3, 4]:
            observations = observations.permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.conv1(observations))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        z = self.fc_enc(x.reshape(x.size(0), -1))
        h = self.fc_dec(z).view(-1, *self._shape)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = torch.sigmoid(self.deconv3(h))
        return h

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        elif observations.max() > 1.5:
            observations = observations / 255.0
        if self.is_hwc and observations.dim() == 4 and observations.shape[-1] in [1, 3, 4]:
            observations = observations.permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.conv1(observations))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        return self.fc_enc(x)


class ContrastiveExtractor(BaseFeaturesExtractor):
    """Contrastive Representation Extractor (Instance Discrimination / SimCLR-style).
    
    Optimizes representation invariance under noise perturbations without requiring
    pixel-level image decoding.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) == 3 and observation_space.shape[2] in [1, 3, 4]:
            n_input = int(observation_space.shape[2])
            self.is_hwc = True
        else:
            n_input = int(observation_space.shape[0])
            self.is_hwc = False
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 64, 64, n_input).permute(0, 3, 1, 2).contiguous() if self.is_hwc else torch.zeros(1, *observation_space.shape)
            n_flat = self.cnn(dummy).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())
        self.proj = nn.Sequential(nn.Linear(features_dim, 128), nn.ReLU(), nn.Linear(128, 64))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        elif observations.max() > 1.5:
            observations = observations / 255.0
        if self.is_hwc and observations.dim() == 4 and observations.shape[-1] in [1, 3, 4]:
            observations = observations.permute(0, 3, 1, 2).contiguous()
        # Light stochastic perturbation during training to promote invariant feature manifold
        if self.training and torch.rand(1).item() < 0.5:
            observations = observations + torch.randn_like(observations) * 0.01
            observations = torch.clamp(observations, 0.0, 1.0)
        return self.fc(self.cnn(observations))
