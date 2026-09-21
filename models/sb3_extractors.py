import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from models.cnn_attention import CBAMModule, SpatialAttentionModule

class ClassicCNNExtractor(BaseFeaturesExtractor):
    """
    SB3 Extractor compatible with CarRacing (4x64x64), Atari/CartPole (4x64x64), and Procgen (64x64x3).
    Uses classic architecture: Conv 32 8x8 s4 -> 64 4x4 s2 -> 64 3x3 s1 -> FC 512.
    Dynamically computes flatten size to support 64x64 or 84x84, HWC or CHW.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # Detect HWC (64,64,3) vs CHW (4,64,64)
        if len(observation_space.shape) == 3 and observation_space.shape[2] in [1, 3, 4]:
            # HWC
            n_input_channels = int(observation_space.shape[2])
            self.is_hwc = True
            h, w = observation_space.shape[0], observation_space.shape[1]
            dummy_shape = (1, h, w, n_input_channels)
        else:
            n_input_channels = int(observation_space.shape[0])
            self.is_hwc = False
            dummy_shape = (1, *observation_space.shape)
        # To handle 64x64, 84x84 or any spatial dimension, use dummy tensor to calculate flatten size
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flatten size
        with torch.no_grad():
            if self.is_hwc:
                # dummy HWC -> needs transpose to CHW for CNN
                h, w, c = observation_space.shape
                dummy_hwc = torch.zeros(1, h, w, c)
                dummy = dummy_hwc.permute(0, 3, 1, 2)  # CHW
            else:
                dummy = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(dummy).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        elif observations.max() > 1.5:
            observations = observations / 255.0
        if self.is_hwc and observations.dim() == 4 and observations.shape[-1] in [1,3,4]:
            # HWC -> CHW
            observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))


class AttentionCNNExtractor(BaseFeaturesExtractor):
    """
    Feature Extractor with CBAM or Spatial Attention.
    use_cbam=True -> Full CBAM, False -> Spatial Attention only.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, use_cbam: bool = True):
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) == 3 and observation_space.shape[2] in [1, 3, 4]:
            n_input_channels = int(observation_space.shape[2])
            self.is_hwc = True
        else:
            n_input_channels = int(observation_space.shape[0])
            self.is_hwc = False
        self.use_cbam = use_cbam

        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4)
        if use_cbam is True:
            self.att1 = CBAMModule(32)
        elif use_cbam is False:
            self.att1 = SpatialAttentionModule(7)
        else:
            self.att1 = nn.Identity()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        if use_cbam is True:
            self.att2 = CBAMModule(64)
        elif use_cbam is False:
            self.att2 = SpatialAttentionModule(7)
        else:
            self.att2 = nn.Identity()

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        if use_cbam is True:
            self.att3 = CBAMModule(64)
        elif use_cbam is False:
            self.att3 = SpatialAttentionModule(7)
        else:
            self.att3 = nn.Identity()

        # Compute flatten size dynamically
        with torch.no_grad():
            if self.is_hwc:
                h, w, c = observation_space.shape
                dummy_hwc = torch.zeros(1, h, w, c)
                dummy = dummy_hwc.permute(0, 3, 1, 2)
            else:
                dummy = torch.zeros(1, *observation_space.shape)
            x = F.relu(self.conv1(dummy))
            x = self.att1(x)
            x = F.relu(self.conv2(x))
            x = self.att2(x)
            x = F.relu(self.conv3(x))
            x = self.att3(x)
            n_flatten = x.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        elif observations.max() > 1.5:
            observations = observations / 255.0
        if self.is_hwc and observations.dim() == 4 and observations.shape[-1] in [1,3,4]:
            observations = observations.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(observations))
        x = self.att1(x)
        x = F.relu(self.conv2(x))
        x = self.att2(x)
        x = F.relu(self.conv3(x))
        x = self.att3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
