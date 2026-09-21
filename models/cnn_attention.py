import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module
    
    Uses max pooling and average pooling along the channel axis to create
    an attention map that focuses on informative spatial regions.
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        x: (batch_size, channels, height, width)
        """
        # Max pooling and average pooling along channel axis
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        
        # Concatenate
        combined = torch.cat([max_pool, avg_pool], dim=1)  # (B, 2, H, W)
        
        # Convolution to generate spatial attention map
        attention_map = self.conv(combined)  # (B, 1, H, W)
        attention_map = self.sigmoid(attention_map)
        
        # Apply attention with residual connection for stabilization (pure spatial attenuates features -> deterministic 0)
        return x * attention_map + x


class ChannelAttentionModule(nn.Module):
    """
    Channel Attention Module (CBAM-style)
    
    Focuses on which feature channels are most informative.
    """
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        x: (batch_size, channels, height, width)
        """
        b, c, _, _ = x.size()
        
        # Average pooling
        avg_pool = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_pool)
        
        # Max pooling
        max_pool = self.max_pool(x).view(b, c)
        max_out = self.fc(max_pool)
        
        # Combine
        attention = self.sigmoid(avg_out + max_out)
        attention = attention.view(b, c, 1, 1)
        
        return x * attention


class CBAMModule(nn.Module):
    """
    CBAM (Convolutional Block Attention Module)
    Combines channel and spatial attention sequentially.
    """
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMModule, self).__init__()
        
        self.channel_attention = ChannelAttentionModule(channels, reduction)
        self.spatial_attention = SpatialAttentionModule(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AttentionCNN(nn.Module):
    """
    CNN with Attention for SAC - Processes 4 grayscale frames 64x64
    
    Architecture:
    Input: (4, 64, 64)
    Conv2d(32, 8x8, stride 4) → ReLU → CBAM/Spatial
    Conv2d(64, 4x4, stride 2) → ReLU → CBAM/Spatial
    Conv2d(64, 3x3, stride 1) → ReLU → CBAM/Spatial
    Flatten → FC(512) → ReLU → FC(3)
    """
    
    def __init__(self, action_dim=3, feature_dim=512, use_cbam=True):
        super(AttentionCNN, self).__init__()
        
        self.use_cbam = use_cbam
        
        # Convolutional layers with attention
        # use_cbam=True -> Full CBAM (channel + spatial)
        # use_cbam=False -> Spatial Attention only (per README)
        # If use_cbam is None or Identity, no attention applied
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        if use_cbam is True:
            self.attention1 = CBAMModule(32)
        elif use_cbam is False:
            self.attention1 = SpatialAttentionModule(kernel_size=7)
        else:
            self.attention1 = nn.Identity()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        if use_cbam is True:
            self.attention2 = CBAMModule(64)
        elif use_cbam is False:
            self.attention2 = SpatialAttentionModule(kernel_size=7)
        else:
            self.attention2 = nn.Identity()
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        if use_cbam is True:
            self.attention3 = CBAMModule(64)
        elif use_cbam is False:
            self.attention3 = SpatialAttentionModule(kernel_size=7)
        else:
            self.attention3 = nn.Identity()
        
        # Calculate output size after convolutions
        self.flatten_size = 64 * 4 * 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, feature_dim)
        self.fc2 = nn.Linear(feature_dim, action_dim)
        
    def forward(self, x):
        """
        Forward pass
        x: (batch_size, 4, 64, 64) - normalized to [0, 1]
        """
        # Convolutions with attention
        x = F.relu(self.conv1(x))
        x = self.attention1(x)
        
        x = F.relu(self.conv2(x))
        x = self.attention2(x)
        
        x = F.relu(self.conv3(x))
        x = self.attention3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def get_attention_maps(self, x):
        """
        Returns intermediate attention maps (useful for visualization)
        Propagates attention correctly between layers.
        """
        x = F.relu(self.conv1(x))
        att1 = self.attention1(x)
        x = att1

        x = F.relu(self.conv2(x))
        att2 = self.attention2(x)
        x = att2

        x = F.relu(self.conv3(x))
        att3 = self.attention3(x)

        return att1, att2, att3


class AttentionCNNWithFeatureDim(nn.Module):
    """
    Flexible version with custom feature_dim for Actor/Critic.
    """
    
    def __init__(self, action_dim=3, feature_dim=512, use_cbam=True):
        super(AttentionCNNWithFeatureDim, self).__init__()
        
        self.use_cbam = use_cbam
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        if use_cbam is True:
            self.attention1 = CBAMModule(32)
        elif use_cbam is False:
            self.attention1 = SpatialAttentionModule(kernel_size=7)
        else:
            self.attention1 = nn.Identity()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        if use_cbam is True:
            self.attention2 = CBAMModule(64)
        elif use_cbam is False:
            self.attention2 = SpatialAttentionModule(kernel_size=7)
        else:
            self.attention2 = nn.Identity()
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        if use_cbam is True:
            self.attention3 = CBAMModule(64)
        elif use_cbam is False:
            self.attention3 = SpatialAttentionModule(kernel_size=7)
        else:
            self.attention3 = nn.Identity()
        
        self.flatten_size = 64 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_size, feature_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.attention1(x)
        
        x = F.relu(self.conv2(x))
        x = self.attention2(x)
        
        x = F.relu(self.conv3(x))
        x = self.attention3(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        return x


class AttentionCNNActor(nn.Module):
    """
    Actor network for SAC using Attention CNN (tanh + log_prob)
    """
    
    def __init__(self, action_dim=3, feature_dim=512, use_cbam=True, action_scale=1.0):
        super(AttentionCNNActor, self).__init__()
        
        self.feature_extractor = AttentionCNNWithFeatureDim(
            feature_dim=feature_dim, 
            use_cbam=use_cbam
        )
        self.fc_mean = nn.Linear(feature_dim, action_dim)
        self.fc_log_std = nn.Linear(feature_dim, action_dim)
        self.action_scale = action_scale
        self.action_dim = action_dim
        
    def forward(self, x):
        """
        Returns action (B, action_dim), log_prob (B,1) with tanh squashing correction
        """
        features = self.feature_extractor(x)
        mean = self.fc_mean(features)
        log_std = torch.clamp(self.fc_log_std(features), -20, 2)
        std = torch.exp(log_std)

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_mean_log_std(self, x):
        features = self.feature_extractor(x)
        mean = self.fc_mean(features)
        log_std = torch.clamp(self.fc_log_std(features), -20, 2)
        return mean, log_std
    
    def get_action(self, x, deterministic=False):
        if deterministic:
            mean, _ = self.get_mean_log_std(x)
            return torch.tanh(mean) * self.action_scale
        else:
            action, _ = self.forward(x)
            return action


class AttentionCritic(nn.Module):
    """
    Critic network (Q-function) for SAC using Attention CNN.
    """
    
    def __init__(self, action_dim=3, feature_dim=512, use_cbam=True):
        super(AttentionCritic, self).__init__()
        
        self.feature_extractor = AttentionCNNWithFeatureDim(
            feature_dim=feature_dim,
            use_cbam=use_cbam
        )
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x, action):
        features = self.feature_extractor(x)
        
        x_action = torch.cat([features, action], dim=-1)
        
        q1 = self.q1(x_action)
        q2 = self.q2(x_action)
        
        return q1, q2
    
    def q1_forward(self, x, action):
        features = self.feature_extractor(x)
        x_action = torch.cat([features, action], dim=-1)
        return self.q1(x_action)


if __name__ == "__main__":
    # Test network
    model = AttentionCNN(action_dim=3, use_cbam=True)
    
    x = torch.randn(2, 4, 64, 64)
    
    output = model(x)
    print("Output shape:", output.shape)
    print("Test completed successfully!")
    
    # Test with attention maps
    att1, att2, att3 = model.get_attention_maps(x)
    print("Attention 1 shape:", att1.shape)
    print("Attention 2 shape:", att2.shape)
    print("Attention 3 shape:", att3.shape)
    
    # Test Actor (API: action, log_prob)
    actor = AttentionCNNActor(action_dim=3, use_cbam=True)
    action, log_prob = actor(x)
    print("Actor action shape:", action.shape)
    print("Actor log_prob shape:", log_prob.shape)
    # Test deterministic
    det_action = actor.get_action(x, deterministic=True)
    print("Actor deterministic action shape:", det_action.shape)
    
    # Test Critic
    critic = AttentionCritic(action_dim=3, use_cbam=True)
    action = torch.randn(2, 3)
    q1, q2 = critic(x, action)
    print("Critic Q1 shape:", q1.shape)
    print("Critic Q2 shape:", q2.shape)
    
    # Compare parameter count
    from models.cnn_classic import ClassicCNN
    classic_model = ClassicCNN(action_dim=3)
    
    classic_params = sum(p.numel() for p in classic_model.parameters())
    attention_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nClassic CNN parameters: {classic_params:,}")
    print(f"Attention CNN parameters: {attention_params:,}")
    print(f"Difference: {attention_params - classic_params:,} ({(attention_params/classic_params - 1)*100:.1f}% increase)")
