import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicCNN(nn.Module):
    """
    Classic CNN for SAC - Processes 4 grayscale frames 64x64
    
    Architecture:
    Input: (4, 64, 64)
    Conv2d(32, 8x8, stride 4) → ReLU
    Conv2d(64, 4x4, stride 2) → ReLU
    Conv2d(64, 3x3, stride 1) → ReLU
    Flatten → FC(512) → ReLU → FC(3)
    """
    
    def __init__(self, action_dim=3, feature_dim=512):
        super(ClassicCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate output size after convolutions
        # Input: 4x64x64
        # Conv1: 32x14x14 ( (64-8)/4 + 1 = 14 )
        # Conv2: 64x6x6 ( (14-4)/2 + 1 = 6 )
        # Conv3: 64x4x4 ( (6-3)/1 + 1 = 4 )
        self.flatten_size = 64 * 4 * 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, feature_dim)
        self.fc2 = nn.Linear(feature_dim, action_dim)
        
    def forward(self, x):
        """
        Forward pass
        x: (batch_size, 4, 64, 64) - normalized to [0, 1]
        """
        # Convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x):
        """
        Returns features before the final layer (useful for debug/visualization)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


class ClassicCNNWithFeatureDim(nn.Module):
    """
    Flexible version allowing custom feature dimensions.
    Useful for SAC Actor and Critic.
    """
    
    def __init__(self, action_dim=3, feature_dim=512):
        super(ClassicCNNWithFeatureDim, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.flatten_size = 64 * 4 * 4
        
        self.fc1 = nn.Linear(self.flatten_size, feature_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


class ClassicCNNActor(nn.Module):
    """
    Actor network for SAC using classic CNN.
    Output: squashed action via tanh + corrected log_prob (SAC)
    """
    
    def __init__(self, action_dim=3, feature_dim=512, action_scale=1.0):
        super(ClassicCNNActor, self).__init__()
        
        self.feature_extractor = ClassicCNNWithFeatureDim(feature_dim=feature_dim)
        self.fc_mean = nn.Linear(feature_dim, action_dim)
        self.fc_log_std = nn.Linear(feature_dim, action_dim)
        self.action_scale = action_scale
        self.action_dim = action_dim
        
    def forward(self, x):
        """
        Forward pass with reparameterization + tanh squashing.
        Returns: action (B, action_dim), log_prob (B, 1)
        Compatible with SACTrainer.update_*: action, log_prob = actor(state)
        """
        features = self.feature_extractor(x)
        mean = self.fc_mean(features)
        log_std = self.fc_log_std(features)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # (B, action_dim)
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale

        # Log prob with tanh squashing correction: log_prob = log N(x_t) - sum log(1 - tanh^2 + eps)
        log_prob = normal.log_prob(x_t)
        # Enforce correction
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # (B, 1)

        return action, log_prob

    def get_mean_log_std(self, x):
        """Returns mean and log_std without sampling (useful for debug)"""
        features = self.feature_extractor(x)
        mean = self.fc_mean(features)
        log_std = torch.clamp(self.fc_log_std(features), -20, 2)
        return mean, log_std
    
    def get_action(self, x, deterministic=False):
        """
        Returns action only (without log_prob), used in select_action/evaluate.
        deterministic=True -> tanh(mean)
        """
        if deterministic:
            mean, _ = self.get_mean_log_std(x)
            return torch.tanh(mean) * self.action_scale
        else:
            action, _ = self.forward(x)
            return action

    def evaluate_log_prob(self, x):
        """Helper for compatibility: returns mean, log_std and log_prob if needed"""
        return self.forward(x)


class ClassicCritic(nn.Module):
    """
    Critic network (Q-function) for SAC using classic CNN.
    """
    
    def __init__(self, action_dim=3, feature_dim=512):
        super(ClassicCritic, self).__init__()
        
        self.feature_extractor = ClassicCNNWithFeatureDim(feature_dim=feature_dim)
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Q2 network (for target network smoothing)
        self.q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x, action):
        features = self.feature_extractor(x)
        
        # Concatenate features with action
        x_action = torch.cat([features, action], dim=-1)
        
        q1 = self.q1(x_action)
        q2 = self.q2(x_action)
        
        return q1, q2
    
    def q1_forward(self, x, action):
        """Returns Q1 only (useful for target computation)"""
        features = self.feature_extractor(x)
        x_action = torch.cat([features, action], dim=-1)
        return self.q1(x_action)


if __name__ == "__main__":
    # Test network
    model = ClassicCNN(action_dim=3)
    
    # Input batch: (batch_size, 4, 64, 64)
    x = torch.randn(2, 4, 64, 64)
    
    output = model(x)
    print("Output shape:", output.shape)
    print("Test completed successfully!")
    
    # Test Actor
    actor = ClassicCNNActor(action_dim=3)
    mean, log_std = actor(x)
    print("Actor mean shape:", mean.shape)
    print("Actor log_std shape:", log_std.shape)
    
    # Test Critic
    critic = ClassicCritic(action_dim=3)
    action = torch.randn(2, 3)
    q1, q2 = critic(x, action)
    print("Critic Q1 shape:", q1.shape)
    print("Critic Q2 shape:", q2.shape)
