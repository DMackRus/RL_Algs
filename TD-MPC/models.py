import torch as T
import torch.nn as nn


class RepresentationModel(nn.Module):
    """
    Given the full state, outputs an embedded latent state.
    Input:  (B, state_dim)
    Output: (B, latent_dim)
    """
    def __init__(self, latent_dim, state_space, hidden_dim=256, image_state=True):
        super().__init__()
        #observations are 64x64 images, so we need to use a convolutional neural network to process them

        if image_state:
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(state_space, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim)
            )

    def forward(self, x):
        # x is a batch of images, so we need to permute the dimensions to match the expected input of the convolutional layers
        # x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return self.net(x)

class LatentDynamics(nn.Module):
    """
    Given a latent state and an action, predicts the next latent state.
    Input:  (B, latent_dim) + (B, action_dim)
    Output: (B, latent_dim)
    """
    def __init__(self, latent_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, z, action):
        x = T.cat([z, action], dim=-1)
        return self.net(x)

class RewardPredictor(nn.Module):
    """
    Given a latent state, predicts the reward.
    Input:  (B, latent_dim + action_dim)
    Output: (B, 1)
    """
    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, action):
        x = T.cat([z, action], dim=-1)
        return self.net(x)

class ValuePredictor(nn.Module):
    """
    Given a latent state and action, predicts the value.
    Input:  (B, latent_dim) + (B, action_dim)
    Output: (B, 1)
    """
    def __init__(self, latent_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # TODO What does elementwise_affine do?
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(alpha=1.0),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, action):
        x = T.cat([z, action], dim=-1)
        return self.net(x)

class PolicyModel(nn.Module):
    """
    Given a latent state, predicts the action to take.
    Input:  (B, latent_dim)
    Output: (B, action_dim)
    """
    def __init__(self, latent_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)