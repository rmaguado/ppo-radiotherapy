import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torchsummary import summary
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FeaturesExtractor3D(nn.Module):
    """
    Structure adapted from C3D https://arxiv.org/pdf/1412.0767
    """

    def __init__(self, observation_shape, features_dim):
        super().__init__()
        self.observation_shape = observation_shape
        n_input_channels = observation_shape[0]

        first_pool_padding = tuple(
            (2 - ((observation_shape[i + 1] - 2) % 2)) for i in range(3)
        )

        self.cnn = nn.Sequential(
            nn.Conv3d(n_input_channels, 32, 3),
            nn.ReLU(),
            nn.MaxPool3d(2, 2, first_pool_padding),
            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(64, 128, 3),
            nn.ReLU(),
            nn.Conv3d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_input = torch.zeros(observation_shape).unsqueeze(0)
            n_flatten = self.cnn(sample_input).shape[1]

        self.mlp = nn.Sequential(
            layer_init(nn.Linear(n_flatten, features_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(features_dim, features_dim)),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.cnn(observations))


class PPO(nn.Module):
    def __init__(self, envs, observation_shape, features_dim, show_summary=False):
        super().__init__()
        self.features_dim = features_dim
        self.observation_shape = observation_shape
        self.action_space = np.prod(envs.single_action_space.shape)
        self.features_extractor = FeaturesExtractor3D(observation_shape, features_dim)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(features_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(features_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, self.action_space),
                std=0.01,
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, self.action_space, dtype=torch.float32)
        )
        if show_summary:
            self.summary()

    def summary(self):
        print("Observation shape: ", self.observation_shape)
        print("Action space: ", self.action_space)
        print("Features dim: ", self.features_dim)
        print("Features extractor: ")
        summary(self.features_extractor, self.observation_shape)

    def get_value(self, x):
        features = self.features_extractor(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.features_extractor(x)
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(features),
        )
