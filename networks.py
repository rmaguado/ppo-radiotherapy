import torch
import torch.nn as nn
from torchsummary import summary
from torch.distributions.normal import Normal
import numpy as np


class FeaturesExtractor3D(nn.Module):
    """
    Structure adapted from C3D https://arxiv.org/pdf/1412.0767
    """

    def __init__(self, observation_shape, features_dim):
        super().__init__()
        self.observation_shape = observation_shape
        n_input_channels = observation_shape[0]

        first_pool_padding = tuple(
            ((observation_shape[i + 1] - 2) % 2) for i in range(3)
        )

        self.cnn = nn.Sequential(
            nn.Conv3d(n_input_channels, 32, 3),
            nn.ReLU(),
            nn.MaxPool3d(2, 2, padding=first_pool_padding),
            nn.Conv3d(32, 64, 3, groups=2),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(64, 128, 3, groups=4),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_input = torch.zeros(observation_shape).unsqueeze(0)
            n_flatten = self.cnn(sample_input).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.cnn(observations))

    def summary(self):
        print("Observation shape: ", self.observation_shape)
        print("Action space: ", self.action_space)
        print("Features dim: ", self.features_dim)
        print("Features extractor: ")
        summary(self.features_extractor, self.observation_shape)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPO_3DCNN(nn.Module):
    def __init__(self, envs, feature_dim=64):
        super().__init__()
        self.features_extractor = FeaturesExtractor3D(
            envs.single_observation_space.shape, feature_dim
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(feature_dim, feature_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(feature_dim, feature_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(feature_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(feature_dim, feature_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(feature_dim, feature_dim)),
            nn.Tanh(),
            layer_init(
                nn.Linear(feature_dim, np.prod(envs.single_action_space.shape)),
                std=0.01,
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

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


class PPO(nn.Module):
    def __init__(self, envs, feature_dim=64):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(), feature_dim
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(feature_dim, feature_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(feature_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(), feature_dim
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(feature_dim, feature_dim)),
            nn.Tanh(),
            layer_init(
                nn.Linear(feature_dim, np.prod(envs.single_action_space.shape)),
                std=0.01,
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )
