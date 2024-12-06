import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FeaturesExtractor3D(nn.Module):
    def __init__(self, observation_shape, features_dim):
        super().__init__()
        self.observation_shape = observation_shape
        n_input_channels = observation_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv3d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_input = torch.zeros(observation_shape).unsqueeze(0)
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            layer_init(nn.Linear(n_flatten, features_dim)), nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class PPO(nn.Module):
    def __init__(self, envs, observation_shape, features_dim):
        super().__init__()
        self.features_dim = features_dim
        self.observation_shape = observation_shape
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
                nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01
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
