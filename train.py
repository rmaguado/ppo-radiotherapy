from stable_baselines3 import PPO

from network import Custom3DCNN
from environment import RadiotherapyEnv


def train_with_sb3():
    env = RadiotherapyEnv()

    policy_kwargs = dict(
        features_extractor_class=Custom3DCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=2048)

    model.learn(total_timesteps=2048, progress_bar=True)


if __name__ == "__main__":
    train_with_sb3()
