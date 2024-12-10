import torch
import gymnasium as gym

from networks import PPO
from environment import RadiotherapyEnv


def generate_trajectory(
    envs,
    model,
    device: torch.device = torch.device("cpu"),
    num_episodes: int = 15,
):
    obs, _ = envs.reset()
    done = False
    for i in range(num_episodes):
        actions, _, _, _ = model.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, done, _, _ = envs.step(actions.cpu().numpy())
        obs = next_obs

    envs.envs[0].export_animation()


def make_env(visionless):
    def thunk():
        return RadiotherapyEnv(visionless=visionless)

    return thunk


def main():
    device = torch.device("cpu")
    model_path = "saves/test.model"
    envs = gym.vector.SyncVectorEnv([make_env(True)])
    observation_shape = envs.single_observation_space.shape
    action_space = envs.single_action_space.shape
    agent = PPO(observation_shape, action_space, 64)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    generate_trajectory(envs, agent, device)


if __name__ == "__main__":
    main()
