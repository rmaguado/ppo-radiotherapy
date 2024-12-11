import torch
import gymnasium as gym

from networks import PPO
from environment import RadiotherapyEnv


def generate_trajectory(
    envs,
    model,
    device: torch.device = torch.device("cpu"),
    num_episodes: int = 15,
    output_file: str = "trajectory",
):
    obs, _ = envs.reset()
    done = False
    for i in range(num_episodes):
        actions, _, _, _ = model.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, done, _, _ = envs.step(actions.cpu().numpy())
        obs = next_obs

    envs.envs[0].export_animation(output_file)


def make_env(visionless):
    def thunk():
        return RadiotherapyEnv(visionless=visionless)

    return thunk


def new_model(envs, device):
    observation_shape = envs.single_observation_space.shape
    action_space = envs.single_action_space.shape
    model = PPO(observation_shape, action_space, 64)
    model.to(device)
    return model


def load_model(model_path, envs, device):
    observation_shape = envs.single_observation_space.shape
    action_space = envs.single_action_space.shape
    model = PPO(observation_shape, action_space, 64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def main():
    device = torch.device("cpu")
    save_name = "20M"
    envs = gym.vector.SyncVectorEnv([make_env(True)])
    agent = load_model(f"saves/{save_name}.model", envs, device)
    # agent = new_model(envs, device)
    agent.eval()
    generate_trajectory(envs, agent, device, output_file=save_name)


if __name__ == "__main__":
    main()
