"""
Modified from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
"""

import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import wandb
import argparse
import omegaconf
from dotenv import load_dotenv

from environment import RadiotherapyEnv


def get_argparser():
    parser = argparse.ArgumentParser(description="PPO agent", add_help=True)
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/default.yaml",
        help="path to the config file",
    )
    parser.add_argument("--output-dir", type=str, help="path to the output directory")
    return parser


def get_config():
    parser = get_argparser()
    args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(args.config_file)
    cfg.batch_size = int(cfg.num_envs * cfg.num_steps)
    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.num_iterations = cfg.total_timesteps // cfg.batch_size
    cfg.output_dir = args.output_dir
    return cfg


def make_env(Env, gamma):
    def thunk():
        env = Env()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10, 10),
            observation_space=env.observation_space,
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


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


class Agent(nn.Module):
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
        return self.critic(x)

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


def set_seeds(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic


def log_training_metrics(
    optimizer,
    global_step,
    start_time,
    clipfracs,
    old_approx_kl,
    approx_kl,
    pg_loss,
    v_loss,
    entropy_loss,
    explained_var,
):
    def log_metric(name, value):
        wandb.log({name: value, "global_step": global_step})

    learning_rate = optimizer.param_groups[0]["lr"]
    sps = int(global_step / (time.time() - start_time))
    log_metric("charts/learning_rate", learning_rate)
    log_metric("losses/value_loss", v_loss.item())
    log_metric("losses/policy_loss", pg_loss.item())
    log_metric("losses/entropy", entropy_loss.item())
    log_metric("losses/old_approx_kl", old_approx_kl.item())
    log_metric("losses/approx_kl", approx_kl.item())
    log_metric("losses/clipfrac", np.mean(clipfracs))
    log_metric("losses/explained_variance", explained_var)
    log_metric("charts/SPS", sps)


def log_episode_metrics(global_step, info):
    def log_metric(name, value):
        wandb.log({name: value, "global_step": global_step})

    log_metric("charts/episodic_return", info["episode"]["r"])
    log_metric("charts/episodic_length", info["episode"]["l"])


def train(
    cfg,
    agent: Agent,
    optimizer: optim.Optimizer,
    envs: gym.vector.SyncVectorEnv,
    device: torch.device,
):
    obs = torch.zeros(
        (cfg.num_steps, cfg.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (cfg.num_steps, cfg.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)

    # TRY NOT TO MODIFY
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(cfg.num_envs).to(device)

    for iteration in range(1, cfg.num_iterations + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            lrnow = frac * cfg.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, cfg.num_steps):
            global_step += cfg.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        log_episode_metrics(global_step, info)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        log_training_metrics(
            optimizer,
            global_step,
            start_time,
            clipfracs,
            old_approx_kl,
            approx_kl,
            pg_loss,
            v_loss,
            entropy_loss,
            explained_var,
        )


if __name__ == "__main__":
    load_dotenv()
    wandb_key = os.environ.get("WANDB_KEY")
    cfg = get_config()
    run_name = f"{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
    wandb.login(key=wandb_key)
    wandb.init(
        project=cfg.wandb_project_name,
        config=cfg,
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

    set_seeds(cfg)

    if cfg.cuda:
        assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda" if cfg.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(RadiotherapyEnv, cfg.gamma) for i in range(cfg.num_envs)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent(envs, RadiotherapyEnv.OBSERVATION_SHAPE, features_dim=128).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    train(cfg, agent, optimizer, envs, device)

    if cfg.save_model:
        model_path = f"{cfg.output_dir}/{run_name}.model"
        torch.save(agent.state_dict(), model_path)

    envs.close()
