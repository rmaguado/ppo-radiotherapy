import os
import wandb
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dotenv import load_dotenv


def setup_tensorboard(cfg, run_name):
    writer = SummaryWriter(f"{cfg.output_dir}/{run_name}")
    return writer


def setup_wandb(cfg, run_name):
    load_dotenv()
    wandb_key = os.environ.get("WANDB_KEY")
    wandb.login(key=wandb_key)
    wandb.init(
        project=cfg.wandb_project_name,
        config=cfg,
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )


def log_wandb(name, value, global_step):
    wandb.log({name: value, "global_step": global_step})


def log_tensorboard(name, value, global_step, writer):
    writer.add_scalar(name, value, global_step)


def get_logger(cfg, run_name):
    if cfg.use_wandb:
        setup_wandb(cfg, run_name)
    if cfg.use_tensorboard:
        writer = setup_tensorboard(cfg, run_name)

    def log_metric(name, value, global_step):
        if cfg.use_wandb:
            log_wandb(name, value, global_step)
        if cfg.use_tensorboard:
            log_tensorboard(name, value, global_step, writer)

    return log_metric


def log_training_metrics(
    optimizer,
    global_step,
    sps,
    clipfracs,
    old_approx_kl,
    approx_kl,
    pg_loss,
    v_loss,
    entropy_loss,
    explained_var,
    logger,
):
    learning_rate = optimizer.param_groups[0]["lr"]
    logger("charts/learning_rate", learning_rate, global_step)
    logger("losses/value_loss", v_loss.item(), global_step)
    logger("losses/policy_loss", pg_loss.item(), global_step)
    logger("losses/entropy", entropy_loss.item(), global_step)
    logger("losses/old_approx_kl", old_approx_kl.item(), global_step)
    logger("losses/approx_kl", approx_kl.item(), global_step)
    logger("losses/clipfrac", np.mean(clipfracs), global_step)
    logger("losses/explained_variance", explained_var, global_step)
    logger("charts/SPS", sps, global_step)
    print(
        f"Step: {global_step}, SPS: {sps:.2f}, LR: {learning_rate:.2e}, V loss: {v_loss.item():.2f}, PG loss: {pg_loss.item():.2f}, Entropy loss: {entropy_loss.item():.2f}, Old approx KL: {old_approx_kl.item():.2f}, Approx KL: {approx_kl.item():.2f}, Clipfrac: {np.mean(clipfracs):.2f}, Explained variance: {explained_var:.2f}"
    )


def log_episode_metrics(global_step, info, logger):
    episodic_return = info["episode"]["r"]
    episodic_length = info["episode"]["l"]
    logger("charts/episodic_return", episodic_return, global_step)
    logger("charts/episodic_length", episodic_length, global_step)
    print(
        f"Step: {global_step}, Episodic_Return: {episodic_return:.2f}, Episodic_Length: {episodic_length:.2f}"
    )
