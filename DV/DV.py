from collections import namedtuple
from dataclasses import dataclass
from easydict import EasyDict
import copy
import math
import numpy as np
import torch
import torch.nn as nn


from network import DAHorizonCritic, DAMlp, IdentityCondition, DiT1d
from diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE


DV_info = namedtuple('DV_info', ['planner_loss', 'critic_loss', 'policy_loss'])


@dataclass
class DVState:
    planner_model: nn.Module
    critic: nn.Module
    policy_model: nn.Module

    target_planner_model: nn.Module
    target_critic: nn.Module
    target_policy_model: nn.Module

    planner_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    policy_optimizer: torch.optim.Optimizer

    planner_scheduler: object = None
    critic_scheduler: object = None
    policy_scheduler: object = None

    global_step: int = 0
    device: torch.device = torch.device("cpu")


def expectile_loss(diff, expectile=0.8):
    weight = torch.where(
        diff > 0,
        torch.full_like(diff, expectile),
        torch.full_like(diff, 1 - expectile),
    )
    return weight * (diff ** 2)


def _cosine_decay_lambda(total_steps: int):
    total_steps = max(int(total_steps), 1)

    def fn(step: int):
        step = min(step, total_steps)
        return 0.5 * (1.0 + math.cos(math.pi * step / total_steps))

    return fn


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def init(config: EasyDict):
    device = torch.device(config.device)

    planner_dim = (
        config.observation_dim
        if config.pipeline_type == "separate"
        else config.observation_dim + config.action_dim
    )

    # --------------- Network Architecture -----------------
    if config.planner_net == "transformer":
        nn_diffusion_planner = DiT1d(
            planner_dim,
            emb_dim=config.planner_emb_dim,
            d_model=config.planner_d_model,
            n_heads=config.planner_d_model // config.planner_d_model_divide,
            depth=config.planner_depth,
            timestep_emb_type="fourier",
        ).to(device)
    else:
        raise NotImplementedError()

    nn_condition_planner = None
    classifier = None

    if config.guidance_type == "MCSS":
        critic = DAHorizonCritic(
            planner_dim,
            emb_dim=config.planner_emb_dim,
            d_model=config.planner_d_model,
            n_heads=config.planner_d_model // config.planner_d_model_divide,
            depth=2,
            norm_type="pre",
        ).to(device)
    else:
        raise NotImplementedError()

    # ----------------- Masking -------------------
    fix_mask = np.zeros((config.planner_horizon, planner_dim), dtype=np.float32)
    fix_mask[0, :config.observation_dim] = 1.0

    loss_weight = np.ones((config.planner_horizon, planner_dim), dtype=np.float32)
    loss_weight[1] = config.planner_next_obs_loss_weight

    fix_mask = torch.tensor(fix_mask, dtype=torch.float32, device=device)
    loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=device)

    # --------------- Diffusion Model with Classifier-Free Guidance --------------------
    diff_lr = 2e-4
    weight_decay = 1e-5

    planner = ContinuousDiffusionSDE(
        nn_diffusion_planner,
        nn_condition_planner,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=classifier,
        ema_rate=config.planner_ema_rate,
        predict_noise=config.planner_predict_noise,
        noise_schedule="linear",
        planner_diffusion_gradient_steps=config.planner_diffusion_gradient_steps,
    )

    planner_optimizer = torch.optim.AdamW(
        planner.model.parameters(),
        lr=diff_lr,
        weight_decay=weight_decay,
    )
    planner_scheduler = torch.optim.lr_scheduler.LambdaLR(
        planner_optimizer,
        lr_lambda=_cosine_decay_lambda(config.planner_diffusion_gradient_steps),
    )

    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=config.critic_learning_rate,
    )
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
        critic_optimizer,
        lr_lambda=_cosine_decay_lambda(config.planner_diffusion_gradient_steps),
    )

    if config.pipeline_type == "separate":
        if config.use_diffusion_policy:
            nn_diffusion_policy = DAMlp(
                config.observation_dim,
                config.action_dim,
                emb_dim=64,
                hidden_dim=config.policy_hidden_dim,
                timestep_emb_type="positional",
            ).to(device)

            nn_condition_policy = IdentityCondition(dropout=0.0).to(device)

            policy = DiscreteDiffusionSDE(
                nn_diffusion_policy,
                nn_condition_policy,
                predict_noise=config.policy_predict_noise,
                x_max=+1.0 * torch.ones((1, config.action_dim), device=device),
                x_min=-1.0 * torch.ones((1, config.action_dim), device=device),
                diffusion_steps=config.policy_diffusion_steps,
                ema_rate=config.policy_ema_rate,
            )

            policy_optimizer = torch.optim.AdamW(
                policy.model.parameters(),
                lr=config.policy_learning_rate,
            )
            policy_scheduler = torch.optim.lr_scheduler.LambdaLR(
                policy_optimizer,
                lr_lambda=_cosine_decay_lambda(config.policy_diffusion_gradient_steps),
            )
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    target_planner_model = copy.deepcopy(planner.model).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    target_policy_model = copy.deepcopy(policy.model).to(device)

    target_planner_model.eval()
    target_critic.eval()
    target_policy_model.eval()

    agent_state = DVState(
        planner_model=planner.model,
        critic=critic,
        policy_model=policy.model,
        target_planner_model=target_planner_model,
        target_critic=target_critic,
        target_policy_model=target_policy_model,
        planner_optimizer=planner_optimizer,
        critic_optimizer=critic_optimizer,
        policy_optimizer=policy_optimizer,
        planner_scheduler=planner_scheduler,
        critic_scheduler=critic_scheduler,
        policy_scheduler=policy_scheduler,
        global_step=0,
        device=device,
    )

    return planner, policy, agent_state


def critic_update(agent_state: DVState, config, planner_batch):
    critic = agent_state.critic
    critic_optimizer = agent_state.critic_optimizer

    critic.train()
    critic_optimizer.zero_grad(set_to_none=True)

    val_pred = critic(planner_batch.obs)
    assert val_pred.shape == planner_batch.val.shape
    loss = ((val_pred - planner_batch.val) ** 2).mean()

    loss.backward()
    critic_optimizer.step()

    if agent_state.critic_scheduler is not None:
        agent_state.critic_scheduler.step()

    soft_update(
        agent_state.target_critic,
        critic,
        tau=1.0 - config.policy_ema_rate,
    )

    return agent_state, loss.detach()


def policy_update(policy, agent_state: DVState, config, policy_batch):
    policy_model = agent_state.policy_model
    policy_optimizer = agent_state.policy_optimizer
    device = agent_state.device

    policy_model.train()
    policy_optimizer.zero_grad(set_to_none=True)

    policy_td_obs = policy_batch.obs[:, 0, :]
    policy_td_next_obs = policy_batch.obs[:, 1, :]
    policy_td_act = policy_batch.act[:, 0, :]

    condition = torch.cat([policy_td_obs, policy_td_next_obs], dim=-1)

    xt, t, eps = policy.add_noise(policy_td_act)

    pred = policy_model(
        xt,
        t,
        condition,
        config.use_policy_condition,
        config.train_condition,
    )
    loss = (pred - eps) ** 2
    loss = loss * policy.loss_weight * (1 - policy.fix_mask)
    loss = loss.mean()

    loss.backward()
    policy_optimizer.step()

    if agent_state.policy_scheduler is not None:
        agent_state.policy_scheduler.step()

    soft_update(
        agent_state.target_policy_model,
        policy_model,
        tau=1.0 - config.policy_ema_rate,
    )

    return agent_state, loss.detach()


def planner_update(planner, agent_state: DVState, config, planner_batch, weighted_tensor=None):
    planner_model = agent_state.planner_model
    planner_optimizer = agent_state.planner_optimizer

    planner_model.train()
    planner_optimizer.zero_grad(set_to_none=True)

    condition = None

    xt, t, eps = planner.add_noise(planner_batch.obs)

    pred = planner_model(
        xt,
        t,
        condition,
        config.use_planner_condition,
        config.train_condition,
    )
    loss = (pred - eps) ** 2
    loss = loss * planner.loss_weight * (1 - planner.fix_mask)

    if config.use_advantage_weighting:
        loss = loss * weighted_tensor.unsqueeze(1)

    loss = loss.mean()

    loss.backward()
    planner_optimizer.step()

    if agent_state.planner_scheduler is not None:
        agent_state.planner_scheduler.step()

    soft_update(
        agent_state.target_planner_model,
        planner_model,
        tau=1.0 - config.planner_ema_rate,
    )

    return agent_state, loss.detach()


def update(planner, policy, agent_state: DVState, config, planner_dataset, policy_dataset):
    planner_batch = planner_dataset.sample(config.batch_size, device=agent_state.device)
    policy_batch = policy_dataset.sample(config.batch_size, device=agent_state.device)

    if config.use_advantage_weighting:
        weighted_tensor = torch.exp((planner_batch.val - 1) * config.weight_factor)
    else:
        weighted_tensor = None

    agent_state, planner_loss = planner_update(
        planner,
        agent_state,
        config,
        planner_batch,
        weighted_tensor=weighted_tensor,
    )
    agent_state, critic_loss = critic_update(
        agent_state,
        config,
        planner_batch,
    )
    agent_state, policy_loss = policy_update(
        policy,
        agent_state,
        config,
        policy_batch,
    )

    agent_state.global_step += 1

    return agent_state, DV_info(
        planner_loss=planner_loss,
        critic_loss=critic_loss,
        policy_loss=policy_loss,
    )