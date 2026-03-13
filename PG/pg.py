from collections import namedtuple
from dataclasses import dataclass
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
from easydict import EasyDict

from network import (
    DAHorizonCritic,
    DAMlp,
    IdentityCondition,
    DiT1d,
    TanhStochasticGRU,
    V,
)
from diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE


prior_DV_info = namedtuple(
    'prior_DV_info',
    [
        'total_loss',
        'prior_loss',
        'const_loss',
        'abs_prior_mean',
        'prior_mean',
        'prior_std',
        'critic_loss'
    ]
)


@dataclass
class PriorDVState:
    prior: nn.Module
    target_prior: nn.Module
    prior_optimizer: torch.optim.Optimizer

    critic_T: nn.Module
    target_critic_T: nn.Module
    critic_T_optimizer: torch.optim.Optimizer

    value_T: nn.Module
    target_value_T: nn.Module
    value_T_optimizer: torch.optim.Optimizer

    prior_scheduler: object = None
    critic_T_scheduler: object = None
    value_T_scheduler: object = None

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

    # ---------------- Planner ----------------
    nn_diffusion_planner = DiT1d(
        planner_dim,
        emb_dim=config.planner_emb_dim,
        d_model=config.planner_d_model,
        n_heads=config.planner_d_model // config.planner_d_model_divide,
        depth=config.planner_depth,
        timestep_emb_type="fourier",
    ).to(device)

    nn_condition_planner = None
    classifier = None

    fix_mask = np.zeros((config.planner_horizon, planner_dim), dtype=np.float32)
    fix_mask[0, :config.observation_dim] = 1.0
    loss_weight = np.ones((config.planner_horizon, planner_dim), dtype=np.float32)
    loss_weight[1] = config.planner_next_obs_loss_weight

    fix_mask = torch.tensor(fix_mask, dtype=torch.float32, device=device)
    loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=device)

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

    # ---------------- Policy ----------------
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

    # ---------------- Critic / Value (teacher, loaded later) ----------------
    critic = DAHorizonCritic(
        planner_dim,
        emb_dim=config.planner_emb_dim,
        d_model=config.planner_d_model,
        n_heads=config.planner_d_model // config.planner_d_model_divide,
        depth=2,
        norm_type="pre",
    ).to(device)

    value = V(config.observation_dim, hidden_dim=256).to(device)

    # ---------------- Prior ----------------
    prior = TanhStochasticGRU(
        config.observation_dim,
        planner_horizon=config.planner_horizon,
        hidden_dim=config.prior_hidden_dim,
        squash_mean=config.prior_squash_mean,
        divergence=config.divergence,
    ).to(device)

    prior_optimizer = torch.optim.Adam(
        prior.parameters(),
        lr=config.prior_learning_rate,
    )
    prior_scheduler = torch.optim.lr_scheduler.LambdaLR(
        prior_optimizer,
        lr_lambda=_cosine_decay_lambda(config.planner_diffusion_gradient_steps),
    )

    # ---------------- latent value / latent critic ----------------
    critic_T = DAHorizonCritic(
        planner_dim,
        emb_dim=config.planner_emb_dim,
        d_model=config.planner_d_model,
        n_heads=config.planner_d_model // config.planner_d_model_divide,
        depth=2,
        norm_type="pre",
    ).to(device)

    critic_T_optimizer = torch.optim.Adam(
        critic_T.parameters(),
        lr=config.critic_learning_rate,
    )
    critic_T_scheduler = torch.optim.lr_scheduler.LambdaLR(
        critic_T_optimizer,
        lr_lambda=_cosine_decay_lambda(config.planner_diffusion_gradient_steps),
    )

    value_T = DAHorizonCritic(
        planner_dim,
        emb_dim=config.planner_emb_dim,
        d_model=config.planner_d_model,
        n_heads=config.planner_d_model // config.planner_d_model_divide,
        depth=2,
        norm_type="pre",
    ).to(device)

    value_T_optimizer = torch.optim.Adam(
        value_T.parameters(),
        lr=config.critic_learning_rate,
    )
    value_T_scheduler = torch.optim.lr_scheduler.LambdaLR(
        value_T_optimizer,
        lr_lambda=_cosine_decay_lambda(config.planner_diffusion_gradient_steps),
    )

    target_prior = copy.deepcopy(prior).to(device)
    target_critic_T = copy.deepcopy(critic_T).to(device)
    target_value_T = copy.deepcopy(value_T).to(device)

    target_prior.eval()
    target_critic_T.eval()
    target_value_T.eval()

    agent_state = PriorDVState(
        prior=prior,
        target_prior=target_prior,
        prior_optimizer=prior_optimizer,
        critic_T=critic_T,
        target_critic_T=target_critic_T,
        critic_T_optimizer=critic_T_optimizer,
        value_T=value_T,
        target_value_T=target_value_T,
        value_T_optimizer=value_T_optimizer,
        prior_scheduler=prior_scheduler,
        critic_T_scheduler=critic_T_scheduler,
        value_T_scheduler=value_T_scheduler,
        global_step=0,
        device=device,
    )

    return planner, policy, critic, value, agent_state


def prior_update(agent_state: PriorDVState, config, planner_batch):
    prior = agent_state.prior
    prior_optimizer = agent_state.prior_optimizer

    if config.use_target:
        value_teacher = agent_state.target_value_T
        critic_teacher = agent_state.target_critic_T
    else:
        value_teacher = agent_state.value_T
        critic_teacher = agent_state.critic_T

    obs_one = planner_batch.obs[:, 0, :]

    prior.train()
    prior_optimizer.zero_grad(set_to_none=True)

    dist = prior(obs_one)
    prior_value = dist.rsample() if hasattr(dist, "rsample") else dist.sample()

    if config.use_tanh_squash:
        prior_value = torch.tanh(prior_value) * config.prior_squash_mean

    info = {
        'abs_prior_mean': prior_value.abs().mean(),
        'prior_mean': dist.mean.mean(),
        'prior_std': dist.stddev.mean(),
    }

    prior_s_concat = torch.cat((obs_one.unsqueeze(1), prior_value), dim=1)

    if config.use_value:
        val = value_teacher(prior_s_concat)
        prior_loss = -val.mean()
    else:
        val = critic_teacher(prior_s_concat)
        prior_loss = -val.mean()

    if config.normalize_q:
        lmbda = (1.0 / val.abs().mean()).detach()
        prior_loss = prior_loss * lmbda

    # KL to standard normal
    std_normal_dist = D.Independent(
        D.Normal(
            loc=torch.zeros_like(dist.mean),
            scale=torch.ones_like(dist.stddev),
        ),
        1
    )

    if config.divergence == 'kl':
        kl = D.kl_divergence(dist, std_normal_dist)
        const_loss = kl.mean()
    else:
        raise NotImplementedError()

    total_loss = prior_loss + config.alpha * const_loss
    total_loss.backward()
    prior_optimizer.step()

    if agent_state.prior_scheduler is not None:
        agent_state.prior_scheduler.step()

    soft_update(agent_state.target_prior, prior, tau=0.005)

    return agent_state, {
        'total_loss': total_loss.detach(),
        'prior_loss': prior_loss.detach(),
        'const_loss': const_loss.detach(),
        'abs_prior_mean': info['abs_prior_mean'].detach(),
        'prior_mean': info['prior_mean'].detach(),
        'prior_std': info['prior_std'].detach(),
    }


def critic_update(planner, critic, value, agent_state: PriorDVState, config, planner_batch):
    obs_one = planner_batch.obs[:, 0, :]

    with torch.no_grad():
        prior = agent_state.target_prior
        dist = prior(obs_one)
        prior_value = dist.sample()

        if config.use_tanh_squash:
            prior_value = torch.tanh(prior_value) * config.prior_squash_mean

        traj = planner.sample_prior_train(
            obs_one,
            prior=prior_value,
            config=config,
        )

        if config.use_value:
            # teacher value on generated trajectory
            value_td_ev = value(traj)[:, 1:]
            val_0 = value_td_ev.mean(dim=1)
        else:
            val_0 = critic(traj)

    prior_s_concat = torch.cat((obs_one.unsqueeze(1), prior_value), dim=1)

    if config.use_value:
        model = agent_state.value_T
        optimizer = agent_state.value_T_optimizer

        model.train()
        optimizer.zero_grad(set_to_none=True)

        val_pred = model(prior_s_concat)
        assert val_pred.shape == val_0.shape
        loss = ((val_pred - val_0) ** 2).mean()

        loss.backward()
        optimizer.step()

        if agent_state.value_T_scheduler is not None:
            agent_state.value_T_scheduler.step()

        soft_update(agent_state.target_value_T, model, tau=0.005)
    else:
        model = agent_state.critic_T
        optimizer = agent_state.critic_T_optimizer

        model.train()
        optimizer.zero_grad(set_to_none=True)

        val_pred = model(prior_s_concat)
        assert val_pred.shape == val_0.shape
        loss = ((val_pred - val_0) ** 2).mean()

        loss.backward()
        optimizer.step()

        if agent_state.critic_T_scheduler is not None:
            agent_state.critic_T_scheduler.step()

        soft_update(agent_state.target_critic_T, model, tau=0.005)

    return agent_state, loss.detach()


def update(planner, critic, value, agent_state: PriorDVState, config, planner_dataset):
    planner_batch = planner_dataset.sample(config.batch_size, device=agent_state.device)

    agent_state, critic_loss = critic_update(
        planner,
        critic,
        value,
        agent_state,
        config,
        planner_batch
    )

    agent_state, info = prior_update(
        agent_state,
        config,
        planner_batch
    )

    agent_state.global_step += 1

    return agent_state, prior_DV_info(
        total_loss=info['total_loss'],
        prior_loss=info['prior_loss'],
        const_loss=info['const_loss'],
        abs_prior_mean=info['abs_prior_mean'],
        prior_mean=info['prior_mean'],
        prior_std=info['prior_std'],
        critic_loss=critic_loss
    )