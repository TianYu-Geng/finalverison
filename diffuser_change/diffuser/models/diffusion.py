"""
该文件实现扩散模型的训练损失与条件采样接口：在轨迹空间上定义前向加噪 `q(x_t|x_0)`、反向一步采样 `p(x_{t-1}|x_t)`，并提供完整反向扩散循环。
它输出的 `GaussianDiffusion` 模块会被训练脚本封装进 `Trainer` 做反向传播，也会在规划/推理阶段被 `Policy` 调用以生成动作-观测序列。
"""

import numpy as np
import torch
from torch import nn
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        """
        做什么：初始化扩散过程的常量缓冲区（beta/alpha 累积量、后验系数等）与损失函数封装，并持有噪声预测网络 `model`。
        配置→实例：把 `horizon/维度/n_timesteps/损失权重` 等配置转换为可训练、可采样的 `GaussianDiffusion` 实例。
        谁调用：由训练脚本的 `diffusion_config(model)` 创建，并交给 `Trainer`/`Policy` 在训练或推理时使用。
        """
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 以下为前向/反向扩散过程中常用的预计算量，均以时间步 t 的索引向量被 extract 后
        # 广播到 batch/空间维度，以便在训练/采样时按批量高效计算。
        # 语义说明：
        #  - sqrt_alphas_cumprod[t] = sqrt(prod_{i<=t} alpha_i)
        #  - sqrt_one_minus_alphas_cumprod[t] = sqrt(1 - prod_{i<=t} alpha_i)
        # 这些值用于将 x0 与噪声按比例混合得到 x_t（见 q_sample）以及从 x_t 反推 x0（见 predict_start_from_noise）。
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        # 反向重建用到的比例因子（1/sqrt(alpha_cumprod) 等）
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # 为后验 q(x_{t-1}|x_t,x_0) 计算所需的系数，并注册为缓冲区
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## 由于扩散链起始阶段后验方差为0，因此对对数计算进行了截断处理。
        # posterior 的方差在 t=0 时可能为 0，需要数值稳定处理
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        # posterior_mean_coef1/2 对应 q(x_{t-1}|x_t,x_0) 的线性系数，使得后验均值可以由
        # x_start 与 x_t 的线性组合得到（见 q_posterior）
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## 得到损失系数矩阵并初始化损失函数封装
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        """
        做什么：构造逐时间步、逐维度的损失系数矩阵（动作维度可单独加权，时间维度按折扣衰减）。
        配置→实例：把 `action_weight/discount/weights_dict` 的配置具体化为 `loss_weights:[H, transition_dim]`。
        谁调用：在 `__init__` 中调用，用于初始化 `self.loss_fn`（训练阶段由 `p_losses` 使用）。
        """
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        做什么：根据 `x_t` 与模型输出（噪声或 `x0`）重建 `x_start`（即 `x0` 的估计）。
        配置→实例：将 `predict_epsilon` 的配置分支落到具体的重建公式，实现“模型输出语义”到 `x0` 的统一接口。
        谁调用：由 `p_mean_variance` 在反向扩散一步中调用。
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            # 当模型直接预测 x0（不是 epsilon）时，模型的输出即为重建值
            return noise

    def q_posterior(self, x_start, x_t, t):
        """
        做什么：计算后验 `q(x_{t-1} | x_t, x_start)` 的均值与方差（以及对数方差）。
        配置→实例：将初始化阶段缓存的后验系数应用到给定 batch 上，实例化出该步的高斯分布参数。
        谁调用：由 `p_mean_variance` 在构造一步采样分布时调用。
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        """
        做什么：根据当前 `x_t` 与条件 `cond` 计算反向一步采样分布的均值与方差。
        配置→实例：调用噪声预测网络 `self.model(x, cond, t)` 得到重建信息，再通过 `q_posterior` 实例化 `p(x_{t-1}|x_t,cond)` 参数。
        谁调用：由 `p_sample` 在执行一步反向扩散采样时调用。
        """
        # 调用噪声预测网络：
        #  - 网络接收带噪输入 x（形状 [B,H,transition_dim]）与条件 cond、时间步 t
        #  - 若 self.predict_epsilon==True，网络应返回 epsilon；否则返回 x0
        # 下面把网络输出统一视为 "noise_or_x0"，并通过 predict_start_from_noise 统一还原 x0
        model_out = self.model(x, cond, t)
        x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)

        # 可选裁剪去噪后结果到 [-1,1]（有些任务/数据归一化到该范围）
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            # 保留原始逻辑：当未启用裁剪时，原代码通过断言引发异常（可能用于提醒用户配置）
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        """
        做什么：从 `p(x_{t-1}|x_t,cond)` 采样得到 `x_{t-1}`（当 `t==0` 时不再加噪声）。
        配置→实例：将 `p_mean_variance` 的分布参数与“是否在最后一步加噪”的规则落到具体的张量采样操作。
        谁调用：由 `p_sample_loop` 在反向扩散循环中逐步调用。
        """
        # 计算 model_mean 与 model_log_variance（p 的高斯参数），并从中采样一个样本
        # x: [B, H, transition_dim] 或同形状
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        # 若 t==0（最后一步），不再添加随机噪声；nonzero_mask 用于在 batch 中按样本屏蔽
        # 形状广播：nonzero_mask -> [B,1,1,...] 与 x 对齐
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # 注意：model_log_variance 是 log(var)，因此 exp(0.5*logvar) = sqrt(var)
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_diffusion=False):
        """
        做什么：从标准高斯初始化 `x_T`，执行完整的反向扩散循环得到最终样本（可选返回中间扩散轨迹）。
        配置→实例：将 `n_timesteps` 与条件注入规则（`apply_conditioning`）组合成一个可调用的采样过程。
        谁调用：由 `conditional_sample`（推理入口）调用；训练中的可视化也会间接调用该接口。
        """
        # 采样入口：从标准正态初始化 x_T，并逐步递推得到 x_{T-1} ... x_0
        device = self.betas.device

        batch_size = shape[0]
        # shape 期望为 (B, H, transition_dim)
        x = torch.randn(shape, device=device)
        # apply_conditioning 会把已知的条件（例如开头几步已观测到的 obs/action）写回到 x 中，
        # 以保证采样过程中这些位置保持不变（典型用于条件生成 / prefix conditioning）。
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        # 逆序遍历时间步，从 T-1 到 0
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps)
            # 每一步采样后再应用一次 conditioning，确保条件位置被强制回写
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            # 返回最终样本与完整的扩散轨迹（轴为 time index）
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, **kwargs):
        """
        做什么：根据条件字典 `cond` 生成一批轨迹样本，返回形如 `[B, H, transition_dim]` 的张量。
        配置→实例：将条件（时间索引→观测向量）与采样长度 `horizon` 配置组合为 `p_sample_loop` 的输入形状与初值约束。
        谁调用：推理/规划阶段由 `Policy.__call__` 调用；训练阶段由 `Trainer.render_samples` 通过 EMA 模型调用。
        """
        # cond 的约定：条件是一个元组/结构，第一项长度为 batch_size（通常是已观测的 prefix）
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        # 最终采样张量形状：[B, H, transition_dim]
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        """
        做什么：从前向扩散分布 `q(x_t|x_0)` 采样得到 `x_t`（即对 `x_start` 加噪到时间步 `t`）。
        配置→实例：把已注册的 `sqrt_alphas_cumprod` 等缓冲区应用到 batch 上，实例化加噪公式。
        谁调用：由 `p_losses` 在训练时构造带噪输入 `x_noisy`。
        """
        # 前向加噪：根据时间步 t 将 x_start 与随机噪声按比例混合得到 x_t
        # noise 形状必须与 x_start 相同；extract 会把每个 batch 的标量系数广播到 x_start.shape
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        """
        做什么：构造带噪样本 `x_noisy`，调用噪声预测网络得到重建，并计算与目标（噪声或 `x_start`）之间的损失。
        配置→实例：把条件注入（`apply_conditioning`）与 `predict_epsilon` 的目标选择落到具体的训练损失计算。
        谁调用：由 `loss` 在每个训练 batch 上调用；训练循环通过 `Trainer.train()` 触发。
        """
        # 训练损失的构造流程：
        # 1) 为每个样本采样 noise，并根据 t 把 x_start 加噪得到 x_noisy
        # 2) 将 conditioning 应用到带噪样本（确保已知位置不被扰动）
        # 3) 模型预测（epsilon 或 x0），再把模型输出同样通过 conditioning（输出的已知位置被覆盖）
        # 4) 根据 predict_epsilon 选择目标（noise 或 x_start）并计算逐维加权损失
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        # 将模型输出的已知/条件位置恢复为条件值（与采样时一致）
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            # 目标是噪声（epsilon prediction objective）
            loss, info = self.loss_fn(x_recon, noise)
        else:
            # 目标直接是 x_start（直接预测 x0）
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond):
        """
        做什么：为 batch 采样随机时间步 `t`，并返回 `p_losses(x, cond, t)` 的结果。
        配置→实例：将 `n_timesteps` 的离散时间范围配置落实为 `t ~ Uniform{0..T-1}` 的采样策略。
        谁调用：由 `Trainer.train()` 在每次迭代中调用（`self.model.loss(*batch)`）。
        """
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)

    def forward(self, cond, *args, **kwargs):
        """
        做什么：提供 `nn.Module` 兼容的推理入口，等价于 `conditional_sample(cond, ...)`。
        配置→实例：将“给定条件直接采样”的使用方式封装为 `model(conditions)` 的调用习惯。
        谁调用：由 `Policy.__call__` 使用（`sample = self.diffusion_model(conditions)`）。
        """
        return self.conditional_sample(cond=cond, *args, **kwargs)
