import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from util import at_least_ndim, SinusoidalEmbedding


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        device = x.device
        dtype = x.dtype

        freqs = torch.arange(
            start=0,
            end=self.dim // 2,
            device=device,
            dtype=torch.float32,
        )
        denom = (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = freqs / denom
        freqs = (1.0 / self.max_positions) ** freqs
        freqs = freqs.to(dtype=dtype)

        x = torch.outer(x, freqs)
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=1)
        return x


class FourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale=16):
        super().__init__()
        freqs = torch.randn(dim // 8) * scale
        self.register_buffer("freqs", freqs)

        self.mlp = nn.Sequential(
            nn.Linear(dim // 4, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        emb = torch.einsum('...i,j->...ij', x, (2 * math.pi * self.freqs.to(x.device, x.dtype)))
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return self.mlp(emb)


SUPPORTED_TIMESTEP_EMBEDDING = {
    "positional": PositionalEmbedding,
    "fourier": FourierEmbedding,
}


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0, norm_type="post"):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = MultiHeadSelfAttention(hidden_size, n_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attn(x)
        x = x + self.mlp(self.norm2(x))
        return x


class DAHorizonCritic(nn.Module):
    def __init__(
        self,
        in_dim: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        norm_type: str = "post",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.d_model = d_model

        self.x_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = SinusoidalEmbedding(d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout, norm_type) for _ in range(depth)
        ])
        self.final_layer = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.constant_(self.x_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.constant_(self.final_layer.bias, 0.0)

    def forward(self, x):
        pos_emb_cache = self.pos_emb(
            torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        )
        x = self.x_proj(x) + pos_emb_cache.unsqueeze(0)

        for block in self.blocks:
            x = block(x)

        x = self.final_layer(x)
        x = x[:, 0, :]
        return x

    def eval_forward(self, x):
        return self.forward(x)


class BaseNNDiffusion(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__()
        assert timestep_emb_type in SUPPORTED_TIMESTEP_EMBEDDING
        timestep_emb_params = timestep_emb_params or {}
        self.map_noise = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](emb_dim, **timestep_emb_params)

    def forward(self, x, noise, condition=None):
        raise NotImplementedError


class DAMlp(BaseNNDiffusion):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        emb_dim: int = 16,
        hidden_dim: int = 256,
        timestep_emb_type: str = "positional",
        timestep_emb_params=None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.Mish(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

        self.mid_layer = nn.Sequential(
            nn.Linear(obs_dim * 2 + act_dim + emb_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

        self.final_layer = nn.Linear(hidden_dim, act_dim)

    def forward(self, x, noise, condition=None, use_condition=None, train_condition=None):
        t = self.time_mlp(self.map_noise(noise))
        x = torch.cat([x, t, condition], dim=-1)
        x = self.mid_layer(x)
        return self.final_layer(x)


def get_mask(mask, mask_shape: tuple, dropout: float, train: bool, device, dtype):
    if train:
        mask = (torch.rand(mask_shape, device=device) > dropout).to(dtype)
    else:
        mask = 1.0 if mask is None else mask
    return mask


class IdentityCondition(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout

    def forward(self, condition, train=True, mask=None):
        mask = at_least_ndim(
            get_mask(mask, (condition.shape[0],), self.dropout, train, condition.device, condition.dtype),
            condition.ndim
        )
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, device=condition.device, dtype=condition.dtype)
        else:
            mask = mask.to(condition.device, condition.dtype)
        return condition * mask


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = MultiHeadSelfAttention(hidden_size, n_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )

        self._init_zero()

    def _init_zero(self):
        last = self.adaLN_modulation[1]
        nn.init.constant_(last.weight, 0.0)
        nn.init.constant_(last.bias, 0.0)

    def forward(self, x, t):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
            self.adaLN_modulation(t), 6, dim=1
        )
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer1d(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self._init_zero()

    def _init_zero(self):
        nn.init.constant_(self.linear.weight, 0.0)
        nn.init.constant_(self.linear.bias, 0.0)
        last = self.adaLN_modulation[1]
        nn.init.constant_(last.weight, 0.0)
        nn.init.constant_(last.bias, 0.0)

    def forward(self, x, t):
        shift, scale = torch.chunk(self.adaLN_modulation(t), 2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT1d(BaseNNDiffusion):
    def __init__(
        self,
        in_dim: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.d_model = d_model

        self.x_proj = nn.Linear(in_dim, d_model)
        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.Mish(),
            nn.Linear(d_model, d_model),
            nn.Mish(),
        )

        self.pos_emb = SinusoidalEmbedding(d_model)
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, dropout) for _ in range(depth)])
        self.final_layer = FinalLayer1d(d_model, in_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.constant_(self.x_proj.bias, 0.0)

        for layer in self.map_emb:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x, noise, condition=None, use_condition=False, train_condition=None):
        pos_emb_cache = self.pos_emb(
            torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        )

        x = self.x_proj(x) + pos_emb_cache.unsqueeze(0)
        emb = self.map_noise(noise)
        if use_condition and condition is not None:
            emb = emb + condition
        emb = self.map_emb(emb)

        for block in self.blocks:
            x = block(x, emb)
        x = self.final_layer(x, emb)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        din,
        dout=1,
        hidden_dims=(256, 256),
        activation=nn.ReLU,
        activate_final: bool = False,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        dims = [din] + list(hidden_dims) + [dout]

        layers = []
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            nn.init.orthogonal_(linear.weight, gain=math.sqrt(2))
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)

            if i < len(dims) - 2:
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                layers.append(activation())
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))

        if activate_final:
            layers.append(activation())

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class TanhTransformedDistribution(D.TransformedDistribution):
    @property
    def mean(self):
        x = self.base_dist.mean
        for transform in self.transforms:
            x = transform(x)
        return x


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dims=(256, 256),
        activation=nn.ReLU,
        tanh_squash: bool = False,
        temperature=1.0,
        log_std_scale: float = 1e-3,
        layer_norm: bool = False,
    ):
        super().__init__()

        self.temparature = temperature
        self.mlp_layer = MLP(
            observation_dim,
            hidden_dims[-1],
            hidden_dims[:-1],
            activation=activation,
            activate_final=True,
            layer_norm=layer_norm,
        )
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.std_layer = nn.Linear(hidden_dims[-1], action_dim)

        nn.init.orthogonal_(self.mean_layer.weight, gain=math.sqrt(2))
        nn.init.constant_(self.mean_layer.bias, 0.0)

        nn.init.orthogonal_(self.std_layer.weight, gain=log_std_scale)
        nn.init.constant_(self.std_layer.bias, 0.0)

        self.tanh_squash = tanh_squash
        self.action_dim = action_dim

    def forward(self, observations):
        x = self.mlp_layer(observations)

        means = self.mean_layer(x)
        if not self.tanh_squash:
            means = torch.tanh(means)

        log_stds = self.std_layer(x)
        log_stds = torch.clamp(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        base = D.Independent(
            D.Normal(loc=means, scale=torch.exp(log_stds) * self.temparature),
            1
        )

        if self.tanh_squash:
            return TanhTransformedDistribution(base, [D.transforms.TanhTransform(cache_size=1)])
        else:
            return base


class TanhDeterministic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dims=(256, 256), squash_mean: float = 1.0):
        super().__init__()
        self.mlp_layer = MLP(
            observation_dim,
            action_dim,
            hidden_dims,
            activation=nn.ReLU,
            activate_final=False,
            layer_norm=True,
        )
        self.squash_mean = squash_mean

    def forward(self, observations):
        return torch.tanh(self.mlp_layer(observations)) * self.squash_mean

    def eval_forward(self, observations):
        return self.forward(observations)


class V(nn.Module):
    def __init__(self, obs_dim, hidden_dim: int = 256):
        super().__init__()
        self.Vnet = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.Vnet(obs)

    def eval_forward(self, x):
        return self.forward(x)


class TanhStochasticGRU(nn.Module):
    def __init__(
        self,
        observation_dim,
        planner_horizon: int = 4,
        hidden_dim: int = 256,
        squash_mean: float = 1.0,
        divergence: str = 'kl',
    ):
        super().__init__()
        self.linear_1 = nn.Linear(observation_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.linear_mean = nn.Linear(hidden_dim, observation_dim)
        self.linear_log_std = nn.Linear(hidden_dim, observation_dim)
        self.squash_mean = squash_mean
        self.planner_horizon = planner_horizon
        self.divergence = divergence

    def forward(self, x):
        batch_size = x.shape[0]
        h = torch.zeros(batch_size, self.gru.hidden_size, device=x.device, dtype=x.dtype)

        x = self.linear_1(x)
        x = self.ln1(x)
        x = F.relu(x)

        means_seq = []
        log_stds_seq = []

        cur = x
        for _ in range(self.planner_horizon - 1):
            h = self.gru(cur, h)
            cur = self.ln2(h)
            cur = F.relu(cur)

            mean = self.linear_mean(cur)
            log_std = self.linear_log_std(cur)

            means_seq.append(mean)
            log_stds_seq.append(log_std)

            cur = torch.zeros_like(cur)

        means = torch.stack(means_seq, dim=1)
        log_stds = torch.stack(log_stds_seq, dim=1)

        if self.divergence == 'pearson_chi2':
            log_stds = -F.softplus(log_stds) - 0.5
        log_stds = torch.clamp(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        base = D.Independent(
            D.Normal(loc=means, scale=torch.exp(log_stds)),
            1
        )
        return base

    def eval_forward(self, x):
        return self.forward(x)


class TanhDeterministicGRU(nn.Module):
    def __init__(
        self,
        observation_dim,
        planner_horizon: int = 4,
        hidden_dim: int = 256,
        squash_mean: float = 1.0,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(observation_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.linear_mean = nn.Linear(hidden_dim, observation_dim)
        self.squash_mean = squash_mean
        self.planner_horizon = planner_horizon

    def forward(self, x):
        batch_size = x.shape[0]
        h = torch.zeros(batch_size, self.gru.hidden_size, device=x.device, dtype=x.dtype)

        x = self.linear_1(x)
        x = self.ln1(x)
        x = F.relu(x)

        means_seq = []
        cur = x
        for _ in range(self.planner_horizon - 1):
            h = self.gru(cur, h)
            cur = self.ln2(h)
            cur = F.relu(cur)

            mean = self.linear_mean(cur)
            means_seq.append(mean)

            cur = torch.zeros_like(cur)

        means = torch.stack(means_seq, dim=1)
        return means

    def eval_forward(self, x):
        return self.forward(x)


class MixtureStochasticGRU(nn.Module):
    def __init__(
        self,
        observation_dim,
        planner_horizon: int = 4,
        hidden_dim: int = 256,
        num_components: int = 5,
        squash_mean: float = 1.0,
        divergence: str = 'kl',
    ):
        super().__init__()
        self.linear_1 = nn.Linear(observation_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.num_components = num_components
        self.linear_mix_logits = nn.Linear(hidden_dim, num_components)
        self.linear_mean = nn.Linear(hidden_dim, observation_dim * num_components)
        self.linear_log_std = nn.Linear(hidden_dim, observation_dim * num_components)

        self.squash_mean = squash_mean
        self.planner_horizon = planner_horizon
        self.divergence = divergence
        self.observation_dim = observation_dim

    def forward(self, x):
        batch_size = x.shape[0]
        obs_dim = x.shape[-1]
        h = torch.zeros(batch_size, self.gru.hidden_size, device=x.device, dtype=x.dtype)

        x = self.linear_1(x)
        x = self.ln1(x)
        x = F.relu(x)

        means_seq = []
        log_stds_seq = []
        logits_seq = []

        cur = x
        for _ in range(self.planner_horizon - 1):
            h = self.gru(cur, h)
            cur = self.ln2(h)
            cur = F.relu(cur)

            logits = self.linear_mix_logits(cur)
            means = self.linear_mean(cur)
            log_stds = self.linear_log_std(cur)

            means_seq.append(means)
            log_stds_seq.append(log_stds)
            logits_seq.append(logits)

            cur = torch.zeros_like(cur)

        means_seq = torch.stack(means_seq, dim=1)
        log_stds_seq = torch.stack(log_stds_seq, dim=1)
        logits_seq = torch.stack(logits_seq, dim=1)

        means_seq = means_seq.view(batch_size, self.planner_horizon - 1, self.num_components, obs_dim)
        log_stds_seq = log_stds_seq.view(batch_size, self.planner_horizon - 1, self.num_components, obs_dim)

        if self.divergence == 'pearson_chi2':
            log_stds_seq = -F.softplus(log_stds_seq) - 0.5
        log_stds_seq = torch.clamp(log_stds_seq, LOG_STD_MIN, LOG_STD_MAX)

        mix_dist = D.Categorical(logits=logits_seq)
        comp_dist = D.Independent(
            D.Normal(loc=means_seq, scale=torch.exp(log_stds_seq)),
            1
        )
        dist = D.MixtureSameFamily(mix_dist, comp_dist)
        return dist

    def eval_forward(self, x):
        return self.forward(x)