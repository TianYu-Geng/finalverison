import math
import os
import random
from typing import Union, Dict, Callable, Optional

import numpy as np
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=x.device, dtype=x.dtype) * (-emb_scale)
        )
        emb = torch.einsum('...i,j->...ij', x, emb)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __call__(self, x):
        return super().__call__(x)


# ================= Noise schedules =================
def linear_noise_schedule(
    t_diffusion,
    beta0: float = 0.1,
    beta1: float = 20.0,
):
    log_alpha = -(beta1 - beta0) / 4.0 * (t_diffusion ** 2) - beta0 / 2.0 * t_diffusion
    alpha = torch.exp(log_alpha)
    sigma = torch.sqrt(1.0 - alpha ** 2)
    return alpha, sigma


def inverse_linear_noise_schedule(
    alpha=None,
    sigma=None,
    logSNR=None,
    beta0: float = 0.1,
    beta1: float = 20.0,
):
    assert (logSNR is not None) or (alpha is not None and sigma is not None)
    lmbda = torch.log(alpha / sigma) if logSNR is None else logSNR

    term = torch.log1p(torch.exp(-2 * lmbda))
    denom = beta0 + torch.sqrt(beta0 ** 2 + 2 * (beta1 - beta0) * term)
    t_diffusion = (2 * term) / denom
    return t_diffusion


def cosine_noise_schedule(t_diffusion, s: float = 0.008):
    if isinstance(t_diffusion, torch.Tensor) and t_diffusion.numel() > 0:
        t_diffusion = t_diffusion.clone()
        t_diffusion.reshape(-1)[-1] = 0.9946

    alpha = torch.cos(
        torch.pi / 2.0 * (t_diffusion + s) / (1 + s)
    ) / math.cos(math.pi / 2.0 * s / (1 + s))
    sigma = torch.sqrt(1.0 - alpha ** 2)
    return alpha, sigma


def inverse_cosine_noise_schedule(
    alpha=None,
    sigma=None,
    logSNR=None,
    s: float = 0.008,
):
    assert (logSNR is not None) or (alpha is not None and sigma is not None)
    lmbda = torch.log(alpha / sigma) if logSNR is None else logSNR

    const = math.cos(math.pi * s / 2.0 / (s + 1))
    inner = torch.exp(
        -0.5 * torch.log1p(torch.exp(-2 * lmbda)) + math.log(const)
    )
    t_diffusion = 2 * (1 + s) / math.pi * torch.arccos(inner) - s
    return t_diffusion


SUPPORTED_NOISE_SCHEDULES = {
    "linear": {
        "forward": linear_noise_schedule,
        "reverse": inverse_linear_noise_schedule,
    },
    "cosine": {
        "forward": cosine_noise_schedule,
        "reverse": inverse_cosine_noise_schedule,
    },
}


def uniform_discretization(
    T: int = 1000,
    eps: float = 1e-3,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    return torch.linspace(eps, 1.0, T, device=device, dtype=dtype)


SUPPORTED_DISCRETIZATIONS = {
    "uniform": uniform_discretization,
}


def uniform_sampling_step_schedule(
    T: int = 1000,
    sampling_steps: int = 10,
    device: Optional[torch.device] = None,
):
    return torch.linspace(
        0, T - 1, sampling_steps + 1, device=device, dtype=torch.int32
    )


def uniform_sampling_step_schedule_continuous(
    trange=None,
    sampling_steps: int = 10,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    if trange is None:
        trange = [1e-3, 1.0]
    return torch.linspace(
        trange[0], trange[1], sampling_steps + 1, device=device, dtype=dtype
    )


SUPPORTED_SAMPLING_STEP_SCHEDULE = {
    "uniform": uniform_sampling_step_schedule,
    "uniform_continuous": uniform_sampling_step_schedule_continuous,
}


def at_least_ndim(x, ndim: int, pad: int = 0):
    """
    Add dimensions to the input tensor to make it at least ndim-dimensional.

    Args:
        x: Union[np.ndarray, torch.Tensor, int, float], input tensor
        ndim: int, minimum number of dimensions
        pad: int, padding direction.
             0: pad in the last dimension
             1: pad in the first dimension

    Returns:
        - np.ndarray or torch.Tensor: reshaped tensor
        - int or float: input value
    """
    if isinstance(x, np.ndarray):
        if ndim > x.ndim:
            if pad == 0:
                return np.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return np.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        return x

    elif isinstance(x, torch.Tensor):
        if ndim > x.ndim:
            if pad == 0:
                return x.reshape(x.shape + (1,) * (ndim - x.ndim))
            else:
                return x.reshape((1,) * (ndim - x.ndim) + tuple(x.shape))
        return x

    elif isinstance(x, (int, float)):
        return x

    else:
        raise ValueError(f"Unsupported type {type(x)}")


class GaussianNormalizer:
    def __init__(self, X, start_dim: int = -1):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        elif not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        total_dims = X.ndim
        if start_dim < 0:
            start_dim = total_dims + start_dim

        axes = tuple(range(start_dim))

        self.mean = torch.mean(X, dim=axes)
        self.std = torch.std(X, dim=axes, unbiased=False)
        self.std = torch.where(
            self.std == 0,
            torch.ones_like(self.std),
            self.std
        )

    def normalize(self, x):
        input_is_numpy = isinstance(x, np.ndarray)

        if input_is_numpy:
            x_t = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            x_t = x
        else:
            x_t = torch.tensor(x, dtype=torch.float32)

        mean = at_least_ndim(self.mean.to(x_t.device, x_t.dtype), x_t.ndim, 1)
        std = at_least_ndim(self.std.to(x_t.device, x_t.dtype), x_t.ndim, 1)
        y = (x_t - mean) / std

        if input_is_numpy:
            return y.cpu().numpy()
        return y

    def unnormalize(self, x):
        input_is_numpy = isinstance(x, np.ndarray)

        if input_is_numpy:
            x_t = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            x_t = x
        else:
            x_t = torch.tensor(x, dtype=torch.float32)

        mean = at_least_ndim(self.mean.to(x_t.device, x_t.dtype), x_t.ndim, 1)
        std = at_least_ndim(self.std.to(x_t.device, x_t.dtype), x_t.ndim, 1)
        y = x_t * std + mean

        if input_is_numpy:
            return y.cpu().numpy()
        return y