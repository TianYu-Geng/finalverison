import math
from copy import deepcopy
from typing import Optional, Union, Callable

import torch
import torch.nn as nn

from network import BaseNNDiffusion, IdentityCondition
from prior_struct.guidance import weak_center_pull_step
from util import (
    SUPPORTED_NOISE_SCHEDULES,
    SUPPORTED_DISCRETIZATIONS,
    SUPPORTED_SAMPLING_STEP_SCHEDULE,
    at_least_ndim,
)


def epstheta_to_xtheta(x, alpha, sigma, eps_theta):
    """
    x_theta = (x - sigma * eps_theta) / alpha
    """
    return (x - sigma * eps_theta) / alpha


def xtheta_to_epstheta(x, alpha, sigma, x_theta):
    """
    eps_theta = (x - alpha * x_theta) / sigma
    """
    return (x - alpha * x_theta) / sigma


class ScoreDiffusion(nn.Module):
    def __init__(self, nn_diffusion, nn_condition):
        super().__init__()
        self.diffusion = nn_diffusion
        self.condition = nn_condition

    def forward(self, xt, t, condition, use_condition, train_condition):
        if use_condition and condition is not None:
            condition = self.condition(condition, train=train_condition)
        output = self.diffusion(xt, t, condition, use_condition, train_condition)
        return output


class ContinuousDiffusionSDE:
    def __init__(
        self,
        nn_diffusion: BaseNNDiffusion,
        nn_condition=None,
        fix_mask=None,
        loss_weight=None,
        classifier=None,
        grad_clip_norm=None,
        ema_rate: float = 0.995,
        optim_params=None,
        epsilon: float = 1e-3,
        noise_schedule="cosine",
        noise_schedule_params=None,
        x_max=None,
        x_min=None,
        predict_noise: bool = True,
        planner_diffusion_gradient_steps=1e6,
    ):
        self.grad_clip_norm = grad_clip_norm
        self.ema_rate = ema_rate

        if nn_condition is None:
            nn_condition = IdentityCondition()

        self.model = ScoreDiffusion(nn_diffusion, nn_condition)
        self.classifier = classifier

        self.fix_mask = fix_mask[None, ...] if fix_mask is not None else 0.0
        self.loss_weight = loss_weight[None, ...] if loss_weight is not None else 1.0

        self.predict_noise = predict_noise
        self.epsilon = epsilon
        self.x_max = x_max
        self.x_min = x_min

        if noise_schedule == "cosine":
            self.t_diffusion = [epsilon, 0.9946]
        else:
            self.t_diffusion = [epsilon, 1.0]

        if isinstance(noise_schedule, str):
            if noise_schedule in SUPPORTED_NOISE_SCHEDULES:
                self.noise_schedule_funcs = SUPPORTED_NOISE_SCHEDULES[noise_schedule]
                self.noise_schedule_params = noise_schedule_params
            else:
                raise ValueError(f"Noise schedule {noise_schedule} is not supported.")
        elif isinstance(noise_schedule, dict):
            self.noise_schedule_funcs = noise_schedule
            self.noise_schedule_params = noise_schedule_params
        else:
            raise ValueError("noise_schedule must be a callable or a string")

    def _to_device_tensor(self, x, ref):
        if isinstance(x, torch.Tensor):
            return x.to(device=ref.device, dtype=ref.dtype)
        return torch.tensor(x, device=ref.device, dtype=ref.dtype)

    def add_noise(self, x0, t=None, eps=None):
        device = x0.device
        dtype = x0.dtype

        if t is None:
            t = (
                torch.rand((x0.shape[0],), device=device, dtype=dtype)
                * (self.t_diffusion[1] - self.t_diffusion[0])
                + self.t_diffusion[0]
            )

        if eps is None:
            eps = torch.randn_like(x0)

        alpha, sigma = self.noise_schedule_funcs["forward"](
            t, **(self.noise_schedule_params or {})
        )
        alpha = at_least_ndim(alpha, x0.ndim)
        sigma = at_least_ndim(sigma, x0.ndim)

        xt = alpha * x0 + sigma * eps

        fix_mask = self._to_device_tensor(self.fix_mask, x0) if isinstance(self.fix_mask, torch.Tensor) else self.fix_mask
        xt = (1.0 - fix_mask) * xt + fix_mask * x0

        return xt, t, eps

    def sample(
        self,
        observation,
        model,
        solver: str = "ddpm",
        planner_horizon: int = 4,
        n_samples: int = 1,
        sample_steps: int = 5,
        sample_step_schedule: Union[str, Callable] = "uniform_continuous",
        temperature: float = 1.0,
        diffusion_x_sampling_steps: int = 0,
        return_intermediates: bool = False,
    ):
        device = observation.device
        dtype = observation.dtype

        prior = torch.zeros(
            (n_samples, planner_horizon - 1, observation.shape[-1]),
            device=device,
            dtype=dtype,
        )
        prior = torch.cat((observation.unsqueeze(1), prior), dim=1)

        xt = torch.randn_like(prior) * temperature
        t_diffusion = self.t_diffusion

        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE:
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    t_diffusion, sample_steps, device=device, dtype=dtype
                )
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")

        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {})
        )

        fix_mask = self._to_device_tensor(self.fix_mask, prior) if isinstance(self.fix_mask, torch.Tensor) else self.fix_mask
        xt = xt * (1.0 - fix_mask) + prior * fix_mask

        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        intermediates = [xt.detach().clone()] if return_intermediates else None

        for i in reversed(loop_steps):
            t = torch.full((n_samples,), sample_step_schedule[i].item(), device=device, dtype=dtype)

            pred = model.diffusion(xt, t, condition=None, use_condition=False, train_condition=False)
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)

            if solver == "ddim":
                xt = (
                    alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i])
                    + sigmas[i - 1] * eps_theta
                )
            else:
                raise NotImplementedError()

            xt = xt * (1.0 - fix_mask) + prior * fix_mask
            if return_intermediates:
                intermediates.append(xt.detach().clone())

        if self.clip_pred():
            xt = torch.clamp(
                xt,
                min=self.x_min if self.x_min is not None else -torch.inf,
                max=self.x_max if self.x_max is not None else torch.inf,
            )
            if return_intermediates:
                intermediates[-1] = xt.detach().clone()

        if return_intermediates:
            return xt, intermediates
        return xt

    def sample_prior(
        self,
        observation,
        prior,
        solver: str = "ddpm",
        planner_horizon: int = 4,
        n_samples: int = 1,
        sample_steps: int = 5,
        sample_step_schedule: Union[str, Callable] = "uniform_continuous",
        temperature: float = 1.0,
        diffusion_x_sampling_steps: int = 0,
        return_intermediates: bool = False,
        prior_init_noise_scale: float = 0.003,
        guidance=None,
        lomap_projector=None,
        lomap_context=None,
    ):
        device = observation.device
        dtype = observation.dtype
        batch_size = observation.shape[0]

        prior = torch.cat((observation.unsqueeze(1), prior), dim=1)

        # prior is already the structured initialization; only add small local noise
        # to non-start position dimensions for visualization diversity.
        noise = torch.zeros_like(prior)
        noise[:, 1:, :2] = (
            prior_init_noise_scale
            * temperature
            * torch.randn_like(prior[:, 1:, :2])
        )
        xt = prior.clone() + noise

        t_diffusion = self.t_diffusion

        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE:
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    t_diffusion, sample_steps, device=device, dtype=dtype
                )
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")

        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {})
        )

        fix_mask = self._to_device_tensor(self.fix_mask, prior) if isinstance(self.fix_mask, torch.Tensor) else self.fix_mask
        xt = xt * (1.0 - fix_mask) + prior * fix_mask

        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        intermediates = [xt.detach().clone()] if return_intermediates else None

        for i in reversed(loop_steps):
            t = torch.full((batch_size,), sample_step_schedule[i].item(), device=device, dtype=dtype)
            xt = xt * (1.0 - fix_mask) + prior * fix_mask

            corridor_ratio = float(guidance.get("corridor_start_ratio", 0.5)) if guidance is not None else 0.5

            pred = self.model.diffusion(xt, t, condition=None, use_condition=False, train_condition=False)
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)
            x_theta = pred if not self.predict_noise else epstheta_to_xtheta(xt, alphas[i], sigmas[i], pred)
            x_theta = x_theta * (1.0 - fix_mask) + prior * fix_mask

            if lomap_projector is not None:
                x_theta = lomap_projector.project(
                    x_theta,
                    step_index=i,
                    total_steps=sample_steps,
                    context=lomap_context,
                )
                x_theta = x_theta * (1.0 - fix_mask) + prior * fix_mask

            if guidance is not None and (float(i) / float(max(sample_steps, 1))) <= corridor_ratio:
                x_theta = weak_center_pull_step(
                    x_theta,
                    guidance,
                    step_ratio=float(i) / float(max(sample_steps, 1)),
                )
                x_theta = x_theta * (1.0 - fix_mask) + prior * fix_mask

            x_theta = torch.nan_to_num(x_theta, nan=0.0, posinf=1e3, neginf=-1e3)

            eps_theta = xtheta_to_epstheta(xt, alphas[i], sigmas[i], x_theta)

            if solver == "ddim":
                xt = (
                    alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i])
                    + sigmas[i - 1] * eps_theta
                )
            else:
                raise NotImplementedError()

            xt = xt * (1.0 - fix_mask) + prior * fix_mask
            xt = torch.nan_to_num(xt, nan=0.0, posinf=1e3, neginf=-1e3)

            if return_intermediates:
                intermediates.append(xt.detach().clone())

        if self.clip_pred():
            xt = torch.clamp(
                xt,
                min=self.x_min if self.x_min is not None else -torch.inf,
                max=self.x_max if self.x_max is not None else torch.inf,
            )
            if return_intermediates:
                intermediates[-1] = xt.detach().clone()

        if return_intermediates:
            return xt, intermediates
        return xt

    def sample_prior01(
        self,
        observation,
        prior,
        solver: str = "ddpm",
        planner_horizon: int = 4,
        n_samples: int = 1,
        sample_steps: int = 5,
        sample_step_schedule: Union[str, Callable] = "uniform_continuous",
        temperature: float = 1.0,
        diffusion_x_sampling_steps: int = 0,
        return_intermediates: bool = False,
    ):
        device = observation.device
        dtype = observation.dtype
        batch_size = observation.shape[0]

        prior = torch.cat((observation.unsqueeze(1), prior), dim=1)
        xt = prior.clone()

        t_diffusion = self.t_diffusion

        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE:
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    t_diffusion, sample_steps, device=device, dtype=dtype
                )
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")

        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {})
        )

        fix_mask = self._to_device_tensor(self.fix_mask, prior) if isinstance(self.fix_mask, torch.Tensor) else self.fix_mask
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        intermediates = [xt.detach().clone()] if return_intermediates else None

        for i in reversed(loop_steps):
            t = torch.full((batch_size,), sample_step_schedule[i].item(), device=device, dtype=dtype)
            pred = self.model.diffusion(xt, t, condition=None, use_condition=False, train_condition=False)
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)

            if solver == "ddim":
                xt = (
                    alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i])
                    + sigmas[i - 1] * eps_theta
                )
            else:
                raise NotImplementedError()

            xt = xt * (1.0 - fix_mask) + prior * fix_mask
            if return_intermediates:
                intermediates.append(xt.detach().clone())

        if self.clip_pred():
            xt = torch.clamp(
                xt,
                min=self.x_min if self.x_min is not None else -torch.inf,
                max=self.x_max if self.x_max is not None else torch.inf,
            )
            if return_intermediates:
                intermediates[-1] = xt.detach().clone()

        if return_intermediates:
            return xt, intermediates
        return xt

    def sample_prior_train(
        self,
        observation,
        prior,
        config,
    ):
        solver = config.planner_solver
        sample_steps = config.planner_sampling_steps_train
        sample_step_schedule = "uniform_continuous"
        diffusion_x_sampling_steps = 0

        device = observation.device
        dtype = observation.dtype
        batch_size = observation.shape[0]

        prior = torch.cat((observation.unsqueeze(1), prior), dim=1)
        xt = prior.clone()

        t_diffusion = self.t_diffusion

        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE:
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    t_diffusion, sample_steps, device=device, dtype=dtype
                )
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")

        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {})
        )

        fix_mask = self._to_device_tensor(self.fix_mask, prior) if isinstance(self.fix_mask, torch.Tensor) else self.fix_mask
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))

        for i in reversed(loop_steps):
            t = torch.full((batch_size,), sample_step_schedule[i].item(), device=device, dtype=dtype)
            pred = self.model.diffusion(xt, t, condition=None, use_condition=False, train_condition=False)
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)

            if solver == "ddim":
                xt = (
                    alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i])
                    + sigmas[i - 1] * eps_theta
                )
            else:
                raise NotImplementedError()

            xt = xt * (1.0 - fix_mask) + prior * fix_mask

        if self.clip_pred():
            xt = torch.clamp(
                xt,
                min=self.x_min if self.x_min is not None else -torch.inf,
                max=self.x_max if self.x_max is not None else torch.inf,
            )

        return xt

    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    def clip_prediction(self, pred, xt, alpha, sigma):
        if self.predict_noise:
            if self.clip_pred():
                upper_bound = (xt - alpha * self.x_min) / sigma if self.x_min is not None else None
                lower_bound = (xt - alpha * self.x_max) / sigma if self.x_max is not None else None

                if lower_bound is not None and upper_bound is not None:
                    pred = torch.maximum(torch.minimum(pred, upper_bound), lower_bound)
                elif upper_bound is not None:
                    pred = torch.minimum(pred, upper_bound)
                elif lower_bound is not None:
                    pred = torch.maximum(pred, lower_bound)
        else:
            if self.clip_pred():
                pred = torch.clamp(
                    pred,
                    min=self.x_min if self.x_min is not None else -torch.inf,
                    max=self.x_max if self.x_max is not None else torch.inf,
                )

        return pred


class DiscreteDiffusionSDE:
    def __init__(
        self,
        nn_diffusion: BaseNNDiffusion,
        nn_condition=None,
        fix_mask=None,
        loss_weight=None,
        classifier=None,
        grad_clip_norm: Optional[float] = None,
        ema_rate: float = 0.995,
        optim_params: Optional[dict] = None,
        epsilon: float = 1e-3,
        diffusion_steps: int = 1000,
        discretization: Union[str, Callable] = "uniform",
        noise_schedule="cosine",
        noise_schedule_params: Optional[dict] = None,
        x_max=None,
        x_min=None,
        predict_noise: bool = True,
    ):
        self.grad_clip_norm = grad_clip_norm
        self.ema_rate = ema_rate

        if nn_condition is None:
            nn_condition = IdentityCondition()

        self.model = ScoreDiffusion(nn_diffusion, nn_condition)
        self.classifier = classifier

        self.fix_mask = fix_mask[None, ...] if fix_mask is not None else 0.0
        self.loss_weight = loss_weight[None, ...] if loss_weight is not None else 1.0

        self.predict_noise = predict_noise
        self.epsilon = epsilon
        self.x_max = x_max
        self.x_min = x_min
        self.diffusion_steps = diffusion_steps

        if 1.0 / diffusion_steps < epsilon:
            raise ValueError("epsilon is too large for the number of diffusion steps")

        if isinstance(discretization, str):
            if discretization in SUPPORTED_DISCRETIZATIONS:
                self.t_diffusion = SUPPORTED_DISCRETIZATIONS[discretization](diffusion_steps, epsilon)
            else:
                self.t_diffusion = SUPPORTED_DISCRETIZATIONS["uniform"](diffusion_steps, epsilon)
        elif callable(discretization):
            self.t_diffusion = discretization(diffusion_steps, epsilon)
        else:
            raise ValueError("discretization must be a callable or a string")

        if not isinstance(self.t_diffusion, torch.Tensor):
            self.t_diffusion = torch.tensor(self.t_diffusion, dtype=torch.float32)

        if isinstance(noise_schedule, str):
            if noise_schedule in SUPPORTED_NOISE_SCHEDULES:
                self.alpha, self.sigma = SUPPORTED_NOISE_SCHEDULES[noise_schedule]["forward"](
                    self.t_diffusion, **(noise_schedule_params or {})
                )
            else:
                raise ValueError(f"Noise schedule {noise_schedule} is not supported.")
        elif isinstance(noise_schedule, dict):
            self.alpha, self.sigma = noise_schedule["forward"](
                self.t_diffusion, **(noise_schedule_params or {})
            )
        else:
            raise ValueError("noise_schedule must be a callable or a string")

        self.logSNR = torch.log(self.alpha / self.sigma)

    def _ensure_schedule_device(self, device, dtype):
        self.t_diffusion = self.t_diffusion.to(device=device, dtype=torch.float32)
        self.alpha = self.alpha.to(device=device, dtype=dtype)
        self.sigma = self.sigma.to(device=device, dtype=dtype)
        self.logSNR = self.logSNR.to(device=device, dtype=dtype)

    def _to_device_tensor(self, x, ref):
        if isinstance(x, torch.Tensor):
            return x.to(device=ref.device, dtype=ref.dtype)
        return torch.tensor(x, device=ref.device, dtype=ref.dtype)

    def add_noise(self, x0, t=None, eps=None):
        device = x0.device
        dtype = x0.dtype
        self._ensure_schedule_device(device, dtype)

        if t is None:
            t = torch.randint(
                low=0,
                high=self.diffusion_steps,
                size=(x0.shape[0],),
                device=device,
                dtype=torch.int64,
            )

        if eps is None:
            eps = torch.randn_like(x0)

        alpha = at_least_ndim(self.alpha[t], x0.ndim)
        sigma = at_least_ndim(self.sigma[t], x0.ndim)

        xt = alpha * x0 + sigma * eps

        fix_mask = self._to_device_tensor(self.fix_mask, x0) if isinstance(self.fix_mask, torch.Tensor) else self.fix_mask
        xt = (1.0 - fix_mask) * xt + fix_mask * x0

        return xt, t, eps

    def sample(
        self,
        prior,
        model,
        solver: str = "ddpm",
        n_samples: int = 1,
        sample_steps: int = 5,
        sample_step_schedule: Union[str, Callable] = "uniform",
        temperature: float = 1.0,
        condition_cfg=None,
        mask_cfg=None,
        diffusion_x_sampling_steps: int = 0,
    ):
        device = prior.device
        dtype = prior.dtype
        self._ensure_schedule_device(device, dtype)

        xt = torch.randn_like(prior) * temperature

        fix_mask = self._to_device_tensor(self.fix_mask, prior) if isinstance(self.fix_mask, torch.Tensor) else self.fix_mask
        xt = xt * (1.0 - fix_mask) + prior * fix_mask

        condition_vec_cfg = (
            model.condition(condition=condition_cfg, train=False, mask=mask_cfg)
            if condition_cfg is not None else None
        )

        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE:
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    self.diffusion_steps, sample_steps, device=device
                )
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")

        sample_step_schedule = sample_step_schedule.to(device=device, dtype=torch.long)

        alphas = self.alpha[sample_step_schedule]
        sigmas = self.sigma[sample_step_schedule]
        logSNRs = torch.log(alphas / sigmas)
        hs = torch.cat((torch.tensor([0.0], device=device, dtype=dtype), logSNRs[:-1] - logSNRs[1:]), dim=0)
        stds_1 = sigmas[:-1] / sigmas[1:] * torch.sqrt(1 - (alphas[1:] / alphas[:-1]) ** 2)
        stds = torch.cat((torch.tensor([0.0], device=device, dtype=dtype), stds_1), dim=0)

        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))

        for i in reversed(loop_steps):
            t = torch.full((n_samples,), sample_step_schedule[i].item(), device=device, dtype=torch.long)
            pred = model.diffusion(xt, t, condition_vec_cfg, use_condition=condition_vec_cfg is not None, train_condition=False)
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)

            if solver == "ddpm":
                xt = (
                    (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta)
                    + torch.sqrt(sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8) * eps_theta
                )
                if i > 1:
                    xt = xt + stds[i] * torch.randn_like(xt)
            else:
                raise NotImplementedError()

            xt = xt * (1.0 - fix_mask) + prior * fix_mask

        if self.clip_pred():
            xt = torch.clamp(
                xt,
                min=self.x_min if self.x_min is not None else -torch.inf,
                max=self.x_max if self.x_max is not None else torch.inf,
            )

        return xt

    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    def clip_prediction(self, pred, xt, alpha, sigma):
        if self.predict_noise:
            if self.clip_pred():
                upper_bound = (xt - alpha * self.x_min) / sigma if self.x_min is not None else None
                lower_bound = (xt - alpha * self.x_max) / sigma if self.x_max is not None else None

                if lower_bound is not None and upper_bound is not None:
                    pred = torch.maximum(torch.minimum(pred, upper_bound), lower_bound)
                elif upper_bound is not None:
                    pred = torch.minimum(pred, upper_bound)
                elif lower_bound is not None:
                    pred = torch.maximum(pred, lower_bound)
        else:
            if self.clip_pred():
                pred = torch.clamp(
                    pred,
                    min=self.x_min if self.x_min is not None else -torch.inf,
                    max=self.x_max if self.x_max is not None else torch.inf,
                )

        return pred

    def guided_sampling(
        self,
        xt,
        t,
        alpha,
        sigma,
        model,
        condition_cfg=None,
        w_cfg: float = 0.0,
        condition_cg=None,
        w_cg: float = 0.0,
        requires_grad: bool = False,
    ):
        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad
        )
        return pred

    def classifier_guidance(
        self,
        xt,
        t,
        alpha,
        sigma,
        model,
        condition=None,
        w: float = 1.0,
        pred=None,
    ):
        if pred is None:
            pred = model["diffusion"](xt, t, None)
        if self.classifier is None or w == 0.0:
            return pred, None
        else:
            log_p, grad = self.classifier.gradients(xt.clone(), t, condition)
            if self.predict_noise:
                pred = pred - w * sigma * grad
            else:
                pred = pred + w * ((sigma ** 2) / alpha) * grad

        return pred, log_p

    def classifier_free_guidance(
        self,
        xt,
        t,
        model,
        condition=None,
        w: float = 1.0,
        pred=None,
        pred_uncond=None,
        requires_grad: bool = False,
    ):
        pred = model["diffusion"](xt, t, condition)
        pred_uncond = 0.0
        return pred
