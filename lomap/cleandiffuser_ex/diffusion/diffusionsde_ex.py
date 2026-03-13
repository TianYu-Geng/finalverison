from typing import Optional, Union, Callable

import numpy as np
import torch
from cleandiffuser.utils import SUPPORTED_SAMPLING_STEP_SCHEDULE
from cleandiffuser.diffusion.diffusionsde import (
    at_least_ndim, epstheta_to_xtheta, xtheta_to_epstheta,
    DiscreteDiffusionSDE, ContinuousDiffusionSDE, SUPPORTED_SOLVERS)

from cleandiffuser_ex.local_manifold import LocalManifold


class DiscreteDiffusionSDEEX(DiscreteDiffusionSDE):
    """
    扩展版 DiscreteDiffusionSDE：在标准 diffusion 采样循环上加了两类“推理期纠偏”：
    1) LoMAP（Local Manifold Approximation & Projection）：FAISS kNN + PCA 局部子空间投影
    2) Low-Density Guidance（LDG）：用 forward-diffuse consistency 构造 density proxy 并对 xt 施加梯度
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # LocalManifold 内部通常维护 PCA 的子空间基 U，并提供 compute_pca / project_points
        self.local_manifold = LocalManifold(device=self.device)

    @torch.no_grad()
    def get_manifold_samples(self, xt, t, n_manifold_samples, prior, faiss_wrapper):
        """
        目的：为“局部流形估计(PCA)”构造邻域样本（论文 Algorithm 2 的 kNN 邻居集合）。

        输入：
          xt: 当前噪声层的 trajectory sample，shape (B, T, D)
          t : 当前扩散时间步（long tensor），shape (B,)
          prior: 已知条件（起点/终点等）shape (B, T, D)
          faiss_wrapper: 存储训练集轨迹 x0(flatten) 的 FAISS 索引

        输出：
          manifold_samples: shape (B, K, T, D)
            - 若 t==0：直接用 xt≈x0 做 kNN
            - 若 t>0：先用 Tweedie 得到 x0|t，再 kNN 找到训练集的 x0 邻居，然后 forward-diffuse 到当前噪声层
        """
        batch_size, *dims = xt.shape  # dims 通常是 [T, D]

        # t==0：已接近 x0，无需 Tweedie / forward-diffuse 邻居
        if t[0].item() == 0:
            x0_t = xt
            x0_t_flat = x0_t.reshape(batch_size, -1)  # (B, T*D)

            # 在训练集 x0(flat) 中做 kNN
            _, idxs = faiss_wrapper.search(x0_t_flat.cpu().numpy(), n_manifold_samples)

            # 取回原始邻居向量（仍是 x0 空间），并 reshape 成 (B, K, T, D)
            neigh_flat = faiss_wrapper.get_original_vectors(idxs)  # numpy (B,K,T*D)
            neigh = torch.from_numpy(neigh_flat).to(self.device).reshape(batch_size, n_manifold_samples, *dims)
            return neigh

        # t>0：LoMAP 标准路径：Tweedie -> kNN(x0) -> forward-diffuse(neighbors) -> return xt_neighbors
        alpha_t = at_least_ndim(self.alpha[t], xt.dim())
        sigma_t = at_least_ndim(self.sigma[t], xt.dim())

        # 1) Tweedie：从 xt 估计 x0|t（用 EMA diffusion 模型更稳定）
        pred = self.model_ema["diffusion"](xt, t, None)
        x0_t = pred if not self.predict_noise else epstheta_to_xtheta(xt, alpha_t, sigma_t, pred)

        # 2) 强制满足已知条件（起点/终点固定）
        x0_t = x0_t * (1. - self.fix_mask) + prior * self.fix_mask

        # 3) 在 x0 空间做 kNN：用 x0_t(flat) 去检索训练集的 x0 邻居
        x0_t_flat = x0_t.reshape(batch_size, -1)
        _, idxs = faiss_wrapper.search(x0_t_flat.cpu().numpy(), n_manifold_samples)

        # 4) 取回训练集邻居（x0），reshape 成 (B,K,T,D)
        neigh_flat = faiss_wrapper.get_original_vectors(idxs)  # numpy (B,K,T*D)
        x0_neighbors = torch.from_numpy(neigh_flat).to(self.device).reshape(batch_size, n_manifold_samples, *dims)

        # 5) forward-diffuse 邻居到同一噪声层（为了 PCA 估计的是“当前噪声层的局部邻域”）
        alpha_k = at_least_ndim(self.alpha[t], x0_neighbors.dim())
        sigma_k = at_least_ndim(self.sigma[t], x0_neighbors.dim())
        xt_neighbors = x0_neighbors * alpha_k + sigma_k * torch.randn_like(x0_neighbors)

        # 6) 邻居也要满足已知条件（与当前 sample 同约束）
        repeat_pattern = [1, n_manifold_samples] + [1] * len(dims)
        prior_rep = prior.unsqueeze(1).repeat(*repeat_pattern)  # (B,K,T,D)
        xt_neighbors = xt_neighbors * (1. - self.fix_mask) + prior_rep * self.fix_mask
        return xt_neighbors


    def compute_rg(self, xt, t, model, prior, forward_level=0.8, n_mc_samples=1):
        """
        Low-Density Guidance 的核心：构造一个“密度代理”rg(xt)。

        思想：如果 xt 对应的数据密度高、模型一致性好，则：
          x0 = denoise(xt) 之后 forward-diffuse 到某个噪声层再 denoise，应当回到接近的 x0_hat。
        用 ||x0 - x0_hat|| 作为 “low-density / inconsistency” 指标，越大说明越可能 OOD。

        返回：
          rg: shape (B,) 每条样本一个标量
        """
        if self.predict_noise:
            raise NotImplementedError  # 这份实现只在预测 x0 的分支支持

        # 当前层去噪得到 x0
        x0 = model["diffusion"](xt, t, None)
        x0 = x0 * (1. - self.fix_mask) + prior * self.fix_mask

        # Monte Carlo 多次采样估计 rg
        rglb_samples = torch.zeros((xt.shape[0], n_mc_samples), device=self.device)
        for i in range(n_mc_samples):
            # forward 到更高噪声层 t_hat = forward_level * diffusion_steps
            diffusion_steps = int(forward_level * self.diffusion_steps)
            fwd_alpha, fwd_sigma = self.alpha[diffusion_steps], self.sigma[diffusion_steps]
            xt_hat = x0 * fwd_alpha + fwd_sigma * torch.randn_like(x0)
            xt_hat = xt_hat * (1. - self.fix_mask) + prior * self.fix_mask

            # 再 denoise 回来得到 x0_hat
            t_hat = torch.full((xt_hat.shape[0],), diffusion_steps, dtype=torch.long, device=self.device)
            x0_hat = model["diffusion"](xt_hat, t_hat, None)
            x0_hat = x0_hat * (1. - self.fix_mask) + prior * self.fix_mask

            # inconsistency：||x0 - stopgrad(x0_hat)||
            diff = x0 - x0_hat.detach()
            rglb_samples[:, i] = diff.reshape(diff.shape[0], -1).norm(p=2.0, dim=1)

        return rglb_samples.mean(dim=-1)
        
    def low_density_guidance(self, xt, t, alpha, sigma, model, w, forward_level, n_mc_samples, prior, pred):
        """
        将 rg(xt) 的梯度注入到 pred（x0 或 eps）的预测里。
        w=0 关闭；w>0 开启。
        """
        if w == 0.0:
            return pred

        # 对 xt 求 rg 的梯度（requires_grad）
        with torch.enable_grad():
            xt = xt.detach().requires_grad_(True)
            rg = self.compute_rg(xt, t, model, prior, forward_level=forward_level, n_mc_samples=n_mc_samples)
            grad = torch.autograd.grad(rg.sum(), xt)[0]

        # 把 grad 映射到 pred 的更新（注意：predict_noise 分支与 predict_x0 分支更新形式不同）
        if self.predict_noise:
            pred = pred - w * sigma * grad
        else:
            pred = pred + w * ((sigma ** 2) / alpha) * grad
        return pred

    def guided_sampling(
            self, xt, t, alpha, sigma,
            model,
            condition_cfg=None, w_cfg: float = 0.0,
            condition_cg=None, w_cg: float = 0.0,
            requires_grad: bool = False,
            # ----------- Low Density Guidance Params ------------ #
            w_ldg: float = 0.0,
            rg_forward_level: float = 0.8,
            n_mc_samples: int = 1,
            prior: torch.Tensor = None,
        ):
        """
        单步“预测器”封装：先 CFG，再 classifier guidance，再 low-density guidance。
        返回 pred（x0/eps）以及 classifier logp（若启用）。
        """

        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad)

        pred, logp = self.classifier_guidance(
            xt, t, alpha, sigma, model, condition_cg, w_cg, pred)

        pred = self.low_density_guidance(
            xt, t, alpha, sigma, model, w_ldg, rg_forward_level, 
            n_mc_samples, prior, pred)

        return pred, logp

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,  # 已知且要“硬固定”的轨迹部分（例如起点/终点 state），shape: (B,T,D)
            # ----------------- sampling ----------------- #
            solver: str = "ddpm",  # 选择反向采样更新公式（ddpm/ddim/ode/sde dpmsolver 等）
            n_samples: int = 1,  # 采样条数（通常等于 batch_size；上层可能传 num_candidates*num_envs）
            sample_steps: int = 5,  # 实际反向采样的离散步数（可小于 diffusion_steps）
            sample_step_schedule: Union[str, Callable] = "uniform",  # 从 diffusion_steps 里挑哪些 t 进行采样的调度策略
            use_ema: bool = True,  # 是否使用 EMA 参数的模型做推理（通常更稳）
            temperature: float = 1.0,  # 初始噪声尺度（>1 更随机，多样性更高）
            # ------------------ guidance ------------------ #
            condition_cfg=None,  # classifier-free guidance 的条件输入（如文本/goal 等，具体看 model["condition"]）
            mask_cfg=None,  # CFG 条件的 mask（哪些维度/时间步有条件）
            w_cfg: float = 0.0,  # CFG guidance 强度（0 表示不用 CFG）
            condition_cg=None,  # classifier guidance 的条件（这里常直接传 None 或某种条件向量）
            w_cg: float = 0.0,  # classifier guidance 强度（0 表示不用 classifier guidance）
            # ----------- Diffusion-X sampling ----------
            diffusion_x_sampling_steps: int = 0,  # 额外的“Diffusion-X”步数（把 loop_steps 前面塞一段重复步）
            # ----------- Warm-Starting -----------
            warm_start_reference: Optional[torch.Tensor] = None,  # warm-start 的参考轨迹（若给定，从它加噪开始采样）
            warm_start_forward_level: float = 0.3,  # warm-start：把参考轨迹 forward-noise 到哪个噪声强度（占比）
            # ---------Manifold Preserved Guidance -------- # 
            faiss_wrapper=None,  # 训练集 x0(flat) 的 FAISS 索引（用于 kNN 检索邻居）
            proj_range=[],  # 在采样的哪些阶段启用投影（例如 [0.5,0.8]）
            proj_mask: Optional[torch.Tensor] = None,  # 每条样本是否启用投影的 mask（True=投影，False=不投影）
            n_manifold_samples: int = 5,  # kNN 的 k（用于 PCA 的邻居数）
            tau: float = 0.95,  # LocalManifold 内部 PCA/子空间更新的平滑系数（越大越“保守/平滑”）
            # ----------- Low-Density Guidance -----------
            w_ldg: float = 0.0,  # low-density guidance 强度（0 关闭）
            rg_forward_level: float = 0.8,  # 计算 rg 时 forward-noise 到的噪声层比例
            n_mc_samples: int = 1,  # rg 的 Monte Carlo 采样次数
            # ------------------ others ------------------ #
            requires_grad: bool = False,  # 是否保留梯度（通常推理 False；做分析/优化时才 True）
            preserve_history: bool = False,  # 是否保存每一步的 xt 到 log["sample_history"]
            **kwargs,
    ):
        assert solver in SUPPORTED_SOLVERS, f"Solver {solver} is not supported."  # solver 必须在支持列表中

        # ===================== Initialization =====================
        log = {
            # 若 preserve_history=True，则预分配数组保存每一步的 xt（注意 shape 用 prior.shape）
            "sample_history": np.empty((sample_steps + 1, *prior.shape)) if preserve_history else None,
        }

        model = self.model if not use_ema else self.model_ema  # 选择使用原模型参数还是 EMA 参数

        prior = prior.to(self.device)  # prior 放到设备上（cpu/cuda）
        if isinstance(warm_start_reference, torch.Tensor):  # 若启用 warm-start
            diffusion_steps = int(warm_start_forward_level * self.diffusion_steps)  # 选一个 forward-noise 的噪声层
            fwd_alpha, fwd_sigma = self.alpha[diffusion_steps], self.sigma[diffusion_steps]  # 对应该层的 alpha/sigma
            # 从参考轨迹 x0 出发 forward diffusion 到 x_t（加噪），作为采样初值
            xt = warm_start_reference * fwd_alpha + fwd_sigma * torch.randn_like(warm_start_reference)
        else:
            diffusion_steps = self.diffusion_steps  # 不 warm-start：从最大扩散步数开始
            xt = torch.randn_like(prior) * temperature  # 初始 xt ~ N(0, I)*temperature（整段轨迹的高斯噪声）
        # 注入 hard condition：mask=1 的位置强制等于 prior（例如 t=0、t=T-1 的 state）
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()  # 保存初值（注意这里索引写法看起来有点怪，但按作者逻辑存）

        # 在是否需要梯度的上下文里准备条件向量（CFG 用）
        with torch.set_grad_enabled(requires_grad):
            # model["condition"] 将 condition_cfg + mask_cfg 编成 condition_vec_cfg（用于 classifier-free guidance）
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg  # classifier guidance 的条件（这里直接透传）

        # ===================== Sampling Schedule ====================
        if isinstance(sample_step_schedule, str):  # schedule 是字符串：查表得到具体策略
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                # 用策略函数把总 diffusion_steps 映射成长度为 sample_steps+1 的离散时间索引序列
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    diffusion_steps, sample_steps
                )
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):  # schedule 是函数：直接调用生成索引序列
            sample_step_schedule = sample_step_schedule(diffusion_steps, sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        alphas = self.alpha[sample_step_schedule]  # 采样用到的 alpha 序列（按 schedule 取子序列）
        sigmas = self.sigma[sample_step_schedule]  # 采样用到的 sigma 序列（按 schedule 取子序列）
        logSNRs = torch.log(alphas / sigmas)  # logSNR 序列（DPM-solver 等需要）
        hs = torch.zeros_like(logSNRs)  # 存储相邻 logSNR 的差
        hs[1:] = logSNRs[:-1] - logSNRs[1:]  # hs[0] 不正确但不会被用到（作者注释）
        stds = torch.zeros((sample_steps + 1,), device=self.device)  # ddpm 采样噪声项的 std
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()
        # 上面 stds 推导来自 DDPM 的一步更新中“额外噪声”的方差重标定

        buffer = []  # DPM-Solver++ 2M 需要缓存上一步的 x_theta

        # ===================== Denoising Loop ========================
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        # loop_steps：如果 diffusion_x_sampling_steps>0，会在最前面插入若干个“i=1”的重复步（某种 heuristic）
        for i in reversed(loop_steps):  # 反向遍历：从大噪声层走到小噪声层

            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.long, device=self.device)
            # 为 batch 中每条样本构造当前扩散时间步 t（所有样本同一 t）

            # guided sampling：得到 pred（x0 或 eps 的预测），以及 classifier logp（如果用到）
            pred, logp = self.guided_sampling(
                xt, t, alphas[i], sigmas[i],
                model, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg, requires_grad,
                w_ldg, rg_forward_level, n_mc_samples, prior
            )

            # clip prediction：对 pred 做裁剪/约束（避免数值爆炸，具体规则在 clip_prediction 内）
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            # noise & data prediction：同时得到 eps_theta 与 x_theta（取决于网络输出形式）
            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)
            # 若网络直接预测噪声：eps_theta=pred；否则用 xt 与 x_theta 反推 eps_theta

            x_theta = pred if not self.predict_noise else epstheta_to_xtheta(xt, alphas[i], sigmas[i], pred)
            # 若网络直接预测 x0：x_theta=pred；否则用 xt 与 eps_theta 反推 x_theta

            # one-step update：根据 solver 把 xt 更新到下一层 xt_{t-1}
            if solver == "ddpm":
                xt = (
                    (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta) +
                    (sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta
                )
                if i > 1:  # ddpm 在中间步骤会再加随机噪声（最后一步通常不加）
                    xt += (stds[i] * torch.randn_like(xt))

            elif solver == "ddim":
                xt = (alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[i - 1] * eps_theta)
                # DDIM 是确定性更新（不额外加噪），相当于 ODE-like 的一条轨迹

            elif solver == "ode_dpmsolver_1":
                xt = (alphas[i - 1] / alphas[i]) * xt - sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta
                # DPM-Solver 的一阶 ODE 形式（用 logSNR 步长 hs）

            elif solver == "ode_dpmsolver++_1":
                xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta
                # DPM-Solver++ 一阶形式（用 x_theta）

            elif solver == "ode_dpmsolver++_2M":
                buffer.append(x_theta)  # 缓存 x_theta 以形成二阶多步更新
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]  # 步长比
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]  # 二阶外推的 D
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * D
                else:
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "sde_dpmsolver_1":
                xt = (
                    (alphas[i - 1] / alphas[i]) * xt -
                    2 * sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta +
                    sigmas[i - 1] * torch.expm1(2 * hs[i]).sqrt() * torch.randn_like(xt)
                )
                # SDE 版本：会显式加随机项（与 ODE 对应）

            elif solver == "sde_dpmsolver++_1":
                xt = (
                    (sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                    alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                    sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt)
                )

            elif solver == "sde_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = (
                        (sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                        alphas[i - 1] * torch.expm1(-2 * hs[i]) * D +
                        sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt)
                    )
                else:
                    xt = (
                        (sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                        alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                        sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt)
                    )

            # ================= LoMAP / Manifold Projection =================
            if faiss_wrapper and proj_range:  # 若给了索引且配置了投影区间
                if (sample_step_schedule[i] > 0) \
                    and (sample_step_schedule[i] >= int(proj_range[0] * (sample_steps - 1))) \
                    and (sample_step_schedule[i] <= int(proj_range[1] * (sample_steps - 1))):
                    # 只在指定区间内启用投影（作者把 proj_range 当成采样步区间比例）
                    assert proj_range[0] < proj_range[1]

                    # 在做投影前先注入 hard condition（保证起点/终点是对的）
                    xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

                    xt_orig = xt.clone()  # 备份：若 proj_mask=False 时回退

                    # 生成用于 PCA 的局部邻域样本（核心：Tweedie -> kNN -> forward-diffuse neighbors）
                    manifold_samples = self.get_manifold_samples(
                        xt,
                        t - 1,  # 注意这里用 t-1：作者希望用“更新后目标层”对应的邻域（实现细节）
                        n_manifold_samples, prior, faiss_wrapper
                    )

                    # 基于邻域样本估计 PCA 子空间（LocalManifold 内部可能维护 U，并用 tau 平滑更新）
                    self.local_manifold.compute_pca(manifold_samples, tau)

                    # 把当前 xt 投影到估计的局部子空间（典型是 U U^T xt）
                    xt_projected = self.local_manifold.project_points(xt)

                    if proj_mask is not None:  # 可选：只对部分样本启用投影（常用于 OOD start-goal 保护）
                        proj_mask = proj_mask.to(xt.device)
                        xt = torch.where(
                            proj_mask.view(-1, *([1] * (xt.dim() - 1))),  # 扩展形状以广播到 (B,T,D)
                            xt_projected,  # True：用投影后的
                            xt_orig        # False：用原始的
                        )
                    else:
                        xt = xt_projected  # 全部投影

            # 每一步结束都强制注入 hard condition（避免 solver / projection 破坏起终点）
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()  # 记录每步结果（作者的索引方式）

        # ================= Post-processing =================
        if self.classifier is not None:  # 若存在 classifier（CumRewClassifier）
            with torch.no_grad():
                t = torch.zeros((n_samples,), dtype=torch.long, device=self.device)  # 用 t=0 评估最终轨迹得分
                logp = self.classifier.logp(xt, t, condition_vec_cg)  # 用 EMA classifier 输出作为 logp/score
            log["log_p"] = logp  # 记录到日志

        if self.clip_pred:  # 若启用了全局裁剪（x_min/x_max）
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log  # 返回最终采样轨迹与日志