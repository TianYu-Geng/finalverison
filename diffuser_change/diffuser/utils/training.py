"""
该文件实现训练期的通用封装：迭代数据集计算扩散损失、执行优化与 EMA 更新，并按频率保存 checkpoint 与渲染采样结果到 `results_folder`。
它影响的主要产物是 `state_<epoch>.pt` 权重文件与若干 `*.png` 可视化图片，后续由推理/规划脚本通过 `utils.load_diffusion()`/`Trainer.load()` 复用。
"""

import os
import copy
import numpy as np
import torch
import einops
import pdb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs

def cycle(dl):
    """
    做什么：将有限长度的 DataLoader 包装为无限迭代器，循环产生 batch。
    配置→实例：把“按 epoch 终止”的 loader 转换为“按训练步数消费”的 batch 流。
    谁调用：`Trainer.__init__` 用它创建 `self.dataloader` 与 `self.dataloader_vis`。
    """
    while True:
        for data in dl:
            yield data

class EMA():
    """
    做什么：维护模型参数的指数滑动平均（EMA）更新逻辑。
    配置→实例：把 `beta` 衰减系数配置具体化为 `ma_params <- beta*ma + (1-beta)*current` 的更新规则。
    谁调用：由 `Trainer` 在训练过程中按 `update_ema_every` 触发更新 `self.ema_model`。
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        """
        做什么：将 `current_model` 的参数融合进 `ma_model`，逐参数执行 EMA 更新。
        配置→实例：把一对模型实例映射为一次 EMA 同步操作。
        谁调用：由 `Trainer.step_ema()` 调用。
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """
        做什么：对单个张量执行 EMA 更新并返回新值。
        配置→实例：将 `beta` 配置应用到一次 `old/new` 融合计算。
        谁调用：由 `EMA.update_model_average` 调用。
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    """
    做什么：封装训练循环、梯度累积、EMA 更新、checkpoint 保存/加载，以及参考轨迹与采样结果的渲染落盘。
    配置→实例：把脚本中的训练超参（batch_size/lr/freq/results_folder 等）实例化为一个可运行的训练器对象。
    谁调用：由 `scripts/train.py` 通过 `trainer_config(...)` 创建并调用 `train(...)`；推理加载通常通过 `Trainer.load(...)` 间接触发。
    """
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        """
        做什么：用当前模型参数覆盖 EMA 模型参数（重置 EMA 状态）。
        配置→实例：将 “EMA 初始化/重置” 从配置状态落到具体的 state_dict 拷贝操作。
        谁调用：在 `Trainer.__init__` 与 `step_ema` 的早期阶段（未到 `step_start_ema`）调用。
        """
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        """
        做什么：按训练步数决定是否更新 EMA；在预热期内仅做重置，之后执行一次 EMA 参数融合。
        配置→实例：把 `step_start_ema/update_ema_every` 的配置落实为训练中的 EMA 调度规则。
        谁调用：由 `train` 在每个 step 根据 `update_ema_every` 触发调用。
        """
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        """
        做什么：执行训练主循环：取 batch、计算 `self.model.loss`、反向传播、优化器更新、EMA/日志/保存/渲染。
        配置→实例：把“训练步数 + 频率超参”组合成可重复运行的训练过程，并将产物写入 `results_folder`。
        谁调用：由 `scripts/train.py` 的入口逻辑调用（通常在构造完模型、数据集、renderer 之后）。
        """

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples(n_samples=self.n_samples)

            self.step += 1

    def save(self, epoch):
        """
        做什么：保存训练 step、当前模型与 EMA 模型的权重到 `state_<epoch>.pt`（可选同步到远端 bucket）。
        配置→实例：将 `results_folder/save_freq/save_parallel/bucket` 的配置落实为一次 checkpoint 落盘与同步行为。
        谁调用：由 `train` 按 `save_freq` 周期调用；推理/恢复训练会读取其产物。
        """
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        """
        做什么：从 `state_<epoch>.pt` 读取并恢复训练 step、模型权重与 EMA 权重。
        配置→实例：将磁盘上的 checkpoint 文件实例化为可继续训练/采样的内存状态。
        谁调用：由 `utils.load_diffusion()` 或脚本中的恢复逻辑调用；规划/推理在加载训练权重时也会用到。
        """
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        """
        做什么：从数据集中抽取一批真实轨迹观测并渲染为参考图片（用于对照采样质量）。
        配置→实例：将 `renderer` 与数据集的反归一化规则应用到 batch，生成并保存 `_sample-reference.png`。
        谁调用：由 `train` 在 step 0 时调用一次（若 `sample_freq` 非 0）。
        """

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        """
        做什么：以可视化用 batch 的条件为输入，调用 EMA 模型采样轨迹并渲染为图片序列。
        配置→实例：将 `n_samples` 重复采样、反归一化与 `renderer.composite` 落盘组合为一次采样可视化过程。
        谁调用：由 `train` 按 `sample_freq` 周期调用。
        """
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(conditions)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)
