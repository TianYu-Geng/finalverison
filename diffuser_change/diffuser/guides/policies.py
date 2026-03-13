"""
该文件提供规划/推理阶段的策略封装：将“按时间索引的条件观测字典”格式化为扩散模型所需张量，并将采样得到的轨迹拆分为动作与观测。
它输出的 `Policy` 通常在 `scripts/plan_*.py` 中被创建并调用一次（开环采样），其结果用于与环境交互或用于渲染/评估。
"""

from collections import namedtuple
# import numpy as np
import torch
import einops
import pdb

import diffuser.utils as utils
# from diffusion.datasets.preprocessing import get_policy_preprocess_fn

Trajectories = namedtuple('Trajectories', 'actions observations')
# GuidedTrajectories = namedtuple('GuidedTrajectories', 'actions observations value')

class Policy:
    """
    做什么：封装扩散模型的条件采样调用与（反）归一化，将采样结果拆分为 `Trajectories(actions, observations)`。
    配置→实例：把“扩散模型 + normalizer”的配置组合实例化为一个可调用策略 `policy(conditions, batch_size=...)`。
    谁调用：由规划脚本（如 `scripts/plan_maze2d.py`）创建；推理/规划流程在需要生成动作序列时调用。
    """

    def __init__(self, diffusion_model, normalizer):
        """
        做什么：保存扩散模型与归一化器引用，并缓存动作维度。
        配置→实例：把上层构造出的 `GaussianDiffusion` 与 `DatasetNormalizer` 连接成策略实例。
        谁调用：由规划脚本在开始规划前调用构造函数。
        """
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

    @property
    def device(self):
        """
        做什么：返回扩散模型参数所在设备。
        配置→实例：将模型的运行设备信息暴露为策略实例的属性。
        谁调用：可被外部脚本/调试代码读取；本文件内部主要使用固定的 `cuda:0` 转移。
        """
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        """
        做什么：对条件观测做归一化、转为 torch 张量并按 `batch_size` 复制扩展。
        配置→实例：把“条件字典（t->obs）”转为扩散模型采样可直接使用的张量字典。
        谁调用：由 `__call__` 在执行采样前调用。
        """
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1):
        """
        做什么：基于条件运行扩散模型反向扩散采样，并返回首个动作与整条采样轨迹（动作/观测）。
        配置→实例：将 `conditions/batch_size` 的调用参数实例化为一次 `diffusion_model(conditions)` 采样，并做反归一化输出。
        谁调用：规划脚本在 t=0 处调用一次（开环）或在需要重采样时调用。
        """


        conditions = self._format_conditions(conditions, batch_size)

        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        sample = self.diffusion_model(conditions)
        sample = utils.to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = Trajectories(actions, observations)
        return action, trajectories
        # else:
        #     return action
