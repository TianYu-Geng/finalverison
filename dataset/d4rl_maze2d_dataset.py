import os
from typing import Tuple, Dict, Optional
import gym
import numpy as np
from tqdm import tqdm
import d4rl
import collections
import wrappers
import torch

from util import at_least_ndim, GaussianNormalizer


Batch = collections.namedtuple(
    'Batch',
    ['obs', 'act', 'rew', 'val']
)


class D4RLMaze2DSeqDataset:
    def __init__(
            self,
            dataset: Dict[str, np.ndarray],
            horizon: int = 1,
            max_path_length: int = 800,
            discount: float = 0.99,
            continous_reward_at_done: bool = False,
            reward_tune: str = "iql",
            center_mapping: bool = True,
            learn_policy: bool = False,
            stride: int = 1,
            device: Optional[torch.device] = None,
    ):
        self.max_path_length = max_path_length
        self.learn_policy = learn_policy
        self.stride = stride
        self.horizon = horizon
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        observations = dataset["observations"].astype(np.float32)
        actions = dataset["actions"].astype(np.float32)
        rewards = dataset["rewards"].astype(np.float32)
        timeouts = dataset["timeouts"].astype(np.float32)

        self.normalizer = GaussianNormalizer(observations)
        normed_observations = self.normalizer.normalize(observations)

        self.o_dim = observations.shape[-1]
        self.a_dim = actions.shape[-1]

        self.indices = []
        self.seq_obs, self.seq_act, self.seq_rew = [], [], []

        path_idx = 0
        self.paths = []

        next_end = [-1] * (timeouts.shape[0] + 1)
        next_start = [-1] * (timeouts.shape[0] + 1)

        for index in reversed(range(timeouts.shape[0])):
            if rewards[index] == 1.0:
                next_end[index] = index
                next_start[index] = next_start[index + 1]
            else:
                next_end[index] = next_end[index + 1]
                next_start[index] = index

        path_start = next_start[0]
        path_end = next_end[path_start]

        padded_len = max_path_length + (horizon - 1) * stride

        if self.learn_policy:
            for path_start in range(0, timeouts.shape[0], max_path_length):
                path_end = min(path_start + max_path_length - 1, timeouts.shape[0] - 1)
                path_length = path_end - path_start + 1

                _seq_obs = np.zeros((padded_len, self.o_dim), dtype=np.float32)
                _seq_act = np.zeros((padded_len, self.a_dim), dtype=np.float32)
                _seq_rew = np.zeros((padded_len, 1), dtype=np.float32)

                _seq_obs[:path_length] = normed_observations[path_start:path_end + 1]
                _seq_act[:path_length] = actions[path_start:path_end + 1]
                _seq_rew[:path_length] = rewards[path_start:path_end + 1][:, None]

                _seq_obs[path_length:] = normed_observations[path_end]
                _seq_act[path_length:] = 0.0
                _seq_rew[path_length:] = 1.0 if continous_reward_at_done else 0.0

                self.seq_obs.append(_seq_obs)
                self.seq_act.append(_seq_act)
                self.seq_rew.append(_seq_rew)

                max_start = path_length - 1
                self.indices += [
                    (path_idx, start, start + (horizon - 1) * stride + 1)
                    for start in range(max_start + 1)
                ]
                self.paths.append((path_start, path_end))
                path_idx += 1

        else:
            while path_end != -1:
                path_start = max(path_start, path_end - max_path_length + 1)
                path_length = path_end - path_start + 1
                assert path_length >= 2

                _seq_obs = np.zeros((padded_len, self.o_dim), dtype=np.float32)
                _seq_act = np.zeros((padded_len, self.a_dim), dtype=np.float32)
                _seq_rew = np.zeros((padded_len, 1), dtype=np.float32)

                _seq_obs[:path_length] = normed_observations[path_start:path_end + 1]
                _seq_act[:path_length] = actions[path_start:path_end + 1]
                _seq_rew[:path_length] = rewards[path_start:path_end + 1][:, None]

                _seq_obs[path_length:] = normed_observations[path_end]
                _seq_act[path_length:] = 0.0
                _seq_rew[path_length:] = 1.0 if continous_reward_at_done else 0.0

                self.seq_obs.append(_seq_obs)
                self.seq_act.append(_seq_act)
                self.seq_rew.append(_seq_rew)

                max_start = path_length - 1
                self.indices += [
                    (path_idx, start, start + (horizon - 1) * stride + 1)
                    for start in range(max_start + 1)
                ]
                self.paths.append((path_start, path_end))
                path_idx += 1

                path_start = next_start[path_end]
                path_end = next_end[path_start]

        self.seq_obs = np.array(self.seq_obs, dtype=np.float32)
        self.seq_act = np.array(self.seq_act, dtype=np.float32)
        self.seq_rew = np.array(self.seq_rew, dtype=np.float32)

        if reward_tune == "iql":
            self.seq_rew += -1.0
        elif reward_tune == "none":
            pass
        else:
            raise ValueError(f"reward_tune: {reward_tune} is not supported.")

        self.seq_val = np.copy(self.seq_rew)

        print(self.seq_obs.shape)
        for i in reversed(range(max_path_length - 1)):
            self.seq_val[:, i] = self.seq_rew[:, i] + discount * self.seq_val[:, i + 1]

        print(f"max discounted return: {self.seq_val.max()}")
        print(f"min discounted return: {self.seq_val.min()}")

        val_min = self.seq_val.min()
        val_max = self.seq_val.max()
        denom = val_max - val_min
        if denom < 1e-8:
            self.seq_val = np.zeros_like(self.seq_val, dtype=np.float32)
        else:
            self.seq_val = (self.seq_val - val_min) / denom

        if center_mapping:
            self.seq_val = self.seq_val * 2.0 - 1.0

        print(f"max normed discounted return: {self.seq_val.max()}")
        print(f"min normed discounted return: {self.seq_val.min()}")

        preprocessed_data = {
            'obs': [],
            'act': [],
            'rew': [],
            'val': [],
        }

        for path_idx, start, end in self.indices:
            if self.learn_policy:
                horizon_state = self.seq_obs[path_idx, start:end:self.stride].copy()
                horizon_state[:, :2] -= horizon_state[0, :2]
            else:
                horizon_state = self.seq_obs[path_idx, start:end:self.stride]

            preprocessed_data['obs'].append(horizon_state.astype(np.float32))
            preprocessed_data['act'].append(
                self.seq_act[path_idx, start:end:self.stride].astype(np.float32)
            )
            preprocessed_data['rew'].append(
                self.seq_rew[path_idx, start:end:self.stride].astype(np.float32)
            )
            preprocessed_data['val'].append(
                np.array(self.seq_val[path_idx, start], dtype=np.float32)
            )

        preprocessed_data['obs'] = torch.from_numpy(np.stack(preprocessed_data['obs'], axis=0)).float()
        preprocessed_data['act'] = torch.from_numpy(np.stack(preprocessed_data['act'], axis=0)).float()
        preprocessed_data['rew'] = torch.from_numpy(np.stack(preprocessed_data['rew'], axis=0)).float()
        preprocessed_data['val'] = torch.from_numpy(np.stack(preprocessed_data['val'], axis=0)).float()

        self.data = Batch(
            preprocessed_data['obs'],
            preprocessed_data['act'],
            preprocessed_data['rew'],
            preprocessed_data['val']
        )

    def to(self, device):
        device = torch.device(device)
        self.device = device
        self.data = Batch(
            self.data.obs.to(device),
            self.data.act.to(device),
            self.data.rew.to(device),
            self.data.val.to(device),
        )
        return self

    def sample(self, batch_size, device: Optional[torch.device] = None, generator: Optional[torch.Generator] = None):
        sample_device = torch.device(device) if device is not None else self.device
        return _sample(self.data, batch_size, sample_device, generator)


def get_batch_item(tree_batch, idx):
    return Batch(
        obs=tree_batch.obs[idx],
        act=tree_batch.act[idx],
        rew=tree_batch.rew[idx],
        val=tree_batch.val[idx],
    )


def _sample(data, batch_size, device, generator: Optional[torch.Generator] = None):
    if data.obs.device != device:
        obs = data.obs.to(device)
        act = data.act.to(device)
        rew = data.rew.to(device)
        val = data.val.to(device)
        data = Batch(obs=obs, act=act, rew=rew, val=val)

    size = data.obs.shape[0]
    idx = torch.randint(
        low=0,
        high=size,
        size=(batch_size,),
        device=device,
        generator=generator
    )

    return Batch(
        obs=data.obs[idx],
        act=data.act[idx],
        rew=data.rew[idx],
        val=data.val[idx],
    )