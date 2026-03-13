from collections import namedtuple
import numpy as np
import torch
import pdb

'''
    get_preprocess_fn 根据配置和环境名，返回一个“处理每条轨迹episode的函数”
    load_environment 加载指定名称的离线环境，返回环境对象
    sequence_dataset 从环境中读取离线轨迹，按episode产出轨迹
    DatasetNormalizer 对observations/actions做标准化，保持训练和采样一致；
    ReplayBuffer 把多条episode组织成固定形状的张量，并提供path_lengths等元数据。
    ReplayBuffer 此时还没有 horizon 的概念，所有 episode 被强行对齐到同一时间轴

'''
from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

'''
    定义样本结构：
        batch：轨迹片段和条件
        valuebatch：轨迹，条件，回报
'''
Batch = namedtuple('Batch', 'trajectories conditions') 
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SequenceDataset(torch.utils.data.Dataset):
    '''
    	1.	把 D4RL 的整条 episode 数据存成统一 shape 的大数组
	    2.	提前枚举所有能切出来的 horizon 子轨迹
	    3.	每次返回一段 [action, obs] 序列 + 起点状态作为条件
    '''

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.horizon = horizon # 采样轨迹样本的长度
        self.max_path_length = max_path_length # episode 在buffer里的最大存储长度
        self.use_padding = use_padding # 是否对不足horizon的短轨迹进行 padding

        # 从环境中读取离线轨迹数据，按 episode 组织成 ReplayBuffer
        # 它把 D4RL 里“每条 episode 长度不一、字段散”的数据，变成 fields 这种整齐结构
        itr = sequence_dataset(env, self.preprocess_fn) # itr 是一个 迭代器，每次 yield 一整条 episode
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode) # 把每条 episode 加入 ReplayBuffer
        fields.finalize()

        # 构建归一化器，并生成可采样的索引列表 --准备工作
        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        # 记录维度信息（给模型使用）
        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        # Dataset持有 ReplayBuffer 形式的 fields 数据
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths

        # 对 observations 和 actions 做归一化
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            二维数组做归一化器更方便统计均值和方差。
            把 [episodes, time, dim] 拉平为 [N, dim]，做归一化后再 reshape 回去。
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
            # 补充：后卖你训练直接用fields.normed_observations和fields.normed_actions，而不用原始值

    def make_indices(self, path_lengths, horizon):
        '''
            Dataset的“样本池”：（episode, start, start+horizon）列表
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            # 计算该 episode 可采样的最大起始位置
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon) # 如果不允许 padding，那么 start 不能超过 path_length - horizon。
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):

        # 根据索引，取出对应轨迹片段的观测和动作
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations) # 条件 = “这段轨迹的起点状态”。
        '''
        trajectories[t] = [action_t, observation_t]
        shape = [horizon, act_dim + obs_dim]
        '''
        trajectories = np.concatenate([actions, observations], axis=-1)
        '''
            Batch(
                trajectories = [H, act+obs],
                conditions   = {0: obs_0}
            )
        '''
        batch = Batch(trajectories, conditions)
        return batch

class GoalDataset(SequenceDataset):
    '''
        0 和 self.horizon - 1 → 表示这些条件对应轨迹中的哪个时间步
        那么下面的意思，就是给定轨迹的起点和终点作为条件。
        简言之：这是一个“已知起点和终点，中间轨迹由模型补全”的
    '''
    def get_conditions(self, observations):
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

class ValueDataset(SequenceDataset):
    '''
        核心：学“这个起点值不值钱”
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        # 计算一个随着时间步增长而不断减小的“权重系数”，用来体现“眼下的奖励比未来的奖励更值钱”这一逻辑
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:] # 从这段 horizon 的起点开始，一直算到 episode 结束的所有 reward。
        discounts = self.discounts[:len(rewards)] # 给每一步reward一个折扣系数
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32) # shape = [1,]，方便torch batch处理
        value_batch = ValueBatch(*batch, value)
        '''
            ValueBatch(
                trajectories = [H, act + obs],
                conditions   = {0: obs_0},
                values       = [1]
            )
        '''
        return value_batch
