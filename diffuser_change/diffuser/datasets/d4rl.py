"""
该文件提供 D4RL 离线环境与数据集的轻量封装，并且修复一下 D4RL 数据集的已知问题。
供 `diffuser.datasets` 的数据管线与训练/规划脚本迭代使用。
"""

import os
import collections
import numpy as np
import gym
import pdb

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    '''
        临时屏蔽 stdout / stderr。
    '''
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## 避免 d4rl 在 import 和 gym.make 时打印大量 warning。
    import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    ## 直接传一个gym.Env实例进来就直接返回
    if type(name) != str:
        return name
    ## 通过 gym.make(name) 来加载环境
    with suppress_output():
        wrapped_env = gym.make(name) # wrapped_env 是 gym 包装后的环境（TimeLimit 等 wrapper 还在）
    env = wrapped_env.unwrapped # 获取最原始的环境实例
    env.max_episode_steps = wrapped_env._max_episode_steps #  用于判断 episode 是否自然结束（尤其是没 timeout 字段的数据集）
    env.name = name # 后面用字符串判断是否是 maze2d / antmaze
    return env

def get_dataset(env):
    
    dataset = env.get_dataset()

    if 'antmaze' in str(env).lower():
        ## AntMaze 的 D4RL 数据 在 episode 切分 / terminal / reward 上是有 bug 的。
        dataset = antmaze_fix_timeouts(dataset) # 修正 terminal / timeout 字段
        dataset = antmaze_scale_rewards(dataset) # 重新缩放 reward 范围
        get_max_delta(dataset) # 计算最大位移（供后续规划使用）

    return dataset

def sequence_dataset(env, preprocess_fn):
    
    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset) # 对数据集做预处理

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list) # 临时存一条 episode 的所有transition

    # 判断数据集中是否有 timeout 字段
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i]) # 当前 transition 是否自然结束
        '''
            两种 episode 结束条件：
                •	terminal（撞了 / 成功）
                •	timeout（自然时间上限）
        '''
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        ## 当前transition的所有字段，被加入data_
        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    """
    Maze2D 的 episode 没有 next_observations 字段，但很多代码假设 (s, a, r, s') 是显式存在的。
    """
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1] # 所有字段都要减去最后一个 transition，字段长度为T-1
    episode['next_observations'] = next_observations
    return episode
