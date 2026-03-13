"""
这个文件的目标是把“每条 episode 的字典数据”存进一个三维数组结构里：
fields[key] 的形状最终会是 [E, T, D]（E=episode 数，T=max_path_length，D=字段维度）。
额外还会有 path_lengths: [E]，记录每条 episode 的真实长度，供后续切 horizon 用。
"""

import numpy as np

def atleast_2d(x):
    """
    把输入数组扩展到至少二维（在末尾轴上补维），统一为形如 `(T, D)` 的表示。
    """
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class ReplayBuffer:
    """
    为离线轨迹提供按 `(episode, step, dim)` 存储的字段容器，支持逐条写入、截断与最终裁剪。
    """

    def __init__(self, max_n_episodes, max_path_length, termination_penalty):
        self._dict = {
            'path_lengths': np.zeros(max_n_episodes, dtype=int),
        } # _dict是字段容器，这里先放一个 path_lengths 字段
        self._count = 0
        ## 下面是buffer第一维和第二维的容量配置
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty # 若 episode 不是 timeout 而是 terminal 提前结束，可在最后一步 reward 上加一个惩罚（通常是负数）。


    def __repr__(self):
        ## 打印出 buffer 里所有字段的形状信息，用于调试确定维度是否正确
        return '[ datasets/buffer ] Fields:\n' + '\n'.join(
            f'    {key}: {val.shape}'
            for key, val in self.items()
        ) 

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        ## 已写入的 episode 数量
        return self._count

    @property
    def n_steps(self):
        ## 所有 episode 的总步数
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        """
        将 `_dict` 中的字段同步为同名对象属性，支持 `buffer.observations` 形式的访问。
        """
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        ## 返回除 path_lengths 外的字段 (key, array) 列表，主要用于 __repr__ 打印 shape
        return {k: v for k, v in self._dict.items()
                if k != 'path_lengths'}.items()

    def _allocate(self, key, array):
        """
        为某个新字段按 `(max_n_episodes, max_path_length, dim)` 分配 float32 存储。
        """
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def add_path(self, path):
        """
        将一条 episode 轨迹（字典）写入 buffer，包括字段写入、长度记录以及可选的终止惩罚修正。
        由 `diffuser/datasets/sequence.py` 在遍历离线轨迹迭代器时逐条调用。
        """
        path_length = len(path['observations'])
        assert path_length <= self.max_path_length

        ## 如果是新字段，就先分配存储空间
        self._add_keys(path)

        ## 对每个字段，写入该 episode 的数据（observations, rewards, terminals 等）
        for key in self.keys:
            array = atleast_2d(path[key])
            if key not in self._dict: self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array

        ## 惩罚提前终止的 episode（非 timeout）
        if path['terminals'].any() and self.termination_penalty is not None:
            assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
            self._dict['rewards'][self._count, path_length - 1] += self.termination_penalty

        ## 记录 episode 长度
        self._dict['path_lengths'][self._count] = path_length

        ## 更新已写入的 episode 计数
        self._count += 1

    def truncate_path(self, path_ind, step):
        """
        将指定 episode 的有效长度裁剪为不超过 `step` 的值（只更新 `path_lengths`）。
        """
        old = self._dict['path_lengths'][path_ind]
        new = min(step, old)
        self._dict['path_lengths'][path_ind] = new

    def finalize(self):
        """
        按已写入的 episode 数 `_count` 裁剪所有字段到紧凑大小，并刷新属性访问。
        由 `diffuser/datasets/sequence.py` 在写入完所有轨迹后调用，之后进入归一化与索引构建流程。
        """
        for key in self.keys + ['path_lengths']:
            self._dict[key] = self._dict[key][:self._count]
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')
