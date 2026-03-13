'''
把 ReplayBuffer 里的多字段离线数据，
在不受 padding 干扰的前提下，统一映射到一个对模型友好的数值空间（通常是 [-1, 1]），
并且提供严格可逆的 unnormalize 接口，供规划/可视化/执行使用。

normalize 是给模型用的，unnormalize 是给“世界”用的。（世界->归一化->世界）
unnormalize 是把数据从模型空间拉回真实物理空间的那一步。
'''
import numpy as np
import scipy.interpolate as interpolate
import pdb

POINTMASS_KEYS = ['observations', 'actions', 'next_observations', 'deltas']

#-----------------------------------------------------------------------------#
#------------------------------ 多子段归一化器 ---------------------------------#
#-----------------------------------------------------------------------------#

class DatasetNormalizer:
    '''
        为数据集中的每个字段构建单独的归一化器实例，并提供统一的归一化/反归一化接口。
    '''
    def __init__(self, dataset, normalizer, path_lengths=None):
        dataset = flatten(dataset, path_lengths) # 归一化统计不是按 episode 统计，而是把所有有效时间步摊平成一个大样本集。

        self.observation_dim = dataset['observations'].shape[1]
        self.action_dim = dataset['actions'].shape[1]

        if type(normalizer) == str:
            # 配置文件的字符可以在这里被 解析 成类对象
            normalizer = eval(normalizer)

        # 对 dataset 里的每一个字段（observations / actions / rewards / terminals …）单独构建一个 normalizer 实例
        self.normalizers = {}
        for key, val in dataset.items():
            try:
                self.normalizers[key] = normalizer(val)
            except:
                print(f'[ utils/normalization ] Skipping {key} | {normalizer}')
            # key: normalizer(val)
            # for key, val in dataset.items()

    def __repr__(self):
        string = ''
        for key, normalizer in self.normalizers.items():
            string += f'{key}: {normalizer}]\n'
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

def flatten(dataset, path_lengths):
    '''
        { key: [E x T x D] } → { key: [(sum path_lengths) x D] }
        把 dataset 里的每个字段都摊平成一个大样本集，方便归一化统计。
        在强化学习中，虽然我们通常设置一个最大步数（max_path_length），
        但实际上机器人可能在第 50 步就到达终点，而另一个可能跑满了 200 步。
        这段代码就是把这些真实长度的数据“去伪存真”后粘在一起。
    '''
    flattened = {}
    for key, xs in dataset.items():
        assert len(xs) == len(path_lengths)
        flattened[key] = np.concatenate([
            x[:length] 
            for x, length in zip(xs, path_lengths)
        ], axis=0)
    return flattened

#-----------------------------------------------------------------------------#
#------------------------------- @TODO: remove? ------------------------------#
#-----------------------------------------------------------------------------#
# 专门为 PointMass 数据集设计的归一化器，基本不用不到，可以删除
class PointMassDatasetNormalizer(DatasetNormalizer):

    def __init__(self, preprocess_fns, dataset, normalizer, keys=POINTMASS_KEYS):

        reshaped = {}
        for key, val in dataset.items():
            dim = val.shape[-1]
            reshaped[key] = val.reshape(-1, dim)

        self.observation_dim = reshaped['observations'].shape[1]
        self.action_dim = reshaped['actions'].shape[1]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizers = {
            key: normalizer(reshaped[key])
            for key in keys
        }

#-----------------------------------------------------------------------------#
#------------------------------ 单字段归一化器 ---------------------------------#
#-----------------------------------------------------------------------------#

class Normalizer:
    '''
        所有normalizer类的基类，定义接口。
    '''

    def __init__(self, X):
        ## 接收一个数据数组 X，计算其最小值和最大值，用于归一化操作
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class DebugNormalizer(Normalizer):
    '''
        调试时完全关闭归一化，确认数值问题是否来自 normalizer。
    '''

    def normalize(self, x, *args, **kwargs):
        return x

    def unnormalize(self, x, *args, **kwargs):
        return x


class GaussianNormalizer(Normalizer):
    '''
        标准化为均值为0，标准差为1的高斯分布
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = self.X.mean(axis=0)
        self.stds = self.X.std(axis=0)
        self.z = 1

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    '''
            f'''means: {np.round(self.means, 2)}\n    '''
            f'''stds: {np.round(self.z * self.stds, 2)}\n'''
        )

    def normalize(self, x):
        return (x - self.means) / self.stds

    def unnormalize(self, x): # 反变换
        return x * self.stds + self.means


class LimitsNormalizer(Normalizer):
    '''
        这是 Diffuser / Mujoco / Maze2D 默认最常用的 normalizer。
    '''

    def normalize(self, x):
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins

class SafeLimitsNormalizer(LimitsNormalizer):
    '''
        某些维度在整个数据集中是常数（比如 z=0 的平面导航）
        会导致 (max - min) = 0，归一化除零。
    '''

    def __init__(self, *args, eps=1, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.mins)):
            if self.mins[i] == self.maxs[i]:
                print(f'''
                    [ utils/normalization ] Constant data in dimension {i} | '''
                    f'''max = min = {self.maxs[i]}'''
                )
                self.mins -= eps
                self.maxs += eps

#-----------------------------------------------------------------------------#
#------------------------------- CDF normalizer ------------------------------#
#-----------------------------------------------------------------------------#

class CDFNormalizer(Normalizer):
    '''
        基于经验累积分布函数（empirical CDF）的归一化器。
        把数据的分布映射到均匀分布，再线性映射到 [-1, 1] 区间。
        反变换时，先把 [-1, 1] 映射回均匀分布，再通过经验逆 CDF 映射回原始分布。
        适用于任意分布的数据，尤其是多峰分布或长尾分布。
    '''

    def __init__(self, X):
        super().__init__(atleast_2d(X))
        self.dim = self.X.shape[1]
        self.cdfs = [
            CDFNormalizer1d(self.X[:, i])
            for i in range(self.dim)
        ]

    def __repr__(self):
        return f'[ CDFNormalizer ] dim: {self.mins.size}\n' + '    |    '.join(
            f'{i:3d}: {cdf}' for i, cdf in enumerate(self.cdfs)
        )

    def wrap(self, fn_name, x):
        shape = x.shape
        ## reshape to 2d
        x = x.reshape(-1, self.dim)
        out = np.zeros_like(x)
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[:, i] = fn(x[:, i])
        return out.reshape(shape)

    def normalize(self, x):
        return self.wrap('normalize', x)

    def unnormalize(self, x):
        return self.wrap('unnormalize', x)

class CDFNormalizer1d:
    '''
        一维经验 CDF 归一化器。
    '''

    def __init__(self, X):
        assert X.ndim == 1
        self.X = X.astype(np.float32)
        quantiles, cumprob = empirical_cdf(self.X)
        self.fn = interpolate.interp1d(quantiles, cumprob)
        self.inv = interpolate.interp1d(cumprob, quantiles)

        self.xmin, self.xmax = quantiles.min(), quantiles.max()
        self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def __repr__(self):
        return (
            f'[{np.round(self.xmin, 2):.4f}, {np.round(self.xmax, 2):.4f}'
        )

    def normalize(self, x):
        x = np.clip(x, self.xmin, self.xmax)
        ## [ 0, 1 ]
        y = self.fn(x)
        ## [ -1, 1 ]
        y = 2 * y - 1
        return y

    def unnormalize(self, x, eps=1e-4):
        '''
            X : [ -1, 1 ]
        '''
        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(
                f'''[ dataset/normalization ] Warning: out of range in unnormalize: '''
                f'''[{x.min()}, {x.max()}] | '''
                f'''x : [{self.xmin}, {self.xmax}] | '''
                f'''y: [{self.ymin}, {self.ymax}]'''
            )

        x = np.clip(x, self.ymin, self.ymax)

        y = self.inv(x)
        return y

def empirical_cdf(sample):

    # 标准的经验累积分布函数计算
    quantiles, counts = np.unique(sample, return_counts=True)
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def atleast_2d(x):
    if x.ndim < 2:
        x = x[:,None]
    return x

