'''
    实验启动器：把命令行参数和配置文件结合起来，形成一个最终的args对象。
    主要功能包括：
    1. 读取配置文件并覆盖默认参数
    2. 处理额外的命令行参数覆盖
    3. 设置随机种子以保证实验的可复现性
    4. 生成实验名称和保存路径
    5. 保存参数配置和git信息以便追踪实验设置
'''
import os
import importlib
import random
import numpy as np
import torch
from tap import Tap # 用类型注解定义参数并解析命令行
import pdb

from .serialization import mkdir
from .git_utils import (
    get_git_rev, # 读当前仓库head的git commit hash，为了实验溯源
    save_git_diff, # 把未提交改动的diff保存到文件
)

def set_seed(seed):
    '''
        设置随机种子以保证实验的可复现性。
        分别固定 Python random、NumPy、PyTorch（CPU）、PyTorch（所有 GPU）的随机数种子。
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def watch(args_to_watch):
    '''
        根据需要监控的参数生成实验名称的格式化函数。
        输入 args_to_watch，通常是形如 [(key, label), ...] 的列表，例如 [('horizon','H'), ('n_diffusion_steps','T')]。
    '''
    def _fn(args):
        exp_name = []
        # 遍历要监控的参数；如果 args 没有该字段就跳过；有就取值。
        for key, label in args_to_watch:
            if not hasattr(args, key):
                continue
            val = getattr(args, key)
            # 如果某个参数是 dict，把它展开成 k-v_k2-v2 这样的片段，拼成单字符串。
            if type(val) == dict:
                val = '_'.join(f'{k}-{v}' for k, v in val.items())
            # 把 label 和 val 拼接成片段，加入 exp_name 列表（把每个 key 转成如 H256、T1000 这种片段，然后用 _ 拼成总实验名）
            exp_name.append(f'{label}{val}')
        # 字符串清洗
        exp_name = '_'.join(exp_name)
        exp_name = exp_name.replace('/_', '/')
        exp_name = exp_name.replace('(', '').replace(')', '')
        exp_name = exp_name.replace(', ', '-')
        return exp_name
    return _fn

def lazy_fstring(template, args):
    '''
        在 config 文件里写好路径模板（比如 "{work_dir}/{timestamp}/result.txt"），
        而程序在运行过程中会产生 timestamp 等变量。通过这个函数，程序能自动把当前运行的变量填入模板，生成唯一的文件夹来保存实验数据。
    '''
    return eval(f"f'{template}'")

class Parser(Tap):
    '''
        自定义参数解析器，继承自 tap.Tap，用于处理复杂的实验配置。
        例如；learning_rate: float = 0.01
        这行代码定义了一个名为 learning_rate 的浮点数参数，默认值为 0.01，并让 Tap 自动将其转化为命令行参数 --learning_rate。
    '''

    def save(self):
        '''
            将解析后的参数保存为 json 文件，生成args.json文件，在savepath路径下。
        '''
        fullpath = os.path.join(self.savepath, 'args.json')
        print(f'[ utils/setup ] Saved args to {fullpath}')
        super().save(fullpath, skip_unpicklable=True)

    def parse_args(self, experiment=None):
        '''
            解析命令行参数并执行一系列设置操作：读取配置、处理额外参数、设置种子、生成实验名称等。
        '''
        args = super().parse_args(known_only=True)
        ## 如果没有 config 脚本，则跳过后续设置
        if not hasattr(args, 'config'): 
            return args
        args = self.read_config(args, experiment) # 从配置文件中读取参数
        self.add_extras(args) # 使用命令行参数覆盖配置文件中的参数
        self.eval_fstrings(args) # 处理配置中带有 'f:' 前缀的延迟评估 f-string
        self.set_seed(args) # 设置随机种子
        self.get_commit(args) # 获取当前 git 提交信息
        self.generate_exp_name(args) # 生成实验名称 logs/.../diffusion/...
        self.mkdir(args) # 创建保存路径
        self.save_diff(args) # 保存未提交的 git diff
        return args

    def read_config(self, args, experiment):
        '''
            从配置文件中读取参数。
        '''
        dataset = args.dataset.replace('-', '_') # 	把 maze2d-large-v1 变成 maze2d_large_v1，因为 Python 变量名不能有 -。
        print(f'[ utils/setup ] Reading config: {args.config}:{dataset}')
        module = importlib.import_module(args.config) # 动态 import，比如 config.maze2d。
        params = getattr(module, 'base')[experiment] # 读取模块里的 base 字典，并按 experiment 索引出对应配置块：例如 base['diffusion']。

        '''
        如果 config 模块里还定义了一个与 dataset 同名的 dict（如 maze2d_large_v1 = {...}），
        则用它对 base 里的参数做覆盖。这样实现“同一套脚本，不同数据集不同默认超参”。
        '''
        if hasattr(module, dataset) and experiment in getattr(module, dataset):
            print(f'[ utils/setup ] Using overrides | config: {args.config} | dataset: {dataset}')
            overrides = getattr(module, dataset)[experiment]
            params.update(overrides)
        else:
            print(f'[ utils/setup ] Not using overrides | config: {args.config} | dataset: {dataset}')

        '''
        把配置参数写入 args，并维护一份 _dict 记录“最终哪些 key 是从 config/overrides 来的”。
        后面 eval_fstrings 会遍历 _dict 处理 f-string。
        '''
        self._dict = {}
        for key, val in params.items():
            setattr(args, key, val)
            self._dict[key] = val

        return args

    def add_extras(self, args):
        '''
            未显示定义在配置文件中的额外命令行参数覆盖。
        '''
        extras = args.extra_args
        if not len(extras):
            return

        print(f'[ utils/setup ] Found extras: {extras}')

        # 额外参数必须成对出现：--key value。并且 key 必须在 config 里存在，否则拒绝（防止写错参数名悄悄无效）。
        assert len(extras) % 2 == 0, f'Found odd number ({len(extras)}) of extras: {extras}'
        for i in range(0, len(extras), 2):
            key = extras[i].replace('--', '')
            val = extras[i+1]
            assert hasattr(args, key), f'[ utils/setup ] {key} not found in config: {args.config}'
            old_val = getattr(args, key)
            old_type = type(old_val)
            print(f'[ utils/setup ] Overriding config | {key} : {old_val} --> {val}')
            
            ## 类型转换处理
            if val == 'None':
                val = None
            elif val == 'latest':
                val = 'latest'
            elif old_type in [bool, type(None)]:
                try:
                    val = eval(val)
                except:
                    print(f'[ utils/setup ] Warning: could not parse {val} (old: {old_val}, {old_type}), using str')
            else:
                val = old_type(val)
            setattr(args, key, val) # 把新值写入 args
            self._dict[key] = val

    def eval_fstrings(self, args):
        '''
            处理配置中带有 'f:' 前缀的延迟评估 f-string。
            例如 config 里写了：data_path: 'f:{work_dir}/data/{dataset}/'
            那么在这里会把它转成真正的 f-string，并用当前 args 里的 work_dir 和 dataset 变量填充进去，生成最终的路径字符串。
            这样就能在配置文件里写出动态路径，这个功能对于生成实验保存路径非常有用。
        '''
        for key, old in self._dict.items():
            if type(old) is str and old[:2] == 'f:':
                val = old.replace('{', '{args.').replace('f:', '')
                new = lazy_fstring(val, args)
                print(f'[ utils/setup ] Lazy fstring | {key} : {old} --> {new}')
                setattr(self, key, new)
                self._dict[key] = new

    def set_seed(self, args):
        '''
            如果参数中包含 seed，则设置随机种子。
        '''
        if not 'seed' in dir(args):
            return
        print(f'[ utils/setup ] Setting seed: {args.seed}')
        set_seed(args.seed)

    def generate_exp_name(self, args):
        '''
            生成实验名称,把exp_name从函数变成字符串。
        '''
        if not 'exp_name' in dir(args):
            return
        exp_name = getattr(args, 'exp_name')
        if callable(exp_name):
            exp_name_string = exp_name(args)
            print(f'[ utils/setup ] Setting exp_name to: {exp_name_string}')
            setattr(args, 'exp_name', exp_name_string)
            self._dict['exp_name'] = exp_name_string

    def mkdir(self, args):
        '''
            创建实验数据保存路径,例如logs/maze2d....../
            logbase-logs文件夹；dataset-数据集名称（maze2d、antmaze等）；
            exp_name-实验名称（例如exp_name = "diffusion/H384_T256"）。
            logs/dataset/exp_name/
        '''
        if 'logbase' in dir(args) and 'dataset' in dir(args) and 'exp_name' in dir(args):
            args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name)
            self._dict['savepath'] = args.savepath
            # 如果有 suffix，就把它加到 savepath 末尾，形成logs/maze2d/diffusion_H384/run1/   
            if 'suffix' in dir(args):
                args.savepath = os.path.join(args.savepath, args.suffix)
            if mkdir(args.savepath):
                print(f'[ utils/setup ] Made savepath: {args.savepath}')
            self.save()

    def get_commit(self, args):
        '''
            获取当前的 git commit hash。
        '''
        args.commit = get_git_rev()

    def save_diff(self, args):
        '''
            保存当前的 git diff 信息到 diff.txt
        '''
        try:
            save_git_diff(os.path.join(args.savepath, 'diff.txt'))
        except:
            print('[ utils/setup ] WARNING: did not save git diff')
