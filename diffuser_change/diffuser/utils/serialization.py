import os
import pickle
import glob
import torch
import pdb

from collections import namedtuple

"""
该模块负责实验相关对象的序列化与反序列化：加载先前保存的配置、模型与训练器状态，
并把它们组装成一个便于恢复训练或做推理的 `DiffusionExperiment` 结构。

输出/影响：返回已恢复的 `DiffusionExperiment`（包含 dataset、renderer、model、diffusion、ema、trainer、epoch），
并在 `trainer.load` 中恢复模型/EMA 权重到训练器。后续评估、采样或可视化脚本会调用这些工具来重现实验结果。
"""

DiffusionExperiment = namedtuple('Diffusion', 'dataset renderer model diffusion ema trainer epoch')

def mkdir(savepath):
    """
    做什么：确保给定目录 `savepath` 存在；在必要时创建目录并返回创建状态。
    配置→实例：把一个路径字符串映射为已创建/已存在的文件系统目录，返回布尔值指示是否新创建。
    谁调用：用于训练/保存流程中在写入检查点或配置快照前确保目标目录存在（由各处保存逻辑间接调用）。
    返回：若目录被创建返回 True，否则返回 False。
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False

def get_latest_epoch(loadpath):
    """
    做什么：扫描给定 `loadpath`（按路径片段传入），查找以 `state_*.pt` 命名的检查点，返回最大的 epoch 编号。
    配置→实例：把一个描述结果目录的路径片段序列映射为最新 checkpoint 的整数 epoch；当找不到时返回 -1。
    谁调用：`load_diffusion(..., epoch='latest')` 会调用此函数以定位要加载的最新检查点编号。
    注意：函数通过文件名解析数字，假定检查点命名遵循 `state_{epoch}.pt` 格式。
    """
    states = glob.glob1(os.path.join(*loadpath), 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    """
    做什么：从磁盘反序列化一个由 `utils.Config` 保存的配置文件（pickle 格式），并返回对应的可调用配置对象。
    配置→实例：把 `loadpath` 的文件路径映射为反序列化得到的配置工厂（例如 dataset_config、model_config 等）。
    谁调用：`load_diffusion` 和其它加载工具用于恢复实验配置（dataset/model/diffusion/trainer）。
    输出：打印加载路径和反序列化对象以便调试。
    """
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    print(config)
    return config

def load_diffusion(*loadpath, epoch='latest', device='cuda:0'):
    """
    做什么：从一组路径加载实验配置、实例化 dataset/model/diffusion/trainer，并恢复指定 epoch 的检查点（或最新）。
    配置→实例：按固定的文件名约定加载 `dataset_config.pkl`、`render_config.pkl`、`model_config.pkl`、`diffusion_config.pkl`、`trainer_config.pkl`；
      对每个反序列化得到的配置调用它们以构造具体实例，并把 `model` 注入到 `diffusion_config` 中，
      最终通过 `trainer_config(diffusion, dataset, renderer)` 得到 `Trainer` 实例。
    谁调用：评估/采样脚本或交互式会话通常调用 `load_diffusion` 来恢复训练/采样环境以生成样本或继续训练。

    重要细节：
    - 当 `epoch=='latest'` 时，函数会调用 `get_latest_epoch` 通过文件名解析定位最新的检查点；
    - 函数会把 `trainer_config._dict['results_folder']` 覆盖为提供的 `loadpath`（用于在 Azure 导入场景下修正绝对路径）；
    - 最后调用 `trainer.load(epoch)` 来把模型权重、优化器状态与 EMA 恢复到 `Trainer` 中。

    返回值：一个 `DiffusionExperiment(dataset, renderer, model, diffusion, ema, trainer, epoch)`，其中 `ema` 来自 `trainer.ema_model`。
    """
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    render_config = load_config(*loadpath, 'render_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    diffusion_config = load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    dataset = dataset_config()
    renderer = render_config()
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    trainer.load(epoch)

    return DiffusionExperiment(dataset, renderer, model, diffusion, trainer.ema_model, trainer, epoch)
