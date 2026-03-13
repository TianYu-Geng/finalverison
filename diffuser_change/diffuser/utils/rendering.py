"""
该文件提供离线轨迹的渲染工具：将观测序列渲染为图片/视频，并支持把多条轨迹拼接后写入磁盘（png/mp4 等）。
它输出的渲染器（如 `MuJoCoRenderer`、`Maze2dRenderer`）由训练流程（`Trainer.render_*`）与规划脚本调用，用于生成 `results/` 或日志目录下的可视化产物。
"""

import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gym
import mujoco_py as mjc
import warnings
import pdb

from .arrays import to_np
from .video import save_video, save_videos

from diffuser.datasets.d4rl import load_environment

#-----------------------------------------------------------------------------#
#------------------------------- helper structs ------------------------------#
#-----------------------------------------------------------------------------#

def env_map(env_name):
    """
    做什么：将部分 D4RL 环境名映射到用于渲染的“全观测”变体环境名。
    配置→实例：把字符串环境配置转换为 `gym.make` 可创建的渲染专用环境名。
    谁调用：`MuJoCoRenderer.__init__` 在传入 `env` 为字符串时调用。
    """
    if 'halfcheetah' in env_name:
        return 'HalfCheetahFullObs-v2'
    elif 'hopper' in env_name:
        return 'HopperFullObs-v2'
    elif 'walker2d' in env_name:
        return 'Walker2dFullObs-v2'
    else:
        return env_name

#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def atmost_2d(x):
    """
    做什么：将输入数组压缩到最多二维（去掉前置的 size=1 维度），以适配渲染函数的输入形状。
    配置→实例：将上游可能带 batch/时间冗余维度的张量转换为渲染侧使用的数组布局。
    谁调用：由 `MuJoCoRenderer.composite` 在拼图时调用。
    """
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

def zipsafe(*args):
    """
    做什么：对多个等长序列做 zip，并在长度不一致时报错。
    配置→实例：把“多路输入参数必须对齐”的约束落实为一次运行时断言。
    谁调用：由 `zipkw` 调用。
    """
    length = len(args[0])
    assert all([len(a) == length for a in args])
    return zip(*args)

def zipkw(*args, **kwargs):
    """
    做什么：将位置参数与关键字参数（逐元素）对齐打包，逐条产出 `(zipped_args, zipped_kwargs)`。
    配置→实例：把“每条轨迹对应一组渲染参数”的调用方式具体化为逐条迭代的参数对齐。
    谁调用：由 `MazeRenderer.composite` 调用。
    """
    nargs = len(args)
    keys = kwargs.keys()
    vals = [kwargs[k] for k in keys]
    zipped = zipsafe(*args, *vals)
    for items in zipped:
        zipped_args = items[:nargs]
        zipped_kwargs = {k: v for k, v in zipsafe(keys, items[nargs:])}
        yield zipped_args, zipped_kwargs

def get_image_mask(img):
    """
    做什么：根据白色背景（RGB=255）计算前景 mask，用于图像合成时覆盖主体像素。
    配置→实例：把“背景像素判定规则”落实为一次像素级布尔 mask 计算。
    谁调用：由 `MuJoCoRenderer.renders` 在将多帧叠加为单张 composite 图时调用。
    """
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def plot2img(fig, remove_margins=True):
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#

class MuJoCoRenderer:
    """
    做什么：将 MuJoCo 环境的状态/观测序列渲染为 RGB 图像，并提供拼图与视频保存接口。
    配置→实例：把环境名或环境实例转换为带 offscreen viewer 的渲染器对象，并推断观测/动作维度用于字段切分。
    谁调用：由 `render_config()` 或脚本创建；训练的 `Trainer.render_*` 与规划脚本在落盘可视化时调用其方法。
    """

    def __init__(self, env):
        """
        做什么：创建用于渲染的环境与 offscreen viewer，并记录观测/动作维度。
        配置→实例：把 `env`（字符串或实例）实例化为 `self.env`，并尝试构造 `self.viewer`。
        谁调用：由训练/规划脚本在启动可视化之前构造。
        """
        if type(env) is str:
            env = env_map(env)
            self.env = gym.make(env)
        else:
            self.env = env
        ## - 1 because the envs in renderer are fully-observed
        ## @TODO : clean up
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)
        except:
            print('[ utils/rendering ] Warning: could not initialize offscreen renderer')
            self.viewer = None

    def pad_observation(self, observation):
        """
        做什么：为单步观测在前面补一个占位维度，匹配渲染环境的 state 布局。
        配置→实例：把“渲染所需 state 维度约定”落实为一次数组拼接。
        谁调用：由 `render` 在 `partial=True` 路径下调用。
        """
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def pad_observations(self, observations):
        """
        做什么：为一段观测序列补全隐藏的 x 位置维度（通过速度积分近似重建 xpos）。
        配置→实例：把“部分观测→渲染全状态”的规则落实为序列级变换，输出可用于 set_state 的 states。
        谁调用：由 `renders` 在 `partial=True` 路径下调用。
        """
        qpos_dim = self.env.sim.data.qpos.size
        ## xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate([
            xpos[:,None],
            observations,
        ], axis=-1)
        return states

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):
        """
        做什么：渲染单步观测/状态为一张 RGB 图像（支持 partial 观测补全与相机参数覆盖）。
        配置→实例：把输入观测与 `dim/render_kwargs` 配置落实为一次 `viewer.render/read_pixels` 调用。
        谁调用：由 `_renders`（序列渲染）间接调用。
        """

        if type(dim) == int:
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [xpos, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def _renders(self, observations, **kwargs):
        """
        做什么：对一段观测序列逐步调用 `render`，返回 `[T, H, W, C]` 图像序列。
        配置→实例：将轨迹序列实例化为逐帧渲染循环。
        谁调用：由 `renders`、`render_rollout`、`render_plan` 等调用。
        """
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        """
        做什么：渲染一条轨迹并将多帧叠加为单张 composite 图（用于静态展示一条轨迹的运动轨迹）。
        配置→实例：把 `partial` 的补全规则与叠加规则（前景覆盖）落实为一次 composite 图构造。
        谁调用：由 `composite` 在批量生成拼图时调用。
        """
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders(samples, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def composite(self, savepath, paths, dim=(1024, 256), **kwargs):
        """
        做什么：将多条轨迹逐条渲染为 composite 图并按行拼接，保存为单张图片（png）。
        配置→实例：把 `paths/ncol(隐式为逐条拼接)/dim` 等配置落实为渲染与 `imageio.imsave` 落盘。
        谁调用：训练的 `Trainer.render_*` 与规划脚本在保存轨迹预览图时调用。
        """

        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [5, 2, 0.5],
            'elevation': 0
        }
        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            img = self.renders(to_np(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images

    def render_rollout(self, savepath, states, **video_kwargs):
        """
        做什么：将单条轨迹渲染为视频并保存到 `savepath`。
        配置→实例：把 `video_kwargs`（fps/codec 等）落实为 `save_video` 的调用参数。
        谁调用：脚本在需要输出视频版本轨迹时调用。
        """
        if type(states) is list: states = np.array(states)
        images = self._renders(states, partial=True)
        save_video(savepath, images, **video_kwargs)

    def render_plan(self, savepath, actions, observations_pred, state, fps=30):
        """
        做什么：将预测轨迹与从环境滚动得到的真实轨迹并排渲染为视频。
        配置→实例：把 `actions/state` 的配置落实为一次真实 rollout（`rollouts_from_state`）并与预测结果拼接落盘。
        谁调用：评估/调试脚本在对比计划与环境动态时调用。
        """
        ## [ batch_size x horizon x observation_dim ]
        observations_real = rollouts_from_state(self.env, state, actions)

        ## there will be one more state in `observations_real`
        ## than in `observations_pred` because the last action
        ## does not have an associated next_state in the sampled trajectory
        observations_real = observations_real[:,:-1]

        images_pred = np.stack([
            self._renders(obs_pred, partial=True)
            for obs_pred in observations_pred
        ])

        images_real = np.stack([
            self._renders(obs_real, partial=False)
            for obs_real in observations_real
        ])

        ## [ batch_size x horizon x H x W x C ]
        images = np.concatenate([images_pred, images_real], axis=-2)
        save_videos(savepath, *images)

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        """
        做什么：渲染反向扩散过程中各时间步的中间轨迹，并保存为视频。
        配置→实例：把 `diffusion_path` 的张量布局与相机配置落实为逐扩散步的渲染循环与 `save_video` 落盘。
        谁调用：调试/可视化脚本在需要观察扩散过程时调用（通常由采样时 `return_diffusion=True` 得到输入）。
        """
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [10, 2, 0.5],
            'elevation': 0,
        }

        diffusion_path = to_np(diffusion_path)

        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f'[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}')

            ## [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[:, :, :self.observation_dim]

            frame = []
            for states in states_l:
                img = self.composite(None, states, dim=(1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
                frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        save_video(savepath, frames, **video_kwargs)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

#-----------------------------------------------------------------------------#
#----------------------------------- maze2d ----------------------------------#
#-----------------------------------------------------------------------------#

## Maze 环境的边界配置，不可以随便改动，地图不变，框的大小也不能变
MAZE_BOUNDS = {
    'maze2d-umaze-v1': (0, 5, 0, 5),
    'maze2d-medium-v1': (0, 8, 0, 8),
    'maze2d-large-v1': (0, 9, 0, 12)
}

## 为dense/sparse版本的 maze2d 环境补全边界配置
MAZE_BOUNDS.update({
    'maze2d-umaze-dense-v1': MAZE_BOUNDS['maze2d-umaze-v1'],
    'maze2d-umaze-sparse-v1': MAZE_BOUNDS['maze2d-umaze-v1'],
    'maze2d-medium-dense-v1': MAZE_BOUNDS['maze2d-medium-v1'],
    'maze2d-medium-sparse-v1': MAZE_BOUNDS['maze2d-medium-v1'],
    'maze2d-large-dense-v1': MAZE_BOUNDS['maze2d-large-v1'],
    'maze2d-large-sparse-v1': MAZE_BOUNDS['maze2d-large-v1'],
})

class MazeRenderer:
    """
    为 Maze 类环境提供基于 matplotlib 的轨迹渲染与拼图保存接口。
    `Maze2dRenderer` 继承并复用；训练与规划脚本通过 `Maze2dRenderer` 间接使用。
    1.把迷宫的墙/空地渲染为背景栅格；
    2.把一条轨迹（点序列）画在背景上，点的颜色表示时间步。
    """

    def __init__(self, env):
        if type(env) is str: env = load_environment(env)
        self._config = env._config
        self._background = self._config != ' ' # 把非空格视为障碍/墙
        self._remove_margins = False
        self._extent = (0, 1, 1, 0) 

    def renders(self, observations, conditions=None, title=None):
        """
        将单条 Maze 轨迹（2D 坐标序列）渲染为一张图像。
        """
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.imshow(self._background * .5,
            extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)

        path_length = len(observations)
        colors = plt.cm.jet(np.linspace(0,1,path_length))
        plt.plot(observations[:,1], observations[:,0], c='black', zorder=10)
        plt.scatter(observations[:,1], observations[:,0], c=colors, zorder=20)
        plt.axis('off')
        plt.title(title)
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img

    def composite(self, savepath, paths, ncol=5, **kwargs):
        """
        将多条 Maze 轨迹渲染结果按网格拼接并保存为单张图片。也就是为什么最终结果是lengh（paths）个图合成一张。
        paths 是一个“轨迹列表”，长度 = 你采了多少条，决定总共有多少个图生成
	    ncol 只是排版参数， 一行 5 个轨迹图；
        """
        assert len(paths) % ncol == 0, "路径数量必须是 ncol 的整数倍"

        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.renders(*path, **kw)
            images.append(img)
        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(images,
            '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        imageio.imsave(savepath, images)
        print(f'Saved {len(paths)} samples to: {savepath}')

class Maze2dRenderer(MazeRenderer):
    """
    针对 D4RL Maze2D 环境的渲染器，负责加载环境并设置背景、边界与维度信息。
    谁调用：由 `render_config()`/训练流程/规划脚本创建；`Trainer.render_*` 与 `scripts/plan_maze2d.py` 会调用其渲染与拼图接口。
    与MazeRenderer的关键区别：D4RL Maze2D 环境的观测是相对坐标（相对于起点），需要根据不同迷宫尺寸做归一化处理。
    同时，maze2d的墙背景来源也不同。
    """

    def __init__(self, env, observation_dim=None):
        """
        加载 Maze2D 环境并初始化观测/动作维度与背景栅格。
        """
        self.env_name = env
        self.env = load_environment(env)
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        self.goal = None
        self._background = self.env.maze_arr == 10 # D4RL Maze2D 环境中，墙的值为10
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, conditions=None, **kwargs):
        """
        将 Maze2D 的观测序列渲染为图像（将环境坐标归一化到绘图坐标系）。
        """
        bounds = MAZE_BOUNDS[self.env_name] # 获取环境边界配置

        observations = observations + .5 # 把相对坐标转换为绝对坐标（起点在(0.5, 0.5)）
        if len(bounds) == 2:
            _, scale = bounds
            observations /= scale # 归一化到 [0,1] 范围
        elif len(bounds) == 4:
            _, iscale, _, jscale = bounds
            observations[:, 0] /= iscale
            observations[:, 1] /= jscale
        else:
            raise RuntimeError(f'Unrecognized bounds for {self.env_name}: {bounds}')

        if conditions is not None:
            conditions /= scale
        return super().renders(observations, conditions, **kwargs) # 调用父类渲染方法

#-----------------------------------------------------------------------------#
#---------------------------------- rollouts ---------------------------------#
#-----------------------------------------------------------------------------#

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')
        state = state[:qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack([
        rollout_from_state(env, state, actions)
        for actions in actions_l
    ])
    return rollouts

def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)
