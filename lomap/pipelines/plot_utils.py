# import cv2                      # 导入 OpenCV，这里目前没有使用，所以先注释掉
import gym                        # 导入 Gym，用于创建强化学习环境
import d4rl                       # 导入 D4RL，注册离线强化学习数据集环境
# import d4rl_ex                  # 额外的 d4rl 扩展，这里未启用
import numpy as np                # 导入 NumPy，用于数值计算
# import mujoco_py as mjc         # MuJoCo Python 接口，这里未启用
from copy import deepcopy         # 深拷贝，避免直接修改原始对象
from d4rl.locomotion import maze_env   # 导入 d4rl 中 maze 环境的定义

from matplotlib import cm         # 导入 matplotlib 的 colormap 模块
from matplotlib import pyplot as plt   # 导入绘图库 pyplot
from matplotlib.colors import ListedColormap  # 导入可自定义离散颜色映射


def plot_maze2d(env_name, ax=None, fix_xy_lim=False):
    """
    绘制 Maze2D 环境的背景地图（障碍物分布）。
    
    参数：
        env_name: 环境名称，例如 'maze2d-umaze-v1'
        ax: matplotlib 的坐标轴对象；若为 None，则新建
        fix_xy_lim: 是否强制设置坐标轴范围
    """
    offset = 0.1   # 坐标显示时给边界留一点空隙，避免贴边

    if ax is None:   # 如果没有传入绘图坐标轴
        _, ax = plt.subplots(1, 1, figsize=(5, 5))   # 新建一个 5x5 的图
        
    env = gym.make(env_name)         # 根据环境名创建 Maze2D 环境
    background = env.maze_arr == 10  # 提取地图中值为 10 的部分，通常表示障碍或墙体
    ax.imshow(background, cmap='Greys')  # 用灰度图显示背景地图

    if fix_xy_lim:   # 如果要求固定显示范围
        if env_name == 'maze2d-umaze-v1':   # U-maze 小地图
            ax.set_xlim(0.5 - offset, 3.5 + offset)  # 设置 x 轴显示范围
            ax.set_ylim(3.5 + offset, 0.5 - offset)  # 设置 y 轴显示范围，反向是为了图像坐标一致
        elif env_name == 'maze2d-medium-v1':   # 中等 maze
            ax.set_xlim(0.5 - offset, 6.5 + offset)
            ax.set_ylim(6.5 + offset, 0.5 - offset)
        elif env_name == 'maze2d-large-v1':    # 大 maze
            ax.set_xlim(0.5 - offset, 10.5 + offset)
            ax.set_ylim(7.5 + offset, 0.5 - offset)
        elif env_name == 'maze2d-simple-v1':   # simple maze
            ax.set_xlim(0.5 - offset, 5.5 + offset)
            ax.set_ylim(5.5 + offset, 0.5 - offset)
        else:
            raise NotImplementedError   # 如果环境未适配，则直接报错
        

def plot_maze2d_observations(observations, ax=None, plot_start=True, goal=None, cmap=None,
                            s=10, alpha=1.0):
    """
    在 Maze2D 地图上绘制轨迹点。

    参数：
        observations: 轨迹观测，形状为 [plan_hor x observation_dim]
                      一般前两维对应位置信息
        ax: matplotlib 坐标轴；若为 None，则新建
        plot_start: 是否高亮起点
        goal: 是否存在目标点标记（这里只是作为开关使用）
        cmap: 颜色映射，默认 Reds
        s: 散点大小
        alpha: 透明度
    """
    if ax is None:   # 如果没有传入 ax
        _, ax = plt.subplots(1, 1, figsize=(5, 5))   # 新建绘图窗口

    if cmap is None:   # 如果没有指定颜色映射
        cmap = cm.Reds # 默认使用 Reds 渐变色

    colors = cmap(np.linspace(0, 1, observations.shape[0]))  
    # 为每个时间步生成一个颜色，颜色从浅到深渐变

    colors[:observations.shape[0]//5] = colors[observations.shape[0]//5]
    # 将前 1/5 的颜色统一成同一种颜色，避免轨迹起始部分颜色过浅不明显

    cmap = ListedColormap(colors)  
    # 将生成好的颜色数组重新包装为一个离散 colormap

    ax.scatter(
        observations[:, 1], observations[:, 0], 
        c=np.arange(observations.shape[0]), cmap=cmap, s=s, alpha=alpha
    )        
    # 绘制轨迹点
    # 注意这里横轴用 observations[:,1]，纵轴用 observations[:,0]
    # 这是因为 Maze2D 中通常把 [row, col] 映射到图像坐标系时需要交换

    if plot_start:   # 如果要高亮起点
        ax.scatter(
            observations[0, 1], observations[0, 0], 
            marker='o', s=500, edgecolors='black', color='white', alpha=1, linewidth=3
        )
        # 先画一个大白圈黑边，作为起点外框

        ax.scatter(
            observations[0, 1], observations[0, 0], 
            marker='o', s=100, edgecolors='black', color='black', alpha=1, linewidth=3
        )
        # 再画一个小黑点，形成“起点”视觉标记

    if goal is not None:   # 如果传入了 goal（这里只是检查非空）
        ax.scatter(
            observations[-1, 1], observations[-1, 0], 
            marker='o', s=500, edgecolors='black', color='white', alpha=1, linewidth=3
        )
        # 先在最后一个点画大白圈黑边，作为目标点外框

        ax.scatter(
            observations[-1, 1], observations[-1, 0], 
            marker='*', s=300, edgecolors='black', color='black', alpha=1
        )
        # 再叠加一个黑色五角星，表示目标点


def plot_antmaze(env_name, ax=None, fix_xy_lim=False):
    """
    绘制 AntMaze 环境的背景障碍图。

    参数：
        env_name: 环境名称，例如 antmaze-large / antmaze-medium
        ax: matplotlib 坐标轴；若为 None，则新建
        fix_xy_lim: 目前这个参数没有实际使用
    """
    offset = 0.1   # 预留边界偏移，目前未使用

    if ax is None:   # 如果未提供 ax
        _, ax = plt.subplots(1, 1, figsize=(5, 5))   # 新建图像
        
    if 'large' in env_name:   # 如果环境名里包含 large
        maze_map = deepcopy(maze_env.HARDEST_MAZE_TEST)
        # 读取 large 对应的大型 maze 地图，并深拷贝避免改原数据
    # elif 'medium': --------debug部分
    elif 'medium' in env_name:
        maze_map = deepcopy(maze_env.BIG_MAZE_TEST)

    for i in range(len(maze_map)):   # 遍历地图每一行
        for j in range(len(maze_map[0])):   # 遍历每一列
            if maze_map[i][j] == 'r' or maze_map[i][j] == 'g':
                maze_map[i][j] = 0
                # 将起点 r 和目标点 g 替换为 0，即视作空地
            
    maze_map = np.array(maze_map)   # 转成 NumPy 数组，便于后续处理
    h, w = maze_map.shape           # 获取地图高度和宽度

    # env = gym.make(env_name)      # 原本可能想直接通过环境读取，但这里没启用
    background = maze_map == 1      # 值为 1 的位置表示墙体/障碍物

    scale = 4.0   # AntMaze 中每个网格对应实际坐标系中的缩放倍数

    # ax.imshow(background, cmap='Greys', extent=[
    #     -(1.5) * scale, 
    #     w * (scale) -(1.5) * scale, 
    #     h * (scale) -(1.5) * scale, 
    #     -(1.5) * scale])
    # 这段是旧版显示方式，y 方向和当前坐标系可能不一致，所以注释掉了

    background = np.flipud(background)
    # 上下翻转背景图，使其与 AntMaze 的实际物理坐标方向一致

    ax.imshow(background, cmap='Greys', extent=[
        -(1.5) * scale, 
        w * (scale) -(1.5) * scale, 
        -(1.5) * scale,
        h * (scale) -(1.5) * scale])
    # 显示背景图，并通过 extent 指定它映射到连续物理坐标系中的范围


def plot_antmaze_observations(observations, ax=None, plot_start=True, goal=None, cmap=None,
                            s=10, alpha=1.0):
    """
    在 AntMaze 背景上绘制轨迹点。

    参数：
        observations: 轨迹观测，形状为 [plan_hor x observation_dim]
                      一般前两维为 x,y 位置
        ax: matplotlib 坐标轴；若为 None，则新建
        plot_start: 是否高亮起点
        goal: 这里未实际使用，仅保留接口
        cmap: 颜色映射，默认 Reds
        s: 散点大小
        alpha: 透明度
    """
    if ax is None:   # 如果没有传入坐标轴
        _, ax = plt.subplots(1, 1, figsize=(5, 5))   # 新建图像

    if cmap is None:   # 如果没有指定颜色映射
        cmap = cm.Reds # 默认用 Reds

    colors = cmap(np.linspace(0, 1, observations.shape[0]))
    # 给每个时间步分配颜色，从浅到深表示时间推进

    colors[:observations.shape[0]//5] = colors[observations.shape[0]//5]
    # 前 1/5 轨迹颜色固定，避免太浅看不清

    cmap = ListedColormap(colors)  
    # 转成离散 colormap

    ax.scatter(
        observations[:, 0], observations[:, 1], 
        c=np.arange(observations.shape[0]), cmap=cmap, s=s, alpha=alpha
    )        
    # 在 AntMaze 中通常直接使用 observations[:,0] 作为 x，observations[:,1] 作为 y
    # 与 Maze2D 不同，这里不需要交换坐标顺序

    if plot_start:   # 如果要高亮起点
        ax.scatter(
            observations[0, 0], observations[0, 1], 
            marker='o', s=100, edgecolors='black', color='white', alpha=1, linewidth=1
        )
        # 先画起点的大白圈

        ax.scatter(
            observations[0, 0], observations[0, 1], 
            marker='o', s=20, edgecolors='black', color='black', alpha=1, linewidth=1
        )
        # 再画中间的小黑点，形成起点标记


# def plot_locomotion_observations(
#     env_name,
#     observations,
#     ax=None,
#     composite_img=None,        # [중요] 이미 생성된 결과 이미지를 받아서 덧그리기용
#     alpha=1.0,                 # [중요] 덧그릴 때 쓸 투명도 (0~1 사이)
#     img_width=1024,
#     img_height=512,
#     skip_frame=1,
#     edge_only=True,
#     custom_render_kwargs=None,
#     joint_color=None,          # <-- [중요] 모든 관절 색상(또는 컬러맵)을 덮어쓸 인자
# ):
#     """
#     observations로부터 수집된 각 스텝의 환경 이미지를 합성하여
#     trajectory를 시각화합니다. composite_img와 alpha 인자를 통해
#     여러 trajectory를 투명하게 겹칠 수 있습니다.

#     Parameters
#     ----------
#     env_name : str
#         MuJoCo 환경 이름 (예: 'hopper-medium-v2')
#     observations : np.ndarray
#         (T, obs_dim) shape의 observation 시퀀스
#     ax : matplotlib axis
#         결과를 그릴 axis. None이면 내부에서 생성
#     composite_img : np.ndarray
#         이전에 만든 결과 이미지를 넘겨 받음 (H x W x 3)
#         None 이면 흰 배경으로 새 이미지 만듦
#     alpha : float
#         이번에 그릴 trajectory(점, edge)에 적용할 alpha(투명도)
#         0.0(투명) ~ 1.0(불투명)
#     img_width : int
#         렌더링할 폭
#     img_height : int
#         렌더링할 높이
#     skip_frame : int
#         몇 프레임마다 합성할 것인지
#     edge_only : bool
#         합성할 때 MuJoCo 배경을 edge 형식으로만 처리할 지 여부
#     custom_render_kwargs : dict
#         카메라 시점 등을 바꾸고 싶을 때 key/value로 넘기는 인자
#     """
#     # ============ 부수적으로 필요한 함수들 ============

#     def mat2euler(r):
#         """3x3 회전행렬 -> roll-pitch-yaw(euler)"""
#         sy = np.sqrt(r[0, 0] * r[0, 0] +  r[1, 0] * r[1, 0])
#         singular = sy < 1e-6
#         if not singular:
#             x = np.arctan2(r[2, 1], r[2, 2])
#             y = np.arctan2(-r[2, 0], sy)
#             z = np.arctan2(r[1, 0], r[0, 0])
#         else:
#             x = np.arctan2(-r[1, 2], r[1, 1])
#             y = np.arctan2(-r[2, 0], sy)
#             z = 0
#         return np.array([x, y, z])

#     def env_map(env_name):
#         """D4RL env_name -> 시각화용 full-obs 환경 이름 맵핑"""
#         if 'halfcheetah' in env_name:
#             return 'HalfCheetahFullObs-v2'
#         elif 'hopper' in env_name:
#             return 'HopperFullObs-v2'
#         elif 'walker2d' in env_name:
#             return 'Walker2dFullObs-v2'
#         else:
#             return env_name

#     def get_2d_from_3d(obj_pos, cam_pos, cam_ori, width, height, fov=90):
#         """
#         3D 좌표(obj_pos)를 카메라 좌표계(cam_pos, cam_ori)에서 투영하여
#         2D 픽셀 좌표를 얻어냄
#         """
#         e = np.array([height/2, width/2, 1])
#         fov = np.array([fov])

#         # 뮤조코 좌표계 -> 일반 CV 좌표계로 변환
#         cam_ori_cv = np.array([cam_ori[1], cam_ori[0], -cam_ori[2]])
#         obj_pos_cv = np.array([obj_pos[1], obj_pos[0], -obj_pos[2]])
#         cam_pos_cv = np.array([cam_pos[1], cam_pos[0], -cam_pos[2]])

#         # 카메라 좌표계로 회전
#         ac_diff = obj_pos_cv - cam_pos_cv
#         x_rot = np.array([
#             [1 ,0, 0],
#             [0, np.cos(cam_ori_cv[0]), np.sin(cam_ori_cv[0])],
#             [0, -np.sin(cam_ori_cv[0]), np.cos(cam_ori_cv[0])]
#         ])
#         y_rot = np.array([
#             [np.cos(cam_ori_cv[1]) ,0, -np.sin(cam_ori_cv[1])],
#             [0, 1, 0],
#             [np.sin(cam_ori_cv[1]), 0, np.cos(cam_ori_cv[1])]
#         ])
#         z_rot = np.array([
#             [np.cos(cam_ori_cv[2]) ,np.sin(cam_ori_cv[2]), 0],
#             [-np.sin(cam_ori_cv[2]), np.cos(cam_ori_cv[2]), 0],
#             [0, 0, 1]
#         ])
#         transform = z_rot.dot(y_rot.dot(x_rot))
#         d = transform.dot(ac_diff)

#         # 원근(투영) 변환
#         fov_rad = np.deg2rad(fov)
#         e[2] *= e[0]*1/np.tan(fov_rad/2.0)
#         bx = e[2]*d[0]/(d[2]) + e[0]
#         by = e[2]*d[1]/(d[2]) + e[1]
#         return bx, by

#     def pad_observations(env, observations):
#         """x 좌표(히든)를 누적 속도로부터 복원하여 붙이기"""
#         qpos_dim = env.sim.data.qpos.size
#         # xvel은 qpos_dim - 1 위치에 있다고 가정 (Hopper/Walker/HalfCheetah 등)
#         xvel_dim = qpos_dim - 1
#         xvel = observations[:, xvel_dim]
#         xpos = np.cumsum(xvel) * env.dt
#         states = np.concatenate([xpos[:,None], observations], axis=-1)
#         return states

#     def set_state(env, state):
#         """env qpos/qvel 세팅"""
#         qpos_dim = env.sim.data.qpos.size
#         qvel_dim = env.sim.data.qvel.size
#         assert(state.size == qpos_dim + qvel_dim)
#         env.set_state(state[:qpos_dim], state[qpos_dim:])

#     def get_image_mask(img):
#         """흰 배경(255,255,255)과 물체의 마스크 분리"""
#         background = (img == 255).all(axis=-1, keepdims=True)
#         mask = ~background.repeat(3, axis=-1)
#         return mask

#     # ============ 메인 로직 ============

#     if ax is None:
#         _, ax = plt.subplots(1, 1, figsize=(5, 5))

#     # 1) 렌더링 옵션 설정
#     render_kwargs = {
#         'trackbodyid': 2,
#         'distance': 3,
#         'lookat': [2, 0, 1],
#         'elevation': 0
#     }
#     if custom_render_kwargs is None:
#         custom_render_kwargs = {}
#     # 사용자 custom_render_kwargs가 있으면 덮어쓰기
#     for k, v in custom_render_kwargs.items():
#         render_kwargs[k] = v

#     width = img_width
#     height = img_height

#     # 2) 환경/뷰어 생성
#     env = gym.make(env_map(env_name))
#     viewer = mjc.MjRenderContextOffscreen(env.sim)
#     for key, val in render_kwargs.items():
#         if key == 'lookat':
#             viewer.cam.lookat[:] = val[:]
#         else:
#             setattr(viewer.cam, key, val)

#     # 카메라 위치/회전 구하기
#     cam_pos = render_kwargs['lookat'].copy()
#     cam_pos[1] = -render_kwargs['distance']
#     cam_ori = mat2euler(env.sim.data.get_camera_xmat('track'))
#     fov = env.sim.model.cam_fovy[0]

#     # 3) 관절 목록 / 색상 설정
#     if 'hopper' in env_name:
#         joints = ['torso', 'thigh', 'foot']
#         joint_colors = ['Reds', 'Greens', 'Blues']
#     elif 'walker2d' in env_name:
#         joints = ['torso', 'thigh', 'foot']
#         joint_colors = ['Reds', 'Greens', 'Blues']
#     elif 'halfcheetah' in env_name:
#         joints = ['torso']
#         joint_colors = ['Reds']
#     else:
#         raise NotImplementedError()

#     if joint_color is not None:
#         joint_colors = [joint_color] * len(joints)

#     # 4) x좌표 복원 & 시뮬레이션 초기 상태 세팅
#     observations = pad_observations(env, observations)
#     imgs = []
#     joint_poses = {k: [] for k in joints}

#     # 5) 각 시점에서 렌더링, body 위치 수집
#     for observation in observations:
#         set_state(env, observation)
#         dim = (width, height)

#         viewer.render(*dim)
#         img = viewer.read_pixels(*dim, depth=False)
#         # OpenGL은 (0,0)이 왼쪽 하단이므로 상하 뒤집어야 함
#         img = img[::-1, :, :]
#         img = np.ascontiguousarray(img, dtype=np.uint8)
#         imgs.append(img)

#         for k in joints:
#             joint_poses[k].append(env.sim.data.get_body_xipos(k).copy())
    
#     # 6) composite_img가 None이면 흰 배경(255) 생성
#     if composite_img is None:
#         composite_img = np.ones_like(imgs[0]) * 255  # H x W x 3

#     # 7) trajectory를 합성(점, background edge 등)해서 composite_img에 덧그림
#     #    alpha blending을 위해, overlay 이미지를 만들어 작업 후 merge
#     #    (OpenCV는 도형 그릴 때 바로 alpha를 지정할 수 없으므로)

#     # (a) 먼저, trajectory 굵은 선(=joint 위치 연결점, 원 등)을 점 찍듯이 겹쳐 그림
#     overlay_trajectory = composite_img.copy()
#     for k, jc in zip(joints, joint_colors):
#         # joints별로 색상 그라디언트를 잡음
#         colors = getattr(cm, jc)(np.linspace(0, 1, observations.shape[0]))  # RGBA, 0~1
#         # 맨 앞부분 색상 고정 or 조정 가능 -> 예시로는 원 코드처럼
#         colors[:observations.shape[0]//5] = colors[observations.shape[0]//5]

#         for t, pos in enumerate(joint_poses[k]):
#             x, y = get_2d_from_3d(pos, cam_pos, cam_ori, width, height, fov)
#             # 원(점) 찍기
#             cv2.circle(
#                 overlay_trajectory,
#                 (int(y), height - int(x)),    # OpenCV는 (col, row) 순서
#                 width // 150,  # 반지름
#                 # colors[t],
#                 colors[t][:3] * 255,
#                 -1  # 채우기
#             )

#     # (b) 점들을 alpha blending
#     composite_img = cv2.addWeighted(overlay_trajectory, alpha, composite_img, 1 - alpha, 0)

#     # (c) 이제 skip_frame 간격으로 실제 이미지 & edge를 합성
#     for t in range(0, len(imgs), skip_frame):
#         # edge + 배경을 그릴 overlay
#         overlay_edge = composite_img.copy()
#         img = imgs[t].copy()

#         if edge_only:
#             # 마스크와 canny를 구해서 edge를 강조
#             mask = get_image_mask(img)
#             # 원하는 색으로 바꾼 뒤 canny
#             img[mask[:, :, 0], 0] = 255   # B
#             img[mask[:, :, 1], 1] = 196   # G
#             img[mask[:, :, 2], 2] = 131   # R
#             edges = cv2.Canny(image=img, threshold1=100, threshold2=200)
#             edge_mask = (edges == 255)[:, :, None].repeat(3, axis=-1)
#             # edge 픽셀만 검정색(또는 다른 색)으로
#             img[edge_mask] = 0

#         # edge or 배경을 overlay에 합성
#         # (흰 배경으로 남아있는 곳에만 img를 씌우고, 나머지는 투명하게)
#         mask = get_image_mask(img)
#         overlay_edge[mask] = img[mask]

#         # 추가로 joints 위치에 outline(테두리)도 그리고 싶으면
#         for k, jc in zip(joints, joint_colors):
#             colors = getattr(cm, jc)(np.linspace(0, 1, observations.shape[0]))
#             colors[:observations.shape[0]//5] = colors[observations.shape[0]//5]
#             pos = joint_poses[k][t]
#             x, y = get_2d_from_3d(pos, cam_pos, cam_ori, width, height, fov)
#             bgr_outline = (0, 0, 0)  # outline 검정
#             bgr_fill = (
#                 colors[t][2]*255,
#                 colors[t][1]*255,
#                 colors[t][0]*255
#             )
#             # 테두리용 원(검정)
#             cv2.circle(overlay_edge, (int(y), height - int(x)), width // 150, bgr_outline, 1)
#             # 안쪽 채운 원
#             cv2.circle(overlay_edge, (int(y), height - int(x)), width // 150, bgr_fill, -1)

#         # (d) 이번 프레임의 edge/배경을 alpha blending
#         composite_img = cv2.addWeighted(overlay_edge, alpha, composite_img, 1 - alpha, 0)

#     # 8) matplotlib으로 시각화
#     # ax.imshow(composite_img[..., ::-1])  # OpenCV BGR -> Matplotlib RGB
#     ax.imshow(composite_img)
#     return composite_img  # [중요] 업데이트된 composite_img 반환







# """ Usage:
# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# composite_img = None
# plot_antmze_observations_v2(
#     args.task.env_name, 
#     ax=ax,
#     observations=normalizer.unnormalize(traj[0, :, :obs_dim].cpu().numpy()),
#     # observations=states,
#     edge_only=False,
#     alpha=1.,
#     img_width=800, 
#     img_height=512,
#     skip_frame=1,
#     joint_color=None,  # None이면 body별로 Reds/Greens/Blues/Purples/Oranges,
#     custom_render_kwargs={
#         "lookat": [20, 10, 0.0],   # 바라볼 좌표 (x, y, z)
#         "distance": 40.0,        # 멀리서 볼수록 더 넓게 보임
#         "azimuth": 90,           # 수평 회전각 (0도면 x축 방향 기준)
#         "elevation": -90,       # 위에서 내려다보는 각도
#     },
# );
# """


# # def plot_antmze_observations_v2(
# #     env_name,
# #     observations,
# #     ax=None,
# #     composite_img=None,  # 이미 그려둔 이미지가 있으면 이어서 그림
# #     alpha=1.0,
# #     img_width=1024,
# #     img_height=512,
# #     skip_frame=1,
# #     edge_only=True,
# #     custom_render_kwargs=None,
# #     joint_color=None,    # None이면 바디별 기본 색, 아니면 모든 바디에 동일 색/컬러맵
# # ):
# #     """
# #     AntFullObsEnv(또는 이를 등록한 'ant-full-obs-v0')에서
# #     수집한 trajectory(observations)를 2D로 투영+합성하여 시각화.

# #     Parameters
# #     ----------
# #     env_name : str
# #         'ant' 문자열이 들어간 환경 이름 (예: 'ant-full-obs-v0')
# #     observations : np.ndarray
# #         (T, obs_dim) shape (rollout에서 수집한 observation 시퀀스)
# #     ax : matplotlib.axes
# #         그릴 axis (None이면 내부에서 plt.subplots 생성)
# #     composite_img : np.ndarray
# #         이전에 만든 (H,W,3) 이미지 (흰 배경 + 그려진 내용)를 넘겨주면,
# #         그 위에 지금의 trajectory를 투명하게 겹칠 수 있음
# #     alpha : float
# #         투명도 (0~1)
# #     img_width : int
# #         렌더링 폭
# #     img_height : int
# #         렌더링 높이
# #     skip_frame : int
# #         몇 step마다 실제 배경(edge)와 합성할지
# #     edge_only : bool
# #         True면 배경을 엣지(edge) 형태로만 표현 (기존 방식)
# #     custom_render_kwargs : dict
# #         카메라 시점(distance, lookat, elevation 등) 변경용
# #     joint_color : str or None
# #         - None이면, 각 바디별로 미리 정의한 색상/컬러맵을 사용
# #         - 문자열(예: 'red', 'beige', 'Reds' 등)이면, 모든 바디에 해당 색 적용
# #     """

# #     # -----------------------------
# #     # 1) 유틸 함수들
# #     # -----------------------------

# #     def mat2euler(r):
# #         """3x3 회전행렬 -> roll-pitch-yaw(euler)"""
# #         sy = np.sqrt(r[0, 0] * r[0, 0] +  r[1, 0] * r[1, 0])
# #         singular = sy < 1e-6
# #         if not singular:
# #             x = np.arctan2(r[2, 1], r[2, 2])
# #             y = np.arctan2(-r[2, 0], sy)
# #             z = np.arctan2(r[1, 0], r[0, 0])
# #         else:
# #             x = np.arctan2(-r[1, 2], r[1, 1])
# #             y = np.arctan2(-r[2, 0], sy)
# #             z = 0
# #         return np.array([x, y, z])

# #     def env_map(name):
# #         if "large" in name.lower():
# #             return 'AntLargeFullObs-v2'
# #         else:
# #             raise NotImplementedError

# #     def get_image_mask(img):
# #         """흰 배경(255,255,255) 부분 vs 물체 부분의 마스크"""
# #         background = (img == 255).all(axis=-1, keepdims=True)
# #         mask = ~background.repeat(3, axis=-1)
# #         return mask

# #     # -----------------------------
# #     # 2) Env 셋업
# #     # -----------------------------
# #     if ax is None:
# #         _, ax = plt.subplots(1, 1, figsize=(5, 5))

# #     # 카메라 기본 설정
# #     render_kwargs = {
# #         'trackbodyid': 0,  # 보통 ant의 root body는 0 또는 1일 수 있음
# #         'distance': 4,     # ant는 좀 더 멀리서 봐야 할 수도 있으니 4~6 정도
# #         'lookat': [0, 0, 0.5],
# #         'elevation': -20,  # 살짝 내려다보는 시점
# #     }
# #     if custom_render_kwargs is None:
# #         custom_render_kwargs = {}
# #     for k, v in custom_render_kwargs.items():
# #         render_kwargs[k] = v

# #     width = img_width
# #     height = img_height

# #    # 2) 환경/뷰어 생성
# #     env = gym.make(env_map(env_name))
# #     viewer = mjc.MjRenderContextOffscreen(env.sim)

# #     for key, val in render_kwargs.items():
# #         if key == 'lookat':
# #             viewer.cam.lookat[:] = val[:]
# #         else:
# #             setattr(viewer.cam, key, val)

# #     cam_pos = render_kwargs['lookat'].copy()
# #     cam_pos[1] = -render_kwargs['distance']
# #     cam_ori = mat2euler(env.sim.data.get_camera_xmat('track'))
# #     fov = env.sim.model.cam_fovy[0]
 
# #     # -----------------------------
# #     # 4) 관측값 -> 상태 세팅 & 렌더링
# #     # -----------------------------
# #     def pad_observations(env, obs):
# #         """
# #         Ant의 경우에는 HalfCheetah처럼 x좌표를 따로 복원하기 쉽지 않으므로,
# #         여기서는 그냥 그대로 반환 (필요하다면 root x속도(qvel[0]) 누적해도 됨).
# #         """
# #         return obs

# #     observations = pad_observations(env, observations)

# #     # (T, obs_dim)
# #     T = observations.shape[0]
# #     imgs = []

# #     # 매 스텝마다 상태 세팅 -> 이미지 렌더링 -> 바디 중심 위치 기록
# #     # AntFullObsEnv 안에서는 `env.set_state(qpos, qvel)`로 전체 상태를 세팅 가능
# #     def set_state(env, state):
# #         """
# #         AntFullObsEnv에서 (qpos_dim + qvel_dim) = 29짜리 state일 경우
# #         qpos_dim=15, qvel_dim=14.
# #         다만, roll-out을 어떻게 수집했는지에 따라
# #         obs가 실제로 qpos+qvel인지, 아니면 부분 관측인지 주의해야 함.
# #         """
# #         qpos_dim = env.sim.data.qpos.size   # 15
# #         qvel_dim = env.sim.data.qvel.size   # 14
# #         assert state.size == qpos_dim + qvel_dim, f"{state.size}"
# #         env.set_state(state[:qpos_dim], state[qpos_dim:])
    
# #     # marker_id = env.sim.model.body_name2id("marker")
# #     for obs_t in observations:
# #         # obs_t가 실제로 qpos+qvel인지, 아니면 _get_obs() 형태인지 확인 필요
# #         # 여기서는 '풀 상태'가 들어있다고 가정
# #         set_state(env, obs_t)

# #         # ###
# #         # pos = env.sim.data.get_body_xipos("torso")
# #         # goal_id = env.sim.model.body_name2id("goal")
# #         # env.sim.model.body_pos[goal_id] = list(env.target_goal) + [0]
# #         # env.sim.forward()
# #         # ###

# #         viewer.render(width, height)
# #         img = viewer.read_pixels(width, height, depth=False)
# #         img = img[::-1, :, :]  # 상하 flip
# #         img = np.ascontiguousarray(img, dtype=np.uint8)
# #         imgs.append(img)


# #     # -----------------------------
# #     # 5) composite_img 준비 & alpha blending
# #     # -----------------------------
# #     if composite_img is None:
# #         composite_img = np.ones_like(imgs[0]) * 255  # 흰 배경

# #     # (a) 우선, 바디별 위치를 원(circle)으로 찍어두는 overlay
# #     overlay_trajectory = composite_img.copy()

# #     # color parsing 함수(단색 vs cmap 구분) - 예시
# #     def parse_color(color_str, length):
# #         if hasattr(cm, color_str):
# #             # matplotlib에 내장된 컬러맵 이름이면
# #             colormap = getattr(cm, color_str)
# #             c_array = colormap(np.linspace(0, 1, length))  # (length,4) RGBA
# #         else:
# #             # 단색
# #             rgba = mcolors.to_rgba(color_str)  # (r,g,b,a)
# #             c_array = np.ones((length,4)) * rgba
# #         return c_array


# #     # 이제 overlay_trajectory를 alpha로 composite_img에 blend
# #     composite_img = cv2.addWeighted(overlay_trajectory, alpha, composite_img, 1 - alpha, 0)

# #     # (b) skip_frame마다 실제 배경(edge) 처리
# #     for t in range(0, len(imgs), skip_frame):
# #         overlay_edge = composite_img.copy()
# #         img = imgs[t].copy()

# #         if edge_only:
# #             mask = get_image_mask(img)
# #             # 원하는 색 감조정 후 Canny
# #             img[mask[:, :, 0], 0] = 255
# #             img[mask[:, :, 1], 1] = 196
# #             img[mask[:, :, 2], 2] = 131
# #             edges = cv2.Canny(img, 100, 200)
# #             edge_mask = (edges == 255)[:, :, None].repeat(3, axis=-1)
# #             img[edge_mask] = 0

# #         # 흰 배경인 곳에만 img를 합성
# #         mask = get_image_mask(img)
# #         overlay_edge[mask] = img[mask]

# #         composite_img = cv2.addWeighted(overlay_edge, alpha, composite_img, 1 - alpha, 0)

# #     # (c) matplotlib에 표시
# #     ax.imshow(composite_img)
# #     return composite_img








# def plot_antmze_observations_v2(
#     env_name,
#     observations,
#     ax=None,
#     composite_img=None,  # 이미 그려둔 이미지가 있으면 이어서 그림
#     alpha=1.0,
#     img_width=1024,
#     img_height=512,
#     skip_frame=1,
#     custom_render_kwargs=None,
#     ant_color=None
# ):
#     """
#     AntFullObsEnv(또는 이를 등록한 'ant-full-obs-v0')에서
#     수집한 trajectory(observations)를 2D로 투영+합성하여 시각화.

#     Parameters
#     ----------
#     env_name : str
#         'ant' 문자열이 들어간 환경 이름 (예: 'ant-full-obs-v0')
#     observations : np.ndarray
#         (T, obs_dim) shape (rollout에서 수집한 observation 시퀀스)
#     ax : matplotlib.axes
#         그릴 axis (None이면 내부에서 plt.subplots 생성)
#     composite_img : np.ndarray
#         이전에 만든 (H,W,3) 이미지 (흰 배경 + 그려진 내용)를 넘겨주면,
#         그 위에 지금의 trajectory를 투명하게 겹칠 수 있음
#     img_width : int
#         렌더링 폭
#     img_height : int
#         렌더링 높이
#     skip_frame : int
#         몇 step마다 실제 배경(edge)와 합성할지
#     custom_render_kwargs : dict
#         카메라 시점(distance, lookat, elevation 등) 변경용
#     """

#     def get_image_mask(img):
#         """흰 배경(255,255,255) 부분 vs 물체 부분의 마스크"""
#         background = (img == 255).all(axis=-1, keepdims=True)
#         mask = ~background.repeat(3, axis=-1)
#         return mask

#     # -----------------------------
#     # 2) Env 셋업
#     # -----------------------------
#     if ax is None:
#         _, ax = plt.subplots(1, 1, figsize=(5, 5))

#     # 카메라 기본 설정
#     render_kwargs = {
#         'trackbodyid': 0,  # 보통 ant의 root body는 0 또는 1일 수 있음
#         'distance': 4,     # ant는 좀 더 멀리서 봐야 할 수도 있으니 4~6 정도
#         'lookat': [0, 0, 0.5],
#         'elevation': -20,  # 살짝 내려다보는 시점
#     }
#     if custom_render_kwargs is None:
#         custom_render_kwargs = {}
#     for k, v in custom_render_kwargs.items():
#         render_kwargs[k] = v

#     width = img_width
#     height = img_height

#     # 2) 환경/뷰어 생성 (벽만 있는 환경)
#     if 'large' in env_name:
#         env_walls = gym.make('AntLarge-v2')
#     elif 'medium' in env_name:
#         env_walls = gym.make('AntMedium-v2')
#     else:
#         raise NotImplementedError
#     viewer_walls = mjc.MjRenderContextOffscreen(env_walls.sim)

#     # Ant를 멀리 보내기 (qpos를 조정)
#     qpos = env_walls.sim.data.qpos.copy()  # 현재 qpos 가져오기
#     qpos[0] = 1000  # x 좌표를 멀리 보냄 (카메라 시야에서 벗어나도록)
#     env_walls.set_state(qpos, env_walls.sim.data.qvel)  # 상태 업데이트

#     # goal rendering
#     goal_id = env_walls.sim.model.body_name2id("goal")
#     env_walls.sim.model.body_pos[goal_id] = list(env_walls.target_goal) + [0]
#     env_walls.sim.forward()

#     for key, val in render_kwargs.items():
#         if key == 'lookat':
#             viewer_walls.cam.lookat[:] = val[:]
#         else:
#             setattr(viewer_walls.cam, key, val)

#     # 벽만 있는 환경 렌더링 (composite_img가 None인 경우에만)
#     if composite_img is None:
#         viewer_walls.render(width, height)
#         img_walls = viewer_walls.read_pixels(width, height, depth=False)
#         img_walls = img_walls[::-1, :, :]  # 상하 flip
#         img_walls = np.ascontiguousarray(img_walls, dtype=np.uint8)
#         composite_img = np.ones_like(img_walls) * 255  # 흰 배경
#         mask_walls = get_image_mask(img_walls)
#         composite_img[mask_walls] = img_walls[mask_walls]

#     # Ant만 있는 환경 렌더링
#     env_ant = gym.make('AntFullObs-v2')
#     viewer_ant = mjc.MjRenderContextOffscreen(env_ant.sim)

#     # Ant와 관련된 geom들의 색깔을 바꾸기
#     if ant_color:
#         for geom_name in env_ant.sim.model.geom_names:
#             if any(s in geom_name for s in ["torso", "leg", "ankle", "aux"]):
#                 geom_id = env_ant.sim.model.geom_name2id(geom_name)
#                 env_ant.sim.model.geom_rgba[geom_id] = ant_color

#     for key, val in render_kwargs.items():
#         if key == 'lookat':
#             viewer_ant.cam.lookat[:] = val[:]
#         else:
#             setattr(viewer_ant.cam, key, val)

#     # -----------------------------
#     # 4) 관측값 -> 상태 세팅 & 렌더링
#     # -----------------------------
#     def pad_observations(env, obs):
#         """
#         Ant의 경우에는 HalfCheetah처럼 x좌표를 따로 복원하기 쉽지 않으므로,
#         여기서는 그대로 반환 (필요하다면 root x속도(qvel[0]) 누적해도 됨).
#         """
#         return obs

#     observations = pad_observations(env_ant, observations)

#     # (T, obs_dim)
#     T = observations.shape[0]
#     imgs = []

#     # 매 스텝마다 상태 세팅 -> 이미지 렌더링 -> 바디 중심 위치 기록
#     def set_state(env, state):
#         """
#         AntFullObsEnv에서 (qpos_dim + qvel_dim) = 29짜리 state일 경우
#         qpos_dim=15, qvel_dim=14.
#         """
#         qpos_dim = env.sim.data.qpos.size   # 15
#         qvel_dim = env.sim.data.qvel.size   # 14
#         assert state.size == qpos_dim + qvel_dim, f"{state.size}"
#         env.set_state(state[:qpos_dim], state[qpos_dim:])
    
#     for obs_t in observations:
#         set_state(env_ant, obs_t)
#         viewer_ant.render(width, height)
#         img = viewer_ant.read_pixels(width, height, depth=False)
#         img = img[::-1, :, :]  # 상하 flip
#         img = np.ascontiguousarray(img, dtype=np.uint8)
#         imgs.append(img)

#     # -----------------------------
#     # 5) composite_img에 Ant 이미지 덧그리기
#     # -----------------------------
#     for t in range(0, len(imgs), skip_frame):
#         img_ant = imgs[t].copy()
#         mask_ant = get_image_mask(img_ant)
#         # alpha blending을 사용하여 Ant 이미지를 합성
#         composite_img[mask_ant] = (
#             alpha * img_ant[mask_ant] + (1 - alpha) * composite_img[mask_ant]
#         ).astype(np.uint8)

#     # (c) matplotlib에 표시
#     ax.imshow(composite_img)
#     return composite_img


















# def plot_locomotion_observations_icml(
#     env_name,
#     observations,
#     ax=None,
#     composite_img=None,        # [중요] 이미 생성된 결과 이미지를 받아서 덧그리기용
#     alpha=1.0,                 # [중요] 덧그릴 때 쓸 투명도 (0~1 사이)
#     img_width=1024,
#     img_height=512,
#     skip_frame=1,
#     edge_only=True,
#     custom_render_kwargs=None,
#     joint_color=None,          # <-- [중요] 모든 관절 색상(또는 컬러맵)을 덮어쓸 인자
# ):
#     """
#     observations로부터 수집된 각 스텝의 환경 이미지를 합성하여
#     trajectory를 시각화합니다. composite_img와 alpha 인자를 통해
#     여러 trajectory를 투명하게 겹칠 수 있습니다.

#     Parameters
#     ----------
#     env_name : str
#         MuJoCo 환경 이름 (예: 'hopper-medium-v2')
#     observations : np.ndarray
#         (T, obs_dim) shape의 observation 시퀀스
#     ax : matplotlib axis
#         결과를 그릴 axis. None이면 내부에서 생성
#     composite_img : np.ndarray
#         이전에 만든 결과 이미지를 넘겨 받음 (H x W x 3)
#         None 이면 흰 배경으로 새 이미지 만듦
#     alpha : float
#         이번에 그릴 trajectory(점, edge)에 적용할 alpha(투명도)
#         0.0(투명) ~ 1.0(불투명)
#     img_width : int
#         렌더링할 폭
#     img_height : int
#         렌더링할 높이
#     skip_frame : int
#         몇 프레임마다 합성할 것인지
#     edge_only : bool
#         합성할 때 MuJoCo 배경을 edge 형식으로만 처리할 지 여부
#     custom_render_kwargs : dict
#         카메라 시점 등을 바꾸고 싶을 때 key/value로 넘기는 인자
#     """
#     # ============ 부수적으로 필요한 함수들 ============

#     def mat2euler(r):
#         """3x3 회전행렬 -> roll-pitch-yaw(euler)"""
#         sy = np.sqrt(r[0, 0] * r[0, 0] +  r[1, 0] * r[1, 0])
#         singular = sy < 1e-6
#         if not singular:
#             x = np.arctan2(r[2, 1], r[2, 2])
#             y = np.arctan2(-r[2, 0], sy)
#             z = np.arctan2(r[1, 0], r[0, 0])
#         else:
#             x = np.arctan2(-r[1, 2], r[1, 1])
#             y = np.arctan2(-r[2, 0], sy)
#             z = 0
#         return np.array([x, y, z])

#     def env_map(env_name):
#         """D4RL env_name -> 시각화용 full-obs 환경 이름 맵핑"""
#         if 'halfcheetah' in env_name:
#             return 'HalfCheetahFullObs-v2'
#         elif 'hopper' in env_name:
#             return 'HopperFullObs-v2'
#         elif 'walker2d' in env_name:
#             return 'Walker2dFullObs-v2'
#         else:
#             return env_name

#     def get_2d_from_3d(obj_pos, cam_pos, cam_ori, width, height, fov=90):
#         """
#         3D 좌표(obj_pos)를 카메라 좌표계(cam_pos, cam_ori)에서 투영하여
#         2D 픽셀 좌표를 얻어냄
#         """
#         e = np.array([height/2, width/2, 1])
#         fov = np.array([fov])

#         # 뮤조코 좌표계 -> 일반 CV 좌표계로 변환
#         cam_ori_cv = np.array([cam_ori[1], cam_ori[0], -cam_ori[2]])
#         obj_pos_cv = np.array([obj_pos[1], obj_pos[0], -obj_pos[2]])
#         cam_pos_cv = np.array([cam_pos[1], cam_pos[0], -cam_pos[2]])

#         # 카메라 좌표계로 회전
#         ac_diff = obj_pos_cv - cam_pos_cv
#         x_rot = np.array([
#             [1 ,0, 0],
#             [0, np.cos(cam_ori_cv[0]), np.sin(cam_ori_cv[0])],
#             [0, -np.sin(cam_ori_cv[0]), np.cos(cam_ori_cv[0])]
#         ])
#         y_rot = np.array([
#             [np.cos(cam_ori_cv[1]) ,0, -np.sin(cam_ori_cv[1])],
#             [0, 1, 0],
#             [np.sin(cam_ori_cv[1]), 0, np.cos(cam_ori_cv[1])]
#         ])
#         z_rot = np.array([
#             [np.cos(cam_ori_cv[2]) ,np.sin(cam_ori_cv[2]), 0],
#             [-np.sin(cam_ori_cv[2]), np.cos(cam_ori_cv[2]), 0],
#             [0, 0, 1]
#         ])
#         transform = z_rot.dot(y_rot.dot(x_rot))
#         d = transform.dot(ac_diff)

#         # 원근(투영) 변환
#         fov_rad = np.deg2rad(fov)
#         e[2] *= e[0]*1/np.tan(fov_rad/2.0)
#         bx = e[2]*d[0]/(d[2]) + e[0]
#         by = e[2]*d[1]/(d[2]) + e[1]
#         return bx, by

#     def pad_observations(env, observations):
#         """x 좌표(히든)를 누적 속도로부터 복원하여 붙이기"""
#         qpos_dim = env.sim.data.qpos.size
#         # xvel은 qpos_dim - 1 위치에 있다고 가정 (Hopper/Walker/HalfCheetah 등)
#         xvel_dim = qpos_dim - 1
#         xvel = observations[:, xvel_dim]
#         xpos = np.cumsum(xvel) * env.dt
#         states = np.concatenate([xpos[:,None], observations], axis=-1)
#         return states

#     def set_state(env, state):
#         """env qpos/qvel 세팅"""
#         qpos_dim = env.sim.data.qpos.size
#         qvel_dim = env.sim.data.qvel.size
#         assert(state.size == qpos_dim + qvel_dim)
#         env.set_state(state[:qpos_dim], state[qpos_dim:])

#     def get_image_mask(img):
#         """흰 배경(255,255,255)과 물체의 마스크 분리"""
#         background = (img == 255).all(axis=-1, keepdims=True)
#         mask = ~background.repeat(3, axis=-1)
#         return mask

#     # ============ 메인 로직 ============

#     if ax is None:
#         _, ax = plt.subplots(1, 1, figsize=(5, 5))

#     # 1) 렌더링 옵션 설정
#     render_kwargs = {
#         'trackbodyid': 2,
#         'distance': 3,
#         'lookat': [2, 0, 1],
#         'elevation': 0
#     }
#     if custom_render_kwargs is None:
#         custom_render_kwargs = {}
#     # 사용자 custom_render_kwargs가 있으면 덮어쓰기
#     for k, v in custom_render_kwargs.items():
#         render_kwargs[k] = v

#     width = img_width
#     height = img_height

#     # 2) 환경/뷰어 생성
#     env = gym.make(env_map(env_name))
#     viewer = mjc.MjRenderContextOffscreen(env.sim)
#     for key, val in render_kwargs.items():
#         if key == 'lookat':
#             viewer.cam.lookat[:] = val[:]
#         else:
#             setattr(viewer.cam, key, val)

#     # 카메라 위치/회전 구하기
#     cam_pos = render_kwargs['lookat'].copy()
#     cam_pos[1] = -render_kwargs['distance']
#     cam_ori = mat2euler(env.sim.data.get_camera_xmat('track'))
#     fov = env.sim.model.cam_fovy[0]

#     # 3) 관절 목록 / 색상 설정
#     if 'hopper' in env_name:
#         joints = ['torso', 'thigh', 'foot']
#         joint_colors = ['Reds', 'Reds', 'Reds']
#     elif 'walker2d' in env_name:
#         joints = ['torso', 'thigh', 'foot']
#         joint_colors = ['Reds', 'Reds', 'Reds']
#     elif 'halfcheetah' in env_name:
#         joints = ['torso']
#         joint_colors = ['Reds']
#     else:
#         raise NotImplementedError()

#     if joint_color is not None:
#         joint_colors = [joint_color] * len(joints)

#     # 4) x좌표 복원 & 시뮬레이션 초기 상태 세팅
#     observations = pad_observations(env, observations)
#     imgs = []
#     joint_poses = {k: [] for k in joints}

#     # 5) 각 시점에서 렌더링, body 위치 수집
#     for observation in observations:
#         set_state(env, observation)
#         dim = (width, height)

#         viewer.render(*dim)
#         img = viewer.read_pixels(*dim, depth=False)
#         # OpenGL은 (0,0)이 왼쪽 하단이므로 상하 뒤집어야 함
#         img = img[::-1, :, :]
#         img = np.ascontiguousarray(img, dtype=np.uint8)
#         imgs.append(img)

#         for k in joints:
#             joint_poses[k].append(env.sim.data.get_body_xipos(k).copy())
    
#     # 6) composite_img가 None이면 흰 배경(255) 생성
#     if composite_img is None:
#         composite_img = np.ones_like(imgs[0]) * 255  # H x W x 3

#     # 7) trajectory를 합성(점, background edge 등)해서 composite_img에 덧그림
#     #    alpha blending을 위해, overlay 이미지를 만들어 작업 후 merge
#     #    (OpenCV는 도형 그릴 때 바로 alpha를 지정할 수 없으므로)

#     # (a) 먼저, trajectory 굵은 선(=joint 위치 연결점, 원 등)을 점 찍듯이 겹쳐 그림
#     overlay_trajectory = composite_img.copy()
#     for k, jc in zip(joints, joint_colors):
#         # joints별로 색상 그라디언트를 잡음
#         colors = getattr(cm, jc)(np.linspace(0, 1, observations.shape[0]))  # RGBA, 0~1
#         # 맨 앞부분 색상 고정 or 조정 가능 -> 예시로는 원 코드처럼
#         colors[:observations.shape[0]//5] = colors[observations.shape[0]//5]

#         for t, pos in enumerate(joint_poses[k]):
#             # if t % 2 == 0:
#                 x, y = get_2d_from_3d(pos, cam_pos, cam_ori, width, height, fov)
#                 # 원(점) 찍기
#                 cv2.circle(
#                     overlay_trajectory,
#                     (int(y), height - int(x)),    # OpenCV는 (col, row) 순서
#                     width // 300,  # 반지름
#                     # colors[t],
#                     colors[t][:3] * 255,
#                     -1  # 채우기
#                 )

#     # (b) 점들을 alpha blending
#     composite_img = cv2.addWeighted(overlay_trajectory, alpha, composite_img, 1 - alpha, 0)
#     # print('hihi')
#     # (c) 이제 skip_frame 간격으로 실제 이미지 & edge를 합성
#     for t in range(0, len(imgs), skip_frame):
#         # edge + 배경을 그릴 overlay
#         overlay_edge = composite_img.copy()
#         img = imgs[t].copy()

#         if edge_only:
#             # 마스크와 canny를 구해서 edge를 강조
#             mask = get_image_mask(img)
#             # 원하는 색으로 바꾼 뒤 canny
#             img[mask[:, :, 0], 0] = 255   # B
#             img[mask[:, :, 1], 1] = 196   # G
#             img[mask[:, :, 2], 2] = 131   # R
#             edges = cv2.Canny(image=img, threshold1=100, threshold2=200)
#             edge_mask = (edges == 255)[:, :, None].repeat(3, axis=-1)
#             # edge 픽셀만 검정색(또는 다른 색)으로
#             img[edge_mask] = 0

#         # edge or 배경을 overlay에 합성
#         # (흰 배경으로 남아있는 곳에만 img를 씌우고, 나머지는 투명하게)
#         mask = get_image_mask(img)
#         overlay_edge[mask] = img[mask]

#         # 추가로 joints 위치에 outline(테두리)도 그리고 싶으면
#         for k, jc in zip(joints, joint_colors):
#             colors = getattr(cm, jc)(np.linspace(0, 1, observations.shape[0]))
#             colors[:observations.shape[0]//5] = colors[observations.shape[0]//5]
#             pos = joint_poses[k][t]
#             x, y = get_2d_from_3d(pos, cam_pos, cam_ori, width, height, fov)
#             bgr_outline = (0, 0, 0)  # outline 검정
#             bgr_fill = (
#                 colors[t][2]*255,
#                 colors[t][1]*255,
#                 colors[t][0]*255
#             )
#             # 테두리용 원(검정)
#             cv2.circle(overlay_edge, (int(y), height - int(x)), width // 150, bgr_outline, 1)
#             # 안쪽 채운 원
#             cv2.circle(overlay_edge, (int(y), height - int(x)), width // 150, bgr_fill, -1)

#         # (d) 이번 프레임의 edge/배경을 alpha blending
#         composite_img = cv2.addWeighted(overlay_edge, alpha, composite_img, 1 - alpha, 0)

#     # 8) matplotlib으로 시각화
#     # ax.imshow(composite_img[..., ::-1])  # OpenCV BGR -> Matplotlib RGB
#     ax.imshow(composite_img)
#     return composite_img  # [중요] 업데이트된 composite_img 반환

