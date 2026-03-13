import heapq                         # Python内置最小堆，用于A*的优先队列
from os.path import join             # 用于拼接路径

import numpy as np

import diffuser.datasets as datasets # diffuser库中加载Maze2D环境的工具
import diffuser.utils as utils       # diffuser工具函数，包括Parser和Renderer


# =========================
# 参数解析器
# =========================
class Parser(utils.Parser):
    # 默认使用 Maze2D large dense 数据集
    dataset: str = 'maze2d-large-dense-v1'

    # diffuser配置文件
    config: str = 'config.maze2d'

    # 是否允许A*对角移动
    astar_allow_diagonal: bool = False

    # 每个cell插值多少步，用于生成连续轨迹
    astar_steps_per_cell: int = 8

    def read_config(self, args, experiment):
        # 读取diffuser配置
        args = super().read_config(args, experiment)

        # 如果运行的是规划任务
        if experiment == 'plan':
            args.prefix = 'plans/astar'
            self._dict['prefix'] = args.prefix

            # 默认使用Maze2D renderer
            if not hasattr(args, 'renderer'):
                args.renderer = 'utils.Maze2dRenderer'
                self._dict['renderer'] = args.renderer

        return args


# =========================
# 提取maze地图的占用栅格
# =========================
def extract_occupancy(env):
    """
    Maze2D内部包含maze_arr

    maze_arr:
        10 -> 墙
        0  -> 可通行

    这里转换为 occupancy grid
    True = obstacle
    False = free
    """

    if not hasattr(env, 'maze_arr'):
        raise AttributeError('Maze2D environment does not expose maze_arr')

    maze = np.asarray(env.maze_arr)

    # 返回布尔地图
    return maze == 10


# =========================
# 将连续坐标映射到最近的free cell
# =========================
def nearest_free_cell(position, occupancy):
    """
    输入：
        position  (连续坐标)
        occupancy (栅格地图)

    输出：
        最近的free grid cell
    """

    # 找到所有free cell
    free_cells = np.argwhere(~occupancy)

    # 将位置四舍五入为grid cell
    rounded = np.rint(position).astype(int)

    # 如果四舍五入的cell本身是free
    if (
        0 <= rounded[0] < occupancy.shape[0]
        and 0 <= rounded[1] < occupancy.shape[1]
        and not occupancy[tuple(rounded)]
    ):
        return tuple(rounded.tolist())

    # 否则找到最近free cell
    distances = np.sum((free_cells - position[None]) ** 2, axis=1)
    nearest = free_cells[np.argmin(distances)]

    return tuple(nearest.tolist())


# =========================
# A*邻居生成
# =========================
def astar_neighbors(cell, occupancy, allow_diagonal=False):
    """
    给定当前cell，生成所有可行邻居
    """

    # 四方向移动
    if allow_diagonal:
        deltas = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    else:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    height, width = occupancy.shape

    for di, dj in deltas:

        ni, nj = cell[0] + di, cell[1] + dj

        # 越界检查
        if ni < 0 or nj < 0 or ni >= height or nj >= width:
            continue

        # 如果是墙则跳过
        if occupancy[ni, nj]:
            continue

        # 对角移动时避免穿墙
        if allow_diagonal and abs(di) + abs(dj) == 2:
            if occupancy[cell[0], nj] or occupancy[ni, cell[1]]:
                continue

        # 返回邻居以及移动cost
        yield (ni, nj), np.hypot(di, dj)


# =========================
# A* 启发函数
# =========================
def astar_heuristic(cell, goal):
    """
    使用欧氏距离作为启发式函数
    """
    return np.linalg.norm(np.subtract(cell, goal))


# =========================
# 回溯路径
# =========================
def reconstruct_path(came_from, current):
    """
    从goal回溯到start
    """
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()

    return path


# =========================
# A* 搜索
# =========================
def astar_search(occupancy, start, goal, allow_diagonal=False):

    # open set (优先队列)
    open_heap = [(astar_heuristic(start, goal), 0.0, start)]

    came_from = {}
    g_score = {start: 0.0}

    closed = set()

    while open_heap:

        # 取当前最小f值节点
        _, current_cost, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        # 找到目标
        if current == goal:
            return reconstruct_path(came_from, current)

        closed.add(current)

        # 扩展邻居
        for neighbor, step_cost in astar_neighbors(current, occupancy, allow_diagonal):

            tentative_cost = current_cost + step_cost

            if tentative_cost >= g_score.get(neighbor, float('inf')):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_cost

            # f = g + h
            f_score = tentative_cost + astar_heuristic(neighbor, goal)

            heapq.heappush(open_heap, (f_score, tentative_cost, neighbor))

    return None


# =========================
# 将离散路径转为连续轨迹
# =========================
def build_sequence(observation, target, cell_path, observation_dim, steps_per_cell, horizon):
    """
    将A*生成的cell path转为连续轨迹
    """

    anchors = [np.asarray(observation[:2], dtype=np.float32)]

    # 中间节点
    if len(cell_path) > 2:
        anchors.extend(np.asarray(cell, dtype=np.float32) for cell in cell_path[1:-1])

    # 终点
    anchors.append(np.asarray(target[:2], dtype=np.float32))

    positions = [anchors[0]]

    # 在anchors之间插值
    for start, end in zip(anchors[:-1], anchors[1:]):

        segment = end - start
        distance = np.linalg.norm(segment)

        # 插值数量
        n_steps = max(1, int(np.ceil(distance * steps_per_cell)))

        for alpha in np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)[1:]:
            positions.append(start + alpha * segment)

    positions = np.asarray(positions, dtype=np.float32)

    # 如果轨迹过长，压缩到horizon
    if len(positions) > horizon:
        indices = np.linspace(0, len(positions) - 1, horizon).astype(int)
        positions = positions[indices]

    # 构建完整state序列
    sequence = np.repeat(np.asarray(observation, dtype=np.float32)[None], len(positions), axis=0)

    sequence[:, :2] = positions

    # 填充速度
    if observation_dim >= 4:

        velocities = np.zeros((len(positions), 2), dtype=np.float32)

        velocities[:-1] = positions[1:] - positions[:-1]

        sequence[:, 2:4] = velocities

    return sequence


# =========================
# main
# =========================

args = Parser().parse_args('plan')

# 加载Maze2D环境
env = datasets.load_environment(args.dataset)

# 初始化renderer
render_config = utils.Config(args.renderer, env=args.dataset)
renderer = render_config()

# 重置环境
observation = env.reset()

# 可选：重新采样目标
if args.conditional:
    print('Resetting target')
    env.set_target()

# 获取目标
target = np.asarray(env._target, dtype=np.float32)

# 提取地图
occupancy = extract_occupancy(env)

# 起点终点cell
start_cell = nearest_free_cell(observation[:2], occupancy)
goal_cell = nearest_free_cell(target, occupancy)

# A*搜索
cell_path = astar_search(
    occupancy,
    start_cell,
    goal_cell,
    allow_diagonal=args.astar_allow_diagonal,
)

if cell_path is None:
    raise RuntimeError(f'A* failed to find a path from {start_cell} to {goal_cell}')

# 构建连续轨迹
observation_dim = int(np.prod(env.observation_space.shape))

sequence = build_sequence(
    observation,
    target,
    cell_path,
    observation_dim,
    steps_per_cell=args.astar_steps_per_cell,
    horizon=args.horizon,
)

plan_observations = sequence[None]

print(f'planner: astar')
print(f'start_cell: {start_cell}')
print(f'goal_cell: {goal_cell}')
print(f'path_cells: {len(cell_path)}')
print(f'plan_horizon: {len(sequence)}')

# 保存图像
save_path = join(args.savepath, 'astar_plan.png')

renderer.composite(save_path, plan_observations, ncol=1)

print(f'saved plan image to: {save_path}')