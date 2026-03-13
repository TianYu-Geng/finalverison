import heapq
from os.path import join

import numpy as np

import diffuser.datasets as datasets
import diffuser.utils as utils


# =========================
# 参数解析器
# =========================
class Parser(utils.Parser):
    # 默认使用 Maze2D large dense 数据集
    dataset: str = 'maze2d-large-dense-v1'

    # diffuser 配置文件
    config: str = 'config.maze2d'

    # 是否允许 A* 对角移动
    astar_allow_diagonal: bool = False

    # 每个 cell 插值多少步，用于生成连续轨迹
    astar_steps_per_cell: int = 8

    # 最终保留多少条路径
    astar_num_paths: int = 5

    # 每条路径长度不得超过最短路径的多少倍
    astar_max_ratio: float = 2

    # 路径去重时的最大重合率
    astar_max_overlap: float = 0.8

    # 先多搜一些候选，再筛掉太相似的
    astar_candidate_multiplier: int = 4

    def read_config(self, args, experiment):
        # 读取 diffuser 配置
        args = super().read_config(args, experiment)

        if experiment == 'plan':
            args.prefix = 'plans/astar_multi'
            self._dict['prefix'] = args.prefix

            # 默认使用 Maze2D renderer
            if not hasattr(args, 'renderer'):
                args.renderer = 'utils.Maze2dRenderer'
                self._dict['renderer'] = args.renderer

        return args


# =========================
# 提取 maze 地图的占用栅格
# =========================
def extract_occupancy(env):
    """
    Maze2D 内部包含 maze_arr

    maze_arr:
        10 -> 墙
        0  -> 可通行

    这里转换为 occupancy grid
    True  = obstacle
    False = free
    """
    if not hasattr(env, 'maze_arr'):
        raise AttributeError('Maze2D environment does not expose maze_arr')

    maze = np.asarray(env.maze_arr)

    return maze == 10


# =========================
# 将连续坐标映射到最近的 free cell
# =========================
def nearest_free_cell(position, occupancy):
    """
    输入：
        position  (连续坐标)
        occupancy (栅格地图)

    输出：
        最近的 free grid cell
    """
    free_cells = np.argwhere(~occupancy)

    rounded = np.rint(position).astype(int)

    if (
        0 <= rounded[0] < occupancy.shape[0]
        and 0 <= rounded[1] < occupancy.shape[1]
        and not occupancy[tuple(rounded)]
    ):
        return tuple(rounded.tolist())

    distances = np.sum((free_cells - position[None]) ** 2, axis=1)
    nearest = free_cells[np.argmin(distances)]

    return tuple(nearest.tolist())


# =========================
# A* 邻居生成
# =========================
def astar_neighbors(cell, occupancy, allow_diagonal=False):
    """
    给定当前 cell，生成所有可行邻居
    """
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

        yield (ni, nj), float(np.hypot(di, dj))


# =========================
# A* 启发函数
# =========================
def astar_heuristic(cell, goal):
    """
    使用欧氏距离作为启发式函数
    """
    return float(np.linalg.norm(np.subtract(cell, goal)))


# =========================
# 回溯路径
# =========================
def reconstruct_path(came_from, current):
    """
    从 goal 回溯到 start
    """
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()

    return path


# =========================
# 计算离散路径长度
# =========================
def path_length(cell_path):
    if cell_path is None or len(cell_path) < 2:
        return 0.0

    total = 0.0
    for a, b in zip(cell_path[:-1], cell_path[1:]):
        di = b[0] - a[0]
        dj = b[1] - a[1]
        total += float(np.hypot(di, dj))

    return total


# =========================
# 单路径 A* 搜索
# =========================
def astar_search_with_cost(occupancy, start, goal, allow_diagonal=False):
    """
    标准 A*，返回最短路径和其长度
    """
    open_heap = [(astar_heuristic(start, goal), 0.0, start)]

    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, current_cost

        closed.add(current)

        for neighbor, step_cost in astar_neighbors(current, occupancy, allow_diagonal):
            tentative_cost = current_cost + step_cost

            if tentative_cost >= g_score.get(neighbor, float('inf')):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_cost

            f_score = tentative_cost + astar_heuristic(neighbor, goal)
            heapq.heappush(open_heap, (f_score, tentative_cost, neighbor))

    return None, float('inf')


# =========================
# 多路径 A* 搜索
# =========================
def astar_multi_paths(
    occupancy,
    start,
    goal,
    allow_diagonal=False,
    num_paths=5,
    max_ratio=2,
):
    """
    返回多条候选路径，每条路径长度满足：
        cost(path) <= max_ratio * shortest_cost

    说明：
    1. 先用标准 A* 求最短路 shortest_cost
    2. 再做受长度上界约束的 best-first path enumeration
    3. 返回的是多条不同路径序列，不保证严格同伦不同
    """
    shortest_path, shortest_cost = astar_search_with_cost(
        occupancy,
        start,
        goal,
        allow_diagonal=allow_diagonal,
    )

    if shortest_path is None:
        return [], float('inf')

    max_cost = max_ratio * shortest_cost

    # 堆元素: (f, g, current, path)
    open_heap = [(astar_heuristic(start, goal), 0.0, start, [start])]

    results = []
    seen_complete_paths = set()

    # 用于抑制重复前缀展开
    best_prefix_cost = {(start,): 0.0}

    while open_heap and len(results) < num_paths:
        f, g, current, path = heapq.heappop(open_heap)

        if g > max_cost:
            continue

        if current == goal:
            path_key = tuple(path)
            if path_key not in seen_complete_paths:
                seen_complete_paths.add(path_key)
                results.append((path, g))
            continue

        for neighbor, step_cost in astar_neighbors(current, occupancy, allow_diagonal):
            # 避免环
            if neighbor in path:
                continue

            new_g = g + step_cost

            # 长度约束
            if new_g > max_cost:
                continue

            h = astar_heuristic(neighbor, goal)

            # 前瞻剪枝
            if new_g + h > max_cost:
                continue

            new_path = path + [neighbor]
            prefix_key = tuple(new_path)

            old_cost = best_prefix_cost.get(prefix_key, float('inf'))
            if new_g >= old_cost:
                continue

            best_prefix_cost[prefix_key] = new_g
            new_f = new_g + h

            heapq.heappush(open_heap, (new_f, new_g, neighbor, new_path))

    return results, shortest_cost


# =========================
# 路径重合率
# =========================
def path_overlap_ratio(path_a, path_b):
    """
    用 cell 集合重合率判断两条路径是否太相似
    """
    set_a = set(path_a)
    set_b = set(path_b)

    inter = len(set_a & set_b)
    denom = max(1, min(len(set_a), len(set_b)))

    return inter / denom


# =========================
# 路径多样性过滤
# =========================
def filter_diverse_paths(path_cost_list, max_overlap=0.8, max_paths=5):
    """
    从候选路径中筛出更有差异的路径
    """
    selected = []

    for path, cost in sorted(path_cost_list, key=lambda x: x[1]):
        keep = True

        for old_path, old_cost in selected:
            overlap = path_overlap_ratio(path, old_path)
            if overlap > max_overlap:
                keep = False
                break

        if keep:
            selected.append((path, cost))

        if len(selected) >= max_paths:
            break

    return selected


# =========================
# 将离散路径转为连续轨迹
# =========================
def build_sequence(observation, target, cell_path, observation_dim, steps_per_cell, horizon):
    """
    将 A* 生成的 cell path 转为连续轨迹
    """
    anchors = [np.asarray(observation[:2], dtype=np.float32)]

    # 中间节点
    if len(cell_path) > 2:
        anchors.extend(np.asarray(cell, dtype=np.float32) for cell in cell_path[1:-1])

    # 终点
    anchors.append(np.asarray(target[:2], dtype=np.float32))

    positions = [anchors[0]]

    # 在 anchors 之间插值
    for start, end in zip(anchors[:-1], anchors[1:]):
        segment = end - start
        distance = np.linalg.norm(segment)

        n_steps = max(1, int(np.ceil(distance * steps_per_cell)))

        for alpha in np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)[1:]:
            positions.append(start + alpha * segment)

    positions = np.asarray(positions, dtype=np.float32)

    # 如果轨迹过长，压缩到 horizon
    if len(positions) > horizon:
        indices = np.linspace(0, len(positions) - 1, horizon).astype(int)
        positions = positions[indices]

    # 构建完整 state 序列
    sequence = np.repeat(
        np.asarray(observation, dtype=np.float32)[None],
        len(positions),
        axis=0,
    )

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

# 加载 Maze2D 环境
env = datasets.load_environment(args.dataset)

# 初始化 renderer
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

# 起点终点 cell
start_cell = nearest_free_cell(observation[:2], occupancy)
goal_cell = nearest_free_cell(target, occupancy)

print(f'observation_xy: {np.asarray(observation[:2])}')
print(f'target_xy: {np.asarray(target[:2])}')
print(f'start_cell: {start_cell}')
print(f'goal_cell: {goal_cell}')

if start_cell == goal_cell:
    print('Warning: start_cell == goal_cell, using trivial path.')

    multi_paths = [([start_cell], 0.0)]
    shortest_cost = 0.0
else:
    num_candidates = max(
        args.astar_num_paths * args.astar_candidate_multiplier,
        args.astar_num_paths,
    )

    candidate_paths, shortest_cost = astar_multi_paths(
        occupancy,
        start_cell,
        goal_cell,
        allow_diagonal=args.astar_allow_diagonal,
        num_paths=num_candidates,
        max_ratio=args.astar_max_ratio,
    )

    if len(candidate_paths) == 0:
        raise RuntimeError(f'A* failed to find a path from {start_cell} to {goal_cell}')

    multi_paths = filter_diverse_paths(
        candidate_paths,
        max_overlap=args.astar_max_overlap,
        max_paths=args.astar_num_paths,
    )

    if len(multi_paths) == 0:
        raise RuntimeError('A* found candidate paths, but all were removed by diversity filtering.')

# 先多搜一些候选，再筛选多样路径
num_candidates = max(
    args.astar_num_paths * args.astar_candidate_multiplier,
    args.astar_num_paths,
)

candidate_paths, shortest_cost = astar_multi_paths(
    occupancy,
    start_cell,
    goal_cell,
    allow_diagonal=args.astar_allow_diagonal,
    num_paths=num_candidates,
    max_ratio=args.astar_max_ratio,
)

if len(candidate_paths) == 0:
    raise RuntimeError(f'A* failed to find a path from {start_cell} to {goal_cell}')

multi_paths = filter_diverse_paths(
    candidate_paths,
    max_overlap=args.astar_max_overlap,
    max_paths=args.astar_num_paths,
)

if len(multi_paths) == 0:
    raise RuntimeError('A* found candidate paths, but all were removed by diversity filtering.')

print('planner: astar_multi')
print(f'start_cell: {start_cell}')
print(f'goal_cell: {goal_cell}')
print(f'shortest_cost: {shortest_cost:.3f}')
print(f'max_ratio: {args.astar_max_ratio:.3f}')
print(f'max_allowed_cost: {args.astar_max_ratio * shortest_cost:.3f}')
print(f'candidate_paths_before_filter: {len(candidate_paths)}')
print(f'final_paths_after_filter: {len(multi_paths)}')

for i, (path_i, cost_i) in enumerate(multi_paths):
    if shortest_cost > 1e-8:
        ratio_str = f'{cost_i / shortest_cost:.3f}'
    else:
        ratio_str = '1.000' if abs(cost_i) <= 1e-8 else 'inf'

    print(
        f'path[{i}]: cells={len(path_i)} '
        f'cost={cost_i:.3f} '
        f'ratio={ratio_str}'
    )

# 构建多条连续轨迹
observation_dim = int(np.prod(env.observation_space.shape))

all_sequences = []
for cell_path, cell_cost in multi_paths:
    sequence = build_sequence(
        observation,
        target,
        cell_path,
        observation_dim,
        steps_per_cell=args.astar_steps_per_cell,
        horizon=args.horizon,
    )
    all_sequences.append(sequence)

# 为了在 Maze2D renderer 中一起显示，shape 需要是 (N, T, obs_dim)
# 如果不同 sequence 长度不同，这里做 padding 到同一长度
max_len = max(seq.shape[0] for seq in all_sequences)

padded_sequences = []
for seq in all_sequences:
    if seq.shape[0] < max_len:
        pad_len = max_len - seq.shape[0]
        pad = np.repeat(seq[-1:],
                        pad_len,
                        axis=0)
        seq = np.concatenate([seq, pad], axis=0)
    padded_sequences.append(seq)

plan_observations = np.stack(padded_sequences, axis=0)

print(f'num_rendered_paths: {plan_observations.shape[0]}')
print(f'plan_horizon: {plan_observations.shape[1]}')
print(f'obs_dim: {plan_observations.shape[2]}')

# 保存 Maze2D 上的多路径图
save_path = join(args.savepath, 'astar_multi_plan.png')
renderer.composite(save_path, plan_observations, ncol=1)

print(f'saved plan image to: {save_path}')