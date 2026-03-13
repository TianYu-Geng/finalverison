import heapq
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

import diffuser.datasets as datasets
import diffuser.utils as utils


# =========================
# 可视化：墙体 + 离散路径
# =========================
def render_walls_and_paths(
    occupancy,
    multi_paths,
    observation,
    target,
    save_path,
    draw_cells=True,
    draw_polyline=False,
):
    h, w = occupancy.shape

    fig, ax = plt.subplots(figsize=(6, 6))

    wall_img = np.where(occupancy, 0.0, 1.0)
    ax.imshow(
        wall_img,
        cmap='gray',
        origin='lower',
        extent=[-0.5, w - 0.5, -0.5, h - 0.5],
    )

    for cell_path, cost in multi_paths:
        pts = np.asarray(cell_path, dtype=np.float32)
        xs = pts[:, 1]
        ys = pts[:, 0]

        if draw_polyline:
            ax.plot(xs, ys, linewidth=2)

        if draw_cells:
            ax.scatter(xs, ys, s=25)

    start_xy = np.asarray(observation[:2], dtype=np.float32)
    goal_xy = np.asarray(target[:2], dtype=np.float32)

    ax.scatter(start_xy[1], start_xy[0], s=120, marker='o', label='start')
    ax.scatter(goal_xy[1], goal_xy[0], s=120, marker='*', label='goal')

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_aspect('equal')
    ax.set_title('Walls + Planned Paths')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# =========================
# 可视化：高分辨率 line-centered prior
# =========================
def render_highres_line_centered_prior(
    occupancy,
    fused_prior_hr,
    scale,
    multi_paths,
    observation,
    target,
    save_path,
):
    h, w = occupancy.shape
    hh, ww = fused_prior_hr.shape

    fig, ax = plt.subplots(figsize=(7, 7))

    wall_hr = np.ones((hh, ww), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            if occupancy[i, j]:
                wall_hr[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale] = 0.15

    ax.imshow(
        wall_hr,
        cmap='gray',
        origin='lower',
        extent=[-0.5, w - 0.5, -0.5, h - 0.5],
    )

    show_map = np.ma.masked_where(wall_hr < 0.2, fused_prior_hr)
    im = ax.imshow(
        show_map,
        cmap='jet',
        origin='lower',
        alpha=0.72,
        extent=[-0.5, w - 0.5, -0.5, h - 0.5],
        interpolation='bilinear',
    )

    for cell_path, _ in multi_paths:
        pts = np.asarray(cell_path, dtype=np.float32)
        xs = pts[:, 1]
        ys = pts[:, 0]
        ax.plot(xs, ys, linewidth=1.5)

    start_xy = np.asarray(observation[:2], dtype=np.float32)
    goal_xy = np.asarray(target[:2], dtype=np.float32)

    ax.scatter(start_xy[1], start_xy[0], s=120, marker='o', label='start')
    ax.scatter(goal_xy[1], goal_xy[0], s=140, marker='*', label='goal')

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_aspect('equal')
    ax.set_title('Line-Centered Gated Soft Corridor Prior')
    ax.legend()

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('prior score')

    plt.tight_layout()
    plt.savefig(save_path, dpi=240)
    plt.close(fig)


# =========================
# 可视化：potential + guidance direction
# =========================
def render_potential_field(
    occupancy,
    potential_hr,
    grad_row_hr,
    grad_col_hr,
    scale,
    multi_paths,
    observation,
    target,
    save_path,
    quiver_stride=10,
):
    h, w = occupancy.shape
    hh, ww = potential_hr.shape

    fig, ax = plt.subplots(figsize=(7, 7))

    wall_hr = np.ones((hh, ww), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            if occupancy[i, j]:
                wall_hr[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale] = 0.15

    ax.imshow(
        wall_hr,
        cmap='gray',
        origin='lower',
        extent=[-0.5, w - 0.5, -0.5, h - 0.5],
    )

    show_map = np.ma.masked_where(wall_hr < 0.2, potential_hr)
    im = ax.imshow(
        show_map,
        cmap='viridis',
        origin='lower',
        alpha=0.70,
        extent=[-0.5, w - 0.5, -0.5, h - 0.5],
        interpolation='bilinear',
    )

    rr = np.arange(0, hh, quiver_stride)
    cc = np.arange(0, ww, quiver_stride)
    RR, CC = np.meshgrid(rr, cc, indexing='ij')

    X = (CC + 0.5) / scale - 0.5
    Y = (RR + 0.5) / scale - 0.5

    U = -grad_col_hr[RR, CC]
    V = -grad_row_hr[RR, CC]

    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=12, width=0.002)

    for cell_path, _ in multi_paths:
        pts = np.asarray(cell_path, dtype=np.float32)
        xs = pts[:, 1]
        ys = pts[:, 0]
        ax.plot(xs, ys, linewidth=1.5)

    start_xy = np.asarray(observation[:2], dtype=np.float32)
    goal_xy = np.asarray(target[:2], dtype=np.float32)

    ax.scatter(start_xy[1], start_xy[0], s=120, marker='o', label='start')
    ax.scatter(goal_xy[1], goal_xy[0], s=140, marker='*', label='goal')

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_aspect('equal')
    ax.set_title('Potential Field and Guidance Direction')
    ax.legend()

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('potential')

    plt.tight_layout()
    plt.savefig(save_path, dpi=240)
    plt.close(fig)


# =========================
# 参数解析器
# =========================
class Parser(utils.Parser):
    dataset: str = 'maze2d-large-dense-v1'
    config: str = 'config.maze2d'

    # A*
    astar_allow_diagonal: bool = False
    astar_steps_per_cell: int = 8
    astar_num_paths: int = 5
    astar_max_ratio: float = 2.0
    astar_max_overlap: float = 0.8
    astar_candidate_multiplier: int = 4

    # high-res line-centered prior / potential
    prior_beta: float = 1.1
    prior_sigma_min: float = 0.8
    prior_sigma_max: float = 2.8
    prior_scale: int = 12
    prior_line_sigma: float = 0.36
    prior_tau: float = 0.35
    prior_eps: float = 1e-6

    def read_config(self, args, experiment):
        args = super().read_config(args, experiment)

        if experiment == 'plan':
            args.prefix = 'plans/astar_multi'
            self._dict['prefix'] = args.prefix

            if not hasattr(args, 'renderer'):
                args.renderer = 'utils.Maze2dRenderer'
                self._dict['renderer'] = args.renderer

        return args


# =========================
# 提取占用栅格
# =========================
def extract_occupancy(env):
    if not hasattr(env, 'maze_arr'):
        raise AttributeError('Maze2D environment does not expose maze_arr')

    maze = np.asarray(env.maze_arr)
    return maze == 10


# =========================
# 最近 free cell
# =========================
def nearest_free_cell(position, occupancy):
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
# A* 邻居
# =========================
def astar_neighbors(cell, occupancy, allow_diagonal=False):
    if allow_diagonal:
        deltas = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    else:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    h, w = occupancy.shape

    for di, dj in deltas:
        ni, nj = cell[0] + di, cell[1] + dj

        if ni < 0 or nj < 0 or ni >= h or nj >= w:
            continue

        if occupancy[ni, nj]:
            continue

        if allow_diagonal and abs(di) + abs(dj) == 2:
            if occupancy[cell[0], nj] or occupancy[ni, cell[1]]:
                continue

        yield (ni, nj), float(np.hypot(di, dj))


# =========================
# A* heuristic
# =========================
def astar_heuristic(cell, goal):
    return float(np.linalg.norm(np.subtract(cell, goal)))


# =========================
# 回溯路径
# =========================
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# =========================
# 单路径 A*
# =========================
def astar_search_with_cost(occupancy, start, goal, allow_diagonal=False):
    open_heap = [(astar_heuristic(start, goal), 0.0, start)]

    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        if current == goal:
            return reconstruct_path(came_from, current), current_cost

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
# 多路径 A*
# =========================
def astar_multi_paths(
    occupancy,
    start,
    goal,
    allow_diagonal=False,
    num_paths=5,
    max_ratio=2.0,
):
    shortest_path, shortest_cost = astar_search_with_cost(
        occupancy,
        start,
        goal,
        allow_diagonal=allow_diagonal,
    )

    if shortest_path is None:
        return [], float('inf')

    max_cost = max_ratio * shortest_cost
    open_heap = [(astar_heuristic(start, goal), 0.0, start, [start])]

    results = []
    seen_complete_paths = set()
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
            if neighbor in path:
                continue

            new_g = g + step_cost
            if new_g > max_cost:
                continue

            h = astar_heuristic(neighbor, goal)
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
    set_a = set(path_a)
    set_b = set(path_b)
    inter = len(set_a & set_b)
    denom = max(1, min(len(set_a), len(set_b)))
    return inter / denom


# =========================
# 多样性过滤
# =========================
def filter_diverse_paths(path_cost_list, max_overlap=0.8, max_paths=5):
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
# 离散路径 -> 连续轨迹
# =========================
def build_sequence(observation, target, cell_path, observation_dim, steps_per_cell, horizon):
    anchors = [np.asarray(observation[:2], dtype=np.float32)]

    if len(cell_path) > 2:
        anchors.extend(np.asarray(cell, dtype=np.float32) for cell in cell_path[1:-1])

    anchors.append(np.asarray(target[:2], dtype=np.float32))

    positions = [anchors[0]]

    for start, end in zip(anchors[:-1], anchors[1:]):
        segment = end - start
        distance = np.linalg.norm(segment)
        n_steps = max(1, int(np.ceil(distance * steps_per_cell)))

        for alpha in np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)[1:]:
            positions.append(start + alpha * segment)

    positions = np.asarray(positions, dtype=np.float32)

    if len(positions) > horizon:
        indices = np.linspace(0, len(positions) - 1, horizon).astype(int)
        positions = positions[indices]

    sequence = np.repeat(
        np.asarray(observation, dtype=np.float32)[None],
        len(positions),
        axis=0,
    )

    sequence[:, :2] = positions

    if observation_dim >= 4:
        velocities = np.zeros((len(positions), 2), dtype=np.float32)
        velocities[:-1] = positions[1:] - positions[:-1]
        sequence[:, 2:4] = velocities

    return sequence


# =========================
# 高分辨率 occupancy / clearance
# =========================
def upsample_occupancy(occupancy, scale):
    return np.kron(
        occupancy.astype(np.uint8),
        np.ones((scale, scale), dtype=np.uint8)
    ).astype(bool)


def build_highres_clearance(occupancy, scale):
    occ_hr = upsample_occupancy(occupancy, scale)
    free_hr = ~occ_hr

    clearance_hr = distance_transform_edt(free_hr).astype(np.float32) / float(scale)
    clearance_hr[occ_hr] = 0.0
    return occ_hr, free_hr, clearance_hr


# =========================
# 连续几何距离：点到线段 / 折线
# =========================
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1

    c1 = vx * wx + vy * wy
    if c1 <= 0.0:
        return float(np.hypot(px - x1, py - y1))

    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return float(np.hypot(px - x2, py - y2))

    t = c1 / c2
    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    return float(np.hypot(px - proj_x, py - proj_y))


def point_to_polyline_distance(px, py, polyline_xy):
    if len(polyline_xy) == 1:
        x, y = polyline_xy[0]
        return float(np.hypot(px - x, py - y))

    dmin = np.inf
    for a, b in zip(polyline_xy[:-1], polyline_xy[1:]):
        d = point_to_segment_distance(px, py, a[0], a[1], b[0], b[1])
        if d < dmin:
            dmin = d
    return float(dmin)


# =========================
# 单路径 line-centered 场
# =========================
def build_single_path_line_field(
    occupancy,
    cell_path,
    scale=12,
    line_sigma=0.28,
    beta=0.9,
    sigma_min=0.8,
    sigma_max=2.5,
):
    h, w = occupancy.shape
    hh, ww = h * scale, w * scale

    occ_hr, free_hr, clearance_hr = build_highres_clearance(occupancy, scale)

    sigma_hr = beta * clearance_hr
    sigma_hr = np.clip(sigma_hr, sigma_min, sigma_max).astype(np.float32)
    sigma_hr[occ_hr] = 1.0

    polyline_xy = np.asarray([(c[1], c[0]) for c in cell_path], dtype=np.float32)

    dist_hr = np.full((hh, ww), np.inf, dtype=np.float32)
    phi_hr = np.zeros((hh, ww), dtype=np.float32)

    for r in range(hh):
        py = (r + 0.5) / scale - 0.5
        for c in range(ww):
            if occ_hr[r, c]:
                continue

            px = (c + 0.5) / scale - 0.5
            d = point_to_polyline_distance(px, py, polyline_xy)
            dist_hr[r, c] = d

            sigma_eff = line_sigma * sigma_hr[r, c]
            sigma_eff = max(sigma_eff, 1e-4)

            phi_hr[r, c] = np.exp(-(d ** 2) / (2.0 * sigma_eff ** 2))

    phi_hr[occ_hr] = 0.0
    return {
        'dist_hr': dist_hr,
        'phi_hr': phi_hr,
        'occ_hr': occ_hr,
        'free_hr': free_hr,
        'clearance_hr': clearance_hr,
        'sigma_hr': sigma_hr,
        'polyline_xy': polyline_xy,
    }


# =========================
# softmax
# =========================
def softmax_stable(x, axis=0):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


# =========================
# 多路径 gated prior + potential + gradient
# =========================
def build_gated_line_centered_prior_and_potential(
    occupancy,
    multi_paths,
    scale=12,
    line_sigma=0.28,
    beta=0.9,
    sigma_min=0.8,
    sigma_max=2.5,
    tau=0.35,
    eps=1e-6,
):
    if len(multi_paths) == 0:
        raise ValueError('multi_paths is empty')

    per_path = []
    for cell_path, cost_k in multi_paths:
        one = build_single_path_line_field(
            occupancy=occupancy,
            cell_path=cell_path,
            scale=scale,
            line_sigma=line_sigma,
            beta=beta,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        per_path.append(one)

    occ_hr = per_path[0]['occ_hr']
    free_hr = per_path[0]['free_hr']

    dist_stack = np.stack([p['dist_hr'] for p in per_path], axis=0)
    phi_stack = np.stack([p['phi_hr'] for p in per_path], axis=0)

    logits = -(dist_stack ** 2) / max(tau ** 2, 1e-8)
    logits[:, occ_hr] = -1e9
    resp_stack = softmax_stable(logits, axis=0)

    fused_prior_hr = np.sum(resp_stack * phi_stack, axis=0)
    fused_prior_hr[occ_hr] = 0.0

    maxv = float(fused_prior_hr.max())
    if maxv > 1e-8:
        fused_prior_hr /= maxv

    potential_hr = -np.log(fused_prior_hr + eps)
    if np.any(free_hr):
        obstacle_fill = float(np.max(potential_hr[free_hr]))
    else:
        obstacle_fill = 0.0
    potential_hr[occ_hr] = obstacle_fill

    grad_row_hr, grad_col_hr = np.gradient(potential_hr)
    grad_row_hr[occ_hr] = 0.0
    grad_col_hr[occ_hr] = 0.0

    return {
        'fused_prior_hr': fused_prior_hr,
        'potential_hr': potential_hr,
        'grad_row_hr': grad_row_hr,
        'grad_col_hr': grad_col_hr,
        'resp_stack': resp_stack,
        'phi_stack': phi_stack,
        'dist_stack': dist_stack,
        'per_path': per_path,
        'occ_hr': occ_hr,
        'free_hr': free_hr,
        'scale': scale,
    }


# =========================
# 双线性采样 + 查询接口
# =========================
def bilinear_sample(arr, x, y, scale):
    hh, ww = arr.shape

    c = (x + 0.5) * scale - 0.5
    r = (y + 0.5) * scale - 0.5

    c0 = int(np.floor(c))
    r0 = int(np.floor(r))
    c1 = c0 + 1
    r1 = r0 + 1

    c0 = np.clip(c0, 0, ww - 1)
    c1 = np.clip(c1, 0, ww - 1)
    r0 = np.clip(r0, 0, hh - 1)
    r1 = np.clip(r1, 0, hh - 1)

    dc = c - c0
    dr = r - r0

    v00 = arr[r0, c0]
    v01 = arr[r0, c1]
    v10 = arr[r1, c0]
    v11 = arr[r1, c1]

    return (
        (1 - dr) * (1 - dc) * v00 +
        (1 - dr) * dc * v01 +
        dr * (1 - dc) * v10 +
        dr * dc * v11
    )


def query_guidance_field(field_dict, x, y):
    scale = field_dict['scale']

    prior = bilinear_sample(field_dict['fused_prior_hr'], x, y, scale)
    potential = bilinear_sample(field_dict['potential_hr'], x, y, scale)
    grad_y = bilinear_sample(field_dict['grad_row_hr'], x, y, scale)
    grad_x = bilinear_sample(field_dict['grad_col_hr'], x, y, scale)

    responsibilities = []
    for k in range(field_dict['resp_stack'].shape[0]):
        rk = bilinear_sample(field_dict['resp_stack'][k], x, y, scale)
        responsibilities.append(float(rk))

    return {
        'prior': float(prior),
        'potential': float(potential),
        'grad_x': float(grad_x),
        'grad_y': float(grad_y),
        'responsibilities': responsibilities,
    }


# =========================
# main
# =========================
args = Parser().parse_args('plan')

env = datasets.load_environment(args.dataset)

render_config = utils.Config(args.renderer, env=args.dataset)
renderer = render_config()

observation = env.reset()

if args.conditional:
    print('Resetting target')
    env.set_target()

target = np.asarray(env._target, dtype=np.float32)
occupancy = extract_occupancy(env)

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
    candidate_paths = multi_paths
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

# =========================
# 构建 gated prior + potential + gradient
# =========================
field_dict = build_gated_line_centered_prior_and_potential(
    occupancy=occupancy,
    multi_paths=multi_paths,
    scale=args.prior_scale,
    line_sigma=args.prior_line_sigma,
    beta=args.prior_beta,
    sigma_min=args.prior_sigma_min,
    sigma_max=args.prior_sigma_max,
    tau=args.prior_tau,
    eps=args.prior_eps,
)

fused_prior_hr = field_dict['fused_prior_hr']
potential_hr = field_dict['potential_hr']
grad_row_hr = field_dict['grad_row_hr']
grad_col_hr = field_dict['grad_col_hr']
resp_stack = field_dict['resp_stack']

print(f'num_paths_for_prior: {resp_stack.shape[0]}')
print(f'fused_prior_hr: min={fused_prior_hr.min():.4f}, max={fused_prior_hr.max():.4f}')
print(f'potential_hr: min={potential_hr.min():.4f}, max={potential_hr.max():.4f}')
print(f'responsibility_shape: {resp_stack.shape}')

# =========================
# 查询起点处 guidance 信息
# 注意：x=col, y=row
# 你前面画图时也是这个约定
# =========================
query_x = float(observation[1])
query_y = float(observation[0])

query_info = query_guidance_field(field_dict, x=query_x, y=query_y)

print('query_guidance_at_start:')
print(f"  prior = {query_info['prior']:.6f}")
print(f"  potential = {query_info['potential']:.6f}")
print(f"  grad_x = {query_info['grad_x']:.6f}")
print(f"  grad_y = {query_info['grad_y']:.6f}")
print(f"  responsibilities = {query_info['responsibilities']}")

# =========================
# 构建多条连续轨迹（保留给后续 diffusion / 其他用途）
# =========================
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

max_len = max(seq.shape[0] for seq in all_sequences)

padded_sequences = []
for seq in all_sequences:
    if seq.shape[0] < max_len:
        pad_len = max_len - seq.shape[0]
        pad = np.repeat(seq[-1:], pad_len, axis=0)
        seq = np.concatenate([seq, pad], axis=0)
    padded_sequences.append(seq)

plan_observations = np.stack(padded_sequences, axis=0)

print(f'num_rendered_paths: {plan_observations.shape[0]}')
print(f'plan_horizon: {plan_observations.shape[1]}')
print(f'obs_dim: {plan_observations.shape[2]}')

# =========================
# 保存图像
# =========================
save_path_paths = join(args.savepath, 'astar_multi_plan_manual.png')
render_walls_and_paths(
    occupancy=occupancy,
    multi_paths=multi_paths,
    observation=observation,
    target=target,
    save_path=save_path_paths,
    draw_cells=True,
    draw_polyline=True,
)
print(f'saved manual path image to: {save_path_paths}')

save_path_prior_hr = join(args.savepath, 'astar_line_centered_gated_prior.png')
render_highres_line_centered_prior(
    occupancy=occupancy,
    fused_prior_hr=fused_prior_hr,
    scale=args.prior_scale,
    multi_paths=multi_paths,
    observation=observation,
    target=target,
    save_path=save_path_prior_hr,
)
print(f'saved line-centered gated prior image to: {save_path_prior_hr}')

save_path_potential = join(args.savepath, 'astar_guidance_potential_field.png')
render_potential_field(
    occupancy=occupancy,
    potential_hr=potential_hr,
    grad_row_hr=grad_row_hr,
    grad_col_hr=grad_col_hr,
    scale=args.prior_scale,
    multi_paths=multi_paths,
    observation=observation,
    target=target,
    save_path=save_path_potential,
    quiver_stride=10,
)
print(f'saved guidance potential image to: {save_path_potential}')