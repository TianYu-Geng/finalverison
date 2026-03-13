import heapq
import numpy as np


def extract_occupancy(env):
    if not hasattr(env, 'maze_arr'):
        raise AttributeError('Maze2D environment does not expose maze_arr')
    maze = np.asarray(env.maze_arr)
    return maze == 10


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


def astar_heuristic(cell, goal):
    return float(np.linalg.norm(np.subtract(cell, goal)))


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


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


def astar_multi_paths(
    occupancy,
    start,
    goal,
    allow_diagonal=False,
    num_paths=5,
    max_ratio=2.0,
):
    shortest_path, shortest_cost = astar_search_with_cost(
        occupancy, start, goal, allow_diagonal=allow_diagonal
    )

    if shortest_path is None:
        return [], float('inf')

    max_cost = max_ratio * shortest_cost
    open_heap = [(astar_heuristic(start, goal), 0.0, start, [start])]

    results = []
    seen_complete_paths = set()
    best_prefix_cost = {(start,): 0.0}

    while open_heap and len(results) < num_paths:
        _, g, current, path = heapq.heappop(open_heap)

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
            heapq.heappush(open_heap, (new_g + h, new_g, neighbor, new_path))

    return results, shortest_cost


def path_overlap_ratio(path_a, path_b):
    set_a = set(path_a)
    set_b = set(path_b)
    inter = len(set_a & set_b)
    denom = max(1, min(len(set_a), len(set_b)))
    return inter / denom


def filter_diverse_paths(path_cost_list, max_overlap=0.8, max_paths=5):
    selected = []
    for path, cost in sorted(path_cost_list, key=lambda x: x[1]):
        keep = True
        for old_path, _ in selected:
            if path_overlap_ratio(path, old_path) > max_overlap:
                keep = False
                break
        if keep:
            selected.append((path, cost))
        if len(selected) >= max_paths:
            break
    return selected


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

    sequence = np.repeat(np.asarray(observation, dtype=np.float32)[None], len(positions), axis=0)
    sequence[:, :2] = positions

    if observation_dim >= 4:
        velocities = np.zeros((len(positions), 2), dtype=np.float32)
        velocities[:-1] = positions[1:] - positions[:-1]
        sequence[:, 2:4] = velocities

    return sequence