from types import SimpleNamespace

import numpy as np
from scipy.ndimage import distance_transform_edt

from diffuser_change.maze2d_prior.astar_utils import (
    astar_multi_paths,
    extract_occupancy,
    filter_diverse_paths,
    nearest_free_cell,
)
from diffuser_change.maze2d_prior.field_utils import (
    bilinear_sample,
    build_gated_line_centered_prior_and_potential,
)

from .types import MapStructPrior


def _cfg(config, name, default):
    return getattr(config, name, default)


def get_start_goal_cells(env, start_xy, goal_xy, occupancy=None):
    if occupancy is None:
        occupancy = extract_occupancy(env.unwrapped if hasattr(env, "unwrapped") else env)
    start_cell = nearest_free_cell(start_xy, occupancy)
    goal_cell = nearest_free_cell(goal_xy, occupancy)
    return occupancy, start_cell, goal_cell


def build_map_struct_prior_from_occupancy(occupancy, start_cell, goal_cell, config):
    occupancy = np.asarray(occupancy, dtype=bool)

    if start_cell == goal_cell:
        multi_paths = [([start_cell], 0.0)]
    else:
        num_paths = _cfg(config, "prior_struct_astar_num_paths", 5)
        num_candidates = max(
            num_paths * _cfg(config, "prior_struct_astar_candidate_multiplier", 4),
            num_paths,
        )
        candidate_paths, _ = astar_multi_paths(
            occupancy=occupancy,
            start=start_cell,
            goal=goal_cell,
            allow_diagonal=_cfg(config, "prior_struct_astar_allow_diagonal", False),
            num_paths=num_candidates,
            max_ratio=_cfg(config, "prior_struct_astar_max_ratio", 2.0),
        )
        if len(candidate_paths) == 0:
            raise RuntimeError(f"A* failed to find a path from {start_cell} to {goal_cell}")
        multi_paths = filter_diverse_paths(
            candidate_paths,
            max_overlap=_cfg(config, "prior_struct_astar_max_overlap", 0.8),
            max_paths=num_paths,
        )
        if len(multi_paths) == 0:
            raise RuntimeError("A* candidates were all removed by diversity filtering.")

    field_dict = build_gated_line_centered_prior_and_potential(
        occupancy=occupancy,
        multi_paths=multi_paths,
        scale=_cfg(config, "prior_struct_scale", 12),
        line_sigma=_cfg(config, "prior_struct_line_sigma", 0.36),
        beta=_cfg(config, "prior_struct_beta", 1.1),
        sigma_min=_cfg(config, "prior_struct_sigma_min", 0.8),
        sigma_max=_cfg(config, "prior_struct_sigma_max", 2.8),
        tau=_cfg(config, "prior_struct_tau", 0.35),
        eps=_cfg(config, "prior_struct_eps", 1e-6),
    )

    support_soft_hr = field_dict["fused_prior_hr"]
    support_threshold = _cfg(config, "prior_struct_support_threshold", 0.08)
    support_hr = support_soft_hr >= support_threshold
    dist_stack = field_dict["dist_stack"]
    distance_to_center_hr = np.min(dist_stack, axis=0).astype(np.float32)

    support_free = support_hr & (~field_dict["occ_hr"])
    distance_to_boundary_hr = distance_transform_edt(support_free).astype(np.float32) / float(field_dict["scale"])
    distance_to_boundary_hr[~support_free] = 0.0

    centerline_potential_hr = distance_to_center_hr.copy()
    if np.any(support_free):
        obstacle_fill = float(np.max(centerline_potential_hr[support_free]))
    else:
        obstacle_fill = 0.0
    centerline_potential_hr[~support_free] = obstacle_fill

    grad_center_row_hr, grad_center_col_hr = np.gradient(centerline_potential_hr)
    grad_center_row_hr[~support_free] = 0.0
    grad_center_col_hr[~support_free] = 0.0

    return MapStructPrior(
        occupancy=occupancy,
        start_xy=np.asarray(start_cell, dtype=np.float32),
        goal_xy=np.asarray(goal_cell, dtype=np.float32),
        start_cell=start_cell,
        goal_cell=goal_cell,
        multi_paths=multi_paths,
        support_hr=support_hr.astype(np.float32),
        support_soft_hr=support_soft_hr.astype(np.float32),
        distance_to_center_hr=distance_to_center_hr,
        distance_to_boundary_hr=distance_to_boundary_hr,
        centerline_potential_hr=centerline_potential_hr.astype(np.float32),
        grad_center_row_hr=grad_center_row_hr.astype(np.float32),
        grad_center_col_hr=grad_center_col_hr.astype(np.float32),
        scale=int(field_dict["scale"]),
        support_threshold=float(support_threshold),
        debug_info={"field_dict": field_dict},
    )


def build_map_struct_prior(env, start_xy, goal_xy, config):
    occupancy, start_cell, goal_cell = get_start_goal_cells(env, start_xy, goal_xy)
    map_prior = build_map_struct_prior_from_occupancy(
        occupancy=occupancy,
        start_cell=start_cell,
        goal_cell=goal_cell,
        config=config,
    )
    map_prior.start_xy = np.asarray(start_xy, dtype=np.float32)
    map_prior.goal_xy = np.asarray(goal_xy, dtype=np.float32)
    return map_prior


def query_map_prior(map_prior: MapStructPrior, x, y):
    scale = map_prior.scale
    support_soft = bilinear_sample(map_prior.support_soft_hr, x, y, scale)
    support_hard = bilinear_sample(map_prior.support_hr, x, y, scale)
    center_distance = bilinear_sample(map_prior.distance_to_center_hr, x, y, scale)
    boundary_distance = bilinear_sample(map_prior.distance_to_boundary_hr, x, y, scale)
    grad_y = bilinear_sample(map_prior.grad_center_row_hr, x, y, scale)
    grad_x = bilinear_sample(map_prior.grad_center_col_hr, x, y, scale)
    return SimpleNamespace(
        support_soft=float(support_soft),
        support_hard=float(support_hard),
        center_distance=float(center_distance),
        boundary_distance=float(boundary_distance),
        grad_x=float(grad_x),
        grad_y=float(grad_y),
    )
