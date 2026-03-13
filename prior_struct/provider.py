import numpy as np

from .fusion import build_structured_prior
from .map_prior import build_map_struct_prior_from_occupancy, get_start_goal_cells
from .pg_adapter import generate_pg_proposals


def build_prior_struct(
    prior_model,
    observation_t,
    env,
    normalizer,
    config,
    planner_batch_size,
    eval_deterministic=True,
    runtime_state=None,
):
    runtime_state = {} if runtime_state is None else runtime_state
    obs_mean = runtime_state.get("obs_mean")
    obs_std = runtime_state.get("obs_std")
    if obs_mean is None or obs_std is None:
        obs_mean = np.asarray(normalizer.mean, dtype=np.float32)
        obs_std = np.asarray(normalizer.std, dtype=np.float32)
        runtime_state["obs_mean"] = obs_mean
        runtime_state["obs_std"] = obs_std
    observation_world = observation_t.detach().cpu().numpy() * obs_std + obs_mean
    goal_xy = env.unwrapped._target
    if hasattr(goal_xy, "detach"):
        goal_xy = goal_xy.detach().cpu().numpy()
    else:
        goal_xy = np.asarray(goal_xy, dtype=np.float32)

    occupancy = runtime_state.get("occupancy")
    occupancy, start_cell, goal_cell = get_start_goal_cells(
        env=env,
        start_xy=observation_world[:2],
        goal_xy=goal_xy,
        occupancy=occupancy,
    )
    runtime_state["occupancy"] = occupancy

    step_index = int(runtime_state.get("step_index", 0))
    rebuild_interval = int(getattr(config, "prior_struct_rebuild_interval", 5))
    map_prior = runtime_state.get("map_prior")
    map_prior_cache = runtime_state.setdefault("map_prior_cache", {})
    last_start_cell = runtime_state.get("start_cell")
    last_goal_cell = runtime_state.get("goal_cell")

    should_rebuild = (
        map_prior is None
        or last_goal_cell != goal_cell
        or (last_start_cell != start_cell and getattr(config, "prior_struct_rebuild_on_cell_change", True))
        or step_index % max(rebuild_interval, 1) == 0
    )

    if should_rebuild:
        cache_key = (
            tuple(start_cell),
            tuple(goal_cell),
            int(getattr(config, "prior_struct_scale", 12)),
            int(getattr(config, "prior_struct_astar_num_paths", 5)),
            bool(getattr(config, "prior_struct_astar_allow_diagonal", False)),
            float(getattr(config, "prior_struct_astar_max_ratio", 2.0)),
            float(getattr(config, "prior_struct_astar_max_overlap", 0.8)),
            float(getattr(config, "prior_struct_support_threshold", 0.08)),
            float(getattr(config, "prior_struct_line_sigma", 0.36)),
            float(getattr(config, "prior_struct_beta", 1.1)),
            float(getattr(config, "prior_struct_sigma_min", 0.8)),
            float(getattr(config, "prior_struct_sigma_max", 2.8)),
            float(getattr(config, "prior_struct_tau", 0.35)),
        )
        map_prior = map_prior_cache.get(cache_key)
        if map_prior is None:
            map_prior = build_map_struct_prior_from_occupancy(
                occupancy=occupancy,
                start_cell=start_cell,
                goal_cell=goal_cell,
                config=config,
            )
            map_prior_cache[cache_key] = map_prior
        runtime_state["map_prior"] = map_prior
        runtime_state["rebuild_count"] = int(runtime_state.get("rebuild_count", 0)) + 1

    map_prior.start_xy = observation_world[:2].astype("float32", copy=True)
    map_prior.goal_xy = goal_xy.astype("float32", copy=True)

    runtime_state["step_index"] = step_index + 1
    runtime_state["start_cell"] = start_cell
    runtime_state["goal_cell"] = goal_cell

    pg_proposal = generate_pg_proposals(
        prior_model=prior_model,
        observation_t=observation_t,
        config=config,
        eval_deterministic=eval_deterministic,
    )
    return build_structured_prior(
        pg_proposal=pg_proposal,
        map_prior=map_prior,
        normalizer=normalizer,
        config=config,
        planner_batch_size=planner_batch_size,
    )
