import numpy as np
import torch

from .types import BranchStructField, PGProposal, StructuredPrior


def _cfg(config, name, default):
    return getattr(config, name, default)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _cached_map_tensor(map_prior, key, ref):
    cache = map_prior.debug_info.setdefault("_tensor_cache", {})
    cache_key = (key, str(ref.device), str(ref.dtype))
    value = cache.get(cache_key)
    if value is None:
        value = torch.as_tensor(getattr(map_prior, key), device=ref.device, dtype=ref.dtype)
        cache[cache_key] = value
    return value


def _cached_branch_tensor(branch_field: BranchStructField, key, ref):
    cache = branch_field.debug_info.setdefault("_tensor_cache", {})
    cache_key = (key, str(ref.device), str(ref.dtype))
    value = cache.get(cache_key)
    if value is None:
        value = torch.as_tensor(getattr(branch_field, key), device=ref.device, dtype=ref.dtype)
        cache[cache_key] = value
    return value


def _bilinear_sample_grid_torch(arr, x, y, scale):
    hh, ww = arr.shape
    c = (x + 0.5) * scale - 0.5
    r = (y + 0.5) * scale - 0.5

    c0 = torch.floor(c).long()
    r0 = torch.floor(r).long()
    c1 = c0 + 1
    r1 = r0 + 1

    c0 = c0.clamp(0, ww - 1)
    c1 = c1.clamp(0, ww - 1)
    r0 = r0.clamp(0, hh - 1)
    r1 = r1.clamp(0, hh - 1)

    dc = c - c0.to(c.dtype)
    dr = r - r0.to(r.dtype)

    v00 = arr[r0, c0]
    v01 = arr[r0, c1]
    v10 = arr[r1, c0]
    v11 = arr[r1, c1]
    return (
        (1.0 - dr) * (1.0 - dc) * v00
        + (1.0 - dr) * dc * v01
        + dr * (1.0 - dc) * v10
        + dr * dc * v11
    )


def score_pg_candidates_with_support(pg_proposal: PGProposal, map_prior, normalizer, config):
    candidates = pg_proposal.candidates_future_obs
    mean_xy = torch.as_tensor(normalizer.mean[:2], device=candidates.device, dtype=candidates.dtype)
    std_xy = torch.as_tensor(normalizer.std[:2], device=candidates.device, dtype=candidates.dtype).clamp_min(1e-6)
    xy = candidates[:, :, :2] * std_xy.view(1, 1, 2) + mean_xy.view(1, 1, 2)

    y = xy[:, :, 0]
    x = xy[:, :, 1]
    boundary_clip = float(_cfg(config, "prior_struct_boundary_clip", 0.5))
    scale = map_prior.scale

    support_soft_field = _cached_map_tensor(map_prior, "support_soft_hr", candidates)
    support_field = _cached_map_tensor(map_prior, "support_hr", candidates)
    boundary_field = _cached_map_tensor(map_prior, "distance_to_boundary_hr", candidates)

    support_soft_vals = _bilinear_sample_grid_torch(support_soft_field, x, y, scale)
    inside_vals = _bilinear_sample_grid_torch(support_field, x, y, scale)
    boundary_vals = _bilinear_sample_grid_torch(boundary_field, x, y, scale).clamp_max(boundary_clip)

    horizon = support_soft_vals.shape[1]
    time_weights = torch.linspace(
        float(_cfg(config, "prior_struct_time_weight_start", 0.8)),
        float(_cfg(config, "prior_struct_time_weight_end", 1.2)),
        horizon,
        device=candidates.device,
        dtype=candidates.dtype,
    )
    time_weights = time_weights / time_weights.sum().clamp_min(1e-6)

    support_soft_means = torch.sum(support_soft_vals * time_weights.view(1, -1), dim=1)
    inside_ratios = torch.mean(inside_vals, dim=1)
    boundary_scores = torch.mean(boundary_vals, dim=1)
    endpoint_support = support_soft_vals[:, -1]

    start_xy = torch.as_tensor(map_prior.start_xy, device=candidates.device, dtype=candidates.dtype)
    goal_xy = torch.as_tensor(map_prior.goal_xy, device=candidates.device, dtype=candidates.dtype)
    start_goal_dist = torch.linalg.norm(goal_xy - start_xy).clamp_min(1e-6)
    end_goal_dist = torch.linalg.norm(xy[:, -1, :] - goal_xy.view(1, 2), dim=1)
    min_goal_dist = torch.min(torch.linalg.norm(xy - goal_xy.view(1, 1, 2), dim=2), dim=1).values
    progress_scores = (start_goal_dist - end_goal_dist) / start_goal_dist
    near_goal_scores = 1.0 - torch.clamp(min_goal_dist / start_goal_dist, 0.0, 1.5)

    support_scores = (
        float(_cfg(config, "prior_struct_support_soft_weight", 1.0)) * support_soft_means
        + float(_cfg(config, "prior_struct_inside_ratio_weight", 2.0)) * inside_ratios
        + float(_cfg(config, "prior_struct_boundary_weight", 0.5)) * boundary_scores
        + float(_cfg(config, "prior_struct_endpoint_support_weight", 1.0)) * endpoint_support
        + float(_cfg(config, "prior_struct_progress_weight", 1.2)) * progress_scores
        + float(_cfg(config, "prior_struct_goal_closeness_weight", 1.5)) * near_goal_scores
    )

    return support_scores, {
        "inside_ratios": _to_numpy(inside_ratios),
        "boundary_scores": _to_numpy(boundary_scores),
        "support_soft_means": _to_numpy(support_soft_means),
        "endpoint_support": _to_numpy(endpoint_support),
        "progress_scores": _to_numpy(progress_scores),
        "near_goal_scores": _to_numpy(near_goal_scores),
        "end_goal_dist": _to_numpy(end_goal_dist),
    }


def score_pg_candidates_by_branch(pg_proposal: PGProposal, map_prior, normalizer, config):
    candidates = pg_proposal.candidates_future_obs
    mean_xy = torch.as_tensor(normalizer.mean[:2], device=candidates.device, dtype=candidates.dtype)
    std_xy = torch.as_tensor(normalizer.std[:2], device=candidates.device, dtype=candidates.dtype).clamp_min(1e-6)
    xy = candidates[:, :, :2] * std_xy.view(1, 1, 2) + mean_xy.view(1, 1, 2)
    y = xy[:, :, 0]
    x = xy[:, :, 1]

    horizon = candidates.shape[1]
    time_weights = torch.linspace(
        float(_cfg(config, "prior_struct_time_weight_start", 0.8)),
        float(_cfg(config, "prior_struct_time_weight_end", 1.2)),
        horizon,
        device=candidates.device,
        dtype=candidates.dtype,
    )
    time_weights = time_weights / time_weights.sum().clamp_min(1e-6)

    per_branch_scores = []
    per_branch_debug = {
        "branch_support_means": [],
        "branch_inside_ratios": [],
        "branch_boundary_scores": [],
        "branch_progress_scores": [],
        "branch_goal_scores": [],
        "branch_path_cost_penalty": [],
    }
    boundary_clip = float(_cfg(config, "prior_struct_boundary_clip", 0.5))
    start_xy = torch.as_tensor(map_prior.start_xy, device=candidates.device, dtype=candidates.dtype)
    goal_xy = torch.as_tensor(map_prior.goal_xy, device=candidates.device, dtype=candidates.dtype)
    start_goal_dist = torch.linalg.norm(goal_xy - start_xy).clamp_min(1e-6)
    branch_costs = torch.as_tensor(
        [float(branch.path_cost) for branch in map_prior.branch_fields],
        device=candidates.device,
        dtype=candidates.dtype,
    )
    branch_costs = branch_costs / branch_costs.min().clamp_min(1e-6)

    for branch_id, branch_field in enumerate(map_prior.branch_fields):
        support_soft_field = _cached_branch_tensor(branch_field, "support_soft_hr", candidates)
        support_field = _cached_branch_tensor(branch_field, "support_hr", candidates)
        boundary_field = _cached_branch_tensor(branch_field, "distance_to_boundary_hr", candidates)

        support_soft_vals = _bilinear_sample_grid_torch(support_soft_field, x, y, map_prior.scale)
        inside_vals = _bilinear_sample_grid_torch(support_field, x, y, map_prior.scale)
        boundary_vals = _bilinear_sample_grid_torch(boundary_field, x, y, map_prior.scale).clamp_max(boundary_clip)

        support_soft_means = torch.sum(support_soft_vals * time_weights.view(1, -1), dim=1)
        inside_ratios = torch.mean(inside_vals, dim=1)
        boundary_scores = torch.mean(boundary_vals, dim=1)
        endpoint_support = support_soft_vals[:, -1]
        end_goal_dist = torch.linalg.norm(xy[:, -1, :] - goal_xy.view(1, 2), dim=1)
        min_goal_dist = torch.min(torch.linalg.norm(xy - goal_xy.view(1, 1, 2), dim=2), dim=1).values
        progress_scores = (start_goal_dist - end_goal_dist) / start_goal_dist
        goal_scores = 1.0 - torch.clamp(min_goal_dist / start_goal_dist, 0.0, 1.5)
        path_cost_penalty = branch_costs[branch_id].expand_as(progress_scores)

        branch_score = (
            float(_cfg(config, "prior_struct_support_soft_weight", 1.0)) * support_soft_means
            + float(_cfg(config, "prior_struct_inside_ratio_weight", 2.0)) * inside_ratios
            + float(_cfg(config, "prior_struct_boundary_weight", 0.5)) * boundary_scores
            + float(_cfg(config, "prior_struct_endpoint_support_weight", 1.0)) * endpoint_support
            + float(_cfg(config, "prior_struct_branch_progress_weight", 1.0)) * progress_scores
            + float(_cfg(config, "prior_struct_branch_goal_weight", 1.0)) * goal_scores
            - float(_cfg(config, "prior_struct_branch_path_cost_weight", 0.35)) * path_cost_penalty
        )
        per_branch_scores.append(branch_score)
        per_branch_debug["branch_support_means"].append(_to_numpy(support_soft_means))
        per_branch_debug["branch_inside_ratios"].append(_to_numpy(inside_ratios))
        per_branch_debug["branch_boundary_scores"].append(_to_numpy(boundary_scores))
        per_branch_debug["branch_progress_scores"].append(_to_numpy(progress_scores))
        per_branch_debug["branch_goal_scores"].append(_to_numpy(goal_scores))
        per_branch_debug["branch_path_cost_penalty"].append(_to_numpy(path_cost_penalty))

    branch_scores = torch.stack(per_branch_scores, dim=1)
    selected_branch_ids = torch.argmax(branch_scores, dim=1)
    selected_branch_scores = branch_scores.gather(1, selected_branch_ids.unsqueeze(1)).squeeze(1)
    per_branch_debug["selected_branch_ids"] = _to_numpy(selected_branch_ids)
    per_branch_debug["branch_scores"] = _to_numpy(branch_scores)
    per_branch_debug["selected_branch_scores"] = _to_numpy(selected_branch_scores)
    return branch_scores, selected_branch_ids, selected_branch_scores, per_branch_debug


def build_structured_prior(pg_proposal: PGProposal, map_prior, normalizer, config, planner_batch_size):
    score_support, support_debug = score_pg_candidates_with_support(
        pg_proposal, map_prior, normalizer, config
    )
    branch_scores, selected_branch_ids, score_branch, branch_debug = score_pg_candidates_by_branch(
        pg_proposal, map_prior, normalizer, config
    )

    score_pg = pg_proposal.score_pg.to(score_support.device, dtype=score_support.dtype)
    score_total = (
        float(_cfg(config, "prior_struct_pg_weight", 0.15)) * score_pg
        + float(_cfg(config, "prior_struct_support_weight", 1.0)) * score_support
        + float(_cfg(config, "prior_struct_support_weight", 1.0)) * score_branch
    )
    topk = min(int(planner_batch_size), int(score_total.shape[0]))
    topk_indices = torch.argsort(score_total, descending=True)[:topk]
    selected_index = int(topk_indices[0].item())
    selected_future_obs = pg_proposal.candidates_future_obs[topk_indices]
    fused_prior_state = selected_future_obs
    selected_branch_id = int(selected_branch_ids[selected_index].item())
    selected_branch_ids_topk = selected_branch_ids[topk_indices]
    selected_branch_field = map_prior.branch_fields[selected_branch_id]
    normalizer_mean = _to_numpy(normalizer.mean)
    normalizer_std = _to_numpy(normalizer.std)

    return StructuredPrior(
        support=map_prior,
        pg_density=pg_proposal,
        fused_prior_state=fused_prior_state,
        selected_future_obs=selected_future_obs,
        score_support=score_support,
        score_total=score_total,
        score_branch=score_branch,
        branch_scores=branch_scores,
        selected_index=selected_index,
        selected_indices=[int(x) for x in _to_numpy(topk_indices).tolist()],
        selected_branch_id=selected_branch_id,
        selected_branch_ids=[int(x) for x in _to_numpy(selected_branch_ids_topk).tolist()],
        guidance={
            "type": "branch_corridor_guidance",
            "enabled": bool(_cfg(config, "prior_struct_enable_guidance", False)),
            "corridor_start_ratio": float(_cfg(config, "corridor_start_ratio", 0.5)),
            "strength": float(_cfg(config, "prior_struct_guidance_strength", 0.02)),
            "max_step": float(_cfg(config, "prior_struct_guidance_max_step", 0.03)),
            "boundary_margin": float(_cfg(config, "prior_struct_guidance_boundary_margin", 0.15)),
            "activation_temp": float(_cfg(config, "prior_struct_guidance_activation_temp", 0.05)),
            "late_stage_power": float(_cfg(config, "prior_struct_guidance_late_stage_power", 1.5)),
            "center_pull_weight": float(_cfg(config, "prior_struct_guidance_center_pull_weight", 1.0)),
            "support_pull_weight": float(_cfg(config, "prior_struct_guidance_support_pull_weight", 1.0)),
            "boundary_pull_weight": float(_cfg(config, "prior_struct_guidance_boundary_pull_weight", 1.0)),
            "support_threshold": float(map_prior.support_threshold),
            "scale": int(map_prior.scale),
            "branch_id": selected_branch_id,
            "support_soft_hr": selected_branch_field.support_soft_hr,
            "distance_to_center_hr": selected_branch_field.distance_to_center_hr,
            "distance_to_boundary_hr": selected_branch_field.distance_to_boundary_hr,
            "grad_center_row_hr": selected_branch_field.grad_center_row_hr,
            "grad_center_col_hr": selected_branch_field.grad_center_col_hr,
            "obs_mean_xy": normalizer_mean[:2],
            "obs_std_xy": normalizer_std[:2],
        },
        debug_info={
            **support_debug,
            **branch_debug,
            "topk_indices": [int(x) for x in _to_numpy(topk_indices).tolist()],
            "selected_branch_id": selected_branch_id,
            "selected_branch_ids_topk": [int(x) for x in _to_numpy(selected_branch_ids_topk).tolist()],
        },
        selected_future_obs_world=None,
    )
