import torch


def _as_tensor(arr, ref):
    if isinstance(arr, torch.Tensor):
        return arr.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(arr, device=ref.device, dtype=ref.dtype)


def _cached_guidance_tensor(guidance, key, ref):
    cache = guidance.setdefault("_tensor_cache", {})
    cache_key = (key, str(ref.device), str(ref.dtype))
    value = cache.get(cache_key)
    if value is None:
        value = _as_tensor(guidance[key], ref)
        cache[cache_key] = value
    return value


def _bilinear_sample(arr, x, y, scale):
    hh, ww = arr.shape
    c = (x + 0.5) * scale - 0.5
    r = (y + 0.5) * scale - 0.5

    c0 = torch.floor(c).long().clamp(0, ww - 1)
    r0 = torch.floor(r).long().clamp(0, hh - 1)
    c1 = (c0 + 1).clamp(0, ww - 1)
    r1 = (r0 + 1).clamp(0, hh - 1)

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


def weak_center_pull_step(xt, guidance, step_ratio=None):
    if guidance is None or not guidance.get("enabled", False):
        return xt

    grad_row = _cached_guidance_tensor(guidance, "grad_center_row_hr", xt)
    grad_col = _cached_guidance_tensor(guidance, "grad_center_col_hr", xt)
    support_soft = _cached_guidance_tensor(guidance, "support_soft_hr", xt)
    center_distance = _cached_guidance_tensor(guidance, "distance_to_center_hr", xt)
    boundary = _cached_guidance_tensor(guidance, "distance_to_boundary_hr", xt)
    mean_xy = _cached_guidance_tensor(guidance, "obs_mean_xy", xt)
    std_xy = _cached_guidance_tensor(guidance, "obs_std_xy", xt).clamp_min(1e-6)
    scale = int(guidance["scale"])

    xy_norm = xt[:, 1:, :2]
    xy_world = xy_norm * std_xy.view(1, 1, 2) + mean_xy.view(1, 1, 2)
    y = xy_world[..., 0]
    x = xy_world[..., 1]

    grad_y = _bilinear_sample(grad_row, x, y, scale)
    grad_x = _bilinear_sample(grad_col, x, y, scale)
    support_vals = _bilinear_sample(support_soft, x, y, scale)
    center_vals = _bilinear_sample(center_distance, x, y, scale)
    boundary_vals = _bilinear_sample(boundary, x, y, scale)

    boundary_margin = float(guidance.get("boundary_margin", 0.15))
    activation_temp = float(guidance.get("activation_temp", 0.05))
    support_threshold = float(guidance.get("support_threshold", 0.08))
    strength = float(guidance.get("strength", 0.02))
    max_step = float(guidance.get("max_step", 0.03))
    late_stage_power = float(guidance.get("late_stage_power", 1.5))
    center_pull_weight = float(guidance.get("center_pull_weight", 1.0))
    support_pull_weight = float(guidance.get("support_pull_weight", 1.0))
    boundary_pull_weight = float(guidance.get("boundary_pull_weight", 1.0))

    near_boundary_weight = torch.sigmoid((boundary_margin - boundary_vals) / max(activation_temp, 1e-6))
    outside_support_weight = torch.sigmoid((support_threshold - support_vals) / max(activation_temp, 1e-6))
    off_center_weight = 1.0 - torch.exp(-center_vals.clamp_min(0.0))

    weight = (
        boundary_pull_weight * near_boundary_weight
        + support_pull_weight * outside_support_weight
        + center_pull_weight * off_center_weight
    ) / max(boundary_pull_weight + support_pull_weight + center_pull_weight, 1e-6)

    corridor_ratio = float(guidance.get("corridor_start_ratio", 0.5))
    if step_ratio is not None and corridor_ratio > 0:
        late_progress = torch.tensor(
            max(0.0, min(1.0, 1.0 - step_ratio / corridor_ratio)),
            device=xt.device,
            dtype=xt.dtype,
        )
        weight = weight * late_progress.pow(late_stage_power)

    grad = torch.stack([grad_y, grad_x], dim=-1)
    grad_norm = torch.linalg.norm(grad, dim=-1, keepdim=True).clamp_min(1e-6)
    direction_to_center = -grad / grad_norm
    delta_world = direction_to_center * weight.unsqueeze(-1) * strength
    delta_world = torch.clamp(delta_world, min=-max_step, max=max_step)
    delta_norm = delta_world / std_xy.view(1, 1, 2)

    xt = xt.clone()
    xt[:, 1:, :2] = xt[:, 1:, :2] + delta_norm
    return xt
