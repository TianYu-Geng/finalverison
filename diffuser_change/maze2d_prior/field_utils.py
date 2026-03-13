import numpy as np
from .geometry_utils import build_highres_clearance, point_to_polyline_distance


def softmax_stable(x, axis=0):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


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

            sigma_eff = max(line_sigma * sigma_hr[r, c], 1e-4)
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
    for cell_path, _ in multi_paths:
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
    obstacle_fill = float(np.max(potential_hr[free_hr])) if np.any(free_hr) else 0.0
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
        responsibilities.append(float(bilinear_sample(field_dict['resp_stack'][k], x, y, scale)))

    return {
        'prior': float(prior),
        'potential': float(potential),
        'grad_x': float(grad_x),
        'grad_y': float(grad_y),
        'responsibilities': responsibilities,
    }