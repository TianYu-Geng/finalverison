import matplotlib.pyplot as plt
import numpy as np


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

    for k, (cell_path, cost) in enumerate(multi_paths):
        pts = np.asarray(cell_path, dtype=np.float32)
        xs = pts[:, 1]
        ys = pts[:, 0]

        if draw_polyline:
            if k == 0:
                ax.plot(xs, ys, linewidth=2, label='A* paths')
            else:
                ax.plot(xs, ys, linewidth=2)

        if draw_cells:
            if k == 0:
                ax.scatter(xs, ys, s=25, label='path cells')
            else:
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

    for k, (cell_path, _) in enumerate(multi_paths):
        pts = np.asarray(cell_path, dtype=np.float32)
        xs = pts[:, 1]
        ys = pts[:, 0]
        if k == 0:
            ax.plot(xs, ys, linewidth=1.8, label='A* paths')
        else:
            ax.plot(xs, ys, linewidth=1.8)

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
    vmax_percentile=95,
    use_log_display=False,
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

    # free 区域 mask
    valid_mask = wall_hr >= 0.2
    valid_vals = potential_hr[valid_mask]

    if valid_vals.size > 0:
        if use_log_display:
            display_map = np.log1p(potential_hr)
            display_vals = display_map[valid_mask]
            vmax = np.percentile(display_vals, vmax_percentile)
            show_map = np.ma.masked_where(~valid_mask, display_map)
            vmin = float(display_vals.min())
        else:
            vmax = np.percentile(valid_vals, vmax_percentile)
            show_map = np.ma.masked_where(~valid_mask, potential_hr)
            vmin = float(valid_vals.min())
    else:
        show_map = np.ma.masked_where(~valid_mask, potential_hr)
        vmin = None
        vmax = None

    im = ax.imshow(
        show_map,
        cmap='viridis',
        origin='lower',
        alpha=0.70,
        extent=[-0.5, w - 0.5, -0.5, h - 0.5],
        interpolation='bilinear',
        vmin=vmin,
        vmax=vmax,
    )

    rr = np.arange(0, hh, quiver_stride)
    cc = np.arange(0, ww, quiver_stride)
    RR, CC = np.meshgrid(rr, cc, indexing='ij')

    X = (CC + 0.5) / scale - 0.5
    Y = (RR + 0.5) / scale - 0.5

    U = -grad_col_hr[RR, CC]
    V = -grad_row_hr[RR, CC]

    # 只在 free 区域画箭头
    free_quiver = valid_mask[RR, CC]
    ax.quiver(
        X[free_quiver],
        Y[free_quiver],
        U[free_quiver],
        V[free_quiver],
        angles='xy',
        scale_units='xy',
        scale=12,
        width=0.002,
    )

    for k, (cell_path, _) in enumerate(multi_paths):
        pts = np.asarray(cell_path, dtype=np.float32)
        xs = pts[:, 1]
        ys = pts[:, 0]
        if k == 0:
            ax.plot(xs, ys, linewidth=1.8, label='A* paths')
        else:
            ax.plot(xs, ys, linewidth=1.8)

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
    cbar.set_label('potential' if not use_log_display else 'log(1+potential)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=240)
    plt.close(fig)