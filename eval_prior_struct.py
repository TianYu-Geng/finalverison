import os
import json
from os.path import join

from absl import app, flags
import gym
import matplotlib.pyplot as plt
import numpy as np

import eval_pg_prior as base_eval

FLAGS = flags.FLAGS

flags.DEFINE_bool("use_prior_struct", True, "Use corridor-constrained prior_struct")
flags.DEFINE_integer("prior_struct_num_candidates", 16, "Number of PG prior candidates")
flags.DEFINE_integer("prior_struct_planner_batch_size", 1, "How many selected priors to send into planner sampling")
flags.DEFINE_float("prior_struct_support_threshold", 0.08, "Support threshold on soft corridor")
flags.DEFINE_float("prior_struct_support_weight", 1.0, "Weight for corridor support score")
flags.DEFINE_float("prior_struct_pg_weight", 0.15, "Weight for PG density score")
flags.DEFINE_float("prior_struct_support_soft_weight", 1.0, "Weight for soft support mean")
flags.DEFINE_float("prior_struct_inside_ratio_weight", 2.0, "Weight for inside-corridor ratio")
flags.DEFINE_float("prior_struct_boundary_weight", 0.5, "Weight for boundary distance score")
flags.DEFINE_float("prior_struct_endpoint_support_weight", 0.0, "Weight for endpoint support score")
flags.DEFINE_float("prior_struct_progress_weight", 0.0, "Weight for progress-to-goal score")
flags.DEFINE_float("prior_struct_goal_closeness_weight", 0.0, "Weight for goal-closeness score")
flags.DEFINE_float("prior_struct_time_weight_start", 0.8, "Early-step weight for candidate support")
flags.DEFINE_float("prior_struct_time_weight_end", 1.2, "Late-step weight for candidate support")
flags.DEFINE_float("prior_struct_boundary_clip", 0.5, "Boundary distance clip")
flags.DEFINE_integer("prior_struct_rebuild_interval", 5, "Rebuild map prior every N env steps")
flags.DEFINE_bool("prior_struct_rebuild_on_cell_change", True, "Rebuild map prior when the current grid cell changes")
flags.DEFINE_bool("prior_struct_debug_runtime", False, "Print runtime stats for prior_struct")
flags.DEFINE_integer("prior_struct_astar_num_paths", 5, "Number of diverse A* paths")
flags.DEFINE_integer("prior_struct_astar_candidate_multiplier", 4, "A* candidate multiplier")
flags.DEFINE_float("prior_struct_astar_max_ratio", 2.0, "A* max cost ratio")
flags.DEFINE_float("prior_struct_astar_max_overlap", 0.8, "A* path overlap limit")
flags.DEFINE_bool("prior_struct_astar_allow_diagonal", False, "Allow diagonal A* moves")
flags.DEFINE_integer("prior_struct_scale", 12, "High-resolution field scale")
flags.DEFINE_float("prior_struct_line_sigma", 0.36, "Line sigma")
flags.DEFINE_float("prior_struct_beta", 1.1, "Support beta")
flags.DEFINE_float("prior_struct_sigma_min", 0.8, "Min sigma")
flags.DEFINE_float("prior_struct_sigma_max", 2.8, "Max sigma")
flags.DEFINE_float("prior_struct_tau", 0.35, "Responsibility tau")
flags.DEFINE_float("prior_struct_eps", 1e-6, "Field epsilon")
flags.DEFINE_bool("prior_struct_enable_guidance", False, "Enable weak center guidance placeholder")
flags.DEFINE_float("prior_struct_guidance_strength", 0.02, "Weak center guidance strength")
flags.DEFINE_float("prior_struct_guidance_max_step", 0.03, "Max world-space weak guidance step")
flags.DEFINE_float("prior_struct_guidance_boundary_margin", 0.15, "Boundary margin for weak guidance activation")
flags.DEFINE_float("prior_struct_guidance_activation_temp", 0.05, "Activation temperature for weak guidance")
flags.DEFINE_bool("save_all_rollouts", True, "Save rollout figure for every episode")
flags.DEFINE_bool("save_summary_json", True, "Save summary json with episode returns")


def render_support_field(occupancy, structured_prior, start_xy, goal_xy, save_path):
    support = structured_prior.support
    h, w = occupancy.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    wall_img = np.where(occupancy, 0.0, 1.0)
    ax.imshow(wall_img, cmap="gray", origin="lower", extent=[-0.5, w - 0.5, -0.5, h - 0.5])
    ax.imshow(
        support.support_soft_hr,
        origin="lower",
        cmap="YlOrRd",
        alpha=0.6,
        extent=[-0.5, w - 0.5, -0.5, h - 0.5],
    )
    for cell_path, _ in support.multi_paths:
        xy = np.asarray([(c[1], c[0]) for c in cell_path], dtype=np.float32)
        ax.plot(xy[:, 0], xy[:, 1], linewidth=1.2, alpha=0.5, color="tab:blue")
    ax.scatter(start_xy[1], start_xy[0], s=100, marker="o")
    ax.scatter(goal_xy[1], goal_xy[0], s=120, marker="*")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_aspect("equal")
    ax.set_title("corridor support")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


def render_candidate_set(occupancy, structured_prior, normalizer, start_xy, goal_xy, save_path):
    candidates = structured_prior.pg_density.candidates_future_obs
    candidates = normalizer.unnormalize(candidates).detach().cpu().numpy()
    selected_index = structured_prior.selected_index
    h, w = occupancy.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    wall_img = np.where(occupancy, 0.0, 1.0)
    ax.imshow(wall_img, cmap="gray", origin="lower", extent=[-0.5, w - 0.5, -0.5, h - 0.5])
    for i in range(candidates.shape[0]):
        xy = candidates[i, :, :2]
        alpha = 0.18 if i != selected_index else 0.95
        width = 1.0 if i != selected_index else 2.8
        color = "tab:orange" if i != selected_index else "tab:red"
        ax.plot(xy[:, 1], xy[:, 0], linewidth=width, alpha=alpha, color=color)
    ax.scatter(start_xy[1], start_xy[0], s=100, marker="o")
    ax.scatter(goal_xy[1], goal_xy[0], s=120, marker="*")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_aspect("equal")
    ax.set_title("PG candidates and selected prior")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


def _trajectory_for_vis(traj, success_step, post_success_steps=10):
    traj = np.asarray(traj, dtype=np.float32)
    if success_step is None:
        return traj
    success_idx = min(int(success_step) + 1 + post_success_steps, len(traj))
    return traj[:success_idx]


def save_all_rollout_visualizations(config, occupancy, eval_info):
    trajectories = eval_info.get("trajectories") or []
    goals = eval_info.get("goals") or []
    success_steps = eval_info.get("success_steps") or []
    episode_returns = eval_info.get("episode_returns") or []

    if len(trajectories) == 0:
        return []

    rollout_dir = join(config.vis_dir, "rollouts")
    os.makedirs(rollout_dir, exist_ok=True)

    saved_paths = []
    for idx, traj in enumerate(trajectories):
        traj = np.asarray(traj, dtype=np.float32)
        start_xy = traj[0, :2]
        goal_xy = None
        if idx < len(goals) and goals[idx] is not None:
            goal_xy = np.asarray(goals[idx], dtype=np.float32)
        if goal_xy is None:
            continue

        success_step = success_steps[idx] if idx < len(success_steps) else None
        traj_vis = _trajectory_for_vis(traj, success_step)
        ep_return = float(episode_returns[idx]) if idx < len(episode_returns) else float("nan")

        save_path = join(
            rollout_dir,
            f"rollout_ep{idx:03d}_{config.env_name}_seed{config.seed}_pg{config.pg_ckpt}.png",
        )
        base_eval.render_maze2d_rollout(
            occupancy=occupancy,
            trajectory=traj_vis,
            start_xy=start_xy,
            goal_xy=goal_xy,
            save_path=save_path,
            title=f"prior_struct rollout | ep={idx} | return={ep_return:.2f}",
        )
        saved_paths.append(save_path)

    print(f"[save] saved {len(saved_paths)} rollout figures to: {rollout_dir}")
    return saved_paths


def save_summary(config, eval_info):
    episode_returns = np.asarray(eval_info.get("episode_returns", []), dtype=np.float32)
    if episode_returns.size == 0:
        return None

    summary = {
        "env_name": config.env_name,
        "seed": int(config.seed),
        "num_episodes": int(len(episode_returns)),
        "return_mean": float(np.mean(episode_returns)),
        "return_std": float(np.std(episode_returns)),
        "return_stderr": float(np.std(episode_returns) / np.sqrt(len(episode_returns))),
        "return_var": float(np.var(episode_returns)),
        "episode_returns": episode_returns.tolist(),
    }
    save_path = join(config.vis_dir, f"summary_{config.env_name}_seed{config.seed}_pg{config.pg_ckpt}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] summary json saved to: {save_path}")
    return save_path


def main(_):
    config = base_eval.build_config()
    config.use_prior_struct = True
    os.makedirs(config.vis_dir, exist_ok=True)

    eval_info, normalizer = base_eval.evaluate_pg_prior_once(config)

    print("[eval final]")
    for k, v in eval_info.items():
        if k in ["trajectories", "goals", "success_steps", "denoise_histories", "priors", "structured_priors", "episode_returns"]:
            continue
        print(f"  {k}: {float(v):.6f}" if isinstance(v, (int, float, np.floating)) else f"  {k}: {v}")

    trajectories = eval_info.get("trajectories")
    if not trajectories:
        return

    env = gym.make(config.env_name)
    _ = env.reset()
    occupancy = base_eval.extract_occupancy(env.unwrapped)
    goals = eval_info.get("goals")
    if goals is None or len(goals) == 0 or goals[0] is None:
        goals = [np.asarray(env.unwrapped._target, dtype=np.float32)]
    eval_info["goals"] = goals

    start_xy, goal_xy, _ = base_eval.save_rollout_visualization(config, occupancy, eval_info)
    if config.save_all_rollouts:
        save_all_rollout_visualizations(config, occupancy, eval_info)
    base_eval.save_denoise_visualization(config, occupancy, eval_info, normalizer, start_xy, goal_xy)
    base_eval.save_prior_visualization(config, occupancy, eval_info, normalizer, start_xy, goal_xy)
    if config.save_summary_json:
        save_summary(config, eval_info)

    structured_priors = eval_info.get("structured_priors")
    if not structured_priors or structured_priors[0] is None:
        print("[warn] structured prior is not available")
        return

    structured_prior = structured_priors[0]
    save_support = join(config.vis_dir, f"support_{config.env_name}_seed{config.seed}_pg{config.pg_ckpt}.png")
    render_support_field(occupancy, structured_prior, start_xy, goal_xy, save_support)
    print(f"[save] support figure saved to: {save_support}")

    save_candidates = join(config.vis_dir, f"candidates_{config.env_name}_seed{config.seed}_pg{config.pg_ckpt}.png")
    render_candidate_set(occupancy, structured_prior, normalizer, start_xy, goal_xy, save_candidates)
    print(f"[save] candidate figure saved to: {save_candidates}")


if __name__ == "__main__":
    app.run(main)
