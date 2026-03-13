import os
import random
from os.path import join

from absl import app, flags

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

# ========================
# 基本配置
# =========================
flags.DEFINE_string('env_name', 'maze2d-large-v1', 'environment name')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_string('device', 'cuda:0', 'Device to use')

# DV / planner / policy / critic
flags.DEFINE_string('planner_net', 'transformer', 'Type of planner network')
flags.DEFINE_bool('rebase_policy', True, 'Rebase policy position')

flags.DEFINE_bool('continous_reward_at_done', True, 'Continous reward at done')
flags.DEFINE_string('reward_tune', 'iql', 'Reward tune')
flags.DEFINE_float('discount', 1.0, 'Discount factor')
flags.DEFINE_integer('planner_d_model_divide', 64, 'divide planner model dimension')

flags.DEFINE_string('planner_solver', 'ddim', 'Planner solver')
flags.DEFINE_integer('planner_emb_dim', 128, 'Planner embedding dimension')
flags.DEFINE_integer('planner_d_model', 256, 'Planner model dimension')
flags.DEFINE_integer('planner_depth', 2, 'Planner depth')
flags.DEFINE_integer('planner_sampling_steps', 20, 'Planner sampling steps')
flags.DEFINE_bool('planner_predict_noise', True, 'Planner predict noise')
flags.DEFINE_float('planner_next_obs_loss_weight', 1, 'Planner next observation loss weight')
flags.DEFINE_float('planner_ema_rate', 0.9999, 'Planner EMA rate')
flags.DEFINE_integer('unet_dim', 32, 'UNet dimension')
flags.DEFINE_integer('use_advantage_weighting', 0, 'Use advantage weighting')

flags.DEFINE_string('policy_solver', 'ddpm', 'Policy solver')
flags.DEFINE_integer('policy_hidden_dim', 256, 'Policy hidden dimension')
flags.DEFINE_integer('policy_diffusion_steps', 10, 'Policy diffusion steps')
flags.DEFINE_integer('policy_sampling_steps', 10, 'Policy sampling steps')
flags.DEFINE_bool('policy_predict_noise', True, 'Policy predict noise')
flags.DEFINE_float('policy_ema_rate', 0.995, 'Policy EMA rate')
flags.DEFINE_float('policy_learning_rate', 0.0003, 'Policy learning rate')
flags.DEFINE_float('critic_learning_rate', 0.0003, 'Critic learning rate')

flags.DEFINE_integer('use_diffusion_policy', 1, 'Use diffusion policy')
flags.DEFINE_integer('invdyn_gradient_steps', 200000, 'Inverse dynamics gradient steps')
flags.DEFINE_integer('policy_diffusion_gradient_steps', 1000000, 'Policy diffusion gradient steps')
flags.DEFINE_integer('planner_diffusion_gradient_steps', 1000000, 'Planner diffusion gradient steps')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('log_interval', 1000, 'Log interval')
flags.DEFINE_integer('eval_interval', 100000, 'Eval interval')
flags.DEFINE_integer('save_interval', 100000, 'Save interval')

flags.DEFINE_integer('num_episodes', 1, 'Number of evaluation episodes')
flags.DEFINE_integer('planner_ckpt', 1000000, 'Planner checkpoint')
flags.DEFINE_integer('critic_ckpt', 200000, 'Critic checkpoint')
flags.DEFINE_integer('policy_ckpt', 1000000, 'Policy checkpoint')
flags.DEFINE_integer('invdyn_ckpt', 200000, 'Inverse dynamics checkpoint')
flags.DEFINE_bool('planner_use_ema', True, 'Use EMA for planner')
flags.DEFINE_float('policy_temperature', 0.5, 'Policy temperature')
flags.DEFINE_bool('policy_use_ema', True, 'Use EMA for policy')

flags.DEFINE_integer('max_path_length', 800, 'Maximum path length')
flags.DEFINE_integer('planner_horizon', 32, 'Planner horizon')
flags.DEFINE_integer('stride', 15, 'Stride')
flags.DEFINE_float('planner_temperature', 1.0, 'Planner temperature')
flags.DEFINE_float('planner_target_return', 1.0, 'Planner target return')
flags.DEFINE_float('planner_w_cfg', 1.0, 'Planner weight for CFG')

# PG prior
flags.DEFINE_integer("prior_hidden_dim", 256, "Prior hidden dimensions")
flags.DEFINE_float('prior_squash_mean', 2.0, 'prior squash mean')
flags.DEFINE_float('prior_learning_rate', 3e-4, 'prior learning rate')
flags.DEFINE_float('alpha', 1.0, 'weight for const loss')
flags.DEFINE_bool('use_tanh_squash', True, 'use squash')
flags.DEFINE_string('divergence', 'kl', 'divergence type')
flags.DEFINE_bool('normalize_q', True, 'normalize q')
flags.DEFINE_bool('use_value', False, 'use value')
flags.DEFINE_bool('use_target', True, 'use target')
flags.DEFINE_bool('eval_deterministic', True, 'eval deterministic')
flags.DEFINE_integer('planner_sampling_steps_train', 5, 'Planner sampling steps train')

flags.DEFINE_string("entity", "entity_name", "entity name")
flags.DEFINE_string('pipeline_type', 'separate', 'Type of pipeline')

flags.DEFINE_integer('pg_ckpt', 1000000, 'PG prior checkpoint')

# 可视化输出
flags.DEFINE_string('vis_dir', './vis_eval_pg', 'directory to save rollout figures')
flags.DEFINE_bool('save_denoise_process', True, 'whether to save denoising process figures')
flags.DEFINE_integer('num_denoise_snapshots', 10, 'number of denoising snapshots to save')
flags.DEFINE_integer('num_vis_samples', 16, 'number of planner samples for denoise visualization')
flags.DEFINE_float('prior_init_noise_scale', 0.003, 'noise scale around prior initialization for denoise visualization')
flags.DEFINE_bool('debug_vis', False, 'print visualization debug info')
flags.DEFINE_bool('save_individual_denoise_steps', True, 'save individual denoise step figures')
flags.DEFINE_bool("use_lomap", False, "Enable LoMAP manifold projection during planner sampling")
flags.DEFINE_string("lomap_store_path", "", "Path to the lightweight LoMAP datastore")
flags.DEFINE_integer("lomap_topk", 8, "Number of nearest trajectories used for local PCA projection")
flags.DEFINE_integer("lomap_prefilter_k", 256, "Endpoint-based prefilter size for LoMAP search")
flags.DEFINE_float("lomap_pca_tau", 0.95, "Explained-variance target for LoMAP PCA")
flags.DEFINE_float("lomap_blend", 1.0, "Interpolation weight between xt and projected xt")
flags.DEFINE_float("lomap_active_start_ratio", 0.5, "Start ratio of reverse denoising steps where LoMAP is active")
flags.DEFINE_float("lomap_active_end_ratio", 1.0, "End ratio of reverse denoising steps where LoMAP is active")
flags.DEFINE_float("corridor_start_ratio", 0.5, "Reverse denoising ratio where corridor guidance starts")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(model_path, map_location='cpu'):
    return torch.load(model_path, map_location=map_location)


def build_config():
    from easydict import EasyDict

    config = EasyDict(FLAGS.flag_values_dict())
    config.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    if config.env_name == 'maze2d-umaze-v1':
        config.max_path_length = 300
    elif config.env_name == 'maze2d-medium-v1':
        config.max_path_length = 600
    elif config.env_name == 'maze2d-large-v1':
        config.max_path_length = 800
    else:
        raise ValueError(f'Unknown environment name: {config.env_name}')

    if config.pg_ckpt < 0:
        config.pg_ckpt = config.planner_ckpt

    return config


def extract_occupancy(env):
    env = env.unwrapped
    if not hasattr(env, 'maze_arr'):
        raise AttributeError('Maze2D environment does not expose maze_arr')
    maze = np.asarray(env.maze_arr)
    return maze == 10


def render_maze2d_rollout(
    occupancy,
    trajectory,
    start_xy,
    goal_xy,
    save_path,
    title='Maze2D Rollout',
):
    h, w = occupancy.shape
    traj = np.asarray(trajectory, dtype=np.float32)
    traj_xy = traj[:, :2]

    xs = traj_xy[:, 1]
    ys = traj_xy[:, 0]

    fig, ax = plt.subplots(figsize=(6, 6))
    wall_img = np.where(occupancy, 0.0, 1.0)
    ax.imshow(
        wall_img,
        cmap='gray',
        origin='lower',
        extent=[-0.5, w - 0.5, -0.5, h - 0.5],
    )

    ax.plot(xs, ys, linewidth=2.0, label='rollout')
    ax.scatter(xs, ys, s=12)

    ax.scatter(start_xy[1], start_xy[0], s=100, marker='o', label='start')
    ax.scatter(goal_xy[1], goal_xy[0], s=120, marker='*', label='goal')

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


def render_denoise_snapshot(
    occupancy,
    xt,
    start_xy,
    goal_xy,
    save_path,
    title='Denoising Snapshot',
):
    xt = np.asarray(xt, dtype=np.float32)
    h, w = occupancy.shape

    if xt.ndim == 2:
        xy_cloud = xt[:, :2]
    elif xt.ndim == 3:
        xy_cloud = xt[:, :, :2].reshape(-1, 2)
    else:
        raise ValueError(f'Unexpected xt shape: {xt.shape}')

    fig, ax = plt.subplots(figsize=(6, 6))
    wall_img = np.where(occupancy, 0.0, 1.0)
    ax.imshow(
        wall_img,
        cmap='gray',
        origin='lower',
        extent=[-0.5, w - 0.5, -0.5, h - 0.5],
    )

    ax.scatter(xy_cloud[:, 1], xy_cloud[:, 0], s=8, alpha=0.18)
    ax.scatter(start_xy[1], start_xy[0], s=90, marker='o', label='start')
    ax.scatter(goal_xy[1], goal_xy[0], s=110, marker='*', label='goal')

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


def render_denoise_grid(
    occupancy,
    xt_list,
    start_xy,
    goal_xy,
    save_path,
    ncols=5,
    title='Denoising Process',
):
    n = len(xt_list)
    nrows = int(np.ceil(n / ncols))
    h, w = occupancy.shape

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)
    wall_img = np.where(occupancy, 0.0, 1.0)

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis('off')
            continue

        xt = np.asarray(xt_list[i], dtype=np.float32)
        if xt.ndim == 2:
            xy_cloud = xt[:, :2]
        elif xt.ndim == 3:
            xy_cloud = xt[:, :, :2].reshape(-1, 2)
        else:
            raise ValueError(f'Unexpected xt shape: {xt.shape}')

        ax.imshow(
            wall_img,
            cmap='gray',
            origin='lower',
            extent=[-0.5, w - 0.5, -0.5, h - 0.5],
        )
        ax.scatter(xy_cloud[:, 1], xy_cloud[:, 0], s=6, alpha=0.16)
        ax.scatter(start_xy[1], start_xy[0], s=32, marker='o')
        ax.scatter(goal_xy[1], goal_xy[0], s=40, marker='*')

        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(-0.5, h - 0.5)
        ax.set_aspect('equal')
        ax.set_title(f'step {i}')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def render_prior_only(
    occupancy,
    prior,
    start_xy,
    goal_xy,
    save_path,
    title='Prior Visualization',
):
    prior = np.asarray(prior, dtype=np.float32)
    h, w = occupancy.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    wall_img = np.where(occupancy, 0.0, 1.0)
    ax.imshow(
        wall_img,
        cmap='gray',
        origin='lower',
        extent=[-0.5, w - 0.5, -0.5, h - 0.5],
    )

    if prior.ndim == 2:
        xy = prior[:, :2]
        ax.plot(xy[:, 1], xy[:, 0], linewidth=2.5, alpha=0.95)
        ax.scatter(xy[:, 1], xy[:, 0], s=18, alpha=0.9)

    elif prior.ndim == 3:
        trajs = prior[:, :, :2]
        for i in range(trajs.shape[0]):
            xy = trajs[i]
            ax.plot(xy[:, 1], xy[:, 0], linewidth=1.0, alpha=0.20)
            ax.scatter(xy[:, 1], xy[:, 0], s=6, alpha=0.14)

        traj_mean = trajs.mean(axis=0)
        dist = ((trajs - traj_mean[None, :, :]) ** 2).sum(axis=(1, 2))
        rep_idx = int(np.argmin(dist))
        xy_center = trajs[rep_idx]
        ax.plot(xy_center[:, 1], xy_center[:, 0], linewidth=2.8, alpha=0.95)

    else:
        raise ValueError(f'Unexpected prior shape: {prior.shape}')

    ax.scatter(start_xy[1], start_xy[0], s=100, marker='o', label='start')
    ax.scatter(goal_xy[1], goal_xy[0], s=120, marker='*', label='goal')

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


def load_dv_models(config, planner, policy, critic):
    planner_ckpt_path = f"./checkpoint/DV/{config.env_name}/{config.seed}/model_{config.planner_ckpt}.pt"
    critic_ckpt_path = f"./checkpoint/DV/{config.env_name}/{config.seed}/model_{config.critic_ckpt}.pt"

    planner_ckpt = load_checkpoint(planner_ckpt_path, map_location=config.device)
    critic_ckpt = load_checkpoint(critic_ckpt_path, map_location=config.device)

    if planner_ckpt.get('target_planner_model', None) is not None:
        planner.model.load_state_dict(planner_ckpt['target_planner_model'])
    else:
        planner.model.load_state_dict(planner_ckpt['planner_model'])

    if planner_ckpt.get('target_policy_model', None) is not None:
        policy.model.load_state_dict(planner_ckpt['target_policy_model'])
    else:
        policy.model.load_state_dict(planner_ckpt['policy_model'])

    if critic_ckpt.get('critic', None) is not None:
        critic.load_state_dict(critic_ckpt['critic'])

    planner.model.eval()
    policy.model.eval()
    critic.eval()

    print(f"[load] planner ckpt: {planner_ckpt_path}")
    print(f"[load] critic  ckpt: {critic_ckpt_path}")


def load_pg_prior(config, agent_state):
    pg_ckpt_path = f'./checkpoint/PG/{config.env_name}/{config.seed}/model_{config.pg_ckpt}.pt'
    pg_ckpt = load_checkpoint(pg_ckpt_path, map_location=config.device)

    if pg_ckpt.get("target_prior", None) is not None:
        agent_state.target_prior.load_state_dict(pg_ckpt["target_prior"])
        prior = agent_state.target_prior
        prior_name = "target_prior"
    elif pg_ckpt.get("prior", None) is not None:
        agent_state.prior.load_state_dict(pg_ckpt["prior"])
        prior = agent_state.prior
        prior_name = "prior"
    else:
        raise ValueError(f"No prior found in PG checkpoint: {pg_ckpt_path}")

    prior.eval()
    print(f"[load] PG ckpt: {pg_ckpt_path} ({prior_name})")
    return prior


def evaluate_pg_prior_once(config):
    from dataset.d4rl_maze2d_dataset import D4RLMaze2DSeqDataset
    try:
        import pg
    except ModuleNotFoundError:
        from PG import pg
    from evaluation import evaluate_prior_with_trajectories

    set_seed(config.seed)
    env = gym.make(config.env_name)

    planner_dataset = D4RLMaze2DSeqDataset(
        env.get_dataset(),
        horizon=config.planner_horizon,
        discount=config.discount,
        continous_reward_at_done=config.continous_reward_at_done,
        reward_tune=config.reward_tune,
        stride=config.stride,
        learn_policy=False,
        center_mapping=True,
        device=config.device,
    ).to(config.device)

    config.observation_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]

    planner, policy, critic, _, agent_state = pg.init(config)

    load_dv_models(config, planner, policy, critic)
    prior = load_pg_prior(config, agent_state)

    normalizer = planner_dataset.normalizer

    eval_info = evaluate_prior_with_trajectories(
        planner,
        policy,
        prior,
        config,
        env,
        normalizer,
        eval_deterministic=config.eval_deterministic,
        critic=critic,
    )
    return eval_info, normalizer


def save_rollout_visualization(config, occupancy, eval_info):
    trajectories = eval_info.get('trajectories', None)
    goals = eval_info.get('goals', None)
    success_steps = eval_info.get('success_steps', None)

    if trajectories is None or len(trajectories) == 0:
        return None, None, None

    traj = np.asarray(trajectories[0], dtype=np.float32)
    start_xy = traj[0, :2]

    if goals is None or len(goals) == 0 or goals[0] is None:
        raise ValueError('Goal is missing for rollout visualization.')
    goal_xy = np.asarray(goals[0], dtype=np.float32)

    post_success_steps = 10
    if success_steps is not None and len(success_steps) > 0 and success_steps[0] is not None:
        success_idx = min(int(success_steps[0]) + 1 + post_success_steps, len(traj))
        traj_vis = traj[:success_idx]
    else:
        traj_vis = traj

    save_path_rollout = join(
        config.vis_dir,
        f'rollout_{config.env_name}_seed{config.seed}_pg{config.pg_ckpt}.png'
    )

    render_maze2d_rollout(
        occupancy=occupancy,
        trajectory=traj_vis,
        start_xy=start_xy,
        goal_xy=goal_xy,
        save_path=save_path_rollout,
        title=f"PG Prior Rollout | return={float(eval_info['return']):.2f}",
    )
    print(f"[save] rollout figure saved to: {save_path_rollout}")
    print("start_xy:", start_xy)
    print("goal_xy:", goal_xy)
    print("traj_end:", traj_vis[-1, :2])

    return start_xy, goal_xy, traj_vis


def save_denoise_visualization(config, occupancy, eval_info, normalizer, start_xy, goal_xy):
    denoise_histories = eval_info.get('denoise_histories', None)
    if not config.save_denoise_process or denoise_histories is None or len(denoise_histories) == 0:
        print("[warn] denoise process is not available")
        return

    hist = denoise_histories[0]
    if hist is None or len(hist) == 0:
        print("[warn] denoise_histories[0] is empty or None")
        return

    num_snapshots = min(config.num_denoise_snapshots, len(hist))
    idxs = np.linspace(0, len(hist) - 1, num_snapshots).astype(int)

    xt_list = []
    for k, idx in enumerate(idxs):
        xt = hist[idx]
        if isinstance(xt, torch.Tensor):
            xt = xt.detach().cpu().numpy()

        if config.debug_vis:
            xt_flat_raw = xt.reshape(-1, xt.shape[-1])
            print(f"\n[denoise step {k}] raw")
            print("xt_xy_min:", xt_flat_raw[:, :2].min(axis=0))
            print("xt_xy_max:", xt_flat_raw[:, :2].max(axis=0))
            print("start_xy:", start_xy)
            print("goal_xy:", goal_xy)

        xt = normalizer.unnormalize(xt)

        if config.debug_vis:
            xt_flat = xt.reshape(-1, xt.shape[-1])
            print(f"[denoise step {k}] world")
            print("xt_xy_min:", xt_flat[:, :2].min(axis=0))
            print("xt_xy_max:", xt_flat[:, :2].max(axis=0))

        xt_list.append(xt)

        if config.save_individual_denoise_steps:
            save_path_xt = join(config.vis_dir, f'denoise_step_{k:02d}.png')
            render_denoise_snapshot(
                occupancy=occupancy,
                xt=xt,
                start_xy=start_xy,
                goal_xy=goal_xy,
                save_path=save_path_xt,
                title=f'Denoising Step {k} | idx={idx}',
            )

    save_path_grid = join(config.vis_dir, 'denoise_grid.png')
    render_denoise_grid(
        occupancy=occupancy,
        xt_list=xt_list,
        start_xy=start_xy,
        goal_xy=goal_xy,
        save_path=save_path_grid,
        ncols=5,
        title='PG Prior Denoising Process',
    )

    print(f"[save] denoising grid saved to: {save_path_grid}")
    if config.save_individual_denoise_steps:
        print(f"[save] denoising snapshots saved to: {config.vis_dir}")


def save_prior_visualization(config, occupancy, eval_info, normalizer, start_xy, goal_xy):
    priors = eval_info.get('priors', None)
    if priors is None or len(priors) == 0 or priors[0] is None:
        return

    prior_vis = np.asarray(priors[0], dtype=np.float32)
    prior_vis = normalizer.unnormalize(prior_vis)

    save_path_prior = join(
        config.vis_dir,
        f'prior_{config.env_name}_seed{config.seed}_pg{config.pg_ckpt}.png'
    )

    render_prior_only(
        occupancy=occupancy,
        prior=prior_vis,
        start_xy=start_xy,
        goal_xy=goal_xy,
        save_path=save_path_prior,
        title='PG Prior',
    )
    print(f"[save] prior figure saved to: {save_path_prior}")


def main(_):
    config = build_config()
    os.makedirs(config.vis_dir, exist_ok=True)

    eval_info, normalizer = evaluate_pg_prior_once(config)

    print("[eval final]")
    for k, v in eval_info.items():
        if k in ['trajectories', 'goals', 'success_steps', 'denoise_histories', 'priors']:
            continue
        if isinstance(v, (int, float, np.floating)):
            print(f"  {k}: {float(v):.6f}")
        else:
            print(f"  {k}: {v}")

    trajectories = eval_info.get('trajectories', None)
    if trajectories is None or len(trajectories) == 0:
        return

    env = gym.make(config.env_name)
    _ = env.reset()
    base_env = env.unwrapped
    occupancy = extract_occupancy(base_env)

    goals = eval_info.get('goals', None)
    if goals is None or len(goals) == 0 or goals[0] is None:
        if not hasattr(base_env, '_target'):
            raise AttributeError('Underlying Maze2D env does not expose _target')
        eval_info['goals'] = [np.asarray(base_env._target, dtype=np.float32)]

    start_xy, goal_xy, _ = save_rollout_visualization(config, occupancy, eval_info)
    save_denoise_visualization(config, occupancy, eval_info, normalizer, start_xy, goal_xy)
    save_prior_visualization(config, occupancy, eval_info, normalizer, start_xy, goal_xy)


if __name__ == '__main__':
    app.run(main)
