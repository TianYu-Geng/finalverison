from typing import Dict

import gym
import numpy as np
import torch
from tqdm import tqdm

from lomap_runtime import build_lomap_projector
from prior_struct import build_prior_struct


def _to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _to_numpy_action(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _sanitize_tensor(x, clamp=1e3):
    if not isinstance(x, torch.Tensor):
        return x
    return torch.nan_to_num(x, nan=0.0, posinf=clamp, neginf=-clamp)


def _build_prior_value(prior, observation_t, config, env, normalizer, eval_deterministic=True, runtime_state=None):
    num_samples = getattr(config, "num_vis_samples", 16)

    if getattr(config, "use_prior_struct", False) and "maze2d" in config.env_name:
        planner_batch_size = getattr(config, "prior_struct_planner_batch_size", 1)
        structured_prior = build_prior_struct(
            prior_model=prior,
            observation_t=observation_t,
            env=env,
            normalizer=normalizer,
            config=config,
            planner_batch_size=planner_batch_size,
            eval_deterministic=eval_deterministic,
            runtime_state=runtime_state,
        )
        return structured_prior.fused_prior_state, structured_prior

    obs_repeat = observation_t.unsqueeze(0).repeat(num_samples, 1)
    dist = prior.eval_forward(obs_repeat)
    prior_value = dist.mean if eval_deterministic else dist.sample()

    if config.use_tanh_squash:
        prior_value = torch.tanh(prior_value) * config.prior_squash_mean

    return prior_value, None


def _select_best_planned_traj(traj_all, critic=None, goal_xy=None):
    if traj_all.shape[0] == 1:
        return traj_all[:1], 0, {}

    debug = {}
    if critic is not None:
        values = critic.eval_forward(traj_all).reshape(-1)
        best_idx = int(torch.argmax(values).item())
        debug["critic_scores"] = values.detach().cpu().numpy()
        return traj_all[best_idx:best_idx + 1], best_idx, debug

    if goal_xy is not None:
        goal_xy_t = torch.as_tensor(goal_xy, device=traj_all.device, dtype=traj_all.dtype)
        final_xy = traj_all[:, -1, :2]
        dist = torch.linalg.norm(final_xy - goal_xy_t.view(1, 2), dim=-1)
        best_idx = int(torch.argmin(dist).item())
        debug["goal_distance"] = dist.detach().cpu().numpy()
        return traj_all[best_idx:best_idx + 1], best_idx, debug

    return traj_all[:1], 0, debug


def _get_lomap_projector(config, device, runtime_cache):
    if not bool(getattr(config, "use_lomap", False)):
        return None
    projector = runtime_cache.get("lomap_projector")
    if projector is None:
        projector = build_lomap_projector(config=config, device=device)
        runtime_cache["lomap_projector"] = projector
    return projector


def _build_lomap_context(structured_prior):
    if structured_prior is None:
        return None
    return {
        "selected_branch_id": int(structured_prior.selected_branch_id),
        "map_prior": structured_prior.support,
        "obs_mean_xy": structured_prior.guidance["obs_mean_xy"],
        "obs_std_xy": structured_prior.guidance["obs_std_xy"],
    }


def evaluate(planner, policy, critic, planner_model, policy_model, config, env, normalizer):
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=config.num_episodes)
    progress = tqdm(range(config.num_episodes))

    device = config.device if isinstance(config.device, torch.device) else torch.device(config.device)
    episode_rewards = []

    planner_model.eval()
    policy_model.eval()
    if critic is not None:
        critic.eval()

    for _ in progress:
        observation, cum_done, ep_reward, t = env.reset(), 0.0, 0.0, 0
        finished = np.zeros(1, dtype=bool)
        prior_runtime_state = {}

        while (not np.all(cum_done)) and t < config.max_path_length + 1:
            observation = normalizer.normalize(observation)
            observation_t = _to_tensor(observation, device=device, dtype=torch.float32)

            obs_repeat = observation_t.unsqueeze(0).repeat(config.planner_num_candidates, 1)

            with torch.no_grad():
                traj = planner.sample(
                    obs_repeat,
                    model=planner_model,
                    solver=config.planner_solver,
                    planner_horizon=config.planner_horizon,
                    n_samples=config.planner_num_candidates,
                    sample_steps=config.planner_sampling_steps,
                    temperature=config.planner_temperature,
                )

                value = critic.eval_forward(traj)
                idx = torch.argmax(value)
                traj = traj[idx].unsqueeze(0)

                policy_prior = torch.zeros((1, config.action_dim), device=device, dtype=torch.float32)
                next_obs_plan = traj[:, 1, :]
                obs_policy = observation_t.clone().unsqueeze(0)
                next_obs_policy = next_obs_plan.clone()

                if config.rebase_policy:
                    next_obs_policy = torch.cat(
                        (next_obs_policy[:, :2] - obs_policy[:, :2], next_obs_policy[:, 2:]),
                        dim=1
                    )
                    obs_policy = torch.cat(
                        (torch.zeros((obs_policy.shape[0], 2), device=device, dtype=obs_policy.dtype), obs_policy[:, 2:]),
                        dim=1
                    )

                action = policy.sample(
                    policy_prior,
                    model=policy_model,
                    solver=config.policy_solver,
                    n_samples=1,
                    sample_steps=config.policy_sampling_steps,
                    condition_cfg=torch.cat([obs_policy, next_obs_policy], dim=-1),
                    temperature=config.policy_temperature,
                )

            action_np = _to_numpy_action(torch.squeeze(action))
            observation, reward, done, _ = env.step(action_np)

            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)

            if any(env_name in config.env_name for env_name in ["halfcheetah", "walker2d", "hopper"]):
                ep_reward += (reward * (1 - cum_done)) if t < config.max_path_length else reward
            elif any(env_name in config.env_name for env_name in ["antmaze", "kitchen"]):
                ep_reward += reward
            elif 'maze2d' in config.env_name:
                finished |= (reward == 1.0)
                ep_reward += finished
            else:
                raise NotImplementedError()

        if any(env_name in config.env_name for env_name in ["halfcheetah", "walker2d", "hopper"]):
            episode_rewards.append(ep_reward)
        elif 'antmaze' in config.env_name:
            episode_rewards.append(np.clip(ep_reward, 0.0, 1.0))
        elif 'maze2d' in config.env_name:
            episode_rewards.append(ep_reward)
        elif 'kitchen' in config.env_name:
            episode_rewards.append(np.clip(ep_reward, 0.0, 4.0))
        else:
            raise NotImplementedError()

        mean = np.mean(np.array(list(map(lambda x: env.get_normalized_score(x), episode_rewards))).reshape(-1) * 100)
        progress.set_description_str(str(mean))

    episode_rewards = list(map(lambda x: env.get_normalized_score(x), episode_rewards))
    episode_rewards = np.array(episode_rewards).reshape(-1) * 100
    mean = np.mean(episode_rewards)
    err = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
    print(mean, err)

    return {"return": mean, "return_std": err}


def evaluate_prior(
    planner,
    policy,
    prior,
    config,
    env,
    normalizer,
    eval_deterministic=True,
    critic=None,
):
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=config.num_episodes)
    progress = tqdm(range(config.num_episodes))

    device = config.device if isinstance(config.device, torch.device) else torch.device(config.device)
    episode_rewards = []

    if hasattr(prior, "eval"):
        prior.eval()
    if hasattr(policy, "model"):
        policy.model.eval()
    if hasattr(planner, "model"):
        planner.model.eval()

    shared_prior_runtime_cache = {}
    lomap_runtime_cache = {}

    for _ in progress:
        observation, cum_done, ep_reward, t = env.reset(), 0.0, 0.0, 0
        finished = np.zeros(1, dtype=bool)
        prior_runtime_state = dict(shared_prior_runtime_cache)

        while (not np.all(cum_done)) and t < config.max_path_length + 1:
            observation = normalizer.normalize(observation)
            observation_t = _to_tensor(observation, device=device, dtype=torch.float32)

            with torch.no_grad():
                lomap_projector = _get_lomap_projector(config, device, lomap_runtime_cache)
                prior_value, structured_prior = _build_prior_value(
                    prior=prior,
                    observation_t=observation_t,
                    config=config,
                    env=env,
                    normalizer=normalizer,
                    eval_deterministic=eval_deterministic,
                    runtime_state=prior_runtime_state,
                )
                obs_repeat = observation_t.unsqueeze(0).repeat(prior_value.shape[0], 1)

                traj = planner.sample_prior(
                    obs_repeat,
                    prior=prior_value,
                    solver=config.planner_solver,
                    planner_horizon=config.planner_horizon,
                    sample_steps=config.planner_sampling_steps,
                    temperature=config.planner_temperature,
                    prior_init_noise_scale=getattr(config, "prior_init_noise_scale", 0.003),
                    guidance=structured_prior.guidance if structured_prior is not None else None,
                    lomap_projector=lomap_projector,
                    lomap_context=_build_lomap_context(structured_prior),
                )
                if traj.shape[0] > 1:
                    goal_xy = getattr(env.unwrapped, "_target", None)
                    traj, _, _ = _select_best_planned_traj(traj, critic=critic, goal_xy=goal_xy)
                else:
                    traj = traj[:1]

                policy_prior = torch.zeros((1, config.action_dim), device=device, dtype=torch.float32)
                next_obs_plan = traj[:, 1, :]
                obs_policy = observation_t.clone().unsqueeze(0)
                next_obs_policy = next_obs_plan.clone()

                if config.rebase_policy:
                    next_obs_policy = torch.cat(
                        (next_obs_policy[:, :2] - obs_policy[:, :2], next_obs_policy[:, 2:]),
                        dim=1
                    )
                    obs_policy = torch.cat(
                        (torch.zeros((obs_policy.shape[0], 2), device=device, dtype=obs_policy.dtype), obs_policy[:, 2:]),
                        dim=1
                    )

                action = policy.sample(
                    policy_prior,
                    model=policy.model,
                    solver=config.policy_solver,
                    n_samples=1,
                    sample_steps=config.policy_sampling_steps,
                    condition_cfg=torch.cat([obs_policy, next_obs_policy], dim=-1),
                    temperature=config.policy_temperature,
                )

            action_np = _to_numpy_action(torch.squeeze(action))
            observation, reward, done, _ = env.step(action_np)

            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)

            if any(env_name in config.env_name for env_name in ["halfcheetah", "walker2d", "hopper"]):
                ep_reward += (reward * (1 - cum_done)) if t < config.max_path_length else reward
            elif any(env_name in config.env_name for env_name in ["antmaze", "kitchen"]):
                ep_reward += reward
            elif 'maze2d' in config.env_name:
                finished |= (reward == 1.0)
                ep_reward += finished
            else:
                raise NotImplementedError()

        if any(env_name in config.env_name for env_name in ["halfcheetah", "walker2d", "hopper"]):
            episode_rewards.append(ep_reward)
        elif 'antmaze' in config.env_name:
            episode_rewards.append(np.clip(ep_reward, 0.0, 1.0))
        elif 'maze2d' in config.env_name:
            episode_rewards.append(ep_reward)
        elif 'kitchen' in config.env_name:
            episode_rewards.append(np.clip(ep_reward, 0.0, 4.0))
        else:
            raise NotImplementedError()

        mean = np.mean(np.array(list(map(lambda x: env.get_normalized_score(x), episode_rewards))).reshape(-1) * 100)
        progress.set_description_str(str(mean))

    episode_rewards = list(map(lambda x: env.get_normalized_score(x), episode_rewards))
    episode_rewards = np.array(episode_rewards).reshape(-1) * 100
    mean = np.mean(episode_rewards)
    err = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
    print(mean, err)

    return {"return": mean, "return_std": err}


@torch.no_grad()
def evaluate_prior_with_trajectories(
    planner,
    policy,
    prior,
    config,
    env,
    normalizer,
    eval_deterministic=True,
    critic=None,
):
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=config.num_episodes)
    progress = tqdm(range(config.num_episodes))

    device = config.device if isinstance(config.device, torch.device) else torch.device(config.device)
    episode_rewards = []
    all_trajectories = []
    all_goals = []
    all_success_steps = []
    all_denoise_histories = []
    all_priors = []
    all_structured_priors = []
    all_plan_candidates = []

    if hasattr(prior, "eval"):
        prior.eval()
    if hasattr(policy, "model"):
        policy.model.eval()
    if hasattr(planner, "model"):
        planner.model.eval()

    shared_prior_runtime_cache = {}
    lomap_runtime_cache = {}

    for _ in progress:
        observation, cum_done, ep_reward, t = env.reset(), 0.0, 0.0, 0

        finished = np.zeros(1, dtype=bool)

        episode_traj = [np.array(observation, copy=True)]
        success_step = None
        episode_denoise_history = None
        episode_prior = None
        episode_structured_prior = None
        prior_runtime_state = dict(shared_prior_runtime_cache)

        base_env = env.unwrapped
        if hasattr(base_env, '_target'):
            goal_xy = np.array(base_env._target, copy=True, dtype=np.float32)
        else:
            goal_xy = None
        all_goals.append(goal_xy)

        while (not np.all(cum_done)) and t < config.max_path_length + 1:
            observation_norm = normalizer.normalize(observation)
            observation_t = _to_tensor(observation_norm, device=device, dtype=torch.float32)

            prior_value, structured_prior = _build_prior_value(
                prior=prior,
                observation_t=observation_t,
                config=config,
                env=env,
                normalizer=normalizer,
                eval_deterministic=eval_deterministic,
                runtime_state=prior_runtime_state,
            )
            obs_repeat = observation_t.unsqueeze(0).repeat(prior_value.shape[0], 1)

            if episode_prior is None:
                episode_prior = prior_value.detach().cpu().numpy().copy()
                episode_structured_prior = structured_prior

            record_intermediates = bool(getattr(config, "save_denoise_process", False))
            lomap_projector = _get_lomap_projector(config, device, lomap_runtime_cache)
            traj_out = planner.sample_prior(
                obs_repeat,
                prior=prior_value,
                solver=config.planner_solver,
                planner_horizon=config.planner_horizon,
                sample_steps=config.planner_sampling_steps,
                temperature=config.planner_temperature,
                return_intermediates=record_intermediates,
                prior_init_noise_scale=getattr(config, "prior_init_noise_scale", 0.003),
                guidance=structured_prior.guidance if structured_prior is not None else None,
                lomap_projector=lomap_projector,
                lomap_context=_build_lomap_context(structured_prior),
            )

            if isinstance(traj_out, tuple):
                traj_all, traj_history = traj_out
            else:
                traj_all = traj_out
                traj_history = None
            traj_all = _sanitize_tensor(traj_all, clamp=1e3)

            if episode_denoise_history is None and traj_history is not None:
                episode_denoise_history = traj_history
            if len(all_plan_candidates) < len(all_trajectories) + 1:
                all_plan_candidates.append(traj_all.detach().cpu().numpy().copy())

            if traj_all.shape[0] > 1:
                goal_xy = getattr(env.unwrapped, "_target", None)
                traj, selected_plan_idx, plan_debug = _select_best_planned_traj(
                    traj_all,
                    critic=critic,
                    goal_xy=goal_xy,
                )
            else:
                traj = traj_all[:1]
                selected_plan_idx = 0
                plan_debug = {}
            traj = _sanitize_tensor(traj, clamp=1e3)
            if episode_structured_prior is not None:
                episode_structured_prior.debug_info["selected_plan_index"] = int(selected_plan_idx)
                episode_structured_prior.debug_info.update(plan_debug)

            policy_prior = torch.zeros((1, config.action_dim), device=device, dtype=torch.float32)
            next_obs_plan = traj[:, 1, :]
            next_obs_plan = _sanitize_tensor(next_obs_plan, clamp=1e3)
            obs_policy = observation_t.clone().unsqueeze(0)
            next_obs_policy = next_obs_plan.clone()

            if config.rebase_policy:
                next_obs_policy = torch.cat(
                    (next_obs_policy[:, :2] - obs_policy[:, :2], next_obs_policy[:, 2:]),
                    dim=1
                )
                obs_policy = torch.cat(
                    (
                        torch.zeros((obs_policy.shape[0], 2), device=device, dtype=obs_policy.dtype),
                        obs_policy[:, 2:]
                    ),
                    dim=1
                )

            action = policy.sample(
                policy_prior,
                model=policy.model,
                solver=config.policy_solver,
                n_samples=1,
                sample_steps=config.policy_sampling_steps,
                condition_cfg=torch.cat([obs_policy, next_obs_policy], dim=-1),
                temperature=config.policy_temperature,
            )
            action = _sanitize_tensor(action, clamp=10.0)

            action_np = _to_numpy_action(torch.squeeze(action))
            if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
                action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
            observation, reward, done, _ = env.step(action_np)
            episode_traj.append(np.array(observation, copy=True))

            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)

            if any(env_name in config.env_name for env_name in ["halfcheetah", "walker2d", "hopper"]):
                ep_reward += (reward * (1 - cum_done)) if t < config.max_path_length else reward
            elif any(env_name in config.env_name for env_name in ["antmaze", "kitchen"]):
                ep_reward += reward
            elif 'maze2d' in config.env_name:
                finished_before = finished.copy()
                finished |= (reward == 1.0)
                ep_reward += finished
                if success_step is None and np.any(finished) and not np.any(finished_before):
                    success_step = t
            else:
                raise NotImplementedError()

        all_trajectories.append(np.array(episode_traj, dtype=np.float32))
        all_success_steps.append(success_step)
        all_denoise_histories.append(episode_denoise_history)
        all_priors.append(episode_prior)
        all_structured_priors.append(episode_structured_prior)

        if getattr(config, "prior_struct_debug_runtime", False) and episode_structured_prior is not None:
            rebuild_count = int(prior_runtime_state.get("rebuild_count", 0))
            total_steps = int(prior_runtime_state.get("step_index", 0))
            print(f"[prior_struct] steps={total_steps} rebuilds={rebuild_count}")

        shared_prior_runtime_cache["occupancy"] = prior_runtime_state.get("occupancy")
        shared_prior_runtime_cache["map_prior_cache"] = prior_runtime_state.get("map_prior_cache", {})

        if any(env_name in config.env_name for env_name in ["halfcheetah", "walker2d", "hopper"]):
            episode_rewards.append(ep_reward)
        elif 'antmaze' in config.env_name:
            episode_rewards.append(np.clip(ep_reward, 0.0, 1.0))
        elif 'maze2d' in config.env_name:
            episode_rewards.append(ep_reward)
        elif 'kitchen' in config.env_name:
            episode_rewards.append(np.clip(ep_reward, 0.0, 4.0))
        else:
            raise NotImplementedError()

        mean = np.mean(
            np.array(list(map(lambda x: env.get_normalized_score(x), episode_rewards))).reshape(-1) * 100
        )
        progress.set_description_str(str(mean))

    episode_rewards_norm = list(map(lambda x: env.get_normalized_score(x), episode_rewards))
    episode_rewards_norm = np.array(episode_rewards_norm).reshape(-1) * 100
    mean = float(np.mean(episode_rewards_norm))
    std = float(np.std(episode_rewards_norm))
    stderr = float(std / np.sqrt(len(episode_rewards_norm)))
    var = float(np.var(episode_rewards_norm))
    print(mean, stderr)

    return {
        "return": mean,
        "return_mean": mean,
        "return_std": std,
        "return_stderr": stderr,
        "return_var": var,
        "episode_returns": episode_rewards_norm.tolist(),
        "trajectories": all_trajectories,
        "goals": all_goals,
        "success_steps": all_success_steps,
        "denoise_histories": all_denoise_histories,
        "priors": all_priors,
        "structured_priors": all_structured_priors,
        "plan_candidates": all_plan_candidates,
    }
