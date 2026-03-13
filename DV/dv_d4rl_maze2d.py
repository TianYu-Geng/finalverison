import os
import sys
import random
from absl import app, flags

import gym
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('.')

FLAGS = flags.FLAGS

# =========================
# 基本运行参数
# =========================
flags.DEFINE_string('env_name', 'maze2d-umaze-v1', 'environment name')
flags.DEFINE_string('mode', 'train', 'Mode of operation')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_string('device', 'cuda:0', 'Device to use')
flags.DEFINE_string('project', 'dv', 'Project name')

# =========================
# 方法结构相关参数
# =========================
flags.DEFINE_string('guidance_type', 'MCSS', 'Type of guidance')
flags.DEFINE_string('planner_net', 'transformer', 'Type of planner network')
flags.DEFINE_string('pipeline_type', 'separate', 'Type of pipeline')
flags.DEFINE_bool('rebase_policy', True, 'Rebase policy position')

# =========================
# 奖励与折扣相关参数
# =========================
flags.DEFINE_bool('continous_reward_at_done', True, 'Continous reward at done')
flags.DEFINE_string('reward_tune', 'iql', 'Reward tune')
flags.DEFINE_float('discount', 1.0, 'Discount factor')
flags.DEFINE_integer('planner_d_model_divide', 64, 'divide planner model dimension')

# =========================
# Planner 扩散模型参数
# =========================
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

# =========================
# Policy 扩散模型参数
# =========================
flags.DEFINE_string('policy_solver', 'ddpm', 'Policy solver')
flags.DEFINE_integer('policy_hidden_dim', 256, 'Policy hidden dimension')
flags.DEFINE_integer('policy_diffusion_steps', 10, 'Policy diffusion steps')
flags.DEFINE_integer('policy_sampling_steps', 10, 'Policy sampling steps')
flags.DEFINE_bool('policy_predict_noise', True, 'Policy predict noise')
flags.DEFINE_float('policy_ema_rate', 0.995, 'Policy EMA rate')
flags.DEFINE_float('policy_learning_rate', 0.0003, 'Policy learning rate')
flags.DEFINE_float('critic_learning_rate', 0.0003, 'Critic learning rate')

# =========================
# 训练步数相关参数
# =========================
flags.DEFINE_integer('use_diffusion_policy', 1, 'Use diffusion policy')
flags.DEFINE_integer('invdyn_gradient_steps', 200000, 'Inverse dynamics gradient steps')
flags.DEFINE_integer('policy_diffusion_gradient_steps', 1000000, 'Policy diffusion gradient steps')
flags.DEFINE_integer('planner_diffusion_gradient_steps', 1000000, 'Planner diffusion gradient steps')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('log_interval', 1000, 'Log interval')
flags.DEFINE_integer('save_interval', 200000, 'Save interval')

# =========================
# 评估和采样相关参数
# =========================
flags.DEFINE_integer('num_episodes', 100, 'Number of episodes')
flags.DEFINE_integer('planner_num_candidates', 50, 'Number of planner candidates')
flags.DEFINE_integer('planner_ckpt', 1000000, 'Planner checkpoint')
flags.DEFINE_integer('critic_ckpt', 200000, 'Critic checkpoint')
flags.DEFINE_integer('policy_ckpt', 1000000, 'Policy checkpoint')
flags.DEFINE_integer('invdyn_ckpt', 200000, 'Inverse dynamics checkpoint')
flags.DEFINE_bool('planner_use_ema', True, 'Use EMA for planner')
flags.DEFINE_float('policy_temperature', 0.5, 'Policy temperature')
flags.DEFINE_bool('policy_use_ema', True, 'Use EMA for policy')

# =========================
# 轨迹长度/规划窗口参数
# =========================
flags.DEFINE_integer('max_path_length', 800, 'Maximum path length')
flags.DEFINE_integer('planner_horizon', 32, 'Planner horizon')
flags.DEFINE_integer('stride', 15, 'Stride')
flags.DEFINE_float('planner_temperature', 1.0, 'Planner temperature')
flags.DEFINE_float('planner_target_return', 1.0, 'Planner target return')
flags.DEFINE_float('planner_w_cfg', 1.0, 'Planner weight for CFG')

# =========================
# wandb 相关参数
# =========================
flags.DEFINE_string("entity", "2107165871", "entity name")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_info_to_scalar_dict(info_dict):
    out = {}
    for k, v in info_dict.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v.item() if v.numel() == 1 else v.float().mean().item()
        elif isinstance(v, np.ndarray):
            out[k] = float(v.mean())
        elif isinstance(v, (int, float, np.floating)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def save_checkpoint(agent_state, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    ckpt = {
        'critic': agent_state.critic.state_dict(),
        'planner_model': agent_state.planner_model.state_dict(),
        'policy_model': agent_state.policy_model.state_dict(),
        'target_critic': agent_state.target_critic.state_dict() if getattr(agent_state, 'target_critic', None) is not None else None,
        'target_planner_model': agent_state.target_planner_model.state_dict() if getattr(agent_state, 'target_planner_model', None) is not None else None,
        'target_policy_model': agent_state.target_policy_model.state_dict() if getattr(agent_state, 'target_policy_model', None) is not None else None,
        'critic_optimizer': agent_state.critic_optimizer.state_dict() if getattr(agent_state, 'critic_optimizer', None) is not None else None,
        'planner_optimizer': agent_state.planner_optimizer.state_dict() if getattr(agent_state, 'planner_optimizer', None) is not None else None,
        'policy_optimizer': agent_state.policy_optimizer.state_dict() if getattr(agent_state, 'policy_optimizer', None) is not None else None,
        'global_step': getattr(agent_state, 'global_step', 0),
    }
    torch.save(ckpt, model_path)


def load_checkpoint(model_path, map_location='cpu'):
    return torch.load(model_path, map_location=map_location)


def main(_):
    from easydict import EasyDict
    import wandb

    from dataset.d4rl_maze2d_dataset import D4RLMaze2DSeqDataset
    import DV
    from evaluation import evaluate

    config = EasyDict(FLAGS.flag_values_dict())
    config.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    set_seed(config.seed)

    # wandb.init(project=config.project, entity=config.entity)
    # wandb.config.update(dict(config))

    if config.env_name == 'maze2d-umaze-v1':
        config.max_path_length = 300
    elif config.env_name == 'maze2d-medium-v1':
        config.max_path_length = 600
    elif config.env_name == 'maze2d-large-v1':
        config.max_path_length = 800
    else:
        raise ValueError('Unknown environment name')

    env = gym.make(config.env_name)

    planner_dataset = D4RLMaze2DSeqDataset(
        env.get_dataset(),
        horizon=config.planner_horizon,
        discount=config.discount,
        continous_reward_at_done=config.continous_reward_at_done,
        reward_tune=config.reward_tune,
        stride=config.stride,
        learn_policy=False,
        center_mapping=(config.guidance_type != "cfg"),
        device=config.device,
    ).to(config.device)

    policy_dataset = D4RLMaze2DSeqDataset(
        env.get_dataset(),
        horizon=config.planner_horizon,
        discount=config.discount,
        continous_reward_at_done=config.continous_reward_at_done,
        reward_tune=config.reward_tune,
        stride=config.stride,
        learn_policy=True,
        center_mapping=(config.guidance_type != "cfg"),
        device=config.device,
    ).to(config.device)

    config.observation_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]

    planner, policy, agent_state = DV.init(config)
    update_fn = DV.update

    if config.mode == "train":
        config.use_planner_condition = False
        config.use_policy_condition = True
        config.train_condition = True

        i = 0
        total_steps = config.planner_diffusion_gradient_steps
        if not hasattr(agent_state, 'global_step'):
            agent_state.global_step = 0

        pbar = tqdm(total=total_steps, desc="training", dynamic_ncols=True)

        while i < total_steps:
            steps_this_round = min(config.log_interval, total_steps - i)
            log_accumulator = {}

            for _ in range(steps_this_round):
                agent_state, info = update_fn(
                    planner,
                    policy,
                    agent_state,
                    config,
                    planner_dataset,
                    policy_dataset,
                )

                info_dict = info._asdict() if hasattr(info, "_asdict") else dict(info)
                info_dict = move_info_to_scalar_dict(info_dict)

                for k, v in info_dict.items():
                    if k not in log_accumulator:
                        log_accumulator[k] = []
                    log_accumulator[k].append(v)

                i += 1
                agent_state.global_step = i

            log_dict = {
                f"train/{k}": float(np.mean(v))
                for k, v in log_accumulator.items()
                if len(v) > 0
            }

            # wandb.log(log_dict, step=i)

            pbar.update(steps_this_round)

            if len(log_dict) > 0:
                pbar.set_postfix({
                    k.replace("train/", ""): round(v, 6)
                    for k, v in log_dict.items()
                })
                print(
                    f"[step {i}/{total_steps}] " +
                    " ".join([f"{k}={v:.6f}" for k, v in log_dict.items()])
                )
            else:
                print(f"[step {i}/{total_steps}] no log info")

            if i % config.save_interval == 0:
                checkpoint_dir = f'./checkpoint/DV/{config.env_name}/{config.seed}/'
                model_path = os.path.join(checkpoint_dir, f'model_{i}.pt')
                save_checkpoint(agent_state, model_path)
                print(f"[checkpoint] saved: {model_path}")

        pbar.close()

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

        critic = agent_state.critic
        if critic_ckpt.get('critic', None) is not None:
            critic.load_state_dict(critic_ckpt['critic'])

        planner.model.eval()
        policy.model.eval()
        critic.eval()

        normalizer = planner_dataset.normalizer

        with torch.no_grad():
            eval_info = evaluate(
                planner,
                policy,
                critic,
                planner.model,
                policy.model,
                config,
                env,
                normalizer,
            )

        # wandb.log({f"eval_final/{k}": v for k, v in eval_info.items()}, step=0)

    else:
        raise ValueError(f"Unsupported mode: {config.mode}")


if __name__ == '__main__':
    app.run(main)