import os
import random
from absl import app, flags

import gym
import numpy as np
import torch
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'maze2d-large-v1', 'environment name')
flags.DEFINE_string('mode', 'train', 'Mode of operation')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_string('device', 'cuda:0', 'Device to use')
flags.DEFINE_string('project', 'pg', 'Project name')

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
flags.DEFINE_integer('planner_diffusion_gradient_steps', 1000000, 'Planner diffusion gradient steps') # 原1000000 10000
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('log_interval', 1000, 'Log interval') # 原1000 100
flags.DEFINE_integer('eval_interval', 100000, 'Eval interval') # 原100000 2000

flags.DEFINE_integer('num_episodes', 100, 'Number of episodes') # 原100 10
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

# PG
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
flags.DEFINE_integer('planner_sampling_steps_train', 5, 'Planner planner_sampling_steps_train steps')

flags.DEFINE_string("entity", "entity_name", "entity name")
flags.DEFINE_string('pipeline_type', 'separate', 'Type of pipeline')

flags.DEFINE_integer('save_interval', 100000, 'Save interval for PG prior checkpoints') # 原100000 2000


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
            if v.numel() == 1:
                out[k] = v.item()
            else:
                out[k] = v.float().mean().item()
        elif isinstance(v, np.ndarray):
            out[k] = float(v.mean())
        elif isinstance(v, (int, float, np.floating)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def load_checkpoint(model_path, map_location='cpu'):
    return torch.load(model_path, map_location=map_location)


# [MOD-2] 新增：保存 PG 模型文件
def save_pg_checkpoint(agent_state, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    ckpt = {
        "prior": agent_state.prior.state_dict() if hasattr(agent_state, "prior") else None,
        "target_prior": agent_state.target_prior.state_dict()
        if hasattr(agent_state, "target_prior") and agent_state.target_prior is not None else None,

        "critic_T": agent_state.critic_T.state_dict() if hasattr(agent_state, "critic_T") else None,
        "target_critic_T": agent_state.target_critic_T.state_dict()
        if hasattr(agent_state, "target_critic_T") and agent_state.target_critic_T is not None else None,

        "value_T": agent_state.value_T.state_dict() if hasattr(agent_state, "value_T") else None,
        "target_value_T": agent_state.target_value_T.state_dict()
        if hasattr(agent_state, "target_value_T") and agent_state.target_value_T is not None else None,

        "prior_optimizer": agent_state.prior_optimizer.state_dict()
        if hasattr(agent_state, "prior_optimizer") and agent_state.prior_optimizer is not None else None,
        "critic_T_optimizer": agent_state.critic_T_optimizer.state_dict()
        if hasattr(agent_state, "critic_T_optimizer") and agent_state.critic_T_optimizer is not None else None,
        "value_T_optimizer": agent_state.value_T_optimizer.state_dict()
        if hasattr(agent_state, "value_T_optimizer") and agent_state.value_T_optimizer is not None else None,

        "global_step": getattr(agent_state, "global_step", 0),
    }

    torch.save(ckpt, model_path)


def main(_):
    from easydict import EasyDict
    # [MOD-3] 删掉 wandb 导入，彻底本地运行
    # import wandb
    from dataset.d4rl_maze2d_dataset import D4RLMaze2DSeqDataset
    import pg
    from evaluation import evaluate_prior

    config = EasyDict(FLAGS.flag_values_dict())
    config.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    set_seed(config.seed)

    if config.env_name == 'maze2d-umaze-v1':
        config.max_path_length = 300
    elif config.env_name == 'maze2d-medium-v1':
        config.max_path_length = 600
    elif config.env_name == 'maze2d-large-v1':
        config.max_path_length = 800
    else:
        raise ValueError('Unknown environment name')

    # wandb.init(project=config.project, entity=config.entity)
    # wandb.config.update(dict(config))

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
    update_fn = pg.update

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

    if config.mode == "train":
        config.use_planner_condition = False
        config.use_policy_condition = True
        config.train_condition = False

        normalizer = planner_dataset.normalizer

        i = 0
        total_steps = config.planner_diffusion_gradient_steps
        pbar = tqdm(total=total_steps, desc="PG training", dynamic_ncols=True)

        while i < total_steps:
            steps_this_round = min(config.log_interval, total_steps - i)
            log_accumulator = {}

            for _ in range(steps_this_round):
                agent_state, info = update_fn(
                    planner,
                    critic,
                    None,
                    agent_state,
                    config,
                    planner_dataset,
                )

                info_dict = info._asdict() if hasattr(info, "_asdict") else dict(info)
                info_dict = move_info_to_scalar_dict(info_dict)

                for k, v in info_dict.items():
                    if k not in log_accumulator:
                        log_accumulator[k] = []
                    log_accumulator[k].append(v)

                i += 1
                if hasattr(agent_state, 'global_step'):
                    agent_state.global_step = i

            pbar.update(steps_this_round)

            log_dict = {
                f"train/{k}": float(np.mean(v))
                for k, v in log_accumulator.items()
                if len(v) > 0
            }

            if len(log_dict) > 0:
                pbar.set_postfix({
                    k.replace("train/", ""): round(v, 6)
                    for k, v in log_dict.items()
                })

            # wandb.log(log_dict, step=i)

            if len(log_dict) > 0:
                print(
                    f"[step {i}/{total_steps}] " +
                    " ".join([f"{k}={v:.6f}" for k, v in log_dict.items()])
                )
            else:
                print(f"[step {i}/{total_steps}] no log info")

            # [MOD-4] 新增：定期保存 PG prior checkpoint
            if i % config.save_interval == 0 or i == total_steps:
                pg_ckpt_dir = f'./checkpoint/PG/{config.env_name}/{config.seed}/'
                pg_ckpt_path = os.path.join(pg_ckpt_dir, f'model_{i}.pt')
                save_pg_checkpoint(agent_state, pg_ckpt_path)
                print(f"[checkpoint] saved: {pg_ckpt_path}")

            if i % config.eval_interval == 0 or i == total_steps:
                print(f"[eval] start evaluation at step {i}")

                if hasattr(agent_state, 'target_prior') and agent_state.target_prior is not None:
                    prior = agent_state.target_prior
                elif hasattr(agent_state, 'prior'):
                    prior = agent_state.prior
                else:
                    raise AttributeError("agent_state must contain target_prior or prior in the PyTorch version.")

                prior.eval()

                eval_info = evaluate_prior(
                    planner,
                    policy,
                    prior,
                    config,
                    env,
                    normalizer,
                    eval_deterministic=config.eval_deterministic,
                )

                eval_log = {f"det_eval_target/{k}": v for k, v in eval_info.items()}
                # wandb.log(eval_log, step=i)

                print(
                    f"[eval step {i}] " +
                    " ".join([f"{k}={v:.6f}" for k, v in eval_log.items()])
                )

        pbar.close()

    # [MOD-5] 新增：支持测试/仅加载 PG prior
    elif config.mode == "eval":
        pg_ckpt_path = f'./checkpoint/PG/{config.env_name}/{config.seed}/model_{config.planner_ckpt}.pt'
        pg_ckpt = load_checkpoint(pg_ckpt_path, map_location=config.device)

        if pg_ckpt.get("target_prior", None) is not None:
            agent_state.target_prior.load_state_dict(pg_ckpt["target_prior"])
            prior = agent_state.target_prior
        elif pg_ckpt.get("prior", None) is not None:
            agent_state.prior.load_state_dict(pg_ckpt["prior"])
            prior = agent_state.prior
        else:
            raise ValueError(f"No prior found in PG checkpoint: {pg_ckpt_path}")

        prior.eval()
        normalizer = planner_dataset.normalizer

        eval_info = evaluate_prior(
            planner,
            policy,
            prior,
            config,
            env,
            normalizer,
            eval_deterministic=config.eval_deterministic,
        )

        print("[eval final] " + " ".join([f"{k}={v:.6f}" for k, v in eval_info.items()]))

    else:
        raise ValueError(f"Unsupported mode: {config.mode}")


if __name__ == '__main__':
    app.run(main)