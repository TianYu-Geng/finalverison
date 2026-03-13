import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]


plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
]

base = {

    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 2e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 50,
        'save_parallel': False,
        
        # ✅ 评估时选 50 个参考起点；对每个起点，模型生成 10 条轨迹作为候选
        'n_reference': 50,
        'n_samples': 10,

        'bucket': None,
        'device': 'cuda',
    },

    'plan': {
        'batch_size': 1, # 调整使得planner在同样条件下生成多条轨迹供选择
        'device': 'cuda',

        ## diffusion model
        'horizon': 256, # 时序长度
        'n_diffusion_steps': 256, # 去噪的步数
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': False, # 如果fasle，终点target使用默认环境提供的固定位置

        ## risk-aware planning defaults
        'risk_lambda': 1.0,
        'risk': {
            'num_candidates': 16,
            'segment_length': 32,
            'prefix_fraction': 0.5,
            'hard_prefix_risk': 2.5,
            'risk_trend_thresh': 0.1,
            'progress_trend_thresh': -0.01,
            'trend_window': 3,
            'restart_mode': 'prefix',  # none|soft|prefix
            'max_soft_restarts': 4,
            'risk_update_every': 8,
            'step_weights': {
                'safe': 0.5,
                'ood': 0.5,
                'dyn': 0.1,
                'llh': 0.25,
            },
        },
        'outer': {
            'likelihood_checks': 4,
            'likelihood_threshold': 1.0,
            'local_risk_threshold': 2.0,
            'global_risk_threshold': 4.5,
            'short_horizon_ratio': 0.5,
        },

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

}

#------------------------ overrides ------------------------#

'''
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
'''

maze2d_umaze_v1 = {
    'diffusion': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
    'plan': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
}

maze2d_large_v1 = {
    'diffusion': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
    'plan': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
}
