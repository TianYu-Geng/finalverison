import diffuser.utils as utils
import pdb

"""
文件概述：
该脚本构建并运行一个基于扩散模型的训练实验：解析命令行配置，实例化数据集、模型、扩散过程与 Trainer，
并执行一次前向/反向检查后进入训练循环。

输出/影响：
    - 在 `args.savepath` 下保存配置快照 (`*_config.pkl`)、训练中间结果与采样输出；
    - 训练期间会更新 `Trainer` 管理的模型/EMA 检查点并触发采样/保存操作。

后续使用者：
    - `Trainer` 在训练过程中使用此处构造的 `diffusion` 与 `model`；
    - 可视化/评估脚本或 `Policy` 会读取保存的检查点与样本来复现实验。
"""


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    """
    tap用法，例如下main就是抓换成命令行的参数 
    --dataset maze2d-large-v1 --config config.maze2d
    """
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d' # 通过read_config读取config/maze2d.py,获取了所有参数

'''
调用的是 utils.Parser.parse_args（）
'diffusion' 这个字符串会作为 experiment 参数传进去，在 config 文件中索引 base['diffusion']
返回的 args 是一个已经“就绪”的实验参数对象。
'''
args = Parser().parse_args('diffusion')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

# utils/Config 是所有对象（dataset / model / diffusion / trainer）的“统一入口”和“调度中枢”
# 后续的 `model_config` 与 `diffusion_config` 使用 `dataset.observation_dim` 与 `dataset.action_dim` 来设置模型尺寸与 transition_dim。
dataset_config = utils.Config(
    args.loader, # args.loader 的作用就是：指定“用哪个数据集类来构造 dataset”；args.loader 被传入 utils.Config 的 _class 参数
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

render_config = utils.Config(
    args.renderer, # args.renderer 的作用就是：指定“用哪个渲染器类来构造 renderer”；args.renderer 被传入 utils.Config 的 _class 参数
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

# observation_dim 和 action_dim 分别表示“单个时间步里，观测向量和动作向量的维度（向量长度）
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

# 定义轨迹扩散模型结构
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)


# 定义扩散过程
diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)


trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size, 
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference, # 画多少条对照轨迹
    n_samples=args.n_samples, # 画多少条“候选轨迹”
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

# 对模型与 diffusion 进行一次快速的前向与反向（sanity check）。
utils.report_parameters(model)
print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0]) #  SequenceDataset 的第 0 条 sample
loss, _ = diffusion.loss(*batch) # 做了一次完整的扩散噪声采样、模型前向和损失计算
loss.backward() # 反向传播测试
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

