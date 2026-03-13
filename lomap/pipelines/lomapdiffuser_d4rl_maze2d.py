'''
该脚本是一个完整的 Diffusion-based trajectory planning pipeline，用于在 D4RL Maze2D 环境中训练、准备数据和推理
'''
# 多行字符串：作为模块说明/脚本注释（不影响执行）

import os                         # 文件路径、目录创建等
import gym                        # OpenAI Gym 环境接口（用于 gym.make / 向量化环境等）
import d4rl                       # D4RL 离线 RL 数据集（为 Maze2D 提供 env.get_dataset()）
import h5py                       # HDF5 数据读写（用于保存检索数据集 dataset.h5）
import hydra                      # 配置管理框架（通过 @hydra.main 注入 args）
import numpy as np                # 数值计算
from tqdm import tqdm             # 进度条
from omegaconf import OmegaConf   # Hydra/OmegaConf 的配置对象工具（此脚本中未直接用到）

import torch                      # PyTorch
import torch.nn as nn             # 神经网络模块（此脚本中未直接用到 nn.*，但通常保留）
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度
from torch.utils.data import DataLoader                 # 数据加载器

from cleandiffuser.classifier import CumRewClassifier   # 累计奖励回归器（作为 guidance 的“打分器”）
from cleandiffuser.dataset.dataset_utils import loop_dataloader  # 无限循环 dataloader（避免手写 epoch）
from cleandiffuser.nn_classifier import HalfJannerUNet1d         # 分类器/回归器网络（用于估计 value/reward）
from cleandiffuser.nn_diffusion import JannerUNet1d              # 扩散轨迹网络（去噪模型）
from cleandiffuser.utils import report_parameters                # 打印参数量等统计

from cleandiffuser_ex.utils import set_seed                      # 统一设置随机种子（numpy/torch/cuda等）
from cleandiffuser_ex.diffusion import DiscreteDiffusionSDEEX    # 训练/采样核心：离散扩散 + (可选)投影/引导
from cleandiffuser_ex.faiss_index_wrapper import FaissIndexIVFWrapper  # FAISS IVF 近邻检索封装（用于投影）
from cleandiffuser_ex.dataset.d4rl_maze2d_dataset import D4RLMaze2DDataset, get_preprocess_fn
# D4RL Maze2D 数据集封装 + 预处理函数构造器


@hydra.main(config_path="../configs/lomapdiffuser/maze2d", config_name="maze2d", version_base=None)
# Hydra 入口：从 ../configs/lomapdiffuser/maze2d/maze2d.yaml 读取配置，注入到 pipeline(args)
def pipeline(args):

    set_seed(args.seed)  # 设置随机种子，确保训练/采样可复现（需 set_seed 内部覆盖 torch/cuda 等）

    # 保存路径 results/<pipeline_name>/<env_name>/
    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'

    if os.path.exists(save_path) is False:  # 若目录不存在则创建
        os.makedirs(save_path)

    # ---------------------- 创建数据集 ----------------------
    env = gym.make(args.task.env_name)  # 创建 Maze2D 环境（用于拿离线数据集与 normalized_score）
    preprocess_fn = get_preprocess_fn(['maze2d_set_terminals'], args.task.env_name)
    # 构造数据预处理函数列表：
    # - 'maze2d_set_terminals' 通常用于设置终止标记/修正 done/terminal 逻辑（Maze2D 常见）
    # - 具体实现决定 obs/act/val 的字段、终止处理方式等

    dataset = D4RLMaze2DDataset(
        env.get_dataset(),               # 从 D4RL env 取出离线轨迹数据（dict of arrays）
        preprocess_fn=preprocess_fn,     # 预处理函数（清洗/重构/裁剪）
        horizon=args.task.horizon,       # 轨迹片段长度 T（训练时采样长度为 T 的子轨迹）
        discount=args.discount)          # 折扣因子（用于计算累计回报/价值标签等）
    # dataset 通常会输出形如：
    # batch["obs"]["state"]: (B,T,obs_dim)
    # batch["act"]:          (B,T,act_dim)
    # batch["val"]:          (B,1) 或 (B,)（累计回报/价值标签，取决于实现）

    # 用于训练 diffusion
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,                   # 随机打乱样本（离线数据通常可打乱）
        num_workers=4,                  # 多进程加载
        pin_memory=True,                # 锁页内存，加速 CPU->GPU 拷贝
        drop_last=True)                 # 丢弃不足一个 batch 的尾部

    obs_dim, act_dim = dataset.o_dim, dataset.a_dim  # 观测维度/动作维度（由数据集封装提供）

    # --------------- 网络架构 -----------------
    # 用于 trajectory denoising（扩散去噪模型 εθ 或 x0θ）
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim,              # 输入通道维度：把 state 与 action 拼成轨迹向量
        model_dim=args.model_dim,        # 主干宽度
        emb_dim=args.model_dim,          # 时间步 embedding 维度
        dim_mult=args.task.dim_mult,     # 多尺度通道倍率
        timestep_emb_type="positional",  # 时间步编码类型（位置编码）
        attention=False,                # 是否用 attention（这里关闭）
        kernel_size=5)                  # 1D 卷积核大小（控制时间局部感受野）

    # 用于 classifier guidance（这里是回归 cumulative reward 的网络）
    nn_classifier = HalfJannerUNet1d(
        args.task.horizon,              # 轨迹长度（HalfJannerUNet1d 的接口通常需要 T）
        obs_dim + act_dim,              # 输入维度同上（state+action）
        out_dim=1,                      # 输出 1 维：预测累计回报/得分
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=args.task.dim_mult,
        timestep_emb_type="positional",
        kernel_size=3)                  # 分类器卷积核更小（更轻量，推理时也更快）

    print(f"======================= Diffusion 模型参数报告 =======================")
    report_parameters(nn_diffusion)     # 打印扩散网络参数量/结构摘要（取决于 report_parameters 实现）
    print(f"======================= Classifier 参数报告 =======================")
    report_parameters(nn_classifier)    # 打印分类器网络参数量/结构摘要
    print(f"==============================================================================")

    # --------------- Classifier Guidance --------------------
    classifier = CumRewClassifier(nn_classifier, device=args.device)
    # 用 nn_classifier 构造 CumRewClassifier：
    # - 内部创建 optimizer、EMA 模型
    # - loss 是 MSE(pred_R, R)
    # - logp() 返回 EMA 输出（用于引导时打分/求梯度）

    # ----------------- Masking -------------------
    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    # fix_mask: (T, D) 的硬约束 mask，典型含义：
    # mask=1 的位置在采样/训练中被“固定”为给定条件（例如起点/终点状态）

    fix_mask[[0, -1], :obs_dim] = 1.
    # 固定第 0 步与第 T-1 步的“观测维度”：
    # 即起点状态与终点状态（goal）作为条件，不让扩散过程改变这两帧的 state

    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    # loss_weight: (T, D) 的加权矩阵，用于对不同时间步/不同维度的损失加权

    loss_weight[0, obs_dim:] = args.action_loss_weight
    # 对第 0 步的 action 维度赋予 action_loss_weight（默认 1.）
    # 注意：这里只改了 t=0 的 action 权重；这是否符合预期要看你的任务：
    # - 若你不关心 action 拟合，可把 action_loss_weight 设小
    # - 若想全时域 action 都加权，应当是 loss_weight[:, obs_dim:] = ...
    # 这里的写法是“只在第一步 action 上调/下调”，通常是实现细节或作者的特定设定

    # --------------- Diffusion Model --------------------
    agent = DiscreteDiffusionSDEEX(
        nn_diffusion, None,                         # 第二个参数 None：通常是条件网络/额外模块占位（看实现）
        fix_mask=fix_mask,                          # 条件固定 mask
        loss_weight=loss_weight,                    # 损失权重
        classifier=classifier,                      # 用于 guidance/训练 classifier
        ema_rate=args.ema_rate,                     # 扩散模型 EMA 衰减率
        device=args.device,
        diffusion_steps=args.diffusion_steps,       # 扩散时间步数（DDPM T）
        predict_noise=args.predict_noise)           # 训练目标：预测噪声 ε 或预测 x0（由实现决定）

    # ---------------------- Training ----------------------
    if args.mode == "train":

        progress_bar = tqdm(total=args.diffusion_gradient_steps, desc="Training Progress")
        # 训练进度条：以 diffusion_gradient_steps 为总步数（不是 epoch）

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        # 扩散模型优化器的余弦退火 LR：T_max = diffusion_gradient_steps

        classifier_lr_scheduler = CosineAnnealingLR(agent.classifier.optim, args.classifier_gradient_steps)
        # 分类器优化器的余弦退火 LR：T_max = classifier_gradient_steps

        agent.train()  # 切换到训练模式（影响 dropout/bn 等；并可能设置内部标志）

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0.}
        # 用于累积 log_interval 内的平均 loss

        for batch in loop_dataloader(dataloader):
            # loop_dataloader 会无限 yield batch（相当于 while True: for batch in dataloader）
            # 所以终止条件由 n_gradient_step >= diffusion_gradient_steps 控制

            obs = batch["obs"]["state"].to(args.device)  # 轨迹观测序列 (B,T,obs_dim)
            act = batch["act"].to(args.device)           # 轨迹动作序列 (B,T,act_dim)
            val = batch["val"].to(args.device)           # 标签：累计回报/价值 (B,1) 或 (B,)

            x = torch.cat([obs, act], -1)                # 拼接成轨迹向量 (B,T,obs_dim+act_dim)

            # ----------- Gradient Step ------------
            log["avg_loss_diffusion"] += agent.update(x)['loss']
            # 更新扩散模型参数：
            # - 内部采样 t，加噪 x_t，做 denoise 预测并算 loss，然后 optimizer.step
            # - 返回 dict，取其中 'loss'

            diffusion_lr_scheduler.step()                # 每步更新一次 LR（与 step 数同步）

            if n_gradient_step <= args.classifier_gradient_steps:
                log["avg_loss_classifier"] += agent.update_classifier(x, val)['loss']
                # 若还在 classifier 训练步数范围内，则更新分类器：
                # - 输入 x 与标签 val
                # - classifier 的 loss 是 MSE(pred_R, val)

                classifier_lr_scheduler.step()           # 分类器 LR 更新

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                log["avg_loss_classifier"] /= args.log_interval
                print(log)                               # 打印平均 loss
                log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0.}  # 重置累积器

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")        # 保存扩散模型 ckpt（按步数）
                agent.save(save_path + f"diffusion_ckpt_latest.pt")                       # 覆盖 latest
                agent.classifier.save(save_path + f"classifier_ckpt_{n_gradient_step + 1}.pt")  # 保存分类器
                agent.classifier.save(save_path + f"classifier_ckpt_latest.pt")                # 覆盖 latest

            n_gradient_step += 1
            progress_bar.update(1)

            if n_gradient_step >= args.diffusion_gradient_steps:
                break                                    # 达到总训练步数则退出

    # ---------------------- Prepare Data (for FAISS projection) ----------------------
    elif args.mode == "prepare_data":

        dataset_size = min(1000000, len(dataset))        # 最多抽 1,000,000 条轨迹片段（受内存影响）
        normalizer = dataset.get_normalizer()            # 数据归一化器（训练/推理一致）

        traj_dataset = np.zeros((dataset_size, args.task.horizon, obs_dim + act_dim), dtype=np.float32)
        # 用于 FAISS 的轨迹库：存归一化后的 x（state+action），形状 (N,T,D)

        sg_dataset = np.zeros((dataset_size, 2, 2), dtype=np.float32)
        # 用于起点-终点的几何检索库：每条样本存 (start_xy, goal_xy)，形状 (N,2,2)

        gen_dl = DataLoader(
            dataset, batch_size=5000, shuffle=True,      # 大 batch 快速生成/搬运
            num_workers=4, pin_memory=True, drop_last=True
        )

        ptr = 0                                          # 写入指针
        with tqdm(total=dataset_size, desc=f"prepare_data: {ptr}/{dataset_size}", leave=False) as pbar:
            for batch in gen_dl:
                obs = batch["obs"]["state"].to(args.device)   # (B,T,obs_dim)
                act = batch["act"].to(args.device)            # (B,T,act_dim)
                x = torch.cat([obs, act], -1)                 # (B,T,D)
                bs = x.shape[0]

                if ptr + bs > dataset_size:                   # 最后一批可能超出目标大小
                    bs = dataset_size - ptr

                traj_dataset[ptr:ptr+bs] = x[:bs].cpu().numpy()
                # 保存“归一化空间”的轨迹（训练用的同一空间），用于后续投影/近邻

                # (x, y pos) unnormalized pos!
                sg_dataset[ptr:ptr+bs] = normalizer.unnormalize(obs.cpu().numpy())[:bs, [0, -1], :2]
                # 从 obs 里取第 0 帧和最后一帧，并取前两维作为 (x,y)
                # 注意这里先 unnormalize，再取 xy，保证几何距离是“真实坐标系”而不是归一化空间

                ptr += bs
                pbar.update(bs)

                if ptr >= dataset_size:
                    break

        with h5py.File(save_path + "dataset.h5", "w") as f:
            f.create_dataset(f"traj_dataset", data=traj_dataset)  # 写入轨迹库
            f.create_dataset(f"sg_dataset", data=sg_dataset)      # 写入 start-goal 库

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":
        agent.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")      # 加载扩散 ckpt（args.ckpt 可为步数或 latest）
        agent.classifier.load(save_path + f"classifier_ckpt_{args.ckpt}.pt")  # 加载分类器 ckpt

        agent.eval()  # 切换到推理模式（dropout/bn等）

        if not args.task.proj_range:
            faiss_wrapper = None
            # 若不启用 proj_range，则不做 FAISS 投影（采样纯靠扩散/引导）
        else:
            with h5py.File(save_path + "dataset.h5", "r") as f:
                traj_dataset = f["traj_dataset"][:]      # (N,T,D) 轨迹库（归一化空间）
                sg_dataset = f["sg_dataset"][:]          # (N,2,2) start-goal 库（真实坐标）

                dim_weights = np.ones((args.task.horizon, obs_dim + act_dim), dtype=np.float32)
                # 维度/时间步加权：影响 FAISS 相似度计算时每个维度的重要性

                dim_weights[:, obs_dim:] = args.task.action_dim_weight
                # 对 action 维度整体赋权（Maze2D 可能 action 不重要，则设 0；multi2d 可能设 1）

                dim_weights[[0, -1], :2] = args.task.sg_weight
                # 让起点与终点的 xy 坐标在相似度里占更大权重（强约束起终点几何接近）

                faiss_wrapper = FaissIndexIVFWrapper(
                    similarity_metric=args.faiss_similarity,  # cosine/inner_product/l2 等
                    nlist=args.faiss_nlist,                   # IVF 聚类中心数
                    data=traj_dataset,                        # 被检索的库
                    dim_weights=dim_weights,                  # 加权
                    device=args.device)                       # 可能把索引放 GPU（取决于实现）

                sg_faiss_wrapper = FaissIndexIVFWrapper(
                    similarity_metric="l2",                   # start-goal 用 L2（几何距离更直接）
                    nlist=args.faiss_nlist,
                    data=sg_dataset,
                    device=args.device)

        env_eval = gym.vector.make(args.task.env_name, args.num_envs, asynchronous=False)
        # 创建向量化评估环境（并行 num_envs 个）

        env_eval.seed(args.seed)                # 设置评估环境随机种子
        normalizer = dataset.get_normalizer()   # 与训练一致的 normalizer（用于 obs/target 归一化）

        episode_rewards = []        # 存每个 episode 的回报（向量化 env -> 每次是 (num_envs,)）
        episode_traj = []           # 存每个 episode 的采样轨迹（含 num_candidates）
        proj_mask_frac_list = []    # 记录投影 mask 为 True 的比例（用于统计）

        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
        # prior: (num_envs,T,D) 作为采样的条件轨迹模板（其中起点/终点被填充，其余为 0）

        for i in tqdm(range(args.num_episodes), desc="Inference Episodes"):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
            # obs: (num_envs, obs_dim) 当前观测
            # ep_reward: 累积回报（这里是标量或向量叠加，下面按 done mask 控制）
            # cum_done: 累积 done 标记（用于并行环境中统计哪些已经结束）
            # t: 环境交互步数计数

            if args.multi_task:
                [e.set_target() for e in env_eval.envs]   # 多任务时每个环境设置不同 target（实现依赖 env）
            targets = np.array([[*e.unwrapped._target, 0, 0] for e in env_eval.envs])
            # 取每个环境内部的 target（Maze2D 通常在 _target 里）
            # 并拼上 0,0（使维度匹配 obs_dim，常见 obs=[x,y,?,?]）

            while not np.all(cum_done) and t < 1000 + 1:
                # 只要还有任一环境未 done 且步数未超过上限，就继续

                if t == 0:
                    # -------- 第 0 步：先采样一条/多条整段轨迹计划 --------

                    prior[:, 0, :obs_dim] = torch.tensor(
                        normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                    # prior 的第 0 帧填入当前 obs（归一化空间）

                    prior[:, -1, :obs_dim] = torch.tensor(
                        normalizer.normalize(targets), device=args.device, dtype=torch.float32)
                    # prior 的最后一帧填入目标 target（归一化空间）

                    proj_mask = torch.ones((prior.shape[0], ), dtype=torch.bool, device=args.device)
                    # proj_mask: (num_envs,) 标记哪些样本要启用投影（默认全 True）

                    if args.task.proj_range and args.task.use_proj_mask:
                        sg = normalizer.unnormalize(prior[:, [0, -1], :obs_dim].cpu().numpy())[:, :, :2]
                        # 把 prior 的起终点从归一化空间还原到真实坐标，并取 xy

                        distances, idxs = sg_faiss_wrapper.search(sg, 1)
                        # 在 sg_dataset 里找最近的一个 (start,goal) 对，返回距离与索引

                        proj_mask = torch.tensor(
                            distances[:, 0] < args.task.proj_mask_threshold, device=args.device)
                        # 若几何距离小于阈值才投影（即：只对“训练集里出现过相似起终点”的任务启用投影）
                        # 目的通常是避免 OOD start-goal 时强行投影到不相关数据导致坏引导

                    traj, log = agent.sample(
                        prior.repeat(args.num_candidates, 1, 1),
                        # 把 prior 复制 num_candidates 份，形状变为 (num_candidates*num_envs,T,D)

                        solver=args.solver,                    # ddpm / ddim 等
                        n_samples=args.num_candidates * args.num_envs,
                        sample_steps=args.sampling_steps,       # 反向去噪步数
                        use_ema=args.use_ema,                   # 是否用 EMA 扩散模型采样
                        w_cg=args.task.w_cg,                    # classifier guidance 强度
                        temperature=args.temperature,           # 采样温度（随机性）
                        faiss_wrapper=faiss_wrapper,            # 轨迹库检索器（用于投影/近邻）
                        proj_range=args.task.proj_range,         # 在去噪时间段中启用投影的区间
                        proj_mask=proj_mask.repeat(args.num_candidates),
                        # 对候选也复制 mask，形状 (num_candidates*num_envs,)

                        n_manifold_samples=args.task.n_manifold_samples,  # 投影近邻采样数
                        tau=args.tau)                        # 投影/融合的平滑系数（具体作用看 sample 实现）

                    # No value guidance in maze2d -> just pick first plan
                    best_obs = traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)[
                            0, torch.arange(args.num_envs), :, :obs_dim]
                    # 将 traj reshape 回 (num_candidates,num_envs,T,D)，选第 0 个候选作为执行计划
                    # 再取 obs_dim 部分作为“计划的观测轨迹”

                    best_obs = normalizer.unnormalize(best_obs.cpu().numpy())
                    # 把计划轨迹转回真实坐标系，用于后续 waypoint 计算与环境交互

                # --------- 执行计划：根据计划轨迹生成动作并 step 环境 ---------
                if t < args.task.horizon - 1:
                    next_waypoint = best_obs[:, t + 1]
                    # 在 horizon 内：取计划的下一帧作为 waypoint
                else:
                    next_waypoint = best_obs[:, -1].copy()
                    next_waypoint[:, 2:] = 0
                    # 超过 horizon 后：一直追最后一个 waypoint，并把非 xy 的维度清零（避免无意义漂移）

                act = next_waypoint[:, :2] - obs[:, :2] + (next_waypoint[:, 2:] - obs[:, 2:])
                # 用“目标 waypoint 与当前 obs 的差”构造动作：
                # - 对 Maze2D，常见 action 就是 dx,dy 或 velocity 类控制
                # - 这里把前两维和后两维的差都加起来，假设 obs 结构为 [x,y,vx,vy] 或 [x,y,?,?]
                # 是否合理取决于 env 的动作定义；这是一种简单的 tracking 控制律

                obs, rew, done, info = env_eval.step(act)
                # 向量化 step：返回下一观测、奖励、done 标记、info

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                # 更新累计 done（只要某个 env done 过，就保持 done）

                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                # 累计回报：
                # - t<1000 时，对已经 done 的 env 不再累加（用 (1-cum_done) mask）
                # - 超过 1000 时直接加 rew（基本不会走到这句，因为 while 条件 t<1001）

            # --------- 统计投影 mask 的启用比例 ---------
            count_true = torch.count_nonzero(proj_mask).item()  # proj_mask 为 True 的个数
            mask_frac = count_true / float(proj_mask.numel())   # True 占比
            proj_mask_frac_list.append(mask_frac)               # 记录到列表

            # we want traj_np shape: (num_envs, num_candidates, horizon, dim)
            traj_np = traj.view(
                args.num_candidates, args.num_envs,
                args.task.horizon, -1
            ).permute(1, 0, 2, 3).cpu().numpy()
            # 将 traj 整理为 (num_envs,num_candidates,T,D) 便于保存与后处理

            episode_traj.append(traj_np)   # 保存本 episode 的计划轨迹

            episode_rewards.append(ep_reward)
            # 保存本 episode 的回报（通常是 (num_envs,)）

        # --------- 汇总所有 episode 的轨迹与回报 ---------
        episode_traj = np.array(episode_traj).reshape(
            args.num_episodes * args.num_envs, args.num_candidates, args.task.horizon, -1)
        # 变形为 (num_episodes*num_envs, num_candidates, T, D)

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        # 用 D4RL 的 normalized_score 归一化每个环境的回报（映射到 benchmark 分数尺度）

        episode_rewards = np.array(episode_rewards).reshape(args.num_episodes * args.num_envs)
        episode_rewards *= 100
        # D4RL 常用报告是 normalized_score * 100（百分制）

        mean = np.mean(episode_rewards)                       # 平均分
        err = np.std(episode_rewards) / np.sqrt(len(episode_rewards))  # 标准误（Std / sqrt(N)）

        proj_mask_fracs = np.array(proj_mask_frac_list)       # 投影 mask 启用比例（按 episode 统计）
        proj_mask_frac_mean = np.mean(proj_mask_fracs)        # 平均启用比例
        proj_mask_frac_std = np.std(proj_mask_fracs)          # 启用比例标准差

        result_str = (
            f"scores: {mean:.1f} +/- {err:.2f}\n"
            f"proj_mask_frac: {proj_mask_frac_mean:.3f} +/- {proj_mask_frac_std:.3f}"
        )
        print(result_str)                                     # 打印最终评估结果

    else:
        raise ValueError(f"Invalid mode: {args.mode}")
        # mode 不是 train/prepare_data/inference 时直接报错


if __name__ == "__main__":
    pipeline()  # Python 入口：执行 hydra 包装后的 pipeline（hydra 会解析命令行覆盖参数）