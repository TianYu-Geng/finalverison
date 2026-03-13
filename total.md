# Total Plan

## 核心结论

当前代码已经按这条链路融合完成：

1. `LoMAP projection`
2. `corridor guidance`
3. `diffusion update`
4. `candidate trajectories`
5. `critic scoring`
6. `best trajectory`

其中真正的工程插入点在 [diffusion.py](/Users/qinghuan/Desktop/diffusion/finalverison/diffusion.py) 的 `ContinuousDiffusionSDE.sample_prior(...)`。

## 已完成的融合

### 1. LoMAP 有效代码抽取

保留为轻量运行时目录：

- [lomap_runtime/__init__.py](/Users/qinghuan/Desktop/diffusion/finalverison/lomap_runtime/__init__.py)
- [lomap_runtime/local_manifold.py](/Users/qinghuan/Desktop/diffusion/finalverison/lomap_runtime/local_manifold.py)
- [lomap_runtime/datastore.py](/Users/qinghuan/Desktop/diffusion/finalverison/lomap_runtime/datastore.py)
- [lomap_runtime/projector.py](/Users/qinghuan/Desktop/diffusion/finalverison/lomap_runtime/projector.py)

作用：

- `local_manifold.py`：局部 PCA 子空间估计与投影
- `datastore.py`：直接读取服务器 LoMAP 目录下的 `dataset.h5`
- `projector.py`：提供统一的 LoMAP projector

当前默认优先搜索：

```text
/home/linux/tianyu/final/lomap/results/lomapdiffuser_d4rl_maze2d/maze2d-large-v1
```

如果该目录存在且包含 `dataset.h5`，不需要额外准备数据。

### 2. 当前系统主链路

入口文件：

- [eval_lomap_prior_struct.py](/Users/qinghuan/Desktop/diffusion/finalverison/eval_lomap_prior_struct.py)

主调用链：

1. [eval_pg_prior.py](/Users/qinghuan/Desktop/diffusion/finalverison/eval_pg_prior.py)
2. [evaluation.py](/Users/qinghuan/Desktop/diffusion/finalverison/evaluation.py)
3. [prior_struct](/Users/qinghuan/Desktop/diffusion/finalverison/prior_struct)
4. [diffusion.py](/Users/qinghuan/Desktop/diffusion/finalverison/diffusion.py)

### 3. 分阶段逻辑

当前 `sample_prior(...)` 的 denoising loop 已经实现为：

- reverse 前半段：对 `x_theta` 做 LoMAP manifold projection
- reverse 后半段：对 `xt` 再叠加 corridor guidance

这比之前“直接对 `xt` 先投影”更接近 LoMAP 原始思想，因为实际被约束的是去噪后的轨迹估计。

### 4. Candidate 选择闭环

这部分已经补齐：

- `prior_struct` 先选 `selected_future_obs`
- planner 输出 `traj_all`
- [evaluation.py](/Users/qinghuan/Desktop/diffusion/finalverison/evaluation.py) 现在优先用 critic 对 `traj_all` 打分
- 再选最佳轨迹进入 policy

也就是说，原来缺失的 `critic scoring -> best trajectory` 这一步已经接回来了。

## 已改动文件

- [diffusion.py](/Users/qinghuan/Desktop/diffusion/finalverison/diffusion.py)
- [evaluation.py](/Users/qinghuan/Desktop/diffusion/finalverison/evaluation.py)
- [eval_pg_prior.py](/Users/qinghuan/Desktop/diffusion/finalverison/eval_pg_prior.py)
- [eval_lomap_prior_struct.py](/Users/qinghuan/Desktop/diffusion/finalverison/eval_lomap_prior_struct.py)
- [prior_struct/fusion.py](/Users/qinghuan/Desktop/diffusion/finalverison/prior_struct/fusion.py)
- [lomap_runtime/__init__.py](/Users/qinghuan/Desktop/diffusion/finalverison/lomap_runtime/__init__.py)
- [lomap_runtime/local_manifold.py](/Users/qinghuan/Desktop/diffusion/finalverison/lomap_runtime/local_manifold.py)
- [lomap_runtime/datastore.py](/Users/qinghuan/Desktop/diffusion/finalverison/lomap_runtime/datastore.py)
- [lomap_runtime/projector.py](/Users/qinghuan/Desktop/diffusion/finalverison/lomap_runtime/projector.py)

## 运行方式

```bash
python eval_lomap_prior_struct.py \
  --env_name=maze2d-large-v1 \
  --seed=0 \
  --planner_ckpt=1000000 \
  --pg_ckpt=1000000 \
  --critic_ckpt=200000 \
  --use_prior_struct=True \
  --use_lomap=True \
  --lomap_store_path=/home/linux/tianyu/final/lomap/results/lomapdiffuser_d4rl_maze2d/maze2d-large-v1 \
  --prior_struct_enable_guidance=True
```

## 还保留但暂未删除的部分

[lomap](/Users/qinghuan/Desktop/diffusion/finalverison/lomap) 原始 research repo 还在仓库里，但当前运行链路已经不依赖它。它现在只作为源码参考。

如果下一步继续整理，可以直接删除整个 [lomap](/Users/qinghuan/Desktop/diffusion/finalverison/lomap) 目录，只保留 `lomap_runtime/`。
