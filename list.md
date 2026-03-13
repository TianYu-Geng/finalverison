# `prior_struct` 双层先验集成方案

## 0. 目标重定义

这次的 `prior_struct` 不应定义成“一条融合后的轨迹”，而应定义成“一个带结构约束的先验场”。

它至少包含三层信息：

- `corridor support`
  - 回答“哪里允许采样”
- `PG density`
  - 回答“允许区域里哪里更像离线数据分布”
- `weak centerline attraction`
  - 回答“采样时如何轻微往中轴回拉，避免贴边、折返、撞墙风险升高”

因此整体逻辑应改成：

1. 先用地图结构构造 corridor support / feasible band
2. 再把 PG prior 变成 corridor-aware prior distribution
3. 最后在实际采样阶段只加很弱的 toward-center guidance

不建议的定义：

- 不把 `prior_struct` 定义成“对单条 PG proposal 做后处理投影”
- 不把 corridor 用成“把轨迹硬拽成 A* centerline”的手段

推荐的定义：

- `prior_struct` 是一个“采样前可查询的结构先验对象”
- 它约束 support、表达 density bias，并提供弱 guidance

## 1. 核心改动文件

### `evaluation.py`

- 保留：
  - rollout 主循环
  - reward / success 统计
  - trajectory / denoise / prior 记录逻辑
- 主要改动：
  - 把当前 `prior.eval_forward(...) -> prior_value -> planner.sample_prior(...)` 这一段替换成统一的 `prior_struct` 构造入口
  - 在采样前先获取 corridor-aware prior candidates / reweighting 结果
  - 在采样时接入一个弱 `center pull` guidance 钩子
- 目标：
  - `evaluation.py` 只负责“在每个 step 请求一个统一 prior，并送入 planner”

### `eval_pg_prior.py`

- 保留：
  - 配置构建
  - DV / PG checkpoint 加载
  - rollout、denoise、prior 可视化框架
- 主要改动：
  - 将 PG prior 推理逻辑抽到单独模块
  - 将地图结构先验和融合逻辑移出，不继续堆进脚本
- 建议：
  - 新建 `eval_prior_struct.py` 作为新的主入口
  - `eval_pg_prior.py` 保留为旧基线或对照实验脚本

### `plan_mub2.py`

- 保留：
  - 作为地图结构先验的原型脚本 / smoke test
- 适合抽离：
  - occupancy 提取
  - start / goal 落格
  - multi A*
  - diverse path 过滤
  - soft corridor / guidance field 构造
  - 结构先验可视化
- 不建议：
  - 让评估主流程直接 import 该脚本
  - 继续把生产逻辑写在 `main()` 里

### `diffusion.py`

- 不再默认完全不改
- 原因：
  - 你当前需要的第三层是“采样阶段的弱中心线引导”
  - 这件事天然发生在 `sample_prior()` 内，而不是只在采样前构造一条轨迹
- 建议：
  - MVP 允许先不改 `diffusion.py`，只做 support-aware candidate filtering
  - 第二步尽快在 `sample_prior()` 中加入弱 guidance 接口，而不是轨迹后处理

---

## 2. 建议新增文件

### `prior_struct/types.py`

- 职责：
  - 定义统一数据结构
- 建议包含：
  - `MapStructPrior`
  - `PGProposal`
  - `StructuredPrior`
- 输入：
  - numpy / torch 基础对象
- 输出：
  - dataclass 风格对象
- 对接：
  - `evaluation.py` 消费 `StructuredPrior` 中的候选 proposal、support 和 weak guidance 参数

### `prior_struct/map_prior.py`

- 职责：
  - 从 Maze2D 地图构建第一层先验
- 输入：
  - `env` 或 `occupancy`
  - `start_xy`
  - `goal_xy`
  - A* / corridor 相关配置
- 输出：
  - `MapStructPrior`
  - 包含 `occupancy`、`start_cell`、`goal_cell`、`multi_paths`
  - 以及：
    - `support_hr`
    - `distance_to_center_hr`
    - `distance_to_boundary_hr`
    - `centerline_potential_hr`
    - `grad_center_row_hr`
    - `grad_center_col_hr`
- 对接：
  - 底层复用 `diffuser_change/maze2d_prior/astar_utils.py`
  - 底层复用 `diffuser_change/maze2d_prior/field_utils.py`

### `prior_struct/pg_adapter.py`

- 职责：
  - 封装第二层 PG prior 推理
  - 生成候选 proposal，并做 corridor-aware filtering / reweighting
- 输入：
  - `prior_model`
  - normalized observation
  - normalizer
  - config
- 输出：
  - `PGProposal`
  - 建议包含：
    - `candidates_future_obs`
    - `candidate_scores_pg`
    - `candidate_scores_support`
    - `candidate_scores_total`
    - `selected_future_obs`
- 对接：
  - 从 `eval_pg_prior.py` 抽取 PG prior 加载和单步 proposal 生成逻辑

### `prior_struct/fusion.py`

- 职责：
  - 进行分层组合，而不是轨迹投影
  - 将 `corridor support + PG density + weak guidance params` 封装成统一结构
- 输入：
  - `PGProposal`
  - `MapStructPrior`
  - `normalizer`
  - fusion config
- 输出：
  - `StructuredPrior`
  - 至少包含：
    - `selected_future_obs`
    - `support_field`
    - `density_field` 或 candidate-level density info
    - `guidance_field`
    - `debug_info`
- 对接：
  - 供 `evaluation.py` 直接调用

### `prior_struct/provider.py`

- 职责：
  - 提供统一入口
- 输入：
  - 当前 observation
  - env
  - normalizer
  - PG prior model
  - config
- 输出：
  - `StructuredPrior`
- 对接：
  - `evaluation.py` 只依赖这一层，不依赖内部细节

### `prior_struct/guidance.py`

- 职责：
  - 采样阶段的弱 center pull guidance
- 输入：
  - 当前采样轨迹 `xt`
  - `MapStructPrior`
  - guidance 强度参数
- 输出：
  - 对当前位置的弱修正项或 guidance gradient
- 对接：
  - 后续接到 `diffusion.py::sample_prior()`

### `eval_prior_struct.py`

- 职责：
  - 新的评估 / 可视化入口
- 输入：
  - 命令行参数
- 输出：
  - 评估结果
  - rollout 图
  - corridor support 图
  - PG candidate / reweighting 图
  - selected proposal 图
  - weak guidance 图
- 对接：
  - 基于 `eval_pg_prior.py` 改造而来

---

## 3. 从现有文件中提取什么逻辑

### 从 `plan_mub2.py` 提取

#### 适合复用

- `extract_occupancy`
- `nearest_free_cell`
- `astar_multi_paths`
- `filter_diverse_paths`
- `build_sequence`
- `build_gated_line_centered_prior_and_potential`
- `query_guidance_field`
- 各类结构先验可视化

#### 需要重构

- 把脚本式顺序流程改成纯函数接口
- 把参数从 `Parser` 风格迁移成显式 config / dataclass
- 把“support 构造”和“centerline guidance 构造”拆开
- 把“场的边界约束”和“中线吸引项”分成两类对象

#### 不建议直接耦合

- `diffuser.datasets`
- `diffuser.utils.Config`
- renderer 初始化逻辑
- `main()` 中串起来的 demo 流程

### 从 `eval_pg_prior.py` 提取

#### 适合复用

- `build_config()`
- `load_dv_models(...)`
- `load_pg_prior(...)`
- `render_maze2d_rollout(...)`
- `render_denoise_snapshot(...)`
- `render_denoise_grid(...)`
- `render_prior_only(...)`

#### 需要重构

- PG prior 单步推理逻辑抽出为 `pg_adapter.py`
- occupancy 提取逻辑与现有 `maze2d_prior` 统一
- prior 可视化从“只画 PG prior”改成：
  - corridor support
  - PG candidate set
  - support-aware selected proposal
  - weak guidance

#### 不建议直接耦合

- 把融合算法直接写进 `eval_pg_prior.py`
- 让这个脚本承担结构先验构建职责

### 从 `evaluation.py` 提取 / 保留

#### 建议保留

- 主评估循环
- episode reward 统计
- normalized score 统计
- trajectory 记录
- goal / success_step / denoise_histories / priors 记录

#### 建议重构

- 当前写死的：
  - `dist = prior.eval_forward(obs_repeat)`
  - `prior_value = ...`
  - `planner.sample_prior(..., prior=prior_value, ...)`
- 改成：
  - `prior_struct = prior_provider.build(...)`
  - `selected_prior = prior_struct.selected_future_obs`
  - `planner.sample_prior(..., prior=selected_prior, guidance=prior_struct.guidance, ...)`

#### 不建议直接耦合

- A* 搜索
- corridor 构造
- 场查询
- 可视化绘图细节

---

## 4. 完整数据流 / 执行流程

### Step 1. 读取 Maze2D 环境

- 输入：
  - `env_name`
  - seed
- 输出：
  - `env`
  - 当前 `observation`
  - `goal_xy`
- 中间表示：
  - world 坐标下的 observation
- 实现位置：
  - `eval_prior_struct.py`
  - `evaluation.py`

### Step 2. 提取 occupancy 并映射起终点网格

- 输入：
  - `env`
  - `observation[:2]`
  - `goal_xy`
- 输出：
  - `occupancy[h, w]`
  - `start_cell`
  - `goal_cell`
- 中间表示：
  - 二值地图
  - 离散网格坐标
- 实现位置：
  - `prior_struct/map_prior.py`

### Step 3. 执行 multi A*

- 输入：
  - `occupancy`
  - `start_cell`
  - `goal_cell`
  - A* 超参数
- 输出：
  - 若干条拓扑可行路径 `multi_paths`
- 中间表示：
  - `[(cell_path, cost), ...]`
- 实现位置：
  - `prior_struct/map_prior.py`

### Step 4. 构造 soft corridor

- 输入：
  - `occupancy`
  - `multi_paths`
  - corridor 超参数
- 输出：
  - `support_hr`
  - `distance_to_center_hr`
  - `distance_to_boundary_hr`
  - `centerline_potential_hr`
  - `grad_center_row_hr`
  - `grad_center_col_hr`
- 中间表示：
  - 高分辨率 support field
  - 高分辨率弱 guidance field
- 实现位置：
  - `prior_struct/map_prior.py`

### Step 5. 调用 PG prior 生成候选 proposal

- 输入：
  - normalized observation
  - PG prior model
- 输出：
  - proposal candidate set
- 中间表示：
  - normalized `future_obs`，形状 `[K, H-1, obs_dim]`
- 实现位置：
  - `prior_struct/pg_adapter.py`

### Step 6. 计算 corridor-aware support score

- 输入：
  - candidate proposals
  - `MapStructPrior`
- 输出：
  - 每个候选的 support score / boundary risk / out-of-support penalty
- 中间表示：
  - candidate-level score table
- 实现位置：
  - `prior_struct/pg_adapter.py`

### Step 7. support-aware filtering / reweighting

- 输入：
  - PG candidate set
  - support score
  - PG 原始 density bias
- 输出：
  - corridor-aware selected proposal
- 中间表示：
  - 候选重排序或重加权分布
- 实现位置：
  - `prior_struct/fusion.py`

### Step 8. 形成最终 `prior_struct`

- 输入：
  - `MapStructPrior`
  - selected proposal
  - weak guidance 配置
- 输出：
  - `StructuredPrior`
- 中间表示：
  - 一个先验场对象，而不是单条融合轨迹
- 实现位置：
  - `prior_struct/fusion.py`
  - `prior_struct/provider.py`

### Step 9. 接回原有规划 / 采样 / evaluation

- 输入：
  - `StructuredPrior.selected_future_obs`
  - `StructuredPrior.guidance`
- 输出：
  - `planner.sample_prior(...)` 的 structured init + weak guidance
- 中间表示：
  - 初始 proposal 仍与旧 prior 兼容
  - guidance 作为附加采样控制项
- 实现位置：
  - `evaluation.py`

---

## 5. 分层先验设计重点

## 5.1 `prior_struct` 的正确语义

`prior_struct` 不应表示“最终融合轨迹”，而应表示一个采样前的分层先验对象：

```python
StructuredPrior(
    support=...,            # corridor support / feasible band
    pg_density=...,         # PG-driven candidate density bias
    selected_future_obs=...,# 当前时刻选中的 structured init
    guidance=...,           # weak center pull during sampling
    debug_info=...,
)
```

其中：

- `support` 负责限制采样空间
- `pg_density` 负责表达数据偏好
- `guidance` 只负责局部稳定，不负责重构全局路径

## 5.2 corridor support 的表示形式

建议包含两层表示：

- 离散层：
  - `multi_paths`
  - `cell_path`
- 连续层：
  - `support_hr`
  - `distance_to_center_hr`
  - `distance_to_boundary_hr`
  - `centerline_potential_hr`
  - `grad_center_row_hr`
  - `grad_center_col_hr`
  - 可查询函数 `query(x, y)`

原因：

- 离散层适合调试和解释拓扑
- 连续层适合：
  - support gating
  - boundary 风险评估
  - 弱中心线 guidance

## 5.3 PG density 的输出形式

PG prior 仍建议输出 normalized 未来轨迹，但语义要改成“候选 proposal 集合”，而不是单一样本：

- `candidates_future_obs`
- 形状建议：`[K, H-1, obs_dim]`
- 每个候选有：
  - `score_pg`
  - `score_support`
  - `score_total`

这样才能把 PG prior 变成：

- corridor support 内的 density bias
- 而不是无约束生成后再硬投影

## 5.4 support 与 density 在哪里结合

建议分两层：

- 候选层：
  - 将 PG candidates 反归一化到 world XY
  - 在连续空间上评估 support consistency
- 选择层：
  - 保留 PG density bias
  - 用 support score 做 gating / reweighting / filtering
- 采样层：
  - 再把选中的 proposal 送回 planner
  - 只加弱 center pull

不建议：

- 直接在离散网格上做硬投影
- 直接把中心线当最终轨迹模板
- 只对单条 proposal 做一次 post-hoc 修正后就结束

## 5.5 三层确定策略

### 第一层：hard/soft support gating

- 做法：
  - 用 multi A* + soft corridor 生成 corridor mask / distance field
  - 对明显落在 corridor 外的 proposal 点赋低权重，或直接判为无效
  - 目标是保证 proposal 总体落在合理拓扑通道里
- 优点：
  - 最符合“先定义允许生成空间”的目标
  - 能有效避免拓扑错误
- 缺点：
  - score 设计要谨慎，避免 corridor 太窄导致候选都被否掉

### 第二层：PG prior inside corridor

- 做法：
  - PG prior 生成一批候选
  - 对每个候选计算 corridor consistency
  - 用 support score 对候选进行过滤、重排或重加权
  - 必要时只做很小幅度的“进 corridor 带内”修正
- 优点：
  - 保留数据分布特性
  - 不需要先改训练
- 缺点：
  - 需要设计 candidate scoring / selection 逻辑

### 第三层：weak center pull during sampling

- 做法：
  - 在实际采样时，对落向 corridor 边缘的点施加很弱的 toward-center 修正
  - 只用于防止贴边、拐角外飘和撞墙风险增加
- 优点：
  - 角色清晰，是局部稳态项
  - 不会主导全局形状
- 缺点：
  - 需要对 `sample_prior()` 增加 guidance 接口
- 工程复杂度：
  - 中

## 5.6 最适合先做的 MVP

最适合的 MVP 不是“轨迹后处理投影”，而是：

- `support-aware candidate filtering`
- 暂时不在 diffusion 内注入 guidance，或者只预留接口
- 先实现：
  - corridor support field
  - PG candidate set
  - candidate support scoring
  - selected proposal 替换旧 prior

第二步再加：

- weak center pull during sampling

## 5.7 如何避免三层互相压制

### 避免 support 过强

- support 主要做 gating，不做“吸附到中心线”
- corridor 宽度不要太窄
- 允许 near-boundary 样本存在，只在明显越界时强惩罚

### 避免 PG density 过强

- 对 corridor 外点给明确 penalty
- 对全局拓扑错误候选直接过滤
- 对靠近边界的 proposal 降低被选中概率

### 避免 center pull 过强

- guidance 只在 sampling 阶段生效
- guidance 系数保持小
- 只修正局部偏移，不改 proposal 全局拓扑

## 5.8 最终 `prior_struct` 的数据结构

建议：

```python
StructuredPrior(
    support={
        "occupancy": ...,
        "multi_paths": ...,
        "support_hr": ...,
        "distance_to_boundary_hr": ...,
        "distance_to_center_hr": ...,
    },
    pg_density={
        "candidates_future_obs": ...,
        "score_pg": ...,
        "score_support": ...,
        "score_total": ...,
    },
    selected_future_obs=...,   # normalized, [B, H-1, obs_dim]
    guidance={
        "type": "weak_center_pull",
        "strength": ...,
        "field": ...,
    },
    debug_info=...,
)
```

这样可以同时满足：

- 对旧 planner 接口兼容
- 明确区分 support / density / guidance 三层职责
- 对调试和可视化友好

---

## 6. 推荐的最小可行实现路径

### MVP 目标

- 改动尽量少
- 跑通 Maze2D
- `prior_struct` 真正替代旧 prior
- 能看到 support / candidate filtering / selected prior 的定性区别

### 实施顺序

#### 第一步：抽离地图结构先验

- 从 `plan_mub2.py` 中抽出：
  - occupancy
  - A*
  - corridor 构造
- 落到：
  - `prior_struct/map_prior.py`

#### 第二步：抽离 PG prior proposal 生成

- 从 `eval_pg_prior.py` 和 `evaluation.py` 中抽出：
  - `prior.eval_forward(...)`
  - tanh squash
  - 多候选 proposal 张量整理
- 落到：
  - `prior_struct/pg_adapter.py`

#### 第三步：先实现 support-aware filtering

- 不做轨迹后处理投影
- 先做：
  - candidate 生成
  - support score 计算
  - candidate filtering / reweighting
- 输出：
  - `selected_future_obs`
- 落到：
  - `prior_struct/fusion.py`

#### 第四步：加统一 provider

- 在：
  - `prior_struct/provider.py`
- 实现：
  - `build_prior_struct(observation, env, normalizer, prior_model, config)`

#### 第五步：改 `evaluation.py`

- 将旧逻辑：
  - `prior.eval_forward(...)`
- 改为：
  - `prior_provider.build(...)`
- 保证最终仍传给：
  - `planner.sample_prior(..., prior=structured_prior.selected_future_obs, ...)`

#### 第六步：新增评估脚本和可视化

- 新建：
  - `eval_prior_struct.py`
- 生成：
  - rollout 图
  - corridor support 图
  - PG candidates 图
  - support score 热图 / 统计
  - selected prior 图

#### 第七步：第二阶段加入 weak center pull

- 进入 `diffusion.py::sample_prior()`
- 新增可选 guidance 参数
- 只在每步 denoise 中对 XY 加很弱的 center guidance
- 不改变全局 proposal 选择逻辑

### MVP 验证标准

- 能在 Maze2D 上正常运行
- `planner.sample_prior()` 的输入已经不是原始 PG prior，而是 support-aware selected prior
- 可视化中能看到：
  - corridor 明确限定可行带
  - PG candidates 体现经验模板
  - selected prior 落在 corridor 内，但未被硬拽成 A*
- evaluation 指标能正常输出
- 第二阶段再验证：
  - weak center guidance 能降低贴边和出 corridor 风险，但不显著改变全局形状

---

## 7. 一个关键现实约束

当前仓库中的 PG prior 在现有评估流程里只显式依赖 `observation`，没有显式接入 `goal`。

这意味着：

- 严格意义上的“基于 start / goal 的数据分布先验”目前并未完全成立
- 当前更接近“从当前状态出发的经验轨迹模板”
- 不重训 PG prior
- 直接使用现有 PG prior 作为 proposal generator
- 再由地图结构先验把 proposal 拉向当前 goal 对应的拓扑 corridor
