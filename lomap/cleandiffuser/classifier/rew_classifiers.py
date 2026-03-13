from typing import Optional  # Optional[T] 表示参数可为 T 或 None（类型标注用途）

from cleandiffuser.nn_classifier import BaseNNClassifier  # 神经网络分类器/回归器的抽象基类（定义 forward 接口等）
from .base import BaseClassifier  # 项目内的 BaseClassifier：封装优化器、EMA、通用训练/推理逻辑


class CumRewClassifier(BaseClassifier):
    """
    累积回报（Cumulative Reward）预测器/“分类器”（实际上是回归器）：
    输入轨迹/样本 x 和扩散噪声条件 noise，预测对应的回报 R（标量或向量），用于训练或引导。
    """

    def __init__(
            self,
            nn_classifier: BaseNNClassifier,     # 具体的网络实现（例如 HalfJannerUNet1d 等），满足 BaseNNClassifier 接口
            device: str = "cpu",                 # 运行设备：'cpu' 或 'cuda:0' 等
            optim_params: Optional[dict] = None, # 优化器参数（学习率、权重衰减等）；None 表示用 BaseClassifier 默认值
    ):
        # 调用父类构造器：负责
        # 1) 把 nn_classifier 放到 device
        # 2) 构建优化器 self.optim（用 optim_params）
        # 3) 构建 EMA 模型 self.model_ema，并设置 EMA 衰减系数
        #
        # 这里的参数含义（需对照 BaseClassifier.__init__ 的签名）：
        # 0.995：EMA 衰减率（越大越平滑，更新越慢）
        # None ：某个可选的 scheduler / clip / cond_dim 之类占位参数（具体看 BaseClassifier 定义）
        super().__init__(nn_classifier, 0.995, None, optim_params, device)

    def loss(self, x, noise, R):
        # 前向预测：用当前训练模型 self.model（非 EMA）预测回报
        # 第三个参数传 None：表示不使用额外条件 c（比如 goal/label），或接口要求占位
        pred_R = self.model(x, noise, None)

        # MSE 损失：最常见的回归目标
        # 期望 R 和 pred_R 形状可广播或一致（例如 [B,1] 或 [B]）
        return ((pred_R - R) ** 2).mean()

    def update(self, x, noise, R):
        # 一步标准训练更新：
        # 1) 清梯度
        self.optim.zero_grad()

        # 2) 计算损失
        loss = self.loss(x, noise, R)

        # 3) 反向传播
        loss.backward()

        # 4) 参数更新
        self.optim.step()

        # 5) EMA 更新：把 self.model 的参数以指数滑动平均的方式写入 self.model_ema
        self.ema_update()

        # 返回可记录的标量日志（方便 tensorboard / wandb）
        return {"loss": loss.item()}

    def logp(self, x, noise, c=None):
        # 给“引导”用的打分函数：
        # 这里命名为 logp，但实际上直接返回 EMA 模型的输出（pred_R）
        # 在某些代码里会把这个输出当作 log-prob 或能量/奖励信号使用；
        # 是否真的是 logp 取决于上层如何用它（例如 guidance = ∇x logp）。
        return self.model_ema(x, noise)