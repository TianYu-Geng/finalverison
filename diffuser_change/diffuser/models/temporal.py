"""
该文件定义扩散模型使用的时间序列 U-Net 噪声预测网络：将轨迹张量与时间步 `t` 映射为同形状的噪声（或 `x0`）预测。
它输出的网络模块（如 `TemporalUnet`）会被 `diffuser/models/diffusion.py:GaussianDiffusion` 在训练损失与反向扩散采样中调用。
"""

import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        # 把时间embedding变成“能加到 feature map 上”的形状
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels), # [B, out_channels]
            Rearrange('batch t -> batch t 1'), # [B, out_channels, 1]
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        # 第一层卷积处理，然后将时间嵌入（通过线性层 + reshape）按通道相加
        # x: [batch, channels, time_steps]（文件中 U-Net 把 horizon 映射到 time dim）
        out = self.blocks[0](x) + self.time_mlp(t)
        # 第二层卷积进一步处理特征
        out = self.blocks[1](out)
        # 残差连接：当输入/输出通道不同时用 1x1 卷积对齐
        return out + self.residual_conv(x)

class TemporalUnet(nn.Module):
    """
    将轨迹张量 `x` 与扩散时间步 `t` 映射为噪声预测（U-Net 结构，1D 卷积沿 horizon 维度建模）。
    """

    def __init__(
        self,
        horizon,
        transition_dim, # 每个时间步的向量维度，通常是 action_dim + observation_dim
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)

        # 编码器 downs：每一级 2 个 residual block + 可选下采样
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        # 这里不改变分辨率，只在最深处加工特征。
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        # 解码器 ups：每一级 2 个 residual block + 可选上采样
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time):
        """
        对输入轨迹做 U-Net 前向计算，并返回与输入同形状的预测张量。
        """

        # 输入 x 的原始形状为: [batch, horizon, transition_dim]
        # 为了沿时间轴做 1D 卷积（把 horizon 视作特征长度），先重排为 [batch, time, channels]
        # 注意：此处 channels 对应原来的 transition_dim 或中间通道数
        x = einops.rearrange(x, 'b h t -> b t h')

        # time: 标量或向量时间步输入（通常为扩散步 t），通过 time_mlp 得到时间 embedding
        t = self.time_mlp(time)
        # 用栈保存 encoder 阶段的特征用于 skip connections
        h = []

        # 下采样路径：每一级有两个残差块，然后可能下采样
        for resnet, resnet2, downsample in self.downs:
            # 每个 resnet 接受输入特征和时间 embedding
            x = resnet(x, t)
            x = resnet2(x, t)
            # 保存用于后续上采样阶段的跳跃连接
            h.append(x)
            # 下采样沿时间/长度维度减半（由 Downsample1d 实现）
            x = downsample(x)

        # 中间瓶颈块（不改变时间分辨率）
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # 上采样路径：每一级先与对应的 skip features 拼接（按通道 dim=1），再走残差块
        for resnet, resnet2, upsample in self.ups:
            # 拼接 encoder 对应的特征（通道数翻倍），dim=1 是通道维
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            # 上采样恢复时间/长度分辨率
            x = upsample(x)

        # 最终 1x1 卷积将通道数投影回原始 transition_dim
        x = self.final_conv(x)

        # 把形状还原为原始的 [batch, horizon, transition_dim]
        x = einops.rearrange(x, 'b t h -> b h t')
        return x

# TemporalValue：用类似的“下采样卷积网络”输出一个标量 value
class TemporalValue(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        time_dim=None,
        out_dim=1,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = time_dim or dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])

        print(in_out)
        for dim_in, dim_out in in_out:

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            horizon = horizon // 2

        fc_dim = dims[-1] * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        # 与 TemporalUnet 类似，先重排时间维度以便使用 Conv1d 风格的模块
        x = einops.rearrange(x, 'b h t -> b t h')

        # 时间嵌入用于与卷积特征拼接/注入（这里在 ResidualTemporalBlock 中相加）
        t = self.time_mlp(time)

        # 逐级下采样并堆叠残差块（不保留 skip connections，因为这是一个 value 回归网络）
        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        # 展平所有时间步与通道，作为全连接层的输入
        x = x.view(len(x), -1)
        # 与时间 embedding 拼接，输出标量/向量值（如 value 或评分）
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out


# class TemporalMixerUnet(nn.Module):

#     def __init__(
#         self,
#         horizon,
#         transition_dim,
#         cond_dim,
#         dim=32,
#         dim_mults=(1, 2, 4, 8),
#     ):
#         super().__init__()
#         # self.channels = channels

#         dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))

#         time_dim = dim
#         self.time_mlp = nn.Sequential(
#             SinusoidalPosEmb(dim),
#             nn.Linear(dim, dim * 4),
#             nn.Mish(),
#             nn.Linear(dim * 4, dim),
#         )
#         self.cond_mlp = nn.Sequential(
#             nn.Linear(cond_dim, dim * 4),
#             nn.GELU(),
#             nn.Linear(dim * 4, dim),
#         )

#         self.downs = nn.ModuleList([])
#         self.ups = nn.ModuleList([])
#         num_resolutions = len(in_out)

#         print(in_out)
#         for ind, (dim_in, dim_out) in enumerate(in_out):
#             is_last = ind >= (num_resolutions - 1)

#             self.downs.append(nn.ModuleList([
#                 ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
#                 ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
#                 nn.Identity(),
#                 Downsample1d(dim_out) if not is_last else nn.Identity()
#             ]))

#             if not is_last:
#                 horizon = horizon // 2

#         mid_dim = dims[-1]
#         self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, kernel_size=5, embed_dim=time_dim, horizon=horizon)
#         self.mid_attn = nn.Identity()
#         self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, kernel_size=5, embed_dim=time_dim, horizon=horizon)

#         for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
#             is_last = ind >= (num_resolutions - 1)

#             self.ups.append(nn.ModuleList([
#                 ResidualTemporalBlock(dim_out * 2, dim_in, kernel_size=5, embed_dim=time_dim, horizon=horizon),
#                 ResidualTemporalBlock(dim_in, dim_in, kernel_size=5, embed_dim=time_dim, horizon=horizon),
#                 nn.Identity(),
#                 Downsample1d(dim_in) if not is_last else nn.Identity()
#             ]))

#             if not is_last:
#                 horizon = horizon * 2

#         self.final_conv = nn.Sequential(
#             # TemporalHelper(dim, dim, kernel_size=5),
#             Conv1dBlock(dim, dim, kernel_size=5),
#             nn.Conv1d(dim, transition_dim, 1),
#         )


#     def forward(self, x, cond, time):
#         '''
#             x : [ batch x horizon x transition ]
#         '''
#         t = self.time_mlp(time)
#         # cond = self.cond_mlp(cond)
#         cond = None

#         h = []

#         # x = x[:,None]
#         # t = torch.cat([t, cond], dim=-1)

#         x = einops.rearrange(x, 'b h t -> b t h')

#         for resnet, resnet2, attn, downsample in self.downs:
#             # print('0', x.shape, t.shape)
#             x = resnet(x, t, cond)
#             # print('resnet', x.shape, t.shape)
#             x = resnet2(x, t, cond)
#             # print('resnet2', x.shape)
#             ##
#             x = einops.rearrange(x, 'b t h -> b t h 1')
#             x = attn(x)
#             x = einops.rearrange(x, 'b t h 1 -> b t h')
#             ##
#             # print('attn', x.shape)
#             h.append(x)
#             x = downsample(x)
#             # print('downsample', x.shape, '\n')

#         x = self.mid_block1(x, t, cond)
#         ##
#         x = einops.rearrange(x, 'b t h -> b t h 1')
#         x = self.mid_attn(x)
#         x = einops.rearrange(x, 'b t h 1 -> b t h')
#         ##
#         x = self.mid_block2(x, t, cond)
#         # print('mid done!', x.shape, '\n')

#         for resnet, resnet2, attn, upsample in self.ups:
#             # print('0', x.shape)
#             x = torch.cat((x, h.pop()), dim=1)
#             # print('cat', x.shape)
#             x = resnet(x, t, cond)
#             # print('resnet', x.shape)
#             x = resnet2(x, t, cond)
#             # print('resnet2', x.shape)
#             ##
#             x = einops.rearrange(x, 'b t h -> b t h 1')
#             x = attn(x)
#             x = einops.rearrange(x, 'b t h 1 -> b t h')
#             ##
#             # print('attn', x.shape)
#             x = upsample(x)
#             # print('upsample', x.shape)
#         # pdb.set_trace()
#         x = self.final_conv(x)

#         # x = x.squeeze(dim=1)

#         ##
#         x = einops.rearrange(x, 'b t h -> b h t')
#         ##
#         return x
