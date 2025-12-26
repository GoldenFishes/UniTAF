"""
25.12.9
这里实现TTS+A2F的UniTextAudioFace联合模型中，tts输出的音频特征到a2f decoder的输入的映射。
主要确保维度和时间长度的一致。 (B, L, 1024) -> (B, L/2, 1024)

25.12.26
我们应当保持UniTextAudioFace联合模型中 a2f decoder完全符合 UniTalkerDecoder，
故而我们需要修改Audio Feature Projector使其 (B, L, 1024) -> (B, L/2, 768) 以符合原始 UniTalker-B-D0-D7.pt 权重中UniTalkerDecoder的输入
    或 (B, L, 1024) -> (B, L/2, 1024) 以符合原始 UniTalker-L-D0-D7.pt 权重中UniTalkerDecoder的输入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck1D(nn.Module):
    """
    分组 1-D  bottleneck：
    1×1 ↓ 压缩通道
    3×1 分组卷积  (kernel=3, stride=1, pad=1)
    1×1 ↑ 恢复通道
    只在最后一个 block 把 stride 设成 2 来做长度减半。
    """
    def __init__(self, in_ch, out_ch, stride=1, groups=8, mid_ratio=0.25):
        super().__init__()
        mid_ch = int(mid_ratio * in_ch)          # 瓶颈通道
        self.conv1 = nn.Conv1d(in_ch,  mid_ch, 1, bias=False)
        self.bn1  = nn.BatchNorm1d(mid_ch)
        self.conv2 = nn.Conv1d(mid_ch, mid_ch, 3, stride=stride, padding=1,
                               groups=groups, bias=False)
        self.bn2  = nn.BatchNorm1d(mid_ch)
        self.conv3 = nn.Conv1d(mid_ch, out_ch, 1, bias=False)
        self.bn3  = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # 残差支路
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False)
            if stride != 1 or in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + self.shortcut(x))

class AudioFeatureProjector(nn.Module):
    """
    用 1-D 卷积把长度降采样到 1/2，同时特征维 in_dim -> out_dim 采样至与后续a2f Decoder一致。
    """
    def __init__(self, in_dim: int = 1024, out_dim: int = 768, groups=8, num_blocks=3):
        super().__init__()
        # 第一步：1024 -> 896（保持长度）
        self.head = Bottleneck1D(in_dim, 256, stride=1, groups=groups)
        # 中间堆叠：全部 stride=1
        self.blocks = nn.Sequential(*[
            Bottleneck1D(256, 256, stride=1, groups=groups)
            for _ in range(num_blocks - 2)
        ])
        # 最后一步：896 -> 768 且 stride=2 完成长度减半
        self.tail = Bottleneck1D(256, out_dim, stride=2, groups=groups)

        self.norm = nn.LayerNorm(out_dim)  # 在特征维度上归一化
        self.audio_scale = nn.Parameter(torch.tensor(1.0))  # 可学习幅度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, L, in_dim) -> (B, L/2, out_dim)
        """
        # x: (B, L, C)
        x = x.transpose(1, 2)  # -> (B, C, L)
        x = self.head(x)
        x = self.blocks(x)
        x = self.tail(x)  # -> (B, 768, L/2)
        x = x.transpose(1, 2)  # -> (B, L/2, 768)
        return self.norm(x) * self.audio_scale