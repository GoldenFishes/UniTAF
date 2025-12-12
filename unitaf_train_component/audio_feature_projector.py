"""
25.12.9
这里实现TTS+A2F的UniTextAudioFace联合模型中，tts输出的音频特征到a2f decoder的输入的映射。
主要确保维度和时间长度的一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioFeatureProjector(nn.Module):
    """
    用 1-D 卷积把长度降采样到 1/2，同时保持特征维 1024 不变。
    """
    def __init__(self, in_dim: int = 1024, out_dim: int = 1024):
        super().__init__()
        #  kernel=2, stride=2  -> 长度减半
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=2, stride=2)

        # 初始化成 1000 倍（让模型自己慢慢降到正常幅度）
        self.audio_scale = nn.Parameter(torch.tensor(1.0))  # 可学习幅度
        self.norm = nn.LayerNorm(out_dim)  # 在特征维度上归一化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, 1024)
        return: (B, L/2, 1024)
        """
        # 降采样
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)       # (B, (L+1)//2, 1024)
        x = self.norm(x) * self.audio_scale

        return x