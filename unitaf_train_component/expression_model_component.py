'''
这里实现用于将口型补齐为情感丰富的表情的表情残差模型 expression model。

Expression module 负责在 “嘴已经对齐” 的前提下，补全情感驱动的非口型表情。学的是：
    Δ_face = f(mouth, emotion, latent)
接收：
    1. Mouth ARKit （来自冻结的A2F）
    2. GPT latent （时间对齐）
    3. emovec / emotion embedding（显式）


GPT latent  ─┐
             ├─ Temporal Encoder ─┐
mouth_motion ┘                    ├─ Fusion → Residual Head
                                  ↑
                              Emotion FiLM
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- FiLM modulation ----------
class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x, cond):
        """
        x:    [B, T, H]
        cond: [B, C]
        """
        gamma, beta = self.linear(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return x * (1 + gamma) + beta

# ---------- Expression Model ----------
class ExpressionModel(nn.Module):
    def __init__(
        self,
        gpt_dim: int,
        face_dim: int,
        emo_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()

        # 1. 输入投影
        self.gpt_proj = nn.Linear(gpt_dim, hidden_dim)
        self.mouth_proj = nn.Linear(face_dim, hidden_dim)

        # 2. Temporal backbone（轻量 Transformer）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 3. Emotion modulation
        self.film = FiLM(emo_dim, hidden_dim)

        # 4. 输出 residual
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, face_dim),
        )

        # 5. 控制 residual 幅度（非常重要）
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        expression_feature,   # GPT latent [B, T, D_gpt]
        mouth_motion,         # [B, T, D_face]
        emotion_contrl,       # [B, D_emo]
    ):
        # 对齐时间长度
        T = mouth_motion.shape[1]
        expression_feature = expression_feature[:, :T]

        # 投影
        x_gpt = self.gpt_proj(expression_feature)
        x_mouth = self.mouth_proj(mouth_motion)

        x = x_gpt + x_mouth

        # Temporal modeling
        x = self.temporal_encoder(x)

        # Emotion modulation
        x = self.film(x, emotion_contrl)

        # Residual
        residual = self.residual_head(x)

        return residual * self.residual_scale














