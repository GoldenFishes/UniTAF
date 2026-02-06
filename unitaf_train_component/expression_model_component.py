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
mouth_motion ┘                    ├─ Fusion → Residual Head → Δ_face
                                  ↑
                              Emotion FiLM
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# FiLM modulation
class FiLM(nn.Module):
    '''
    Feature-wise Linear Modulation 感情作为调制而不是内容

    Feature-wise Linear Modulation（特征逐通道线性调制）
    输入是情感条件向量 cond，对来自序列的中间表示 x
    进行通道方向的 scale + bias 调制。

    这种调制在图像条件生成、音频条件生成等任务中非常常见，
    它允许情感影响隐藏特征的显著性，而不会破坏时间结构。
    该机制本质是 affine 变换作用在特征维度上。
    '''
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        # cond_dim -> hidden_dim * 2（分别生成 gamma 和 beta）
        self.linear = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x, cond):
        """
        x:    [B, T, H]
        cond: [B, C]
        """
        # 线性生成 scale (gamma) 和 bias (beta)
        gamma, beta = self.linear(cond).chunk(2, dim=-1)
        # 扩展时间维度
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        # 特征维度上进行 FiLM 调制
        return x * (1 + gamma) + beta

# Cross Attention Block (用于 cross_attn 模式)
class CrossAttnBlock(nn.Module):
    '''
    一个跨注意力融合块，它包含：
      1. Self-Attention（自注意力）：让当前表示自身内部进行交互
      2. Cross-Attention（跨注意力）：让当前表示的 query 去注意另一个序列的 key/value
      3. 前馈网络（Feed Forward）：进一步组合交互后的信息

    Cross-Attention 让一个模态可以“问询”另一个模态，
    从而直接整合另一个模态的关键信息（更精细的多模态融合）:contentReference[oaicite:1]{index=1}。
    """
    '''
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        # 自注意力层：在 x 自身内部进行信息交换
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 跨注意力层：Query 来自 x，Key/Value 来自 cross_input
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # layernorm 和残差结构是 Transformer 稳定性设计的一部分
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # 前馈模块（增强非线性）
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, cross_input):
        """
        x:           当前主模态表示 (batch, T, hidden)
        cross_input: 跨注意力模态表示 (batch, T, hidden)
        """
        # self-attention
        z = self.self_attn(x, x, x)[0]  # x 对自身做注意力，以捕捉序列内部结构关系
        x = self.norm1(x + z)  # 残差 + layernorm
        # cross-attention
        # 用 x 作为 Query，cross_input 作为 Key/Value
        # 这一步是典型的跨模态注意力融合机制，结构类似于 Transformer 中 decoder cross-attn
        z2 = self.cross_attn(x, cross_input, cross_input)[0]
        # 融合 cross attention 结果
        x = self.norm2(x + z2)
        # feed forward
        z3 = self.ff(x)
        x = self.norm3(x + z3)
        return x


class ExpressionModel(nn.Module):
    '''
    这是一个条件残差生成器（conditional residual generator）
    目标是在口型已对齐的前提下，补齐非口型表情自由度。

    mode:
        - 'film':  传统的时序 + FiLM 控制
        - 'cross_attn': 使用跨注意力（Cross Attention）进行模态融合
    '''
    def __init__(
        self,
        gpt_dim: int,
        face_dim: int,
        emo_dim: int,
        # film参数
        hidden_dim: int = 256,
        num_layers: int = 4,
        # cross_attn参数
        nhead: int = 4,
        dropout: float = 0.1,
        # 模式控制
        mode: str = "film",
    ):
        super().__init__()

        self.mode = mode

        # 输入投影网络（共有）
        self.gpt_proj = nn.Linear(gpt_dim, hidden_dim)
        self.mouth_proj = nn.Linear(face_dim, hidden_dim)
        self.emo_proj = nn.Linear(emo_dim, hidden_dim)

        # film 模式结构
        self.film = FiLM(emo_dim, hidden_dim)  # 仅用于 film
        self.transformer_film = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers
        )

        # cross_attn 模式结构
        self.cross_layers = nn.ModuleList(
            [CrossAttnBlock(d_model=hidden_dim, nhead=nhead, dropout=dropout)
             for _ in range(num_layers)]
        )

        # 输出层（所有模式通用）
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, face_dim)
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def temporal_align(self, x, target_len):
        """
        x: [B, T, D]
        target_len: int, 目标时间步长度
        """
        B, T, D = x.shape
        if T == target_len:
            return x
        x = F.interpolate(x.permute(0, 2, 1), size=target_len, mode="linear", align_corners=False)
        x = x.permute(0, 2, 1)
        return x


    def forward(
        self,
        expression_feature,   # GPT latent [B, T, D_gpt]    语音语义 + 韵律 + 情绪的隐空间
        mouth_motion,         # [B, T, D_face]              已经是几何空间（非常具体）
        emotion_contrl,       # [B, D_emo]
    ):
        print(f"[Expression Model] "
              f"expression_feature: {expression_feature.shape} , "
              f"mouth_motion: {mouth_motion.shape}]")
        B, T, _ = mouth_motion.shape

        # 1. 投影到隐藏空间
        x_gpt = self.gpt_proj(expression_feature)  # 语音特征
        x_mouth = self.mouth_proj(mouth_motion)  # 嘴部运动
        emo_broad = emotion_contrl.unsqueeze(1).expand(B, T, -1)
        x_emo = self.emo_proj(emo_broad)  # emotion 广播到 T

        print(f"[Expression Model] 投影后 shapes -> "
              f"x_gpt: {x_gpt.shape}, "
              f"x_mouth: {x_mouth.shape}, "
              f"x_emo: {x_emo.shape}")

        # 增加一个插值，对齐时间
        T_target = x_mouth.shape[1]  # 以 mouth_motion 为准
        x_gpt = self.temporal_align(x_gpt, T_target)
        x_emo = self.temporal_align(x_emo, T_target)

        print(f"[Expression Model] 时间对齐后 shapes -> "
              f"x_gpt: {x_gpt.shape}, "
              f"x_emo: {x_emo.shape}, "
              f"x_mouth: {x_mouth.shape}")

        # 融合
        x = x_gpt + x_mouth + x_emo

        # 2. 不同模式下的内部时序 / 跨模态融合
        if self.mode == "film":
            # film: temporal sequence + FiLM
            # temporal 编码
            x = self.transformer_film(x)
            print(f"[Expression Model] film 模式 transformer 输出 shape -> x: {x.shape}")
            # 情感 FiLM
            x = self.film(x, emotion_contrl)
            print(f"[Expression Model] film 模式 FiLM 输出 shape -> x: {x.shape}")

        elif self.mode == "cross_attn":
            # cross_attn: 每层 cross-attention + self-attention
            for block in self.cross_layers:
                x = block(x, x_mouth)
            print(f"[Expression Model] cross_attn 模式 cross_layers 输出 shape -> x: {x.shape}")

        else:
            raise ValueError(f"Unknown Expression Model mode: {self.mode}")

        # 3. 输出 表情残差
        residual = self.out_proj(x)
        print(f"[Expression Model] 输出 residual shape -> {residual.shape}")
        return self.residual_scale * residual














