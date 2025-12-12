# yapf: disable
import torch
import torch.nn as nn

from .base_model import BaseModel
from .model_utils import (
    TCN, PeriodicPositionalEncoding, enc_dec_mask, init_biased_mask,
    linear_interpolation,
)

# yapf: enable

# ---------- 梯度钩子：逐层打印范数 ----------
def hook_fn(name):
    def fn(grad):
        gnorm = grad.norm().item() if grad is not None else 0.0
        print(f'{name:40s} grad_norm = {gnorm:.8f}')
        return grad      # 必须返回，否则梯度传不下去
    return fn

class UniTalkerDecoderTCN(BaseModel):

    def __init__(self, args) -> None:
        super().__init__()
        # 1. 可学习的“身份-风格”向量表，大小 = identity_num×64
        self.learnable_style_emb = nn.Embedding(args.identity_num, 64)

        in_dim = out_dim = args.decoder_dimension
        # 2. 把音频特征维度映射到解码器内部维度
        self.audio_feature_map = nn.Linear(args.audio_encoder_feature_dim, in_dim)

        # 3. 核心：TCN 时序卷积，输入 = 音频特征⊕风格向量
        #    卷积核感受野一旦覆盖整段序列，就一次性输出全部帧，无需循环
        self.tcn = TCN(in_dim + 64, out_dim)

        # 4. 插值位置标志，2 表示在解码后再插值到目标帧数
        self.interpolate_pos = args.interpolate_pos

    def forward(self,
                hidden_states: torch.Tensor,
                style_idx: torch.Tensor,
                frame_num: int = None):
        batch_size = len(hidden_states)
        if style_idx is None:
            # 没给 ID 就用最后那个“枢纽身份”向量（PIE 策略）
            style_embedding = self.learnable_style_emb.weight[-1].unsqueeze(0).repeat(batch_size, 1)
        else:
            style_embedding = self.learnable_style_emb(style_idx)  # [B, 64]

        # 5. 音频先过线性映射 → [B, T, in_dim]
        feature = self.audio_feature_map(hidden_states).transpose(1, 2)  # [B, in_dim, T]
        # # ---- 手动检查局部梯度 ----
        # dummy_loss = feature.sum()  # 造一个简单 loss
        # grad_test = torch.autograd.grad(
        #     outputs=dummy_loss,
        #     inputs=hidden_states,
        #     retain_graph=True,
        #     create_graph=False,
        #     allow_unused=True
        # )[0]
        # print('局部梯度 d(feature)/d(hidden_states)  norm =', grad_test.norm().item())  # 270.
        # feature.register_hook(lambda g: print(f'audio_feature_map grad_norm: {g.norm().item():.8f}'))

        # 6. 把风格向量拼到每一帧，一起送给 TCN
        #    TCN 内部是因果/空洞卷积，纯并行，一次前向就出完整序列
        feature = self.tcn(feature, style_embedding)  # 返回 [B, out_dim, T]
        # feature.register_hook(lambda g: print(f'tcn_out grad_norm: {g.norm().item():.8f}'))


        feature = feature.transpose(1, 2)  # [B, T, out_dim]

        # 7. 如果下游需要固定帧数，再做线性插值
        if self.interpolate_pos == 2:
            feature = linear_interpolation(feature, output_len=frame_num)
        #    feature.register_hook(lambda g: print(f'linear_interp grad_norm: {g.norm().item():.8f}'))  # 如果不需要可省

        return feature  # 一次拿到全部帧


class UniTalkerDecoderTransformer(BaseModel):

    def __init__(self, args) -> None:
        super().__init__()
        out_dim = args.decoder_dimension
        # 1. 身份风格向量表，维度直接等于模型宽度
        self.learnable_style_emb = nn.Embedding(args.identity_num, out_dim)

        # 2. 音频特征映射
        self.audio_feature_map = nn.Linear(args.audio_encoder_feature_dim, out_dim)

        # 3. 周期性位置编码，代替 Transformer 默认的绝对位置
        self.PPE = PeriodicPositionalEncoding(out_dim, period=args.period, max_seq_len=3000)

        # 4. 带偏置的注意力 mask（下文解释）
        self.biased_mask = init_biased_mask(n_head=4, max_seq_len=3000, period=args.period)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=out_dim,
            nhead=4,
            dim_feedforward=2 * out_dim,
            batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=1)
        self.interpolate_pos = args.interpolate_pos

    def forward(self, hidden_states: torch.Tensor, style_idx: torch.Tensor,
                frame_num: int):
        # 1. 取对应身份向量并复制到每一帧 → [B, frame_num, out_dim]
        obj_embedding = self.learnable_style_emb(style_idx).unsqueeze(1).repeat(1, frame_num, 1)

        # 2. 音频特征映射
        hidden_states = self.audio_feature_map(hidden_states)  # [B, T_audio, out_dim]

        # 3. 给身份向量加位置编码，作为 Transformer 的 **target**
        style_input = self.PPE(obj_embedding)  # [B, frame_num, out_dim]

        # 4. 生成 attention mask
        #    biased_mask 是下三角矩阵，但注意：**它只用在 self-attention 里，
        #    防止同一序列内看到“未来”位置，并不依赖上一帧的真实预测值**
        tgt_mask = self.biased_mask[:frame_num, :frame_num].to(style_input.device)

        # 5. 音频→memory，身份→target，一次性并行解码
        memory_mask = enc_dec_mask(hidden_states.device, frame_num, hidden_states.shape[1])
        feat_out = self.transformer_decoder(
            tgt=style_input,  # 一次性完整序列
            memory=hidden_states,  # 音频全程不变
            tgt_mask=tgt_mask,
            memory_mask=memory_mask)  # [B, frame_num, out_dim]

        # 6. 后插值
        if self.interpolate_pos == 2:
            feat_out = linear_interpolation(feat_out, output_len=frame_num)

        return feat_out  # 同样一次拿到全部帧
