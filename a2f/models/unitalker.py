import os
import sys
from omegaconf import OmegaConf

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 两个dirname
print("project_root", project_root)
sys.path.insert(0, project_root)

import os.path as osp
import torch
import torch.nn as nn

from a2f.dataset.dataset_config import dataset_config
from a2f.models.wav2vec2 import Wav2Vec2Model
from a2f.models.base_model import BaseModel
# from models.hubert import HubertModel
from a2f.models.wavlm import WavLMModel


class OutHead(BaseModel):

    def __init__(self, args, out_dim: int):
        super().__init__()
        head_layer = []

        in_dim = args.decoder_dimension
        for i in range(args.headlayer - 1):
            head_layer.append(nn.Linear(in_dim, in_dim))
            head_layer.append(nn.Tanh())
        head_layer.append(nn.Linear(in_dim, out_dim))
        self.head_layer = nn.Sequential(*head_layer)

    def forward(self, x):
        return self.head_layer(x)


class UniTalker(BaseModel):
    """UniTalker主模型：音频驱动面部动画生成模型"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        # 初始化音频编码器
        if 'wav2vec2' in args.MODEL.audio_encoder_repo:
            # 使用Wav2Vec2作为音频编码器
            self.audio_encoder = Wav2Vec2Model.from_pretrained(
                args.MODEL.audio_encoder_repo)
        elif 'wavlm' in args.MODEL.audio_encoder_repo:
            # 使用WavLM作为音频编码器
            self.audio_encoder = WavLMModel.from_pretrained(
                args.MODEL.audio_encoder_repo)
        else:
            raise ValueError("wrong audio_encoder_repo")

        # 冻结音频特征提取器参数
        self.audio_encoder.feature_extractor._freeze_parameters()
        if args.MODEL.freeze_wav2vec:
            # 如果设置冻结整个wav2vec2，冻结所有参数
            self.audio_encoder._freeze_wav2vec2_parameters()

        # 初始化解码器
        if args.MODEL.decoder_type == 'conv':
            # 使用卷积解码器（TCN时序卷积网络）
            from a2f.models.unitalker_decoder import UniTalkerDecoderTCN as Decoder
        elif args.MODEL.decoder_type == 'transformer':
            # 使用Transformer解码器
            from a2f.models.unitalker_decoder import \
                UniTalkerDecoderTransformer as Decoder
        else:
            ValueError('unknown decoder type ')
        self.decoder = Decoder(args.MODEL)

        # 初始化PCA层（主成分分析）
        if False: # args.NETWORK.use_pca:
            pca_layer_dict = {}
            from a2f.models.pca import PCALayer
            # 为每个数据集创建PCA层
            for dataset_name in args.DATA.dataset:
                if dataset_config[dataset_name]['pca']:
                    annot_type = dataset_config[dataset_name]['annot_type']
                    dirname = dataset_config[dataset_name]['dirname']
                    pca_path = osp.join(args.DATA.data_root, dirname, 'pca.npz')
                    pca_layer_dict[annot_type] = PCALayer(pca_path)
            self.pca_layer_dict = nn.ModuleDict(pca_layer_dict)
        else:
            self.pca_layer_dict = {}

        # 保存PCA相关参数
        self.pca_dim = args.NETWORK.pca_dim
        self.use_pca = args.NETWORK.use_pca
        self.interpolate_pos = args.MODEL.interpolate_pos

        # 初始化输出头（Output Heads）
        out_head_dict = {}
        for dataset_name in args.dataset:
            annot_type = dataset_config[dataset_name]['annot_type']
            out_dim = dataset_config[dataset_name]['annot_dim']

            # 如果使用PCA且该标注类型有PCA层，调整输出维度
            if (args.NETWORK.use_pca is True) and (annot_type in self.pca_layer_dict):
                pca_dim = self.pca_layer_dict[annot_type].pca_dim
                out_dim = min(args.pca_dim, out_dim, pca_dim)

            # 初始化输出投影层
            if args.MODEL.headlayer == 1:
                # 单层线性输出头
                out_projection = nn.Linear(args.MODEL.decoder_dimension, out_dim)
                # 初始化权重和偏置为零
                nn.init.constant_(out_projection.weight, 0)
                nn.init.constant_(out_projection.bias, 0)
            else:
                # 多层输出头
                out_projection = OutHead(args.MODEL, out_dim)
            out_head_dict[annot_type] = out_projection

        self.out_head_dict = nn.ModuleDict(out_head_dict)
        self.identity_num = args.MODEL.identity_num
        return

    def forward(self,
                audio,                # 输入音频 [batch_size, audio_length]
                template,             # 面部模板 [batch_size, vertice_dim]
                face_motion,          # 真实面部运动（训练时使用）[batch_size, seq_len, vertice_dim]
                style_idx,            # 风格索引 [batch_size]
                annot_type: str,      # 标注类型（决定使用哪个输出头）
                fps: float = None):   # 帧率（用于计算帧数）

        # 计算帧数
        if face_motion is not None:
            # 如果有真实面部运动数据，直接使用其帧数
            frame_num = face_motion.shape[1]
        else:
            # 否则根据音频长度和帧率计算帧数
            frame_num = round(audio.shape[-1] / 16000 * fps)

        # 音频编码：提取音频特征
        hidden_states = self.audio_encoder(
            audio, frame_num=frame_num, interpolate_pos=self.interpolate_pos)
        hidden_states = hidden_states.last_hidden_state  # 获取最后一层隐藏状态
        # print(f"hidden_states.shape = {hidden_states.shape}")  # hidden_states.shape = torch.Size([B, L, 768])

        # 解码器：将音频特征转换为面部运动参数
        decoder_out = self.decoder(hidden_states, style_idx, frame_num)
        # print(f"decoder_out.shape = {decoder_out.shape}")  # torch.Size([B, L, 256])

        # 输出处理
        if (not self.use_pca) or (annot_type not in self.pca_layer_dict):
            # 不使用PCA的情况：直接通过输出头得到最终运动参数
            out_motion = self.out_head_dict[annot_type](decoder_out)
            out_motion = out_motion + template  # 加上模板得到绝对坐标
            out_pca, gt_pca = None, None
        else:
            # 使用PCA的情况：先得到PCA系数，再解码为运动参数
            out_pca = self.out_head_dict[annot_type](decoder_out)
            # 将PCA系数解码为面部运动参数
            out_motion = self.pca_layer_dict[annot_type].decode(
                out_pca, self.pca_dim)
            out_motion = out_motion + template  # 加上模板得到绝对坐标

            # 训练时计算真实数据的PCA系数（用于损失计算）
            if face_motion is not None:
                gt_pca = self.pca_layer_dict[annot_type].encode(
                    face_motion - template, self.pca_dim)
            else:
                gt_pca = None

        # 返回：最终运动参数、PCA系数（如果使用）、真实PCA系数（训练时）
        return out_motion, out_pca, gt_pca


if __name__ == "__main__":
    '''
    python a2f/models/unitalker.py 
    '''
    from a2f.utils.utils import (
        get_average_meter_dict, get_logger, get_parser, get_audio_encoder_dim,
        load_ckpt, seed_everything, filter_unitalker_state_dict
    )
    # audio_path = osp.join(project_root, "examples/voice_zhongli.waw")

    unitalker_config_path = osp.join(project_root, "a2f/config/unitalker.yaml")
    config = OmegaConf.load(unitalker_config_path)
    config["MODEL"]["identity_num"] = 10
    config["MODEL"]["audio_encoder_feature_dim"] = get_audio_encoder_dim(config["MODEL"]["audio_encoder_repo"])
    # print(f"audio_encoder_feature_dim : {config.MODEL.audio_encoder_feature_dim}")  # 768
    config["MODEL"]["use_pca"] = False
    config["dataset"] = ["D0"]

    model = UniTalker(config)
    model = model.cuda()

    # 构造虚假audio传入
    batch_size = 2
    audio_length = 24000 * 3  # 3秒音频，采样率24kHz
    vertice_dim = 5023  # 面部顶点维度（示例）
    seq_len = 75  # 帧数（3秒 * 25fps）

    # 构造测试数据
    audio = torch.randn(batch_size, audio_length).cuda()
    template = torch.randn(batch_size, vertice_dim).cuda()
    face_motion = torch.randn(batch_size, seq_len, vertice_dim).cuda()  # 训练时使用
    style_idx = torch.randint(0, config["MODEL"]["identity_num"], (batch_size,)).cuda()

    # 前向传播
    out_motion, out_pca, gt_pca = model(
        audio=audio,
        template=template,
        face_motion=face_motion,  # 训练时传入真实数据，推理时为None
        style_idx=style_idx,
        annot_type="3DMM",  # 假设的标注类型
        fps=25.0
    )
    print(f"out_motion: {out_motion.shape}, out_pca: {out_pca.shape}, gt_pca: {gt_pca.shape}")


























