"""
这里独立实现UniTalker的Decoder部分
"""
from omegaconf import OmegaConf
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


import torch
import torch.nn as nn

from a2f.models.base_model import BaseModel
from a2f.models.unitalker import OutHead
from a2f.dataset.dataset_config import dataset_config


class UniTalkerDecoder(BaseModel):
    def __init__(self, cfg):

        self.unitalker_config = OmegaConf.load("a2f/config/unitalker.yaml")

        if self.unitalker_config["MODEL"]["decoder_type"] == 'conv':
            from a2f.models.unitalker_decoder import UniTalkerDecoderTCN as Decoder
        elif self.unitalker_config["MODEL"]["decoder_type"] == 'transformer':
            from a2f.models.unitalker_decoder import UniTalkerDecoderTransformer as Decoder

        self.decoder = Decoder(self.unitalker_config["MODEL"])

        # 初始化PCA层（主成分分析）
        if self.unitalker_config["NETWORK"]["use_pca"]:
            from a2f.models.pca import PCALayer
            pca_layer_dict = {}
            # 为每个数据集创建PCA层
            for dataset_name in cfg["dataset_config"]["dataset_list"]:
                if dataset_config[dataset_name]['pca']:
                    annot_type = dataset_config[dataset_name]['annot_type']
                    dirname = dataset_config[dataset_name]['dirname']
                    pca_path = os.path.join(cfg["dataset_config"]["dataset_root_path"], dirname, 'pca.npz')
                    pca_layer_dict[annot_type] = PCALayer(pca_path)
            self.pca_layer_dict = nn.ModuleDict(pca_layer_dict)
        else:
            self.pca_layer_dict = {}

        self.pca_dim = self.unitalker_config["NETWORK"]["pca_dim"]
        self.use_pca = self.unitalker_config["NETWORK"]["use_pca"]
        self.interpolate_pos = self.unitalker_config["MODEL"]["interpolate_pos"]

        # 初始化输出头（Output Heads） 为每个不同数据集（标注）创建不同输出头
        out_head_dict = {}
        for dataset_name in cfg["dataset_config"]["dataset_list"]:
            annot_type = dataset_config[dataset_name]['annot_type']
            out_dim = dataset_config[dataset_name]['annot_dim']

            # 如果使用PCA且该标注类型有PCA层，调整输出维度
            if (self.use_pca is True) and (annot_type in self.pca_layer_dict):
                pca_dim = self.pca_layer_dict[annot_type].pca_dim
                out_dim = min(self.pca_dim, out_dim, pca_dim)

            # 初始化输出投影层
            if self.unitalker_config["MODEL"]["headlayer"] == 1:
                # 单层线性输出头
                out_projection = nn.Linear(self.unitalker_config["MODEL"]["decoder_dimension"], out_dim)
                # 初始化权重和偏置为零
                nn.init.constant_(out_projection.weight, 0)
                nn.init.constant_(out_projection.bias, 0)
            else:
                # 多层输出头
                out_projection = OutHead(self.unitalker_config["MODEL"], out_dim)
            out_head_dict[annot_type] = out_projection

        self.out_head_dict = nn.ModuleDict(out_head_dict)

        self.loss_module = None  # 正常不加载Loss计算模块，如果使用到再加载

    # UniTalker Decoder前向过程
    def forward(
        self,
        audio_feature,          # 输入音频特征 []
        template,               # 面部模板 [batch_size, vertice_dim]
        face_motion,            # 真实面部运动（训练时使用）[batch_size, seq_len, vertice_dim]
        style_idx,              # 风格索引 [batch_size]
        annot_type: str,        # 标注类型（决定使用哪个输出头）
        fps: float = 25         # 帧率（用于计算帧数），如果face_motion中有帧数，否则
    ):
        '''
        接收音频特征，生成最终面部表情
        '''
        # 计算帧数
        if face_motion is not None:
            # 如果有真实面部运动数据，直接使用其帧数
            frame_num = face_motion.shape[1]
        else:
            # 否则根据音频计算帧率
            frame_num = fps

        # 解码器：将音频特征转换为面部运动参数
        decoder_out = self.decoder(audio_feature, style_idx, frame_num)

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

    def compute_loss(self, out_motion, data, annot_type, out_pca, gt_pca):
        '''
        参考 UniTalker/main/train_eval_loop.train_epoch() 中 loss计算部分
        '''
        if self.loss_module is None:
            from a2f.loss.loss import UniTalkerLoss
            self.loss_module = UniTalkerLoss(self.unitalker_config["LOSS"]).cuda()

        rec_loss = self.loss_module(out_motion, data, annot_type)
        if out_pca is not None:
            pca_loss = self.loss_module.pca_loss(out_pca, gt_pca)
        else:
            pca_loss = torch.tensor(0.0)

        return rec_loss, pca_loss


