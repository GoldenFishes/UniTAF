"""
这里独立实现UniTalker的Decoder部分
"""
from omegaconf import OmegaConf
import os
import sys

# 只在直接执行时立即添加路径
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


import torch
import torch.nn as nn

from a2f.models.base_model import BaseModel
from a2f.models.unitalker import OutHead
from unitaf_train.unitaf_dataset_support_config import unitaf_dataset_support_config


class UniTalkerDecoder(BaseModel):
    def __init__(self, cfg):  # cfg需要使用OmegaConf读取，同时具备属性和字典调用方式
        super().__init__()

        self.model_cfg = cfg["UniTalker"]
        self.dataset_cfg = cfg["dataset_config"]
        self.loss_cfg = OmegaConf.load("a2f/config/unitalker.yaml")["LOSS"]

        # 选择具体Decoder
        if self.model_cfg.decoder_type == 'conv':
            from a2f.models.unitalker_decoder import UniTalkerDecoderTCN as Decoder
        elif self.model_cfg.decoder_type == 'transformer':
            from a2f.models.unitalker_decoder import UniTalkerDecoderTransformer as Decoder
        self.decoder = Decoder(self.model_cfg)

        if self.model_cfg.use_pca:
            from a2f.models.pca import PCALayer
            pca_layer_dict = {}
            # 为每个数据集创建PCA层 TODO: 在推理时需要从其他地方获得pca层的类型而不是从数据集配置中
            for dataset_name in self.dataset_cfg.dataset_list:
                # 数据集config中也会记录是否使用pca, 仅有外部配置和数据集配置均使用pca时,才会为改数据集配置pca
                if unitaf_dataset_support_config[dataset_name]['pca']:
                    annot_type = unitaf_dataset_support_config[dataset_name]['annot_type']
                    dirname = unitaf_dataset_support_config[dataset_name]['dirname']
                    pca_path = os.path.join(self.dataset_cfg.dataset_root_path, dirname, 'pca.npz')
                    pca_layer_dict[annot_type] = PCALayer(pca_path)
            self.pca_layer_dict = nn.ModuleDict(pca_layer_dict)
        else:
            self.pca_layer_dict = {}

        self.pca_dim = self.model_cfg["pca_dim"]
        self.use_pca = self.model_cfg["use_pca"]
        self.interpolate_pos = self.model_cfg["interpolate_pos"]

        # 初始化输出头（Output Heads） 为每个不同数据集（标注）创建不同输出头
        out_head_dict = {}
        for dataset_name in self.dataset_cfg["dataset_list"]:
            annot_type = unitaf_dataset_support_config[dataset_name]['annot_type']
            out_dim = unitaf_dataset_support_config[dataset_name]['annot_dim']

            # 如果使用PCA且该标注类型有PCA层，调整输出维度
            if (self.use_pca is True) and (annot_type in self.pca_layer_dict):
                pca_dim = self.pca_layer_dict[annot_type].pca_dim
                out_dim = min(self.pca_dim, out_dim, pca_dim)

            # 初始化输出投影层
            if self.model_cfg["headlayer"] == 1:
                # 单层线性输出头
                out_projection = nn.Linear(self.model_cfg["decoder_dimension"], out_dim)
                # 初始化权重和偏置为零
                nn.init.constant_(out_projection.weight, 0)
                nn.init.constant_(out_projection.bias, 0)
            else:
                # 多层输出头
                out_projection = OutHead(self.model_cfg, out_dim)
            out_head_dict[annot_type] = out_projection

        self.out_head_dict = nn.ModuleDict(out_head_dict)

        self.loss_module = None  # 正常不加载Loss计算模块，如果使用到再加载

    # UniTalker Decoder前向过程
    def forward(
        self,
        audio_feature,          # 输入音频特征 在原始UniTalker传入为 [B, L, 768]
        template,               # 面部模板 [batch_size, vertice_dim]
        face_motion,            # 真实面部运动（训练时使用）[batch_size, seq_len, vertice_dim]
        style_idx,              # 风格索引 [batch_size]
        annot_type: str,        # 标注类型（决定使用哪个输出头）
        fps: float = 25         # 帧率（用于计算帧数），如果face_motion中有帧数，否则
    ):
        '''
        接收音频特征，生成最终面部表情
        '''
        # print("[UniTalker Decoder] audio_feature shape:", audio_feature.shape)  # torch.Size([16, 240, 1024])
        # print("[UniTalker Decoder] face_motion shape:", face_motion.shape)  # torch.Size([16, 240, 61])
        face_motion = face_motion.unsqueeze(1)

        # 计算帧数
        if face_motion is not None:
            # 如果有真实面部运动数据，直接使用其帧数
            frame_num = face_motion.shape[1]
        else:
            # 否则根据音频计算帧率
            frame_num = fps

        # 解码器：将音频特征转换为面部运动参数
        decoder_out = self.decoder(audio_feature, style_idx, frame_num)  # torch.Size([B, L, 256])
        # print("[UniTalker Decoder] decoder_out shape:", decoder_out.shape)  # torch.Size([16, 240, 256])

        # 输出处理
        if (not self.use_pca) or (annot_type not in self.pca_layer_dict):
            # 不使用PCA的情况：直接通过输出头得到最终运动参数
            out_motion = self.out_head_dict[annot_type](decoder_out)
            out_motion = out_motion + template  # 加上模板得到绝对坐标
            out_pca, gt_pca = None, None

            # print("[DEBUG] 不使用PCA out_motion max=", out_motion.abs().max().item(),  # 检查这里是否有梯度
            #       "mean=", out_motion.mean().item(),
            #       "requires_grad=", out_motion.requires_grad,
            #       "grad_fn=", out_motion.grad_fn)

        else:
            # 使用PCA的情况：先得到PCA系数，再解码为运动参数
            out_pca = self.out_head_dict[annot_type](decoder_out)
            # 将PCA系数解码为面部运动参数
            out_motion = self.pca_layer_dict[annot_type].decode(
                out_pca, self.pca_dim)
            out_motion = out_motion + template  # 加上模板得到绝对坐标

            # print("[DEBUG] 使用PCA out_motion max=", out_motion.abs().max().item(),  # 检查这里是否有梯度
            #       "mean=", out_motion.mean().item(),
            #       "requires_grad=", out_motion.requires_grad,
            #       "grad_fn=", out_motion.grad_fn)

            # 训练时计算真实数据的PCA系数（用于损失计算）
            if face_motion is not None:
                gt_pca = self.pca_layer_dict[annot_type].encode(
                    face_motion - template, self.pca_dim)
            else:
                gt_pca = None

        # print("[UniTalker Decoder] out_motion.shape", out_motion.shape)  # torch.Size([16, 240, 61])

        # 返回：最终运动参数、PCA系数（如果使用）、真实PCA系数（训练时）
        return out_motion, out_pca, gt_pca

    def compute_loss(self, out_motion, data, annot_type, out_pca, gt_pca):
        '''
        参考 UniTalker/main/train_eval_loop.train_epoch() 中 loss计算部分

        out_motion 模型生成的面部动作
        data 数据集中的GT 动作
        '''
        if self.loss_module is None:
            from a2f.loss.loss import UniTalkerLoss
            self.loss_module = UniTalkerLoss(self.loss_cfg).cuda()

        rec_loss = self.loss_module(out_motion, data, annot_type)
        if out_pca is not None:
            pca_loss = self.loss_module.pca_loss(out_pca, gt_pca)
        else:
            pca_loss = torch.tensor(0.0)

        return rec_loss, pca_loss

if __name__ == '__main__':
    '''
    python unitaf_train_component/unitalker_decoder_component.py
    '''
    from omegaconf import OmegaConf
    import numpy as np

    config = {
        "UniTalker": {
            # UniTalker Decoder配置, 参数与UniTalker项目的config/unitalker.yaml一致
            "interpolate_pos": 1,
            "decoder_dimension": 256,
            "decoder_type": "conv",
            "period": 30,
            "headlayer": 1,
            # UniTalker Network
            "use_pca": True,
            "pca_dim": 256,
            # A2F Loss计算时设置
            "pca_weight": 0.01,
            # 以下需要从外部获得并更新：
            "audio_encoder_feature_dim": 1024,  # 假设是1024，根据不同的TTS模型的输出决定
            "identity_num": 20,  # 假设是20，需要根据不同数据集决定
        },
        # 数据集类
        "dataset_config": {
            "dataset_root_path": "/home/zqg/project/data/UniTAF Dataset",
            "dataset_list": ["D12"]  # 这里测试也传数据集是用于指导模型选择何种输出头
        },
    }

    checkpoint = torch.load("a2f/pretrained_models/UniTalker-L-D0-D7.pt", map_location='cpu')
    config["UniTalker"]["identity_num"] = len(checkpoint['decoder.learnable_style_emb.weight'])  # 根据权重内容更新identity_num

    unitalker_decoder_config = OmegaConf.create(config)

    model = UniTalkerDecoder(unitalker_decoder_config)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model = model.cuda()

    # 打印模型结构
    print("=" * 80)
    print("Model architecture:")
    print("=" * 80)
    print(model)

    # 打印模型的所有层及其参数
    print("\n" + "=" * 80)
    print("Model layers and parameters:")
    print("=" * 80)
    for name, module in model.named_children():
        print(f"\n{name}: {type(module).__name__}")

        # 如果是ModuleDict或ModuleList，打印其内容
        if isinstance(module, (nn.ModuleDict, nn.ModuleList)):
            for sub_name, sub_module in module.named_children():
                print(f"  {sub_name}: {type(sub_module).__name__}")
                if hasattr(sub_module, 'parameters') and list(sub_module.parameters()):
                    print(f"    Parameters: {sum(p.numel() for p in sub_module.parameters())}")

    # 打印模型参数详情
    print("\n" + "=" * 80)
    print("Detailed parameters:")
    print("=" * 80)
    for name, param in model.named_parameters():
        print(f"{name}: shape={param.shape}, dtype={param.dtype}, requires_grad={param.requires_grad}")


    # 检查权重加载
    print("\n" + "=" * 80)
    print("Checking weight loading:")
    print("=" * 80)
    model.load_state_dict(checkpoint, strict=False)

    # 比较加载的权重和模型期望的权重
    print("\nModel state dict keys:")
    model_keys = set(model.state_dict().keys())
    print(f"Total keys in model: {len(model_keys)}")

    # print("\nCheckpoint keys:")
    checkpoint_keys = set(checkpoint.keys())
    # print(f"Total keys in checkpoint: {len(checkpoint_keys)}")

    # 打印权重期望key时排除 audio_encoder
    print("\nMissing keys in checkpoint (model has but checkpoint doesn't):")
    missing = model_keys - checkpoint_keys
    missing_filtered = [key for key in sorted(missing) if 'audio_encoder' not in key]
    for key in missing_filtered:
        print(f"  - {key}")

    print("\nUnexpected keys in checkpoint (checkpoint has but model doesn't):")
    unexpected = checkpoint_keys - model_keys
    unexpected_filtered = [key for key in sorted(unexpected) if 'audio_encoder' not in key]
    for key in unexpected_filtered:
        print(f"  - {key}")

    # 打印加载的权重形状
    print("\n" + "=" * 80)
    print("Loaded weights shapes (filtered):")
    print("=" * 80)
    for key in sorted(checkpoint.keys()):
        # 跳过 audio_encoder 相关的键
        if 'audio_encoder' in key:
            continue

        if key in model.state_dict():
            model_shape = model.state_dict()[key].shape
            checkpoint_shape = checkpoint[key].shape
            match = "✓" if model_shape == checkpoint_shape else "✗"
            print(f"{match} {key}: model={model_shape}, checkpoint={checkpoint_shape}")
        else:
            print(f"✗ {key}: not in model, shape={checkpoint[key].shape}")



    # 构造输入
    audio_feature = torch.randn(1, 300, 1024).cuda()

    dataset_name = config["dataset_config"]["dataset_list"][0]  # "D12"
    # dirname = unitaf_dataset_support_config[dataset_name]['dirname']  # 子数据集文件夹
    # id_template_path = os.path.join(config["dataset_config"]["dataset_root_path"], dirname, 'id_template.npy')
    # template = np.load(id_template_path)
    # template = torch.from_numpy(template).float().cuda()
    template = torch.randn(1, 1).cuda()

    style_idx = torch.tensor([0], dtype=torch.long, device='cuda')
    annot_type = unitaf_dataset_support_config[dataset_name]['annot_type']

    # 测试模型前向过程
    out_motion, out_pca, gt_pca =model(
        audio_feature = audio_feature,
        template = template,
        face_motion = None,
        style_idx = style_idx,
        annot_type = annot_type
    )

    print(out_motion.shape)  # torch.Size([1, 300, 61])  长度对的上，这意味着，如果TTS假设以N个token表示一秒，那么A2F也会以N FPS表示
    if out_pca:
        print(out_pca.shape)
    if gt_pca:
        print(gt_pca.shape)


