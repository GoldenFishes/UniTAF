'''
这里实现 文本-音频-表情 联合模型的组装
'''
from omegaconf import OmegaConf
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

class UniTextAudioFaceModel(nn.Module):
    def __init__(
        self,
        cfg,
        device: torch.device,
        mode: str = 'train',
    ):
        super().__init__()
        '''
        根据 tts_model_name 与 a2f_model_name 中模型名称来判断初始化那些模型
        '''
        self.cfg = cfg
        self.mode = mode

        # 训练模式下，一些组件在dataset中加载，主干加载需要训练的模块
        if mode == 'train':
            if "IndexTTS2" in cfg["tts_model"]:
                # 导入IndexTTS2组件
                from indextts.utils.front import TextNormalizer, TextTokenizer
                # 初始化文本tokenizer
                self.tokenizer = TextTokenizer(str("checkpoints/bpe.model"), TextNormalizer())
                indextts_cfg = OmegaConf.load(Path("checkpoints/config.yaml"))
                # 初始化IndexTTS核心的UniVoice gpt model
                self.tts_model = self.build_indextts2_model(
                    cfg=indextts_cfg,
                    tokenizer=self.tokenizer,
                    base_checkpoint="checkpoints/gpt.pth",
                    device=device,
                )
                # 初始化IndexTTS的s2mel model
                self.s2mel = self.build_indextts2_s2mel(
                    cfg=indextts_cfg,
                    device=device,
                )  # 这里s2mel以eval模式加载

            if "UniTalker" in cfg["a2f_model"]:
                # 加载UniTalker Decoder并保存权重
                self.a2f_model = self.build_unitalker_decoder_model(
                    cfg=cfg,
                    checkpoint=Path("a2f/pretrained_models/UniTalker-L-D0-D7.pt"),
                    device=device,
                )

    # 工具方法 --------------------------------------------------------------------------------------
    def build_indextts2_model(
        self,
        cfg,
        tokenizer,
        base_checkpoint,
        device,
    ):
        '''
        这里实例化IndexTTS2中核心的GPT模型并加载权重
        '''
        from indextts.gpt.model_v2 import UnifiedVoice
        # 确保模型配置中的词汇表大小与分词器的词汇表大小匹配
        vocab_size = tokenizer.vocab_size
        if cfg.gpt.number_text_tokens != vocab_size:
            cfg.gpt.number_text_tokens = vocab_size

        model = UnifiedVoice(**cfg.gpt)
        checkpoint = torch.load(base_checkpoint, map_location="cpu")
        raw_state_dict = checkpoint.get("model", checkpoint)

        # 过滤和修复状态字典
        filtered_state_dict = {}
        for key, value in raw_state_dict.items():
            if key.startswith("inference_model."):
                continue  # 跳过推理专用模块
            if ".lora_" in key:
                continue  # 跳过 LoRA 适配器权重
            new_key = key.replace(".base_layer.", ".")  # 修复某些层名
            if new_key == "gpt.wte.weight":
                continue  # 跳过文本嵌入权重
            filtered_state_dict[new_key] = value
        state_dict = filtered_state_dict

        # 处理词汇表大小变化的情况：变化时从状态字典中取出旧的权重，计算新旧权重形状的最小交集（slices）
        # 将旧权重的相应部分复制到新参数中，新词汇的嵌入保持随机初始化，将处理后的参数放回状态字典。
        resizable_keys = {
            "text_embedding.weight": model.text_embedding.weight,
            "text_head.weight": model.text_head.weight,
            "text_head.bias": model.text_head.bias,
        }
        for key, param in resizable_keys.items():
            weight = state_dict.pop(key, None)
            if weight is None:
                continue
            with torch.no_grad():
                slices = tuple(min(a, b) for a, b in zip(param.shape, weight.shape))
                if param.ndim == 1:
                    param[: slices[0]].copy_(weight[: slices[0]])
                else:
                    param[: slices[0], : slices[1]].copy_(weight[: slices[0], : slices[1]])
            state_dict[key] = param.detach().clone()

        # 加载权重
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Warn] Missing keys during load: {missing}")
        if unexpected:
            print(f"[Warn] Unexpected keys during load: {unexpected}")


        return model.to(device)

    def build_indextts2_s2mel(
        self,
        cfg,
        device,
    ):
        """
        这里初始化用于将IndexTTS核心gpt生成的semantic token转化为mel的模块，这里不需要训练该模块直接.eval
        """
        from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
        s2mel_path = os.path.join("checkpoints", cfg.s2mel_checkpoint)
        s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        s2mel = s2mel.to(device)
        return s2mel.eval()

    def build_unitalker_decoder_model(
        self,
        cfg,
        checkpoint,
        device,
    ):
        '''
        这里实例化UniTalker Decoder模型并加载权重
        - cfg 需要经过OmegaConf加载的
        '''
        from unitaf_train_component.unitalker_decoder_component import UniTalkerDecoder

        checkpoint = torch.load(checkpoint, map_location='cpu')
        # 向配置中追加 identity_num 参数
        cfg["UniTalker"]["identity_num"] = len(
            checkpoint['decoder.learnable_style_emb.weight'])  # 根据权重内容更新identity_num
        if not cfg["UniTalker"]["audio_encoder_feature_dim"]:
            # 向配置中追加 audio_encoder_feature_dim
            cfg["UniTalker"]["audio_encoder_feature_dim"] = 1024  # 设置默认为1024 来自IndexTTS.s2mel.models['gpt_layer']的维度
            # 实际中这里最好在外部提前判断

        model = UniTalkerDecoder(cfg)  # 实例化UniTalker Decoder
        model.load_state_dict(checkpoint, strict=False)

        return model.to(device)



if __name__ == '__main__':
    """
    测试联合模型的流程 python unitaf_train/UniTAF.py
    """
    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # unitaf_train的父目录
    sys.path.insert(0, project_root)

    from omegaconf import OmegaConf

    train_config = {
        # 模型类型，这里用于指导训练器类训练哪些模型
        "tts_model": ["IndexTTS2"],
        "a2f_model": ["UniTalker"],
        # 模型配置
        "IndexTTS2": {
            # TTS Loss计算时设置
            "use_duration_control": False,
            "duration_dropout": 0.3,
            "text_loss_weight": 0.2,
            "mel_loss_weight": 0.8,
        },
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

    cfg = OmegaConf.create(train_config)

    unitaf = UniTextAudioFaceModel(cfg, device=torch.device("cuda:0"), mode="train")

    # 打印模型结构
    print("=" * 80)
    print("Model architecture:")
    print("=" * 80)
    print(unitaf)

    # 打印模型的所有层及其参数
    print("\n" + "=" * 80)
    print("Model layers and parameters:")
    print("=" * 80)
    for name, module in unitaf.named_children():
        print(f"\n{name}: {type(module).__name__}")

        # 如果是ModuleDict或ModuleList，打印其内容
        if isinstance(module, (nn.ModuleDict, nn.ModuleList)):
            for sub_name, sub_module in module.named_children():
                print(f"  {sub_name}: {type(sub_module).__name__}")
                if hasattr(sub_module, 'parameters') and list(sub_module.parameters()):
                    print(f"    Parameters: {sum(p.numel() for p in sub_module.parameters())}")

    # # 打印模型参数详情
    # print("\n" + "=" * 80)
    # print("Detailed parameters:")
    # print("=" * 80)
    # for name, param in unitaf.named_parameters():
    #     print(f"{name}: shape={param.shape}, dtype={param.dtype}, requires_grad={param.requires_grad}")










