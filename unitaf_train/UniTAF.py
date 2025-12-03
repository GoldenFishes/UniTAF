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
        cfg: Dict,
        device: torch.device,
    ):
        '''
        根据 tts_model_name 与 a2f_model_name 中模型名称来判断初始化那些模型
        '''
        self.cfg = cfg

        if "IndexTTS2" in cfg["tts_model"]:
            # 导入IndexTTS2组件
            from indextts.utils.front import TextNormalizer, TextTokenizer

            # 初始化文本tokenizer
            self.tokenizer = TextTokenizer(str("checkpoints/bpe.model"), TextNormalizer())
            cfg = OmegaConf.load(Path("checkpoints/config.yaml"))
            self.tts_model = self.build_indextts2_model(
                cfg=cfg,
                tokenizer=self.tokenizer,
                base_checkpoint="checkpoints/gpt.pth",
                device=device,
            )

        if "UniTalker" in cfg["a2f_model"]:
            # 加载UniTalker Decoder并保存权重
            self.a2f_model = self.build_unitalker_decoder_model(
                cfg=cfg["UniTalker"],
                checkpoint=Path("a2f/pretrained_models/UniTalker-L-D0-D7.pt"),
                device=device,
            )

        if "IndexTTS2" in cfg["tts_model"] and "UniTalker" in cfg["t2f_model"]:
            # TODO：初始化特征投影层，用于将IndexTTS2的audio feature 投影到 UniTalker decoder 的接收维度
            self.projector =




    # 工具方法 --------------------------------------------------------------------------------------
    def build_indextts2_model(
        self,
        cfg,
        tokenizer,
        base_checkpoint,
        device,
    ):
        '''
        这里实例化模型并加载权重
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

        print(f"IndexTTS2 model:\n")
        print(f"{model.summary()}")

        return model.to(device)


    def build_unitalker_decoder_model(
        self,
        cfg,
        checkpoint,
        device,
    ):
        '''
        这里实例化模型并加载权重
        '''
        from unitaf_train_component.unitalker_decoder_component import UniTalkerDecoder
        from a2f.utils.utils import load_ckpt

        model = UniTalkerDecoder(cfg)  # 实例化UniTalker Decoder
        load_ckpt(model, checkpoint, re_init_decoder_and_head=False)  # 是否重新初始化decoder和head

        print(f"UniTalker Decoder:\n")
        print(f"{model.summary()}")

        return model.to(device)



if __name__ == '__main__':
    """
    测试联合模型的流程 python unitaf_train/unitaf.py
    """
    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # unitaf_train的父目录
    sys.path.insert(0, project_root)









