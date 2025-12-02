'''
这里实现 文本-音频-表情 联合模型的组装
'''


from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


import torch
import torch.nn as nn



class UniTextAudioFaceModel(nn.Module):
    def __init__(
        self,
        tts_model_name: list,
        a2f_model_name: list,
        device: torch.device,
    ):
        '''
        根据 tts_model_name 与 a2f_model_name 中模型名称来判断初始化那些模型
        '''
        if "IndexTTS2" in tts_model_name:
            # 初始化IndexTTS2的模型
            from omegaconf import OmegaConf
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

        if "UniTalker" in self.train_config["a2f_model"]:



    # 工具方法 --------------------------------------------------------------------------------------
    def build_indextts2_model(
        self,
        cfg,
        tokenizer,
        base_checkpoint,
        device,
    ):
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







