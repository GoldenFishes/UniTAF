'''
这里存放IndexTTS 2的训练脚本所需的组件，但是又在官方仓库中没有合适的独立实现的，在这里实现。
'''

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torchaudio

from transformers import SeamlessM4TFeatureExtractor
from indextts.utils.maskgct_utils import build_semantic_codec, build_semantic_model

class SemanticExtractor:
    def __init__(self, stats_path: Path, device: torch.device):
        self.device = device
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            path_=stats_path
        )
        self.semantic_model = self.semantic_model.to(device)
        self.semantic_mean = self.semantic_mean.to(device)
        self.semantic_std = self.semantic_std.to(device)
        self.semantic_model.eval()

    @torch.inference_mode()
    def extract(self, waveform: torch.Tensor, sr: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.squeeze(0).cpu().numpy()
        inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        outputs = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = outputs.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat, attention_mask
