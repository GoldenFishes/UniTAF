"""
（25.12.1）
这里实现UniTAF数据集的dataset
"""


import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


import torchaudio

import numpy as np
import torch
from torch.utils.data import Dataset

# 导入unitaf项目支持的数据集字典
from unitaf_dataset_support_config import unitaf_dataset_support_config




class UniTAFDataset(Dataset):

    def __init__(
        self,
        dataset_config,
        dataset_support_config = unitaf_dataset_support_config,
        dataset_type = "train",
    ):
        self.dataset_config = dataset_config
        # 根据dataset_config["dataset"]设置的子数据集名称 如"DXX" 在dataset_support_config中获取相应sub_dataset_config
        self.sub_dataset_config = dataset_support_config[dataset_config["dataset"]]

        # 初始化用于数据预处理的模型
        self._init_process_model()

        # 使用prepare_dataset方法得到提取后的数据集
        self.samples= self.prepare_dataset(dataset_type)  # List[Dict]
        # 记录子数据集的路径
        self.sub_dataset_path = os.path.join(
            self.dataset_config["dataset_root_path"],
            self.sub_dataset_config["dirname"]
        )
        print("[DEBUG] prepare_dataset() 中 sub_dataset_path:", self.sub_dataset_path)
        '''
        期望从数据集中获取到的每个样本sample格式为：
        {
            "speaker_id": "UID101841495240",
            "sample_id": "00047",
            "pair_id": "2_3"
            "chunk": {
                "2": {
                    "face_type": "qxsk_inhouse_blendshape_weight",
                    "face_fps": 25,
                    "duration": 9.6,
                    "face_path": "train/UID101841495240##00047_2.npy",
                    "audio_path": "train/UID101841495240##00047_2.wav",
                    "text_path": "train/UID101841495240##00047_2.txt",
                    "text_emotion_vector": [0, 0, 0, 0, 0, 0, 0, 1.0],
                },
                "3": {...},
            }
        }
        经过处理后，一个样本sample仅包含两个chunk，前一个chunk会用来当作条件，后一个chunk作为生成目标
        '''

    def _init_process_model(self):
        '''
        该方法用于根据 dataset_config 中模型类型来判断初始化哪些模型作为数据预处理模型
        '''
        if "IndexTTS2" in self.dataset_config["tts_model"]:
            # 初始化IndexTTS2预处理模块
            from indextts.gpt.model_v2 import UnifiedVoice
            from indextts.utils.front import TextNormalizer, TextTokenizer
            from unitaf_train_component.indextts2_train_component import SemanticExtractor
            from indextts.utils.maskgct_utils import build_semantic_codec, build_semantic_model
            from huggingface_hub import hf_hub_download
            import safetensors.torch

            self.tokenizer = TextTokenizer

            stats_path = Path("checkpoints/wav2vec2bert_stats.pt")
            self.semantic_extractor = SemanticExtractor(stats_path, self.dataset_config["device"])

            # config来源于index-tts/checkpoints/config.yaml
            semantic_codec_config = {
                'codebook_size': 8192,
                'hidden_size': 1024,
                'codebook_dim': 8,
                'vocos_dim': 384,
                'vocos_intermediate_dim': 2048,
                'vocos_num_layers': 12
            }

            self.semantic_codec = build_semantic_codec(semantic_codec_config)
            semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
            safetensors.torch.load_model(self.semantic_codec, semantic_code_ckpt)
            self.semantic_codec = self.semantic_codec.to(self.dataset_config["device"])
            self.semantic_codec.eval()

            # config来源于index-tts/checkpoints/config.yaml
            gpt_config = {
                "model_dim": 1280,
                "max_mel_tokens": 1815,
                "max_text_tokens": 600,
                "heads": 20,
                "use_mel_codes_as_input": True,
                "mel_length_compression": 1024,
                "layers": 24,
                "number_text_tokens": 12000,
                "number_mel_codes": 8194,
                "start_mel_token": 8192,
                "stop_mel_token": 8193,
                "start_text_token": 0,
                "stop_text_token": 1,
                "train_solo_embeddings": False,
                "condition_type": "conformer_perceiver",
                "condition_module": {
                    "output_size": 512,
                    "linear_units": 2048,
                    "attention_heads": 8,
                    "num_blocks": 6,
                    "input_layer": "conv2d2",
                    "perceiver_mult": 2
                },
                "emo_condition_module": {
                    "output_size": 512,
                    "linear_units": 1024,
                    "attention_heads": 4,
                    "num_blocks": 4,
                    "input_layer": "conv2d2",
                    "perceiver_mult": 2
                }
            }

            gpt = UnifiedVoice(gpt_config)
            ckpt = torch.load("checkpoints/gpt.pth", map_location="cpu")
            state = ckpt.get("model", ckpt)
            gpt.load_state_dict(state, strict=False)
            self.gpt = gpt.to(self.dataset_config["device"])
            self.gpt.eval()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        if isinstance(idx, (list, np.ndarray)):
            return [self._get_single_item(i) for i in idx]  # 批量获取：返回样本列表
        else:
            return self._get_single_item(idx)  # 单个获取：返回单个样本

    # 处理单个样本
    def _get_single_item(self, idx: int):
        sample = self.samples[idx]

        try:
            # 使用prepare_model_input
            dataset_sample = self.prepare_model_input(sample)
            return dataset_sample

        except Exception as e:
            print(f"[UniTAFDataset] 处理 samples[{idx}] 出错: {e}")
            # 返回下一个样本，避免因单个损坏样本导致训练中断
            return self.__getitem__((idx + 1) % len(self.samples))

    # 获取数据集样本字典
    def prepare_dataset(self, dataset_type="train") -> List[Dict]:
        """
        读取我们数据集json文件。
        将连续chunk例如[0,1,2,3,4]分为多个重叠的sample,使得每个sample为chunk[0,1],chunk[1,2],chunk[2,3],chunk[3,4]
        """
        # 获取对应支持数据集中的参数及路径
        json_path = os.path.join(
            self.dataset_config["dataset_root_path"],
            self.sub_dataset_config["dirname"],
            f"{dataset_type}.json",
        )
        print("[DEBUG] prepare_dataset() 中 json_path:", json_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 获取每一个样本
        samples = []
        for item in raw_data["data"]:
            chunks = item["chunk"]
            chunk_keys = list(chunks.keys())

            # 如果chunk数量少于2个，无法形成chunk对，跳过
            if len(chunk_keys) < 2:
                continue

            # 对chunk键进行排序，确保顺序正确
            # 将字符串转换为整数排序，然后再转换回字符串
            try:
                sorted_chunk_keys = sorted(chunk_keys, key=lambda x: int(x))
            except ValueError:
                sorted_chunk_keys = sorted(chunk_keys) # 如果chunk键不是数字，按字符串排序

            # 创建chunk对: (0,1), (1,2), (2,3), (3,4), ...
            for i in range(len(sorted_chunk_keys) - 1):
                chunk1_key = sorted_chunk_keys[i]
                chunk2_key = sorted_chunk_keys[i + 1]

                # 创建新的chunk字典，包含两个连续的chunk
                new_chunk = {
                    chunk1_key: chunks[chunk1_key],
                    chunk2_key: chunks[chunk2_key]
                }

                # 为每个chunk对创建新的样本
                # FIXME:这里sample中先不包含"speaker_id_list",如果后面UniTalker模块需要再修复这里
                processed_item = {
                    "speaker_idx": item["speaker_idx"],
                    "sample_id": item['sample_id'],
                    "pair_id": f"{chunk1_key}_{chunk2_key}",  # 添加pair_id标识
                    "chunk": new_chunk
                }
                samples.append(processed_item)

        print(f"[DEBUG] 总共生成 {len(samples)} 个chunk对样本")
        return samples

    # 根据sample中的文件路径，预处理成模型的直接输入
    def prepare_model_input(self, sample):
        """
        将获取到的sample处理成模型的直接输入
        """
        # 分别获得sample中的前一个和后一个chunk，前一个一般当作参考条件，后一个当作生成目标
        chunks = sample["chunk"]
        # 获取chunk的键（默认sample中的chunk只有两个，且prepare_dataset()中保证排序正确）
        chunk_keys = list(chunks.keys())
        sorted_keys = sorted(chunk_keys, key=lambda x: int(x))

        first_chunk = chunks[sorted_keys[0]]
        second_chunk = chunks[sorted_keys[1]]

        # 初始化空返回字典
        output = {}
        # 填入时长
        output["duration"] = second_chunk["duration"]
        output["speaker_id"] = second_chunk["speaker_id"]
        output["sample_id"] = second_chunk["sample_id"]

        if "IndexTTS2" in self.dataset_config["tts_model"]:
            '''
            获取样本文件得到
                tts_condition:      送给TTS的音频风格条件
                text_ids:           文本ids
                audio_codes:        数据集中audio的GT token, 形状: (B, T), 包含零填充 [t1, t2, t3, 0, 0, 0]
                emo_vec:            数据集中标注的emo_vec
                tts_condition_len:  条件音频token长度
                text_ids_len:       文本token长度
                audio_code_len:     GT音频token长度
            '''
            # 1. 加载并处理文本,获取第二个chunk的目标文本
            text = self.load_text_file(second_chunk["text_path"])
            text = self.clean_text(text)
            text_tokens = self.tokenizer.tokenize(text)  # 对文本进行分词
            if not text_tokens:
                return None
            text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)  # 将分词转换为ID序列

            # 2. 处理条件音频,获取第一个chunk的条件音频作为tts_condition
            # 加载音频文件并重采样到24kHz
            cond_waveform, cond_sr = self.load_audio(first_chunk["audio_path"], target_sr=24000)
            cond_feat, cond_attention_mask = self.semantic_extractor.extract(cond_waveform, cond_sr)

            # 3. 处理目标音频,获取第二个chunk的目标音频作为GT,audio_codes
            # 加载音频文件并重采样到24kHz
            gt_waveform, st_sr = self.load_audio(second_chunk["audio_path"], target_sr=24000)
            gt_feat, gt_attention_mask = self.semantic_extractor.extract(gt_waveform, st_sr)  # 使用SeamlessM4T特征提取器从音频中提取语义特征

            # 在推理模式下进行特征提取
            with torch.inference_mode():
                # 提取GT音频token：使用语义编解码器对特征进行量化，得到离散token
                gt_semantic_code, _ = self.semantic_codec.quantize(gt_feat)
                if gt_semantic_code.dim() > 1:  # 如果维度大于1，进行降维处理
                    gt_semantic_code = gt_semantic_code.squeeze(0)
                gt_lengths = gt_attention_mask.sum(dim=1).long()

                # 计算条件特征的有效长度（基于注意力掩码）
                cond_lengths = cond_attention_mask.sum(dim=1).long()
                # 提取条件特征：使用GPT模型获取语音条件向量
                conditioning = self.gpt.get_conditioning(
                    cond_feat.transpose(1, 2), cond_lengths.to(cond_feat.device)
                )

                # 如果数据集中有情感向量
                if second_chunk.get("text_emotion_vector", None) is not None:
                    emo_vec = second_chunk["text_emotion_vector"]
                else:
                    # 提取情感向量：使用GPT模型获取情感特征
                    emo_vec = self.gpt.get_emovec(gt_feat, gt_lengths.to(gt_feat.device))

            # 条件音频
            output["tts_condition"] = conditioning
            output["tts_condition_len"] = cond_lengths
            # 文本token
            output["text_ids"] = text_ids
            output["text_ids_len"] = len(text_ids.size)
            # GT音频
            output["audio_codes"] = gt_semantic_code
            output["audio_codes_len"] = gt_lengths
            # 情感向量
            output["emotion_vector"] = emo_vec

        if "UniTalker" in dataset_config["a2f_model"]:
            '''
            获取样本中文件路径得到
                face_path 表情文件路径
                face_type 表情标注类型，flame，arkit这种
                face_fps
                
            '''
            # TODO:这里是否需要预处理
            output["face_path"] = second_chunk["face_path"]
            output["face_type"] = second_chunk["face_type"]
            output["face_fps"] = second_chunk["face_fps"]

        return output


    # ------------------------------------------------------------------------------------------------------
    # 工具方法
    def clean_text(self, text):
        if "IndexTTS2" in self.dataset_config["tts_model"]:
            # 1. 去除文本首尾的空白字符（空格、制表符、换行符等）
            text = text.strip()
            # 2. 将全角空格替换为普通空格
            text = text.replace("\u3000", " ")
            # 3. 将不间断空格（non-breaking space）替换为普通空格
            text = text.replace("\xa0", " ")
            return text.strip()


    def load_audio(self, audio_path, target_sr: int) -> Tuple[torch.Tensor, int]:
        full_path = os.path.join(
            self.dataset_config["dataset_root_path"],
            self.sub_dataset_config["dirname"],
            audio_path
        )

        wav, sr = torchaudio.load(full_path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr
        return wav, sr

    def load_text_file(self, text_path):
        full_path = os.path.join(
            self.dataset_config["dataset_root_path"],
            self.sub_dataset_config["dirname"],
            text_path
        )

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"加载文本文件失败: {full_path}, 错误: {e}")
            return ""



if __name__ == '__main__':
    '''
    python unitaf_train/unitaf_dataset.py
    '''
    dataset_config = {
        # 模型类型，这里用于指导dataset类采用哪种预处理方法来准备模型的输入
        "tts_model": ["IndexTTS2"],
        "a2f_model": ["UniTalker"],
        # 数据集根目录
        "dataset_root_path": "data/UniTAF Dataset",
        "dataset": "D12",  # 支持多数据集训练，对应unitaf_dataset_support_config中具体数据集
        # 预处理组件所在设备
        "device": "cuda:0",
    }

    dataset = UniTAFDataset(dataset_config, dataset_type="train")