"""
（25.12.1）
这里实现UniTAF数据集的dataset
"""


import json
import sys
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
        dataset_name,
        dataset_support_config = unitaf_dataset_support_config,
        dataset_type = "train",
    ):
        self.dataset_config = dataset_config
        # 根据dataset_config["dataset"]设置的子数据集名称 如"DXX" 在dataset_support_config中获取相应sub_dataset_config
        self.sub_dataset_config = dataset_support_config[dataset_name]  # sub_dataset_config对应UniTalker中具体数据集参数配置
        # 记录子数据集的路径
        self.sub_dataset_path = os.path.join(
            self.dataset_config["dataset_root_path"],
            self.sub_dataset_config["dirname"]
        )
        # print("[DEBUG] prepare_dataset() 中 sub_dataset_path:", self.sub_dataset_path)

        # 初始化用于数据预处理的模型
        self._init_process_model()

        # 使用prepare_dataset方法得到提取后的数据集
        self.samples= self.prepare_dataset(dataset_type)  # List[Dict]
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
            from omegaconf import OmegaConf
            from indextts.gpt.model_v2 import UnifiedVoice
            from indextts.utils.front import TextNormalizer, TextTokenizer
            from unitaf_train_component.indextts2_train_component import SemanticExtractor
            from indextts.utils.maskgct_utils import build_semantic_codec, build_semantic_model
            from huggingface_hub import hf_hub_download
            import safetensors.torch

            # config来源于index-tts/checkpoints/config.yaml
            indextts2_cfg = OmegaConf.load(Path("checkpoints/config.yaml"))

            self.tokenizer = TextTokenizer(
                str(Path("checkpoints/bpe.model")),
                TextNormalizer(),
            )

            stats_path = Path("checkpoints/wav2vec2bert_stats.pt")
            self.semantic_extractor = SemanticExtractor(stats_path, self.dataset_config["device"])

            self.semantic_codec = build_semantic_codec(indextts2_cfg.semantic_codec)
            semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
            safetensors.torch.load_model(self.semantic_codec, semantic_code_ckpt)
            self.semantic_codec = self.semantic_codec.to(self.dataset_config["device"])
            self.semantic_codec.eval()

            gpt = UnifiedVoice(**indextts2_cfg.gpt)
            ckpt = torch.load("checkpoints/gpt.pth", map_location="cpu")
            state = ckpt.get("model", ckpt)
            gpt.load_state_dict(state, strict=False)
            self.gpt = gpt.to(self.dataset_config["device"])
            self.gpt.eval()

        if "UniTalker" in self.dataset_config["a2f_model"]:
            id_template_path = os.path.join(self.sub_dataset_path, 'id_template.npy')
            self.id_template_list = np.load(id_template_path)

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
        print("[UniTAFDataset] prepare_dataset() 中加载json路径:", json_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        info = raw_data["info"]

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
                processed_item = {
                    "speaker_id": info["speaker_id_list"][item["speaker_idx"]],
                    "speaker_idx": item["speaker_idx"],
                    "sample_id": item['sample_id'],
                    "pair_id": f"{chunk1_key}_{chunk2_key}",  # 添加pair_id标识
                    "chunk": new_chunk
                }
                samples.append(processed_item)

        print(f"[UniTAFDataset] 总共生成 {len(samples)} 个chunk对样本")
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
        output["speaker_idx"] = sample["speaker_idx"]
        output["sample_id"] = sample["sample_id"]

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
            output["text_ids_len"] = torch.tensor(len(text_ids), dtype=torch.long)
            # GT音频
            output["audio_codes"] = gt_semantic_code
            output["audio_codes_len"] = gt_lengths
            # 情感向量
            output["emotion_vector"] = emo_vec

        if "UniTalker" in self.dataset_config["a2f_model"]:
            '''
            获取样本中文件路径得到
                speaker_idx 说话人idx
                face_data 表情文件路径
                face_template UniTalker对于表情数据的偏置量
                face_type 表情标注类型，flame，arkit这种
                face_fps 表情帧率
            '''
            scale = self.sub_dataset_config["scale"]

            face_data = self.load_npy_array(second_chunk["face_path"])
            face_data = self.scale_and_offset(face_data, scale, 0.0)
            face_data = face_data.reshape(len(face_data),-1).astype(np.float32)

            face_type = second_chunk["face_type"]
            face_fps = second_chunk["face_fps"]
            speaker_idx = sample["speaker_idx"]

            id_template = self.id_template_list[speaker_idx]
            id_template = self.scale_and_offset(id_template, scale, 0.0)
            id_template = id_template.reshape(1, -1).astype(np.float32)

            # 获取音频信息（用于对齐）
            audio_duration = second_chunk["duration"]

            if "IndexTTS2" in self.dataset_config["tts_model"]:
                '''
                如果tts是IndexTTS2，则tts默认采用24k hz，压缩率1024，
                1个音频token ≈ 1024个原始音频样本 ≈ 1024/24000 ≈ 0.04267秒。
                
                在 face_fps = 25 下，1个音频token ≈ 0.04267 × 25 ≈ 1.0667个面部帧
                此时，面部数据应该大致与音频token对齐
                '''
                if face_fps != 25:
                    # 重采样face_fps至于25fps
                    face_data = self.resample_face_data_to_target_fps(
                        face_data=face_data,
                        source_fps=face_fps,
                        target_fps=25,
                        duration=audio_duration
                    )
                    face_fps = 25

            output["speaker_id"] = speaker_idx
            output["face_data"] = face_data
            output["face_template"] = id_template
            output["face_type"] = face_type
            output["face_fps"] = face_fps

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
        else:
            return text.strip()

    def load_audio(self, audio_path, target_sr: int) -> Tuple[torch.Tensor, int]:
        # 修复路径分隔符
        audio_path = audio_path.replace('\\', '/')

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
        # 修复路径分隔符
        text_path = text_path.replace('\\', '/')

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

    def load_npy_array(self, npy_path):
        # 修复路径分隔符
        npy_path = npy_path.replace('\\', '/')

        full_path = os.path.join(
            self.dataset_config["dataset_root_path"],
            self.sub_dataset_config["dirname"],
            npy_path
        )

        try:
            npy_array = np.load(full_path)
            return npy_array
        except Exception as e:
            print(f"加载npy数组文件失败: {full_path}, 错误: {e}")
            return None

    def resample_face_data_to_target_fps(self, face_data, source_fps, target_fps, duration=None):
        """
        将面部数据从源FPS重采样到目标FPS

        Args:
            face_data: numpy array, shape (num_frames, num_features)
            source_fps: 原始帧率
            target_fps: 目标帧率
            duration: 可选，持续时间（秒），用于确定输出长度

        Returns:
            重采样后的面部数据
        """
        import numpy as np
        from scipy import interpolate

        # 如果数据为空，返回空数组
        if len(face_data) == 0:
            print(f"[WARNING] 面部数据为空")
            return np.zeros((0,))

        # 计算持续时间
        if duration is None:
            # 根据帧数计算持续时间
            duration = len(face_data) / source_fps

        print(f"[DEBUG] 面部数据重采样: {source_fps}FPS -> {target_fps}FPS, 时长: {duration:.2f}秒")
        print(f"[DEBUG] 原始数据形状: {face_data.shape}")

        # 创建原始时间轴（秒）
        original_time = np.linspace(0, duration, len(face_data))

        # 创建目标时间轴
        target_num_frames = int(np.round(duration * target_fps))
        if target_num_frames <= 0:
            target_num_frames = 1

        target_time = np.linspace(0, duration, target_num_frames)

        # 处理不同维度的数据
        if face_data.ndim == 1:
            # 一维数据（如单个参数）
            face_data = face_data.reshape(-1, 1)
            num_features = 1
        else:
            # 多维数据（如 blendshape 权重）
            num_features = face_data.shape[1]

        # 初始化重采样后的数据
        resampled_data = np.zeros((target_num_frames, num_features))

        # 对每个特征进行线性插值
        for i in range(num_features):
            # 创建插值函数
            interp_func = interpolate.interp1d(
                original_time,
                face_data[:, i],
                kind='linear',  # 线性插值
                fill_value="extrapolate"  # 外推边界值
            )
            # 插值到新时间点
            resampled_data[:, i] = interp_func(target_time)

        # 恢复原始维度
        if num_features == 1:
            resampled_data = resampled_data.flatten()

        print(f"[DEBUG] 重采样后形状: {resampled_data.shape}")

        return resampled_data

    def scale_and_offset(self,
                         data: np.ndarray,
                         scale: float = 1.0,
                         offset: np.ndarray = 0.0):
        '''
        UniTalker的一些工具方法
        '''

        return data * scale + offset




if __name__ == '__main__':
    '''
    测试 UniTAFDataset
    python unitaf_train/unitaf_dataset.py
    
    该脚本测试结果打印中collate_fn部分有误是正常的，实际训练中会以UniTAFTrainer中的collate_batch实现为准
    '''
    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # unitaf_train的父目录
    sys.path.insert(0, project_root)

    from torch.utils.data import DataLoader

    dataset_config = {
        # 模型类型，这里用于指导dataset类采用哪种预处理方法来准备模型的输入
        "tts_model": ["IndexTTS2"],
        "a2f_model": ["UniTalker"],
        # 数据集根目录
        "dataset_root_path": "/home/zqg/project/data/UniTAF Dataset",  # 使用绝对路径
        # 预处理组件所在设备
        "device": "cuda:0",
    }

    print("开始测试 UniTAFDataset...")
    print("-" * 50)

    try:
        # 1. 测试数据集初始化
        print("1. 初始化数据集...")
        dataset = UniTAFDataset(dataset_config, dataset_name="D12", dataset_type="train")  # 对应{dataset_type}.json文件
        print(f"✓ 数据集初始化成功")
        print(f"  数据集大小: {len(dataset)} 个样本")
        print(f"  子数据集路径: {dataset.sub_dataset_path}")

        # 2. 测试样本访问
        print("\n2. 测试样本访问...")
        if len(dataset) > 0:
            try:
                # 测试单个样本
                print(f"  获取第一个样本...")
                sample_0 = dataset[0]
                print(f"  ✓ 获取成功")
                print(f"  样本键: {list(sample_0.keys())}")

                # 显示每个键的类型和形状
                print(f"\n  样本内容:")
                for key, value in sample_0.items():
                    if isinstance(value, torch.Tensor):
                        print(f"    {key}: Tensor {tuple(value.shape)}")
                    elif isinstance(value, list):
                        print(f"    {key}: List[{len(value)}]")
                    else:
                        print(f"    {key}: {type(value).__name__}")

                # 测试批量访问
                print(f"\n  测试批量获取...")
                indices = [0, 1, 2]
                batch_samples = dataset[indices]
                print(f"  ✓ 批量获取成功, 获取了 {len(batch_samples)} 个样本")

            except Exception as e:
                print(f"  ✗ 样本访问失败: {e}")
                import traceback

                traceback.print_exc()
        else:
            print("  ✗ 数据集为空")

        # 3. 测试多个样本
        print("\n3. 测试多个样本...")
        num_samples_to_test = min(5, len(dataset))
        successful_samples = 0

        for i in range(num_samples_to_test):
            try:
                sample = dataset[i]
                if sample is not None:
                    successful_samples += 1
            except Exception as e:
                print(f"  样本 {i} 失败: {e}")

        print(f"  成功处理 {successful_samples}/{num_samples_to_test} 个样本")

        # 4. 测试DataLoader
        print("\n4. 测试DataLoader...")
        try:
            # 自定义collate函数
            def custom_collate_fn(batch):
                """处理不同长度的数据"""
                collated = {}
                for key in batch[0].keys():
                    if isinstance(batch[0][key], torch.Tensor):
                        # 对于tensor，直接stack
                        collated[key] = torch.stack([item[key] for item in batch])
                    elif isinstance(batch[0][key], list):
                        # 对于列表，保持原样
                        collated[key] = [item[key] for item in batch]
                    else:
                        # 其他类型
                        collated[key] = [item[key] for item in batch]
                return collated


            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=True,
                collate_fn=custom_collate_fn,
                num_workers=0  # 先设置为0避免多进程问题
            )

            # 获取一个batch
            for batch in dataloader:
                print(f"  ✓ DataLoader成功创建batch")
                print(f"    Batch键: {list(batch.keys())}")

                # 检查batch中每个字段的形状
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"    {key}: {tuple(value.shape)}")
                    elif isinstance(value, list):
                        print(f"    {key}: {len(value)}个元素")
                break

        except Exception as e:
            print(f"  ✗ DataLoader测试失败: {e}")
            import traceback

            traceback.print_exc()

        # 5. 验证样本数据
        print("\n5. 验证样本数据完整性...")
        if len(dataset) > 0:
            sample = dataset[0]

            # 检查必要字段
            required_fields = ["tts_condition", "text_ids", "audio_codes", "emotion_vector"]
            missing_fields = [field for field in required_fields if field not in sample]

            if missing_fields:
                print(f"  ✗ 缺少必要字段: {missing_fields}")
            else:
                print(f"  ✓ 所有必要字段都存在")

                # 检查字段类型和有效性
                checks = [
                    ("tts_condition", torch.Tensor, "condition向量"),
                    ("text_ids", list, "文本ID列表"),
                    ("audio_codes", torch.Tensor, "音频token"),
                    ("emotion_vector", (list, torch.Tensor, np.ndarray), "情感向量"),
                ]

                for field, expected_type, desc in checks:
                    value = sample[field]
                    if isinstance(expected_type, tuple):
                        valid = any(isinstance(value, t) for t in expected_type)
                    else:
                        valid = isinstance(value, expected_type)

                    if valid:
                        print(f"    {field}({desc}): ✓ 类型正确")
                    else:
                        print(f"    {field}({desc}): ✗ 类型错误, 实际类型: {type(value)}")

        print("\n" + "=" * 50)
        print("测试完成!")

    except FileNotFoundError as e:
        print(f"✗ 文件未找到错误: {e}")
        print("请检查:")
        print(f"  1. dataset_root_path: {dataset_config['dataset_root_path']}")
        print(f"  2. dataset: {dataset_config['dataset']}")
        print(f"  3. 确保有对应的train.json文件")

    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        print("请确保所有依赖包已安装:")
        print("  pip install torch torchaudio transformers huggingface-hub safetensors")

    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback

        traceback.print_exc()

'''
2. 测试样本访问...
  获取第一个样本...
/home/shared/miniconda3/envs/zqg-indextts/lib/python3.11/site-packages/torchaudio/_backend/utils.py:213: UserWarning: In2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec` under the hood. Some parameters like ``normalize``, ``format``, ``buffer_size``, and ``backend`` will be ignored. We recommend that you port your code to rely directly on TorchCodec's decoder instead: https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder.
  warnings.warn(
/home/shared/miniconda3/envs/zqg-indextts/lib/python3.11/site-packages/torchaudio/_backend/ffmpeg.py:88: UserWarning: torio.io._streaming_media_decoder.StreamingMediaDecoder has been deprecated. This deprecation is part of a large refactoring effort to transition TorchAudio into a maintenance phase. The decoding and encoding capabilities of PyTorch for both audio and video are being consolidated into TorchCodec. Please see https://github.com/pytorch/audio/issues/3902 for more information. It will be removed from the 2.9 release.
  s = torchaudio.io.StreamReader(src, format, None, buffer_size)
  ✓ 获取成功
  样本键: ['duration', 'sample_id', 'tts_condition', 'tts_condition_len', 'text_ids', 'text_ids_len', 'audio_codes', 'audio_codes_len', 'emotion_vector', 'face_path', 'face_type', 'face_fps']

  样本内容:
    duration: float
    sample_id: str
    tts_condition: Tensor (1, 32, 1280)
    tts_condition_len: Tensor (1,)
    text_ids: List[84]
    text_ids_len: Tensor ()
    audio_codes: Tensor (399,)
    audio_codes_len: Tensor (1,)
    emotion_vector: List[8]
    face_path: str
    face_type: str
    face_fps: int

  测试批量获取...
  ✓ 批量获取成功, 获取了 3 个样本

3. 测试多个样本...
  成功处理 5/5 个样本

4. 测试DataLoader...
  ✓ DataLoader成功创建batch
    Batch键: ['duration', 'sample_id', 'tts_condition', 'tts_condition_len', 'text_ids', 'text_ids_len', 'audio_codes', 'audio_codes_len', 'emotion_vector', 'face_path', 'face_type', 'face_fps']
    duration: 2个元素
    sample_id: 2个元素
    tts_condition: (2, 1, 32, 1280)
    tts_condition_len: (2, 1)
    text_ids: 2个元素
    text_ids_len: (2,)
    audio_codes: (2, 399)
    audio_codes_len: (2, 1)
    emotion_vector: 2个元素
    face_path: 2个元素
    face_type: 2个元素
    face_fps: 2个元素

5. 验证样本数据完整性...
  ✓ 所有必要字段都存在
    tts_condition(condition向量): ✓ 类型正确
    text_ids(文本ID列表): ✓ 类型正确
    audio_codes(音频token): ✓ 类型正确
    emotion_vector(情感向量): ✓ 类型正确

==================================================
测试完成!

'''
