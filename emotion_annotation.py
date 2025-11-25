'''
该脚本基于IndexTTS-2模块，实现提取文本和音频中的情感向量。（用于数据集情感向量的标注）

对数据集中的音频和文本分别进行情感向量标注：
- 文本中推断的情感向量含义为，IndexTTS-2认为该文本本身应该包含什么样的情感
- 音频中分析的情感向量含义为，IndexTTS-2识别到实际数据集音频中的情感
'''

import os
import librosa
import torch
import json
from tqdm import tqdm
from omegaconf import OmegaConf
from indextts.infer_v2 import QwenEmotion



class EmotionAnnotator:
    def __init__(
            self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, device=None,
            use_cuda_kernel=None, use_deepspeed=False, use_accel=False, use_torch_compile=False
    ):
        """
        Args:
            cfg_path (str): 配置文件路径
            model_dir (str): 模型目录路径

            以下参数暂时不会使用到：
            use_fp16 (bool): 是否使用fp16精度
            device (str): 使用的设备（如'cuda:0', 'cpu'）。如果为None，将根据CUDA或MPS可用性自动设置
            use_cuda_kernel (None | bool): 是否使用BigVGan自定义融合激活CUDA内核，仅适用于CUDA设备
            use_deepspeed (bool): 是否使用DeepSpeed
            use_accel (bool): 是否对GPT2使用加速引擎
            use_torch_compile (bool): 是否使用torch.compile进行优化
        """
        # if device is not None:
        #     self.device = device
        #     self.use_fp16 = False if device == "cpu" else use_fp16
        #     self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir

        # 初始化情感分析模型（Qwen）
        self.qwen_emo = QwenEmotion(os.path.join(self.model_dir, self.cfg.qwen_emo_path))


    def text_emotion_annotation(self, emo_text):
        # 调用情感分析模型
        emo_dict = self.qwen_emo.inference(emo_text)
        # 将有序字典转换为向量列表（顺序非常重要！）
        emo_vector = list(emo_dict.values())

        return emo_vector

    def read_text_file(self, text_path):
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            return content
        except Exception as e:
            print(f"读取文本文件失败 {text_path}: {e}")
            return ""

    def process_json_file(self, dataset_path, input_json_path, output_json_path=None):
        """
        处理JSON文件，为所有文本生成情感向量
        Args:
            dataset_path: 数据集根目录
            input_json_path: 输入JSON文件路径
            output_json_path: 输出JSON文件路径（如果为None则覆盖原文件）
        """
        # 读取原始JSON文件
        try:
            with open(input_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"读取JSON文件失败: {e}")
            return None

        print(f"开始处理JSON文件: {input_json_path}")
        print(f"总样本数: {len(data.get('data', []))}")

        # 统计信息
        total_chunks = 0
        processed_chunks = 0
        failed_chunks = 0

        # 遍历所有数据
        speaker_data_list = data.get('data', [])
        for speaker_data in tqdm(speaker_data_list, desc="处理说话人", unit="speaker"):

            # 遍历所有chunk
            chunks = speaker_data.get('chunk', {})
            for chunk_id, chunk_data in chunks.items():
                total_chunks += 1

                # 获取文本文件路径
                text_path = chunk_data.get('text_path', '')

                if not text_path:
                    print(f"警告: chunk {chunk_id} 没有文本路径")
                    failed_chunks += 1
                    continue

                # 拼接完整路径
                full_text_path = os.path.join(dataset_path, text_path)

                if not os.path.exists(full_text_path):
                    print(f"警告: 文本文件不存在 {full_text_path}")
                    failed_chunks += 1
                    continue

                # 读取文本内容
                text_content = self.read_text_file(full_text_path)

                if not text_content:
                    print(f"警告: 文本文件为空 {full_text_path}")
                    failed_chunks += 1
                    continue

                # 生成情感向量
                # print(f"  处理 chunk {chunk_id}: {text_content[:50]}...")
                try:
                    emotion_vector = self.text_emotion_annotation(text_content)
                except Exception as e:
                    print(f"警告：情感向量生成错误 {e}")
                    failed_chunks += 1
                    continue


                # 更新情感向量
                chunk_data['text_emotion_vector'] = emotion_vector
                processed_chunks += 1

        # 打印统计信息
        print(f"\n处理完成!")
        print(f"总chunk数: {total_chunks}")
        print(f"成功处理: {processed_chunks}")
        print(f"处理失败: {failed_chunks}")

        # 保存结果
        output_path = output_json_path or input_json_path
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"结果已保存到: {output_path}")
        except Exception as e:
            print(f"保存结果失败: {e}")

        return data




if __name__ == "__main__":
    print("开始测试 EmotionAnnotator 类...")

    annotator = EmotionAnnotator()

    # test_text = "今天天气真好，心情愉快！"
    # result = annotator.text_emotion_annotation(test_text)
    # print(result)

    annotator.process_json_file(
        dataset_path="../data/UniTAF Dataset",
        input_json_path="../data/UniTAF Dataset/all_in_one_final.json",
        output_json_path="../data/UniTAF Dataset/all_in_one_final_emotion.json",
    )
















