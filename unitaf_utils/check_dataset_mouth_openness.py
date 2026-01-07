'''
用于遍历数据集验证开口程度的指标分布
'''


import sys
import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from unitaf_train.unitaf_dataset import UniTAFDataset
from unitaf_train.unitaf_dataset_support_config import unitaf_dataset_support_config
from unitaf_train_component.render_webui import ARkitBlendShapeWebUI


class DatasetTester:
    def __init__(self):
        # 初始化dataset
        self.dataset = UniTAFDataset(
            dataset_config={
                # 模型类型，这里用于指导dataset类采用哪种预处理方法来准备模型的输入
                "tts_model": ["IndexTTS2"],
                "a2f_model": ["UniTalker"],
                # 数据集根目录
                "dataset_root_path": "/home/zqg/project/data/UniTAF Dataset",  # 使用绝对路径
                # 预处理组件所在设备
                "device": "cuda:0",
            },
            dataset_name="D12",
            dataset_support_config=unitaf_dataset_support_config,
            dataset_type="train"
        )
        # 初始化用于口型指标计算的工具，主要使用ARkitBlendShapeWebUI.compute_mouth_openness_single()
        self.engine = ARkitBlendShapeWebUI()


    def check_dataset_mouth_openness(self, output_dir):
        '''
        遍历数据集所有样本的所有帧，计算：
        - mouth_openness
        - norm_openness


        获取数据集中所有样本的每一帧的mouth_openness与norm_openness，并绘制分布图。
        '''
        os.makedirs(output_dir, exist_ok=True)

        # 用与保存统计数据
        mouth_openness_list = []
        norm_openness_list = []

        # 遍历数据集中所有样本
        for i in tqdm(range(len(self.dataset)), desc="遍历数据集"):
            sample = self.dataset.samples[i]  # 我们只需要获取原始sample，不需要dataset的预处理，所以不直接获取dataset[i]
            '''
            这里sample为：
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
            一个样本sample仅包含两个chunk，前一个chunk会用来当作条件，后一个chunk作为生成目标。我们取后一个chunk来统计。
            '''
            chunks = sample["chunk"]
            chunk_keys = list(chunks.keys())
            sorted_keys = sorted(chunk_keys, key=lambda x: int(x))
            second_chunk = chunks[sorted_keys[1]]  # 获得后一个chunk的内容

            # 获取每个样本中face arkit参数
            face_data = self.dataset.load_npy_array(second_chunk["face_path"])
            scale = self.dataset.sub_dataset_config["scale"]
            face_data = self.dataset.scale_and_offset(face_data, scale, 0.0)

            face_data = face_data.reshape(len(face_data), -1).astype(np.float32)  # (240, 61)  30fps * 8s
            # print("after scale & reshape  :", face_data.shape)  # -> (N, 61)

            # 逐帧处理
            for frame_idx in range(len(face_data)):
                bs_single = torch.as_tensor(face_data[frame_idx], dtype=torch.float32)
                # print("bs_single              :", bs_single.shape)  # -> torch.Size([61])

                vertices_single = self.engine.bs2vertices(bs_single)  # ✔ (V*3)
                vertices_single = vertices_single.view(-1, 3)  # (V,3)
                # print("vertices_single        :", vertices_single.shape)  # -> torch.Size([1220, 3])
                # vertices = vertices.to(device)

                mouth_openness, norm_openness = self.engine.compute_mouth_openness_single(vertices_single)

                mouth_openness_list.append(float(mouth_openness))
                norm_openness_list.append(float(norm_openness))

        all_mouth_openness = np.array(mouth_openness_list)
        all_norm_openness = np.array(norm_openness_list)

        print("总帧数: ", len(all_mouth_openness))

        # ---------- 绘图并保存 ----------
        fig_path1 = os.path.join(output_dir, "mouth_openness_distribution.png")
        plt.figure()
        plt.hist(all_mouth_openness, bins=80)
        plt.title("mouth_openness distribution")
        plt.xlabel("mouth_openness")
        plt.ylabel("count")
        plt.savefig(fig_path1)
        plt.close()

        fig_path2 = os.path.join(output_dir, "norm_openness_distribution.png")
        plt.figure()
        plt.hist(all_norm_openness, bins=80)
        plt.title("norm_openness distribution")
        plt.xlabel("norm_openness")
        plt.ylabel("count")
        plt.savefig(fig_path2)
        plt.close()

        print("已保存到: ", output_dir)
        print(" - ", fig_path1)
        print(" - ", fig_path2)

        return all_mouth_openness, all_norm_openness












if __name__ == '__main__':
    '''
    python -m unitaf_utils.check_dataset_mouth_openness
    '''

    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # unitaf_train的父目录
    sys.path.insert(0, project_root)

    tester = DatasetTester()
    tester.check_dataset_mouth_openness(output_dir="outputs/UniTAF Dataset Metrics")