'''
用于遍历数据集验证开口程度的指标分布

1. 计算嘴部开合度（mouth_openness）
    遍历数据集中的所有样本,每个样本取后一个chunk（作为生成目标）。
    逐帧计算嘴部开合度,加载面部ARKit混合形状参数
    转换为3D顶点坐标并计算嘴部开合度量值

2. 计算归一化开合度（norm_openness）
    自动寻找最优归一化区间 [min_val, max_val]
    采用加权平衡算法，确保归一化后的数据分布均衡
    使用二分搜索优化归一化区间

3. 计算损失函数权重
    根据归一化开合度计算每个帧的权重系数
    权重公式：weight = base_weight + alpha * |norm_openness - 0.5|^gamma
    特性：
        嘴部接近闭合或最大张开时权重增大;
        嘴部半开时权重减小;
        鼓励模型更关注极端开合状态

4. 可视化分析
    生成四个分布图：
        mouth_openness_distribution.png: 原始开合度分布
        norm_openness_distribution.png: 归一化开合度分布
        weighting_factor_distribution.png: 权重系数分布
        weighted_in_norm_distribution.png: 加权后的归一化开合度分布


同时，我们使用加权平衡方法寻找最佳归一化区间（考虑数据分布的统计特征），实现于DatasetTester.find_balanced_range_weighted()

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

        mouth_openness_cache = os.path.join(output_dir, "all_mouth_openness.npy")

        # ============================================================
        # 1. 如果已经计算过 mouth_openness，直接加载
        # ============================================================
        if os.path.exists(mouth_openness_cache):
            print(f"[INFO] 发现缓存，直接加载: {mouth_openness_cache}")
            all_mouth_openness = np.load(mouth_openness_cache)

        # ============================================================
        # 2. 否则重新遍历数据集计算
        # ============================================================
        else:
            print("[INFO] 未发现缓存，开始遍历数据集计算 mouth_openness")

            mouth_openness_list = []

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

                    # 指定正确的 min_val 与 max_val
                    mouth_openness, _ = self.engine.compute_mouth_openness_single(
                        vertices_single,
                        min_val = 0.0280,  # 这里 min_val,max_val 只影响 返回的 norm_openness, 然而我们此时只需要mouth_openness，故这里传什么无所谓
                        max_val = 0.0453
                    )

                    mouth_openness_list.append(float(mouth_openness))

            all_mouth_openness = np.array(mouth_openness_list)

            np.save(mouth_openness_cache, all_mouth_openness)
            print(f"[INFO] mouth_openness 已保存到: {mouth_openness_cache}")

        print("总帧数: ", len(all_mouth_openness))

        # ============================================================
        # 3. 派生计算（始终重新算）
        # ============================================================
        # 归一化区间覆盖率0.95
        coverage = 0.95
        # w = base_weight + alpha * np.abs(norm - 0.5) ** gamma
        alpha = 3.0
        gamma = 2.0
        base_weight = 0.1
        # 搜索合理归一化区间迭代次数
        max_iter = 50

        # 计算合理的归一化区间
        min_val, max_val = self.find_balanced_range_weighted(
                all_mouth_openness,
                coverage=coverage,
                alpha=alpha,
                gamma=gamma,
                base_weight=base_weight,
                max_iter=max_iter,
            )
        print(f"自动计算得到的归一化区间: [{min_val:.6f}, {max_val:.6f}]")

        all_norm_openness = np.clip(
            (all_mouth_openness - min_val) / (max_val - min_val),
            0.0,
            1.0
        )

        weight_np = self.compute_weight_from_norm_mouth_openness(
            all_norm_openness,
            alpha=alpha,
            gamma=gamma,
            base_weight=base_weight,
        )

        # ============================================================
        # 4. 绘图
        # ============================================================
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

        fig_path3 = os.path.join(output_dir, "weighting_factor_distribution.png")
        plt.figure()
        plt.hist(weight_np, bins=80, color='orange', alpha=0.7)
        plt.title("weighting_factor distribution")
        plt.xlabel("weighting_factor")
        plt.ylabel("count")
        plt.savefig(fig_path3)
        plt.close()

        fig_path4 = os.path.join(output_dir, "weighted_in_norm_distribution.png")
        plt.figure()
        plt.hist(
            all_norm_openness,
            bins=80,
            weights=weight_np,  # 关键就在这里 norm_openness[i]  ←→  weight_np[i] 一一对应
        )
        plt.title("Weighted norm_openness distribution")
        plt.xlabel("norm_openness")
        plt.ylabel("count * weighting_factor")
        plt.savefig(fig_path4)
        plt.close()


        print("已保存到: ", output_dir)
        print(" - ", fig_path1)
        print(" - ", fig_path2)
        print(" - ", fig_path3)
        print(" - ", fig_path4)

        return all_mouth_openness, all_norm_openness

    def compute_weight_from_norm_mouth_openness(
        self,
        norm_openness_list,
        alpha,
        gamma,
        base_weight,
    ):
        '''
        从归一化后的norm_mouth_openness开口数据中得到用于loss计算的加权系数：

        该加权系数计算模拟 a2f.loss.loss.BlendShapeLoss_61.weighted_vertices_loss_by_mouth_openness()的实现

        norm_openness_list: (T)，在任意时刻T中 norm_openness的值
        - ≈ 0.0（接近闭嘴）或 ≈ 1.0（接近最大张开）：
            → 权重增大，犯错代价更高
        - ≈ 0.5（半开半合）：
            → 权重接近 0，减弱顶点空间loss的作用

        weight = 0 + alpha * |norm_openness - 0.5|^gamma

        其中：
        - alpha 控制放大强度
        - gamma 控制权重曲线的陡峭程度

        计算得到的每个时刻T下的 weight
        '''

        # 将列表转换为PyTorch张量
        norm_openness_tensor = torch.tensor(norm_openness_list, dtype=torch.float32)

        # 计算加权
        weight = base_weight + alpha * torch.abs(norm_openness_tensor - 0.5) ** gamma  # shape (N,)

        # 转回 numpy 方便绘图
        weight_np = weight.numpy()

        return weight_np

    def find_balanced_range_weighted(
        self,
        x,
        coverage=0.95,
        alpha=3.0,
        gamma=2.0,
        base_weight=0.1,
        max_iter=50,
    ):
        '''
        带 coverage 约束的 r 搜索
        '''
        x = np.asarray(x)
        median = np.median(x)

        # ---------- coverage 约束 ----------
        low_q = (1 - coverage) / 2
        high_q = 1 - low_q

        lo = np.quantile(x, low_q)
        hi = np.quantile(x, high_q)

        # 最小 r：至少覆盖 quantile
        r_min = max(median - lo, hi - median)

        # 最大 r：保证 clip 充分发生
        r_max = np.max(np.abs(x - median)) * 2

        for _ in range(max_iter):
            r = 0.5 * (r_min + r_max)
            min_val = median - r
            max_val = median + r

            diff = self.weighted_imbalance(
                x, min_val, max_val, alpha, gamma, base_weight
            )

            if diff > 0:
                # 左侧加权过大 → 区间太窄 → 放大
                r_min = r
            else:
                # 右侧加权过大
                r_max = r

        r = 0.5 * (r_min + r_max)
        return median - r, median + r

    def weighted_imbalance(self, x, min_val, max_val, alpha=3.0, gamma=2.0, base_weight=0.1):
        '''
        在满足 coverage 的前提下，用 二分搜索 r
        定义加权不平衡函数（关键）
        '''
        norm = (x - min_val) / (max_val - min_val)
        norm = np.clip(norm, 0.0, 1.0)

        w = base_weight + alpha * np.abs(norm - 0.5) ** gamma

        left = w[norm < 0.5].sum()
        right = w[norm > 0.5].sum()

        return left - right


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