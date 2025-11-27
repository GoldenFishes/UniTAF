'''
该脚本用于实现评估模型情感控制的实验
'''
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import spatial


# 预防 _kaldifst动态库找不到 的提前导入路径的方法
def setup_kaldifst_dll_paths():
    """设置 kaldifst 的 DLL 搜索路径"""
    try:
        import site
        site_packages_list = site.getsitepackages()

        site_packages = None
        for path in site_packages_list:
            if 'site-packages' in path and os.path.exists(path):
                site_packages = path
                break

        if not site_packages:
            site_packages = os.path.join(sys.prefix, "Lib", "site-packages")

        dll_paths = [
            site_packages,
            os.path.join(site_packages, "kaldifst", "bin"),
            os.path.join(site_packages, "kaldifst", "lib")
        ]

        for path in dll_paths:
            if os.path.exists(path):
                try:
                    os.add_dll_directory(path)
                except (AttributeError, OSError):
                    current_path = os.environ.get('PATH', '')
                    if path not in current_path:
                        os.environ['PATH'] = path + ';' + current_path
        print(f"kaldifst dll paths: {dll_paths}")
    except Exception as e:
        print(f"Warning: 设置 kaldifst DLL 路径失败: {e}")

# 在所有包之前导入 _kaldifst_dll 路径 防止找不到报错
# setup_kaldifst_dll_paths()
# import hashlib

from tqdm import tqdm
import shutil
import itertools
from indextts.infer_v2 import IndexTTS2


"""
# 1. 用一个参考音频文件合成新语音（语音克隆）
text = "Translate for me, what is a surprise!"
tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text=text,
    output_path="gen.wav",
    verbose=True
)

# 2. 使用一个独立的情感参考音频文件来调节语音合成
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"
tts.infer(
    spk_audio_prompt='examples/voice_07.wav',
    text=text,
    output_path="gen.wav",
    emo_audio_prompt="examples/emo_sad.wav",
    # 当指定情感参考音频文件时，你可以选择设置 emo_alpha 来调整对输出的影响程度。
    emo_alpha=0.9,  # 有效范围为 0.0 - 1.0，默认值为 1.0
    verbose=True
)

# 3. 也可以省略情感参考音频，改用 一个8浮点表，指定每种情绪的强度，顺序如下：
# [开心、  愤怒、  悲伤、  害怕、    厌恶、       忧郁、       惊讶、    平静]
# [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
text = "哇塞！这个爆率也太高了！欧皇附体了！"
tts.infer(
    spk_audio_prompt='examples/voice_10.wav',
    text=text,
    output_path="gen.wav",
    emo_vector=[0, 0, 0, 0, 0, 0, 0.45, 0],
    use_random=False, # 还可以使用 use_random 参数在推理过程中引入随机性; 随机采样会降低语音合成的声音克隆精度.
    verbose=True
)

# 4. 或者，也可以启用 use_emo_text 根据提供的文字脚本引导情绪。
# 还文字脚本会自动转换为情感向量。建议在使用文本情感模式时使用大约 0.6（或更低）的 emo_alpha，这样语音听起来更自然。
text = "快躲起来！是他要来了！他要来抓我们了！"
tts.infer(
    spk_audio_prompt='examples/voice_12.wav',
    text=text,
    output_path="gen.wav",
    emo_alpha=0.6,
    use_emo_text=True,  # 根据提供的文字脚本引导情绪
    use_random=False,
    verbose=True
)

# 5. 也可以通过 emo_text 参数直接提供特定的文本情感描述。
# 你的情感文本随后会自动转换为情感向量。这样你就能分别控制文本脚本和文本情感描述：
text = "快躲起来！是他要来了！他要来抓我们了！"
emo_text = "你吓死我了！你是鬼吗？"
tts.infer(
    spk_audio_prompt='examples/voice_12.wav',
    text=text,
    output_path="gen.wav",
    emo_alpha=0.6,
    use_emo_text=True,
    emo_text=emo_text,  # 情感控制选择从传入的情感文本中推断，不传额外用于推断的情感文本时则直接从目标文本中推断。
    use_random=False,
    verbose=True
)

# 6. IndexTTS-2仍然支持一代的拼音功能，支持精确标注发音，例如：
text = "之前你做DE5很好，所以这一次也DEI3做DE2很好才XING2，如果这次目标完成得不错的话，我们就直接打DI1去银行取钱。"
"""


class IndexTTSExperiment:
    def __init__(
        self,
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_fp16=False, use_cuda_kernel=False, use_deepspeed=False,
    ):
        self.tts = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=use_fp16,
            use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=use_deepspeed
        )

        # 定义说话人音频数据
        self.spk_audio_data = {
            # "mabaoguo": "examples/voice_mabaoguo.wav",
            "furina": "examples/voice_furina.wav",
            "zhongli": "examples/voice_zhongli.wav",
        }

        # 定义带有不同情感色彩的文本数据
        self.text_data = {
            "平静,温和": "清晨的阳光透过窗帘洒在书桌上，新的一天开始了。窗外鸟儿欢快地歌唱，空气中弥漫着淡淡的花香。",
            # "喜悦,兴奋": "突然，电话铃声响起！是我期待已久的好消息——我获得了梦寐以求的工作机会！这真是太棒了！我忍不住在房间里跳起舞来，心中充满了无限的喜悦和期待！",
            # "焦虑,紧张": "然而，兴奋过后，一丝不安涌上心头。我真的能胜任这份工作吗？面对新的环境和挑战，我感到有些忐忑。手心开始冒汗，心跳也不由自主地加快了。",
            # "悲伤,低落": "就在这时，窗外下起了雨。雨滴敲打着窗户，仿佛在诉说着我的忧虑。我想起了去年这个时候，也是在这样的雨天，我失去了最亲爱的外婆... 泪水模糊了双眼。",
            # "愤怒,激动": "不！我不能这样消沉下去！为什么每次遇到困难就要退缩？我受够了这种懦弱的自己！我要振作起来，证明给所有人看！",
            # "坚定,充满希望": "雨停了，彩虹出现在天边。我深吸一口气，告诉自己：人生就像这四季更替，有晴有雨，有起有落。我要勇敢地迎接每一个挑战！",
            # "温柔,感恩": "感谢那些曾经帮助过我的人，感谢生活中的每一次经历。无论是快乐还是痛苦，都让我成为了更好的自己。",
            # "幽默,轻松": "说起来，人生就像坐过山车——有时候你会尖叫，有时候你会大笑，但最重要的是，你要享受整个过程！",
            # "平和,睿智": "夜幕降临，星光点点。我静静地坐在窗前，心中充满了平静。明天，将是全新的一天，带着希望和勇气继续前行。"
        }

        # 专门用于不同情感控制测试的中性文本
        self.neutral_texts = {
            "中性文本1": "今天下午三点钟，我需要去超市购买一些日常用品和食物。",
            "中性文本2": "从明天开始，图书馆的开放时间将会延长到晚上十点钟。"
        }


        # 具有代表性的情感向量 供实验3使用
        self.emotion_vectors_example = {
            "happy": [0.8, 0, 0, 0, 0, 0, 0.2, 0],
            "angry": [0, 0.8, 0, 0, 0, 0, 0.2, 0],
            "sad": [0, 0, 0.8, 0, 0, 0.2, 0, 0],
            "afraid": [0, 0, 0, 0.8, 0, 0, 0.2, 0],
            "disgusted": [0, 0, 0, 0, 0.8, 0, 0.2, 0],
            "melancholic": [0, 0, 0.2, 0, 0, 0.8, 0, 0],
            "surprised": [0.2, 0, 0, 0, 0, 0, 0.8, 0],
            "calm": [0, 0, 0, 0, 0, 0, 0, 1.0]
        }

        # 实际情感向量映射 供实验4使用
        self.emotion_vectors_map = {
            "happy":        [1.0, 0, 0, 0, 0, 0, 0, 0],
            "angry":        [0, 1.0, 0, 0, 0, 0, 0, 0],
            "sad":          [0, 0, 1.0, 0, 0, 0, 0, 0],
            "afraid":       [0, 0, 0, 1.0, 0, 0, 0, 0],
            "disgusted":    [0, 0, 0, 0, 1.0, 0, 0, 0],
            "melancholic":  [0, 0, 0, 0, 0, 1.0, 0, 0],
            "surprised":    [0, 0, 0, 0, 0, 0, 1.0, 0],
            "calm":         [0, 0, 0, 0, 0, 0, 0, 1.0]
        }

        # 情感名称列表（用于组合）
        self.emotion_names = list(self.emotion_vectors_map.keys())

        # 控制emo_alpha值程度的列表
        self.emo_alpha_values = [1.0, 0.8, 0.6, 0.4, 0.2]


    def experiment1_no_emotion_control(self, output_dir="outputs/experiment1_no_emotion"):
        """
        实验1: 无情感控制语音克隆任务
        遍历所有说话人和文本，进行基础的语音克隆
        """
        print("=" * 50)
        print("开始实验1: 无情感控制语音克隆")
        print("=" * 50)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        total_tasks = len(self.spk_audio_data) * len(self.text_data)

        with tqdm(total=total_tasks, desc="实验1进度") as pbar:
            for spk_name, spk_audio_path in self.spk_audio_data.items():
                for emotion_label, text in self.text_data.items():
                    # 生成输出文件名
                    safe_emotion_label = emotion_label.replace(",", "_").replace(" ", "")
                    output_filename = f"{spk_name}_{safe_emotion_label}.wav"
                    output_path = os.path.join(output_dir, output_filename)

                    # 执行语音合成
                    self.tts.infer(
                        spk_audio_prompt=spk_audio_path,
                        text=text,
                        output_path=output_path,
                        verbose=False  # 关闭详细输出以避免干扰进度条
                    )

                    pbar.update(1)
                    pbar.set_postfix({
                        "说话人": spk_name,
                        "情感": emotion_label[:10] + "..." if len(emotion_label) > 10 else emotion_label
                    })

        print(f"实验1完成！结果保存在: {output_dir}")

    def experiment2_self_inferred_emotion(self, output_dir="outputs/experiment2_self_inferred_emotion"):
        """
        实验2: 根据目标文本自我推断情感的语音克隆任务
        use_emo_text=true，但不额外传emo_text
        """
        print("=" * 50)
        print("开始实验2: 自我推断情感语音克隆")
        print("=" * 50)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        total_tasks = len(self.spk_audio_data) * len(self.text_data)

        with tqdm(total=total_tasks, desc="实验2进度") as pbar:
            for spk_name, spk_audio_path in self.spk_audio_data.items():
                for emotion_label, text in self.text_data.items():
                    # 生成输出文件名
                    safe_emotion_label = emotion_label.replace(",", "_").replace(" ", "")
                    output_filename = f"{spk_name}_{safe_emotion_label}.wav"
                    output_path = os.path.join(output_dir, output_filename)

                    # 执行语音合成（启用文本情感推断）
                    self.tts.infer(
                        spk_audio_prompt=spk_audio_path,
                        text=text,
                        output_path=output_path,
                        emo_alpha=0.6,  # 建议使用较低的情感强度
                        use_emo_text=True,
                        use_random=False,
                        verbose=False
                    )

                    pbar.update(1)
                    pbar.set_postfix({
                        "说话人": spk_name,
                        "情感": emotion_label[:10] + "..." if len(emotion_label) > 10 else emotion_label
                    })

        print(f"实验2完成！结果保存在: {output_dir}")

    def experiment3_explicit_emotion_control(self, output_dir="outputs/experiment3_explicit_emotion"):
        """
        实验3: 显式控制情感标签的语音克隆任务
        使用emo_vector控制同一个文本的不同情感输出
        """
        print("=" * 50)
        print("开始实验3: 显式情感控制语音克隆")
        print("=" * 50)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 为每个说话人和文本生成多种情感版本
        total_tasks = len(self.spk_audio_data) * len(self.text_data) * len(self.emotion_vectors_example)

        with tqdm(total=total_tasks, desc="实验3进度") as pbar:
            for spk_name, spk_audio_path in self.spk_audio_data.items():
                for emotion_label, text in self.text_data.items():
                    for emo_name, emo_vector in self.emotion_vectors_example.items():
                        # 生成输出文件名
                        safe_emotion_label = emotion_label.replace(",", "_").replace(" ", "")
                        output_filename = f"{spk_name}_{safe_emotion_label}_{emo_name}.wav"
                        output_path = os.path.join(output_dir, output_filename)

                        # 执行语音合成（使用显式情感向量）
                        self.tts.infer(
                            spk_audio_prompt=spk_audio_path,
                            text=text,
                            output_path=output_path,
                            emo_vector=emo_vector,
                            use_random=False,
                            verbose=False
                        )

                        pbar.update(1)
                        pbar.set_postfix({
                            "说话人": spk_name,
                            "文本情感": emotion_label[:8] + "..." if len(emotion_label) > 8 else emotion_label,
                            "控制情感": emo_name
                        })

        print(f"实验3完成！结果保存在: {output_dir}")

    def _generate_emotion_transition_vectors(self, emotion1, emotion2, steps=5):
        """
        生成两种情绪之间的平滑过渡向量

        Args:
            emotion1: 第一种情绪名称
            emotion2: 第二种情绪名称
            steps: 过渡步数（包括起点和终点）

        Returns:
            list: 包含所有过渡向量的列表
        """
        vectors = []
        # 获取基础情绪向量
        vec1 = self.emotion_vectors_map[emotion1]
        vec2 = self.emotion_vectors_map[emotion2]

        # 生成过渡向量
        for i in range(steps):
            alpha = i / (steps - 1) if steps > 1 else 0
            # 线性插值
            transition_vector = [
                vec1[j] * (1 - alpha) + vec2[j] * alpha
                for j in range(len(vec1))
            ]
            vectors.append(transition_vector)

        return vectors

    def experiment4_emotion_transition(self, output_dir="outputs/experiment4_emotion_transition", steps=5):
        """
        实验4: 两种情绪之间的平滑过渡实验

        Args:
            output_dir: 输出目录
            steps: 过渡步数
        """
        print("=" * 50)
        print("开始实验4: 情绪平滑过渡实验")
        print("=" * 50)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成所有情绪对组合
        emotion_pairs = list(itertools.combinations(self.emotion_names, 2))

        # 计算总任务数
        total_tasks = len(self.spk_audio_data) * len(self.text_data) * len(emotion_pairs) * steps

        with tqdm(total=total_tasks, desc="实验4进度") as pbar:
            for spk_name, spk_audio_path in self.spk_audio_data.items():
                for emotion_label, text in self.text_data.items():
                    for emotion1, emotion2 in emotion_pairs:
                        # 生成过渡向量
                        transition_vectors = self._generate_emotion_transition_vectors(
                            emotion1, emotion2, steps
                        )

                        for step, emo_vector in enumerate(transition_vectors):
                            # 计算当前步骤的混合比例
                            alpha = step / (steps - 1) if steps > 1 else 0

                            # 生成输出文件名
                            safe_emotion_label = emotion_label.replace(",", "_").replace(" ", "")
                            output_filename = (
                                f"{spk_name}_{safe_emotion_label}_"
                                f"{emotion1}_to_{emotion2}_step{step + 1}_of_{steps}_"
                                f"alpha{alpha:.1f}.wav"
                            )
                            output_path = os.path.join(output_dir, output_filename)

                            # 执行语音合成
                            self.tts.infer(
                                spk_audio_prompt=spk_audio_path,
                                text=text,
                                output_path=output_path,
                                emo_vector=emo_vector,
                                use_random=False,
                                verbose=False
                            )

                            pbar.update(1)
                            pbar.set_postfix({
                                "说话人": spk_name,
                                "文本情感": emotion_label[:8] + "..." if len(emotion_label) > 8 else emotion_label,
                                "情绪过渡": f"{emotion1}→{emotion2}",
                                "步骤": f"{step + 1}/{steps}"
                            })

        print(f"实验4完成！结果保存在: {output_dir}")

        # 打印实验统计信息
        self._print_experiment4_stats(emotion_pairs, steps)

    def _print_experiment4_stats(self, emotion_pairs, steps):
        """打印实验4的统计信息"""
        print("\n" + "=" * 50)
        print("实验4统计信息:")
        print("=" * 50)
        print(f"说话人数量: {len(self.spk_audio_data)}")
        print(f"文本数量: {len(self.text_data)}")
        print(f"情绪对组合数量: {len(emotion_pairs)}")
        print(f"每个过渡的步数: {steps}")
        print(f"总生成文件数: {len(self.spk_audio_data) * len(self.text_data) * len(emotion_pairs) * steps}")
        print(f"情绪对组合:")
        for i, (emo1, emo2) in enumerate(emotion_pairs):
            print(f"  {i + 1:2d}. {emo1:12} → {emo2:12}")

    def experiment4_specific_transition(self, emotion_pairs, output_dir="outputs/experiment4_specific_transition", steps=5):
        """
        实验4的变体：只针对特定的情绪对进行过渡实验

        Args:
            emotion_pairs: 指定的情绪对列表，如 [('happy', 'sad'), ('angry', 'calm')]
            output_dir: 输出目录
            steps: 过渡步数
        """
        print("=" * 50)
        print("开始实验4变体: 特定情绪对过渡实验")
        print("=" * 50)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 计算总任务数
        total_tasks = len(self.spk_audio_data) * len(self.text_data) * len(emotion_pairs) * steps

        with tqdm(total=total_tasks, desc="实验4变体进度") as pbar:
            for spk_name, spk_audio_path in self.spk_audio_data.items():
                for emotion_label, text in self.text_data.items():
                    for emotion1, emotion2 in emotion_pairs:
                        # 验证情绪名称
                        if emotion1 not in self.emotion_vectors_map or emotion2 not in self.emotion_vectors_map:
                            print(f"警告: 未知情绪名称 '{emotion1}' 或 '{emotion2}'，跳过")
                            continue

                        # 生成过渡向量
                        transition_vectors = self._generate_emotion_transition_vectors(
                            emotion1, emotion2, steps
                        )

                        for step, emo_vector in enumerate(transition_vectors):
                            # 计算当前步骤的混合比例
                            alpha = step / (steps - 1) if steps > 1 else 0

                            # 生成输出文件名
                            safe_emotion_label = emotion_label.replace(",", "_").replace(" ", "")
                            output_filename = (
                                f"{spk_name}_{safe_emotion_label}_"
                                f"{emotion1}_to_{emotion2}_step{step + 1}_of_{steps}_"
                                f"alpha{alpha:.1f}.wav"
                            )
                            output_path = os.path.join(output_dir, output_filename)

                            # 执行语音合成
                            self.tts.infer(
                                spk_audio_prompt=spk_audio_path,
                                text=text,
                                output_path=output_path,
                                emo_vector=emo_vector,
                                use_random=False,
                                verbose=False
                            )

                            pbar.update(1)
                            pbar.set_postfix({
                                "说话人": spk_name,
                                "文本情感": emotion_label[:8] + "..." if len(emotion_label) > 8 else emotion_label,
                                "情绪过渡": f"{emotion1}→{emotion2}",
                                "步骤": f"{step + 1}/{steps}"
                            })

        print(f"实验4变体完成！结果保存在: {output_dir}")

    def experiment5_emo_alpha_sensitivity(self, output_dir="outputs/experiment5_emo_alpha_sensitivity"):
        """
        实验5: 测试每种基础情绪在不同emo_alpha值下的敏感性

        使用每种基础情绪的1.0强度向量，但通过emo_alpha控制整体情感强度
        """
        print("=" * 50)
        print("开始实验5: emo_alpha敏感性测试")
        print("=" * 50)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 计算总任务数
        total_tasks = len(self.spk_audio_data) * len(self.neutral_texts) * len(self.emotion_vectors_map) * len(
            self.emo_alpha_values)

        with tqdm(total=total_tasks, desc="实验5进度") as pbar:
            for spk_name, spk_audio_path in self.spk_audio_data.items():
                for emotion_label, text in self.neutral_texts.items():
                    for emo_name, emo_vector in self.emotion_vectors_map.items():
                        for emo_alpha in self.emo_alpha_values:
                            # 生成输出文件名
                            safe_emotion_label = emotion_label.replace(",", "_").replace(" ", "")
                            output_filename = (
                                f"{spk_name}_{safe_emotion_label}_"
                                f"{emo_name}_alpha{emo_alpha}.wav"
                            )
                            output_path = os.path.join(output_dir, output_filename)

                            # 执行语音合成（使用emo_alpha控制情感强度）
                            self.tts.infer(
                                spk_audio_prompt=spk_audio_path,
                                text=text,
                                output_path=output_path,
                                emo_vector=emo_vector,
                                emo_alpha=emo_alpha,  # 关键参数：控制情感强度
                                use_random=False,
                                verbose=False
                            )

                            pbar.update(1)
                            pbar.set_postfix({
                                "说话人": spk_name,
                                "文本情感": emotion_label[:8] + "..." if len(emotion_label) > 8 else emotion_label,
                                "控制情感": emo_name,
                                "emo_alpha": emo_alpha
                            })

        print(f"实验5完成！结果保存在: {output_dir}")
        self._print_experiment5_stats()

    def _print_experiment5_stats(self):
        """打印实验5的统计信息"""
        print("\n" + "=" * 50)
        print("实验5统计信息:")
        print("=" * 50)
        print(f"说话人数量: {len(self.spk_audio_data)}")
        print(f"文本数量: {len(self.text_data)}")
        print(f"基础情绪数量: {len(self.emotion_vectors_map)}")
        print(f"emo_alpha值: {self.emo_alpha_values}")
        print(
            f"总生成文件数: {len(self.spk_audio_data) * len(self.text_data) * len(self.emotion_vectors_map) * len(self.emo_alpha_values)}")

        print(f"\n测试的情绪:")
        for i, emo_name in enumerate(self.emotion_vectors_map.keys()):
            print(f"  {i + 1:2d}. {emo_name:12} -> {self.emotion_vectors_map[emo_name]}")

    def compare_tts_audio_and_GT_audio(
        self,
        num_of_samples = 15,  # 只实验数据集中前N个样本
        dataset_path = "/home/zqg/project/data/UniTAF Dataset",
        dataset_json = "all_in_one_final_emotion.json",
        output_dir = "outputs/compare_tts_audio_and_GT_audio",
    ):
        '''
        面对数据不对齐问题（TTS生成的audio不等于数据集中的GT audio）:
        https://e6rsxnypod.feishu.cn/wiki/G33owIEw4iEYRpkSUw8cyPrqn6g#share-Q4mDdNvIcoct1Zx86GNcph9EnDR 1.4.1节
        因此我们这里比较TTS生成的audio与GT audio的差别有多大
        '''

        # 加载dataset中JSON内容
        json_path = os.path.join(dataset_path, dataset_json)
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        comparison_results = []

        # 1.遍历 ------------------------------------------------------
        # 遍历并获取每一个样本
        for i, sample in zip(range(num_of_samples), tqdm(raw_data["data"][:num_of_samples], desc=f"遍历数据集中前{num_of_samples}个样本")):
            '''
            获得每个样本后，使用第一个chunk作为spk_prompt, 对第二个chunk的文本进行voice clone (每个样本中的不同chunk来自同一说话人),
            得到第二个chunk的tts输出音频。并与第二个chunk的数据集GT音频进行比较。
            '''
            speaker_idx = sample["speaker_idx"]
            sample_id = sample["sample_id"]
            chunks = sample["chunk"]

            # 检查是否有至少2个chunk
            if len(chunks) < 2:
                continue

            # 获取第一个chunk作为speaker prompt
            first_chunk_key = list(chunks.keys())[0]
            first_chunk = chunks[first_chunk_key]

            # 获取第二个chunk作为目标
            second_chunk_key = list(chunks.keys())[1]
            second_chunk = chunks[second_chunk_key]

            # 修复路径问题：将反斜杠转换为正斜杠
            def fix_path(path):
                """修复路径中的反斜杠问题"""
                return path.replace('\\', '/') if isinstance(path, str) else path

            # 获取推理所需输入的路径
            spk_audio_prompt_path = os.path.join(dataset_path, fix_path(first_chunk["audio_path"]))
            text_path = os.path.join(dataset_path, fix_path(second_chunk["text_path"]))
            GT_audio_path = os.path.join(dataset_path, fix_path(second_chunk["audio_path"]))

            # 生成输出文件名
            tts_output_filename = f"sample{i:03d}_spk{speaker_idx}_{sample_id}_chunk{second_chunk_key}_tts.wav"
            gt_output_filename = f"sample{i:03d}_spk{speaker_idx}_{sample_id}_chunk{second_chunk_key}_gt.wav"
            comparison_plot_filename = f"sample{i:03d}_spk{speaker_idx}_{sample_id}_chunk{second_chunk_key}_comparison.png"

            tts_output_path = os.path.join(output_dir, tts_output_filename)
            gt_output_path = os.path.join(output_dir, gt_output_filename)
            plot_output_path = os.path.join(output_dir, comparison_plot_filename)

            # 2. 生成 ------------------------------------------------------
            # 执行推理得到tts_audio
            self.tts.infer(
                spk_audio_prompt=spk_audio_prompt_path,
                text=self.load_text(text_path),
                output_path=tts_output_path,  # 直接保存到指定路径
                verbose=False
            )

            # 复制GT音频到输出目录(保留原始GT音频)
            shutil.copy2(GT_audio_path, gt_output_path)


            # 3. 比较 ------------------------------------------------------
            # 加载TTS和GT音频
            tts_audio, tts_sr = librosa.load(tts_output_path, sr=None)
            gt_audio, gt_sr = librosa.load(gt_output_path, sr=None)

            # 统一采样率
            target_sr = 16000
            if tts_sr != target_sr:
                tts_audio = librosa.resample(tts_audio, orig_sr=tts_sr, target_sr=target_sr)
            if gt_sr != target_sr:
                gt_audio = librosa.resample(gt_audio, orig_sr=gt_sr, target_sr=target_sr)

            # 数值计算相似性指标
            similarity_metrics = self.calculate_audio_similarity(tts_audio, gt_audio, target_sr)

            # 可视化推理结果与GT音频的波形比较
            self.visualize_audio_comparison(
                tts_audio, gt_audio, target_sr,
                plot_output_path,
                f"Sample {i}: Spk{speaker_idx}_{sample_id}"
            )

            # 保存比较结果
            comparison_result = {
                "sample_index": i,
                "speaker_idx": speaker_idx,
                "sample_id": sample_id,
                "first_chunk": first_chunk_key,
                "second_chunk": second_chunk_key,
                "tts_audio_path": tts_output_path,
                "gt_audio_path": gt_output_path,
                "plot_path": plot_output_path,
                "text": self.load_text(text_path),
                "similarity_metrics": similarity_metrics,
                "audio_durations": {
                    "tts": len(tts_audio) / target_sr,
                    "gt": len(gt_audio) / target_sr
                }
            }
            comparison_results.append(comparison_result)

            print(f"\n样本 {i}: speaker_{speaker_idx}_{sample_id}")
            print(f"  文本: {self.load_text(text_path)[:50]}...")
            print(
                f"  时长 - TTS: {comparison_result['audio_durations']['tts']:.2f}s, GT: {comparison_result['audio_durations']['gt']:.2f}s")
            print(f"  相似性指标:")
            for metric, value in similarity_metrics.items():
                print(f"    {metric}: {value:.4f}")


        # 保存总的比较结果摘要
        if comparison_results:
            summary_path = os.path.join(output_dir, "comparison_summary.json")

            # 转换NumPy类型为Python原生类型以便JSON序列化
            serializable_results = self.convert_to_serializable(comparison_results)

            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            # 生成总体统计报告
            self.generate_summary_report(comparison_results, output_dir)

            print(f"\n处理完成！共成功处理 {len(comparison_results)} 个样本")
            print(f"所有输出文件保存在: {output_dir}")
        else:
            print("\n没有成功处理的样本")

        return comparison_results

    def convert_to_serializable(self, obj):
        """将对象中的NumPy类型转换为Python原生类型以便JSON序列化"""
        if isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def calculate_audio_similarity(self, audio1, audio2, sr):
        """计算音频相似性指标"""

        # 确保音频长度一致（取较短的长度）
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]

        metrics = {}

        # 1. 波形相似性 - 余弦相似度
        metrics['waveform_cosine_similarity'] = 1 - spatial.distance.cosine(audio1, audio2)

        # 2. 均方根误差 (RMSE)
        metrics['rmse'] = np.sqrt(np.mean((audio1 - audio2) ** 2))

        # 3. 信噪比 (SNR)
        noise = audio1 - audio2
        signal_power = np.mean(audio2 ** 2)
        noise_power = np.mean(noise ** 2)
        metrics['snr_db'] = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

        # 4. 提取MFCC特征并比较
        mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr, n_mfcc=13)

        # MFCC余弦相似度（取第一维）
        metrics['mfcc_cosine_similarity'] = 1 - spatial.distance.cosine(mfcc1[0], mfcc2[0])

        # MFCC均方误差
        metrics['mfcc_mse'] = np.mean((mfcc1 - mfcc2) ** 2)

        # 5. 提取梅尔频谱图
        mel1 = librosa.feature.melspectrogram(y=audio1, sr=sr)
        mel2 = librosa.feature.melspectrogram(y=audio2, sr=sr)

        # 转换为对数尺度
        log_mel1 = librosa.power_to_db(mel1, ref=np.max)
        log_mel2 = librosa.power_to_db(mel2, ref=np.max)

        # 梅尔频谱图相似度
        metrics['mel_spectrogram_similarity'] = 1 - spatial.distance.cosine(
            log_mel1.flatten()[:1000],  # 取前1000个元素避免维度太高
            log_mel2.flatten()[:1000]
        )

        return metrics

    def visualize_audio_comparison(self, tts_audio, gt_audio, sr, save_path, title):
        """可视化音频比较"""

        # 创建子图
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(title, fontsize=14)

        # 确保音频长度一致
        min_len = min(len(tts_audio), len(gt_audio))
        tts_audio = tts_audio[:min_len]
        gt_audio = gt_audio[:min_len]
        time_axis = np.linspace(0, min_len / sr, min_len)

        # 1. 波形对比
        axes[0].plot(time_axis, gt_audio, alpha=0.7, label='GT Audio', color='blue')
        axes[0].plot(time_axis, tts_audio, alpha=0.7, label='TTS Audio', color='red')
        axes[0].set_title('Waveform Comparison')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 频谱图对比
        # GT音频频谱图
        S_gt = librosa.stft(gt_audio)
        S_db_gt = librosa.amplitude_to_db(np.abs(S_gt), ref=np.max)
        librosa.display.specshow(S_db_gt, sr=sr, x_axis='time', y_axis='hz', ax=axes[1], cmap='viridis')
        axes[1].set_title('GT Audio Spectrogram')
        axes[1].set_ylabel('Frequency (Hz)')

        # 3. TTS音频频谱图
        S_tts = librosa.stft(tts_audio)
        S_db_tts = librosa.amplitude_to_db(np.abs(S_tts), ref=np.max)
        librosa.display.specshow(S_db_tts, sr=sr, x_axis='time', y_axis='hz', ax=axes[2], cmap='viridis')
        axes[2].set_title('TTS Audio Spectrogram')
        axes[2].set_ylabel('Frequency (Hz)')
        axes[2].set_xlabel('Time (s)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def generate_summary_report(self, results, output_dir):
        """生成总体统计报告"""
        if not results:
            return

        # 计算各项指标的统计信息
        metrics_names = list(results[0]['similarity_metrics'].keys())
        summary = {}

        for metric in metrics_names:
            values = [r['similarity_metrics'][metric] for r in results]
            summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

        # 保存统计报告
        report_path = os.path.join(output_dir, "statistical_summary.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

    def load_text(self, text_path):
        """加载文本文件"""
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"加载文本失败 {text_path}: {e}")
            return ""


if __name__ == "__main__":
    '''
    根目录执行：
    uv run inference.py
    '''
    # 初始化实验类
    experiment = IndexTTSExperiment(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False
    )

    # 实验1: 无情感控制
    # experiment.experiment1_no_emotion_control()

    # 实验2: 自我推断情感
    # experiment.experiment2_self_inferred_emotion()

    # 实验3: 显式情感控制
    # experiment.experiment3_explicit_emotion_control()

    # 实验4（变体）:只针对特定情绪对
    # specific_pairs = [
    #     ('happy', 'sad'), ('angry', 'calm'), ('surprised', 'afraid')
    # ]
    # experiment.experiment4_specific_transition(specific_pairs, steps=5)


    # 实验5：emo_alpha控制不同情感的强度
    # experiment.experiment5_emo_alpha_sensitivity()

    # 实验：比较TTS audio与GT audio差异
    experiment.compare_tts_audio_and_GT_audio()


    # 不使用批量实验，单独测试
    # text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"
    # experiment.tts.infer(
    #     spk_audio_prompt='examples/voice_07.wav',
    #     text=text,
    #     output_path="gen.wav",
    #     emo_audio_prompt="examples/emo_sad.wav",
    #     # 当指定情感参考音频文件时，你可以选择设置 emo_alpha 来调整对输出的影响程度。
    #     emo_alpha=0.9,  # 有效范围为 0.0 - 1.0，默认值为 1.0
    #     verbose=True
    # )

    # text = "快躲起来！是他要来了！他要来抓我们了！"
    # experiment.tts.infer(
    #     spk_audio_prompt='examples/voice_12.wav',
    #     text=text,
    #     output_path="gen.wav",
    #     emo_alpha=0.6,
    #     use_emo_text=True,  # 根据提供的文字脚本引导情绪
    #     use_random=False,
    #     verbose=True
    # )

