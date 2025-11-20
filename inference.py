'''
该脚本用于实现评估模型情感控制的实验
'''
import os
import sys

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


# 模型初始化
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True,
    use_cuda_kernel=True,
    use_deepspeed=False
)
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
            "mabaoguo": "examples/voice_mabaoguo.wav",
            "furina": "examples/voice_furina.wav",
            "zhongli": "examples/voice_zhongli.wav",
        }

        # 定义带有不同情感色彩的文本数据
        self.text_data = {
            "平静,温和": "清晨的阳光透过窗帘洒在书桌上，新的一天开始了。窗外鸟儿欢快地歌唱，空气中弥漫着淡淡的花香。",
            "喜悦,兴奋": "突然，电话铃声响起！是我期待已久的好消息——我获得了梦寐以求的工作机会！这真是太棒了！我忍不住在房间里跳起舞来，心中充满了无限的喜悦和期待！",
            "焦虑,紧张": "然而，兴奋过后，一丝不安涌上心头。我真的能胜任这份工作吗？面对新的环境和挑战，我感到有些忐忑。手心开始冒汗，心跳也不由自主地加快了。",
            "悲伤,低落": "就在这时，窗外下起了雨。雨滴敲打着窗户，仿佛在诉说着我的忧虑。我想起了去年这个时候，也是在这样的雨天，我失去了最亲爱的外婆... 泪水模糊了双眼。",
            "愤怒,激动": "不！我不能这样消沉下去！为什么每次遇到困难就要退缩？我受够了这种懦弱的自己！我要振作起来，证明给所有人看！",
            "坚定,充满希望": "雨停了，彩虹出现在天边。我深吸一口气，告诉自己：人生就像这四季更替，有晴有雨，有起有落。我要勇敢地迎接每一个挑战！",
            "温柔,感恩": "感谢那些曾经帮助过我的人，感谢生活中的每一次经历。无论是快乐还是痛苦，都让我成为了更好的自己。",
            "幽默,轻松": "说起来，人生就像坐过山车——有时候你会尖叫，有时候你会大笑，但最重要的是，你要享受整个过程！",
            "平和,睿智": "夜幕降临，星光点点。我静静地坐在窗前，心中充满了平静。明天，将是全新的一天，带着希望和勇气继续前行。"
        }

        # 情感向量映射
        self.emotion_vectors_map = {
            "happy": [0.8, 0, 0, 0, 0, 0, 0.2, 0],
            "angry": [0, 0.8, 0, 0, 0, 0, 0.2, 0],
            "sad": [0, 0, 0.8, 0, 0, 0.2, 0, 0],
            "afraid": [0, 0, 0, 0.8, 0, 0, 0.2, 0],
            "disgusted": [0, 0, 0, 0, 0.8, 0, 0.2, 0],
            "melancholic": [0, 0, 0.2, 0, 0, 0.8, 0, 0],
            "surprised": [0.2, 0, 0, 0, 0, 0, 0.8, 0],
            "calm": [0, 0, 0, 0, 0, 0, 0, 1.0]
        }

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
        total_tasks = len(self.spk_audio_data) * len(self.text_data) * len(self.emotion_vectors_map)

        with tqdm(total=total_tasks, desc="实验3进度") as pbar:
            for spk_name, spk_audio_path in self.spk_audio_data.items():
                for emotion_label, text in self.text_data.items():
                    for emo_name, emo_vector in self.emotion_vectors_map.items():
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



if __name__ == "__main__":
    # 初始化实验类
    experiment = IndexTTSExperiment(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False
    )

    # 实验1: 无情感控制
    experiment.experiment1_no_emotion_control()

    # 实验2: 自我推断情感
    experiment.experiment2_self_inferred_emotion()

    # 实验3: 显式情感控制
    experiment.experiment3_explicit_emotion_control()










