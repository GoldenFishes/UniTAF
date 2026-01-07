'''
该脚本用于实现评估模型的实验
- UniTAFExperiment 主要用于评估联合模型的生成能力实验
'''
import os
from omegaconf import OmegaConf
from pathlib import Path
import copy
import torch
from unitaf_train.UniTAF import UniTextAudioFaceModel

# UniTAF默认的模型配置，在实验类中需要根据experiment_config在unitaf_default_config基础上填入相应组件的不同权重
unitaf_default_config = {
    # 模型类型，这里用于指导训练器类训练哪些模型
    "tts_model": ["IndexTTS2"],
    "a2f_model": ["UniTalker"],
    # 模型配置
    "IndexTTS2": {
        # TTS Loss计算时设置
        "use_duration_control": False,
        "duration_dropout": 0.3,
        "text_loss_weight": 0.2,
        "mel_loss_weight": 0.8,
    },
    "UniTalker": {
        # UniTalker Decoder配置, 参数与UniTalker项目的config/unitalker.yaml一致
        "interpolate_pos": 1,
        "decoder_dimension": 256,
        "decoder_type": "conv",
        "period": 30,
        "headlayer": 1,
        # UniTalker Network
        "use_pca": True,
        "pca_dim": 256,
        # A2F Loss计算时设置
        "pca_weight": 0.01,
        # 以下需要从外部获得并更新：
        "audio_encoder_feature_dim": 768,  # 原始UniTalker Decoder接收特征维度是768，我们需要经过projector使得音频特征输出相同维度
        "identity_num": 20,  # 假设是20，需要根据不同数据集决定
    },
    # 数据集类
    "dataset_config": {
        "dataset_root_path": "/home/zqg/project/data/UniTAF Dataset",
        "dataset_list": ["D12"]  # 这里传数据集是用于指导模型选择何种输出头
    },
    # 加载指定的部分模块的微调权重。
    "finetune_checkpoint": {}   # 在实验类中组装的model_config应当自定义好这里的微调权重
}

class UniTAFExperiment:
    def __init__(self, experiment_config):
        self.experiment_config = experiment_config  # 实验配置
        self.model = None

    def setup_model(self, model_config=unitaf_default_config):
        '''
        设置模型，这里config应当是模型的config
        '''
        cfg = OmegaConf.create(model_config)
        self.model = UniTextAudioFaceModel(cfg, device=torch.device("cuda:0"), mode="inference")

    def run_experiment(self):
        '''
        对 experiment_config["weights"] 中的每一组权重进行评估

        遍历 self.experiment_config["weights"] 得到字典，从中获取需要评估的权重。
        遍历字典的Key，key为某个权重的名称，其value用于替换unitaf_default_config["finetune_checkpoint"]的内容，得到属于改权重的unitaf_config
        为每个权重的模型，运行评估 run_eval 方法
        '''
        weights_dict = self.experiment_config["weights"]
        output_root = self.experiment_config["output_dir"]
        os.makedirs(output_root, exist_ok=True)

        # 1. 遍历 self.experiment_config["weights"]
        for weight_name, ckpt_dict in weights_dict.items():
            print(f"\n========== Testing Weight: {weight_name} ==========")

            # 深拷贝默认配置，避免污染
            unitaf_config = copy.deepcopy(unitaf_default_config)
            # 替换模型配置文件中的 finetune_checkpoint
            unitaf_config["finetune_checkpoint"] = weights_dict[weight_name]

            # 设置输出目录为输出路径的 "weight_name" 子路径
            exp_output_dir = os.path.join(output_root, weight_name)
            os.makedirs(exp_output_dir, exist_ok=True)

            # 构建模型
            self.setup_model(unitaf_config)

            # 运行评估
            self.run_eval(exp_output_dir)

            # 释放显存
            del self.model
            self.model = None
            torch.cuda.empty_cache()

    def run_eval(self, output_dir):
        '''
        对当前模型进行评估
        1. 遍历 self.experiment_config["eval_dataset_path"] 下 "text_prompt" 所有子文件夹，每个文件夹名称为{TASK_NAME}_prompt代表一种文本生成任务名称，
            在 {TASK_NAME}_prompt 文件夹下的每个{SAMPLE_NAME}.txt文件均为一条要生成的目标文本提示，
            获取每个任务的目标文本提示
        2. 遍历 self.experiment_config["eval_dataset_path"] 下 "voice_prompt/base_voice_prompt"
            其中每个{VOICE_NAME}.wav文件均为提供VoiceClone音色控制的音频提示
        3. 执行生成，调用model.indextts2_unitalker_inference()
            需要传入spk_audio_prompt为音频提示，传入text为目标文本提示，传入tts_output_path与a2f_output_path为完整的路径，
            tts_output_path与a2f_output_path路径应当为：
                output_dir / "{TASK_NAME}" / "{SAMPLE_NAME}_by_{VOICE_NAME}.wav" 与
                output_dir / "{TASK_NAME}" / "{SAMPLE_NAME}_by_{VOICE_NAME}.npz"
        '''
        eval_root = Path(self.experiment_config["eval_dataset_path"])
        # 获取数据集内容的路径
        text_prompt_root = eval_root / "text_prompt"
        voice_prompt_root = eval_root / "voice_prompt" / "base_voice_prompt"
        assert text_prompt_root.exists(), f"{text_prompt_root} not found"
        assert voice_prompt_root.exists(), f"{voice_prompt_root} not found"

        # 1. 遍历所有任务（{TASK_NAME}_prompt）
        task_dirs = sorted([d for d in text_prompt_root.iterdir() if d.is_dir()])

        # 2. 遍历所有音色提示
        voice_wavs = sorted(voice_prompt_root.glob("*.wav"))

        print(f"Found {len(task_dirs)} tasks")
        print(f"Found {len(voice_wavs)} voice prompts")

        for task_dir in task_dirs:
            # 限制进行的任务：
            if task_dir.name in ["mixed-lingual_in-context_prompt", "emotion_prompt"]:  # "hardcase_prompt", "mixed-lingual_in-context_prompt", "emotion_prompt"
                print(f"\n--------- start task: {task_dir.name} ---------")
                # TASK_NAME
                task_name = task_dir.name.replace("_prompt", "")
                print(f"\n---- Task: {task_name} ----")

                # 输出目录：output_dir / TASK_NAME
                task_output_dir = Path(output_dir) / task_name
                task_output_dir.mkdir(parents=True, exist_ok=True)

                # 该任务下所有文本样本
                txt_files = sorted(task_dir.glob("*.txt"))

                for txt_path in txt_files:
                    sample_name = txt_path.stem

                    # 读取目标文本
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text_prompt = f.read().strip()

                    if len(text_prompt) == 0:
                        print(f"[Skip] Empty text: {txt_path}")
                        continue

                    for voice_wav in voice_wavs:
                        voice_name = voice_wav.stem

                        # 输出文件名
                        tts_output_path = task_output_dir / f"{sample_name}_by_{voice_name}.wav"
                        a2f_output_path = task_output_dir / f"{sample_name}_by_{voice_name}.npz"

                        print(
                            f"Generating | Task={task_name} | Sample={sample_name} | Voice={voice_name}"
                        )

                        # 3. 执行生成
                        self.model.indextts2_unitalker_inference(
                            spk_audio_prompt=str(voice_wav),
                            text=text_prompt,
                            tts_output_path=str(tts_output_path),
                            a2f_output_path=str(a2f_output_path),
                            # ------------------------
                            emo_alpha=0.6,
                            use_emo_text=True,
                            emo_text=text_prompt,  # 情感控制选择从传入的情感文本中推断，不传额外用于推断的情感文本时则直接从目标文本中推断。
                            verbose=False,  # 音频生成过程是否打印
                            render=True,  # 是否渲染表情
                        )


if __name__ == '__main__':
    """
    根目录下运行 python unitaf_inference.py
    """
    # 将测试限制在固定卡上
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"


    experiment_config = {
        "weights": {
            # FIXME:这里权重Key名不应包含()，否则会导致渲染结果保存不下来，不明原因。暂时使用_代替
            # "vertex": {
            #     "audio_feature_projector": "./unitaf_ckpt/UniTAF-A2F(顶点空间loss,40epoch验证收敛)-加载Adapter预训练权重(约束AudioFeature_step_74140)_251231/checkpoint-80000/audio_feature_projector.pt",
            #     "a2f_model": "./unitaf_ckpt/UniTAF-A2F(顶点空间loss,40epoch验证收敛)-加载Adapter预训练权重(约束AudioFeature_step_74140)_251231/checkpoint-80000/a2f_model.pt",
            # },
            "vertex_mouth": {
                "audio_feature_projector": "./unitaf_ckpt/UniTAF-A2F(口型状态顶点空间加权loss)-加载Adapter预训练权重(约束AudioFeature_step_74140)_260104/checkpoint-74140/audio_feature_projector.pt",
                "a2f_model": "./unitaf_ckpt/UniTAF-A2F(口型状态顶点空间加权loss)-加载Adapter预训练权重(约束AudioFeature_step_74140)_260104/checkpoint-74140/a2f_model.pt",
            },
            # "coeff+vertex": {
            #     "audio_feature_projector": "./unitaf_ckpt/UniTAF-A2F(系数空间loss+顶点空间loss)-加载Adapter预训练权重(约束AudioFeature_step_74140)_251231/checkpoint-74140/audio_feature_projector.pt",
            #     "a2f_model": "./unitaf_ckpt/UniTAF-A2F(系数空间loss+顶点空间loss)-加载Adapter预训练权重(约束AudioFeature_step_74140)_251231/checkpoint-74140/a2f_model.pt",
            # },
            "coeff+vertex_mouth": {
                "audio_feature_projector": "./unitaf_ckpt/UniTAF-A2F(系数空间loss+口型状态顶点空间加权loss)-加载Adapter预训练权重(约束AudioFeature_step_74140)_251231/checkpoint-74140/audio_feature_projector.pt",
                "a2f_model": "./unitaf_ckpt/UniTAF-A2F(系数空间loss+口型状态顶点空间加权loss)-加载Adapter预训练权重(约束AudioFeature_step_74140)_251231/checkpoint-74140/a2f_model.pt",
            },
        },
        "eval_dataset_path": "/home/zqg/project/data/TTS-Eval Dataset",
        "output_dir": "outputs/UniTAF_Evaluation",
    }
    experiment = UniTAFExperiment(experiment_config)
    experiment.run_experiment()

