'''
这里实现 文本-音频-表情 联合模型的组装

一般情况下联合模型必定会由TTS模块（tts_model）,A2F模块（a2f_model）和
连接两者的特征投影层（audio_feature_projector）组成.

需要注意的是，在训练过程中，这里tts_model一般仅为TTS中自回归transformer部分！
其余的TTS音频编码器/解码器都放在Dataset类中以推理模式进行数据预处理。

在推理过程中，tts_model一般为原本完整的tts模型

'''
import sys
import os
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgen.native_function_generation import self_to_out_signature


class UniTextAudioFaceModel(nn.Module):
    def __init__(
        self,
        cfg,
        device: torch.device,
        mode: str = 'train',
    ):
        super().__init__()
        '''
        根据 tts_model_name 与 a2f_model_name 中模型名称来判断初始化哪些模型
        '''
        self.cfg = cfg
        self.mode = mode

        # 训练模式下，一些组件在dataset中加载，主干加载需要训练的模块
        if mode == 'train':
            if "IndexTTS2" in cfg["tts_model"]:
                '''
                在训练过程中,self.tts_model为核心的自回归transformer模型，前置的音频encoder大部分放在dataset中进行预处理。
                '''
                # 导入IndexTTS2组件
                from indextts.utils.front import TextNormalizer, TextTokenizer
                # 初始化文本tokenizer
                self.tokenizer = TextTokenizer(str("checkpoints/bpe.model"), TextNormalizer())
                indextts_cfg = OmegaConf.load(Path("checkpoints/config.yaml"))
                # 初始化IndexTTS核心的UniVoice gpt model
                ckpt_path = self.cfg.get("finetune_checkpoint", {}).get("tts_model") or "checkpoints/gpt.pth"
                self.tts_model = self.build_indextts2_model(
                    cfg=indextts_cfg,
                    tokenizer=self.tokenizer,
                    base_checkpoint=ckpt_path,
                    device=device,
                )
                # 初始化IndexTTS的s2mel model
                self.s2mel = self.build_indextts2_s2mel(
                    cfg=indextts_cfg,
                    device=device,
                )  # 这里s2mel以eval模式加载

            if "UniTalker" in cfg["a2f_model"]:
                # 加载UniTalker Decoder并保存权重
                ckpt = self.cfg.get("finetune_checkpoint", {}).get("a2f_model") or "a2f/pretrained_models/UniTalker-B-D0-D7.pt"
                self.a2f_model = self.build_unitalker_decoder_model(
                    cfg=cfg,
                    checkpoint=Path(ckpt),
                    device=device,
                )
                # 如果同时存在UniTalker和IndexTTS2,则初始化中间特征的投影层
                if "IndexTTS2" in cfg["tts_model"]:
                    self.audio_feature_projector = self.build_audio_feature_projector(self.cfg)

                # 如果训练a2f时，只训练adapter部分（a2f.audio_feature_projector）
                if cfg["train_a2f_adapter_only"]:
                    import torchaudio
                    # 用于将为IndexTTS准备的24k采样音频重采样为16k，作为UniTalker Encoder输入
                    self.resample = torchaudio.transforms.Resample(orig_freq=24_000, new_freq=16_000).to(device)  # 放到跟模型同一设备

                    # 此时我们需要约束projector输出特征与unitalker_encoder输出特征相同
                    self.unitalker_encoder = self.build_unitalker_encoder_model(device=device)

        # 推理模式下
        if mode == 'inference':
            if "IndexTTS2" in cfg["tts_model"]:
                '''
                推理模式下self.tts_model对应整个IndexTTS2，其中我们训练的核心自回归transformer权重应当替换加载self.tts_model.gpt
                '''
                self.tts_model = self.build_inference_indextts2_model()  # 在UniTAFIndexTTS2中就已经预设好训练模式了

            if "UniTalker" in cfg["a2f_model"]:
                # 尝试获取配置文件中a2f微调权重,如果没有,则载入预训练的权重
                ckpt = self.cfg.get("finetune_checkpoint", {}).get("a2f_model") or "a2f/pretrained_models/UniTalker-B-D0-D7.pt"
                # 加载UniTalker Decoder并保存权重
                self.a2f_model = self.build_unitalker_decoder_model(
                    cfg=cfg,
                    checkpoint=Path(ckpt),
                    device=device,
                )
                self.a2f_model.eval()

                # ----对UniTalker进行包修复----
                # UniTalker loss计算所需的包 chumpy 最后一次更新停留在 2018 年，内部用了 Python 3 已废弃的 inspect.getargspec，在 Python≥3.11 会报：
                # AttributeError: module 'inspect' has no attribute 'getargspec'
                import inspect
                inspect.getargspec = inspect.getfullargspec  # 这里临时补丁修复，否则会影响UniTalker Loss计算
                # 防止 chumpy 用旧的用法时尝试报错
                import numpy as np
                # 重建被删的别名
                np.bool = bool
                np.int = int
                np.float = float
                np.complex = complex
                np.object = object
                np.str = str
                np.unicode = str
                # -------------------------

                # 加载loss_modeule
                from a2f.loss.loss import UniTalkerLoss
                self.a2f_loss_module = UniTalkerLoss(OmegaConf.load("a2f/config/unitalker.yaml")["LOSS"]).cuda()
                # 如果同时存在UniTalker和IndexTTS2,则初始化中间特征的投影层
                if "IndexTTS2" in cfg["tts_model"]:
                    self.audio_feature_projector = self.build_audio_feature_projector(self.cfg)
                    self.audio_feature_projector.eval().cuda()

    def indextts2_unitalker_inference(
        self,
        # TTS部分（IndexTTS2）所需参数
        spk_audio_prompt,
        text,
        tts_output_path,
        a2f_output_path,
        emo_audio_prompt=None,
        emo_alpha=1.0,
        emo_vector=None,
        use_emo_text=False,
        emo_text=None,
        use_random=False,
        interval_silence=200,
        verbose=False,
        max_text_tokens_per_segment=120,
        stream_return=False,
        more_segment_before=0,
        # a2f部分所需参数
        render=False,
    ):
        '''
        UniTAF联合模型的推理流程，接收所有原始IndexTTS2的参数
        (默认IndexTTS2以24k HZ采样率，UniTalker Decoder设置为25fps)

        Args：
        以下为控制TTS生成的参数
            spk_audio_prompt： reference audio 语音克隆参考音频
            text： 目标文本，生成根据这个文本对应的语音
            tts_output_path： 生成的音频路径 .wav 后缀
            a2f_output_path： 生成的表情路径 .npz 后缀
            emo_audio_prompt： 传入用于控制情感的提示音频（如果没有则一般从spk_audio_prompt中获得）
            emo_alpha： 情感控制作用系数，使用文本情感模式时使用大约 0.6（或更低）的 emo_alpha
            emo_vector： 直接指定具体的情感向量，传入时以该emo_vector为准
            use_emo_text： 是否从文本中推断情感，如果是优先从emo_text中推断情感，否则从目标文本中推断情感。
            emo_text： 当use_emo_text时，如果传入emo_text，根据从emo_text中推断出的情感来控制生成
            use_random： 随机情感控制
            interval_silence： 静音间隔，一般保持默认即可，在流式生成中，多个片段拼接时会插入静音间隔来保证声音自然
            verbose： 是否打印中间结果
            max_text_tokens_per_segment： 流式生成对输入进行切分时，每个分段的最大切分token数
            stream_return： 是否流式返回
            more_segment_before：默认为0 暂时不知道是什么意思

        以下是控制A2F部分的参数
            rander：是否进行渲染
            **generation_kwargs： IndexTTS2.infer_generator中接收的其他参数，暂时不详
        '''
        import torchaudio

        # 必须是IndexTTS2与UniTalker Decoder的联合模型的推理模式
        assert "IndexTTS2" in self.cfg["tts_model"] and "UniTalker" in self.cfg["a2f_model"]
        assert self.mode == "inference"

        # 1. tts生成
        tts_gen = self.tts_model.infer(spk_audio_prompt, text,
              emo_audio_prompt, emo_alpha,
              emo_vector,
              use_emo_text, emo_text, use_random, interval_silence,
              verbose, max_text_tokens_per_segment, stream_return, more_segment_before)

        print("获得结果")
        # 取最后一次 yield 的值
        if stream_return:
            # 流式模式：tts_result 是生成器
            for sr, wav, audio_feature in tts_gen:
                pass  # 循环会获取最后一个值
        else:
            # 非流式模式：tts_result 已经是 (sr, wav_data, audio_feature) 元组
            sr, wav, audio_feature = tts_gen

        # 保存音频
        if tts_output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(tts_output_path):
                os.remove(tts_output_path)
                print(">> remove old wav file:", tts_output_path)
            if os.path.dirname(tts_output_path) != "":
                os.makedirs(os.path.dirname(tts_output_path), exist_ok=True)

            torchaudio.save(tts_output_path, wav.type(torch.int16), sr)  # 保存为16位PCM
            print(">> wav file saved to:", tts_output_path)


        # 2. projector
        audio_feature = F.pad(audio_feature, (0, 0, 0, 1), mode='constant', value=0)  # (B, L+1, 1024)
        audio_feature = self.audio_feature_projector(audio_feature)

        # 3. a2f生成
        '''
        face_template: 用于表示说话人静态的脸形，因此是一个与标注格式的表示形状FLAME相同的初始偏移量
            但是在arkit中我们不需要这个，故而保持和我们标注格式相同维度的0向量即可
        identy: 在训练时，网络为每一个训练对象（identity）学了一个可学习的风格嵌入。decoder.learnable_style_emb.weight[i]，i 就是 style_idx。
            推理时，模型拿到这个索引后，把对应的那一列风格向量抽出来，再去调制音频-动作映射，于是同一段音频在不同身份下会生成不同的口型/表情风格。
            我们这里传最后一个身份索引 args.identity_num-1 来表示"通用/平均"的风格。
        '''
        # TODO: annot_type由外部传入，而不在此处写死，同时template随之变化
        B, T, _ = audio_feature.shape
        face_template = torch.zeros(B, 61, device=audio_feature.device)  # 与"qxsk_inhouse_blendshape_weight"标注格式相同
        identity = torch.full((B,),self.a2f_model.model_cfg["identity_num"] - 1,
                              dtype=torch.long, device=audio_feature.device)

        out_motion, _, _ = self.a2f_model(
            audio_feature=audio_feature,
            template=face_template,
            face_motion=None,
            style_idx=identity,
            annot_type="qxsk_inhouse_blendshape_weight",
            fps=25
        )

        # 处理 out_motion，确保它是 Tensor 并且在正确的设备上
        print(f"out_motion 类型: {type(out_motion)}")
        print(f"out_motion 设备: {getattr(out_motion, 'device', 'N/A')}")

        # 获取到顶点数据
        out = self.a2f_loss_module.get_vertices(out_motion.cuda(), annot_type="qxsk_inhouse_blendshape_weight")

        # 处理out以便保存
        if isinstance(out, torch.Tensor):
            if out.is_cuda:  # 如果 out 是 CUDA tensor，先移到 CPU
                out_cpu = out.cpu()
            else:
                out_cpu = out
            out_np = out_cpu.detach().numpy()  # 分离计算图并转换为 numpy
        else:
            out_np = out  # 如果已经是 numpy 或其他类型

        if a2f_output_path:
            if os.path.isfile(a2f_output_path):
                os.remove(a2f_output_path)
                print(">> remove old wav file:", a2f_output_path)
            if os.path.dirname(a2f_output_path) != "":
                os.makedirs(os.path.dirname(a2f_output_path), exist_ok=True)
            # 保存结果为npz文件
            np.savez(a2f_output_path, out_np)
            print(f">> face results save to {a2f_output_path}")

        if tts_output_path and a2f_output_path and render:
            from unitaf_train_component.render import render_npz_video
            parent_dir = Path(a2f_output_path).parent
            parent_str = parent_dir.as_posix()
            # 进行渲染
            render_npz_video(
                out_np=out_np,
                audio_path=tts_output_path,  # 原始 wav 完整路径；None=无声
                out_dir=parent_str,  # 想把视频/图片保存文件夹
                annot_type="qxsk_inhouse_blendshape_weight",  # 你当时传给 get_vertices 的同一字符串
                save_images=False,  # False=直接出 mp4；True=出逐帧 png
                device="cuda"  # 或 "cpu"
            )


    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """
        保存：只返回需要训练的参数
        """
        sd = {}
        if hasattr(self, 'tts_model') and self.training:
            sd.update(self.tts_model.state_dict(prefix='tts_model.'))
        if hasattr(self, 'audio_feature_projector') and self.training:
            sd.update({f'audio_feature_projector.{k}': v for k, v in self.audio_feature_projector.state_dict().items()})
        if hasattr(self, 'a2f_model') and self.training:
            # 跳过 loss_module（不保存）
            for name, param in self.a2f_model.named_parameters():
                if 'loss_module' not in name:
                    sd[f'a2f_model.{name}'] = param
            for name, buffer in self.a2f_model.named_buffers():
                if 'loss_module' not in name:
                    sd[f'a2f_model.{name}'] = buffer
        return sd

    # 工具方法 --------------------------------------------------------------------------------------
    def build_indextts2_model(self, cfg, tokenizer, base_checkpoint, device):
        '''
        训练模式专用
        这里实例化IndexTTS2中核心的GPT模型并加载权重
        '''
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

    def build_inference_indextts2_model(self):
        '''
        如果有自己微调gpt权重，则替换配置文件中的权重路径
        '''
        from unitaf_train_component.indextts2_inference_component import UniTAFIndexTTS2
        indextts2_cfg = OmegaConf.load("checkpoints/config.yaml")  # 加载IndexTTS2配置
        gpt_ckpt = self.cfg.get("finetune_checkpoint", {}).get("tts_model")
        # print("[UniTextAudioFaceModel.__init__()] gpt_ckpt:", gpt_ckpt)

        tts_model = UniTAFIndexTTS2(
            cfg=indextts2_cfg,
            gpt_ckpt=gpt_ckpt,
            model_dir="checkpoints",
            use_cuda_kernel=False,  # 禁用CUDA内核
            use_torch_compile=True  # 启用torch.compile优化
        )
        return tts_model

    def build_indextts2_s2mel(self, cfg, device):
        """
        这里初始化用于将IndexTTS核心gpt生成的semantic token转化为mel的模块，这里不需要训练该模块直接.eval
        """
        from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
        s2mel_path = os.path.join("checkpoints", cfg.s2mel_checkpoint)
        s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        s2mel = s2mel.to(device)
        return s2mel.eval()

    def build_unitalker_encoder_model(self, device):
        '''
        这里实例化UniTalker Decoder模型并加载权重
        '''
        from a2f.models.wavlm import WavLMModel
        model = WavLMModel.from_pretrained('microsoft/wavlm-base-plus')
        model.feature_extractor._freeze_parameters()  # 冻结UniTalker音频特征提取器
        return model.to(device)

    def build_unitalker_decoder_model(self, cfg, checkpoint, device):
        '''
        这里实例化UniTalker Decoder模型并加载权重
        - cfg 需要经过OmegaConf加载的
        '''
        from unitaf_train_component.unitalker_decoder_component import UniTalkerDecoder

        checkpoint = torch.load(checkpoint, map_location='cpu')
        # 向配置中追加 identity_num 参数
        cfg["UniTalker"]["identity_num"] = len(
            checkpoint['decoder.learnable_style_emb.weight'])  # 根据权重内容更新identity_num
        if not cfg["UniTalker"]["audio_encoder_feature_dim"]:
            # 向配置中追加 audio_encoder_feature_dim , 实际中这里最好在外部提前判断
            cfg["UniTalker"]["audio_encoder_feature_dim"] = 768  # 设置默认为768 与官方原始权重UniTalker-L-D0-D7.pt保持一致。注：UniTalker-B-D0-D7.pt 则为768

        # print("decoder.audio_feature_map.weight shape:",
        #       checkpoint['decoder.audio_feature_map.weight'].shape)   # torch.Size([256, 768])
        # print('[build_unitalker_decoder] cfg["UniTalker"]["audio_encoder_feature_dim"] =',
        #       cfg["UniTalker"]["audio_encoder_feature_dim"])  # 768

        # 实例化UniTalkerDecoder
        model = UniTalkerDecoder(cfg)
        model.load_state_dict(checkpoint, strict=False)

        return model.to(device)

    def build_audio_feature_projector(self, cfg):
        """
        in_dim与tts输出音频特征维度相同
        out_dim与a2f接收音频特征维度相同
        这里实例化用于将TTS的audio feature在时间长度和特征维度上与A2F Decoder对齐的投影层
        """
        from unitaf_train_component.audio_feature_projector import AudioFeatureProjector

        if "IndexTTS2" in cfg["tts_model"]:
            in_dim = 1024  # 与我们取IndexTTS2中 self.s2mel.models["gpt_layer"](mel_latent) 相同

        if "UniTalker" in cfg["a2f_model"]:
            out_dim = cfg["UniTalker"]["audio_encoder_feature_dim"]  # 与UniTalker Decoder接收维度相同， 由外部配置文件决定

        proj = AudioFeatureProjector(in_dim, out_dim)

        ckpt_path = cfg.get("finetune_checkpoint", {}).get("audio_feature_projector")
        if ckpt_path: # 如果存在权重
            state_dict = torch.load(ckpt_path, map_location='cpu')
            proj.load_state_dict(state_dict)
        else:
            # 对所有 Conv1d 做 Kaiming
            for m in proj.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    # bias 已经是 False，无需处理
        return proj



# ---------- 一次性检查工具方法 ----------
def dump_param_count(model, name: str):
    """
    通用版：model 可以是 nn.Module，也可以是任意 Python 对象。
    遇到 nn.Module 就累计参数，其他属性递归扫描。
    """
    def _count_any(obj, memo):
        total = 0
        if isinstance(obj, nn.Module):
            for p in obj.parameters():
                if id(p) not in memo:
                    memo.add(id(p))
                    total += p.numel()
            return total
        # 非 Module -> 递归扫属性
        for attr_name in dir(obj):
            if attr_name.startswith('_'):
                continue
            attr = getattr(obj, attr_name, None)
            if isinstance(attr, nn.Module) and id(attr) not in memo:
                memo.add(id(attr))
                total += _count_any(attr, memo)
        return total
    total = _count_any(model, set())
    print(f"{name:25s}  init params : {total:,}")

def dump_state_dict_count(sd: dict, name: str):
    total = sum(v.numel() for v in sd.values())
    print(f"{name:25s}  state_dict  : {total:,}")

def print_model_arch(obj, name="model", indent=0):
    """
    通用模型结构打印：
      - nn.Module 直接 print
      - 非 Module 则递归扫属性，遇到 Module 就打印
    """
    prefix = " " * indent

    if isinstance(obj, nn.Module):
        # 是 Module，直接打印
        print(prefix + f"{name} ({obj.__class__.__name__}):")
        print(obj)
        return

    # 非 Module，递归扫属性
    print(prefix + f"{name} ({obj.__class__.__name__}) -> scanning attributes:")
    for attr_name in dir(obj):
        if attr_name.startswith('_'):
            continue
        attr = getattr(obj, attr_name, None)
        # if isinstance(attr, nn.Module):
        print_model_arch(attr, attr_name, indent + 2)


if __name__ == '__main__':
    """
    测试联合模型的流程 python unitaf_train/UniTAF.py
    加载模型权重后可以打印查看结构等，请逐步解注释后续打印的方法。
    """
    # 将测试限制在固定卡上
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # unitaf_train的父目录
    sys.path.insert(0, project_root)

    from omegaconf import OmegaConf

    train_config = {
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
            "dataset_list": ["D12"]  # 这里测试也传数据集是用于指导模型选择何种输出头
        },
        # 加载指定的部分模块的微调权重。
        "finetune_checkpoint": {
            # "tts_model": "./unitaf_ckpt/UniTAF-A2F(lr_1e-4)- LoRA-TTS(lr_5e-7_rank_128)/checkpoint-20000/tts_model.pt",
            "audio_feature_projector": "./unitaf_ckpt/UniTAF-A2F(lr_1e-4)-加载Adapter预训练权重(Adatpre特征损失约束_step_74140)/checkpoint-74140/audio_feature_projector.pt",
            "a2f_model": "./unitaf_ckpt/UniTAF-A2F(lr_1e-4)-加载Adapter预训练权重(Adatpre特征损失约束_step_74140)/checkpoint-74140/a2f_model.pt",
        }
    }

    cfg = OmegaConf.create(train_config)

    # unitaf = UniTextAudioFaceModel(cfg, device=torch.device("cuda:0"), mode="train")  # 训练模式
    unitaf = UniTextAudioFaceModel(cfg, device=torch.device("cuda:0"), mode="inference")  # 推理模式

    # # 1. 打印模型结构 FIXME: 推理状态下 tts_model不是nn.Model类型这里会打不出来
    # print("=" * 80)
    # print("Model architecture:")
    # print("=" * 80)
    # print_model_arch(unitaf, "unitaf")

    # # 2. 打印模型的所有层及其参数  FIXME: 推理状态下 tts_model不是nn.Model类型这里会打不出来
    # print("\n" + "=" * 80)
    # print("Model layers and parameters:")
    # print("=" * 80)
    # for name, module in unitaf.named_children():
    #     print(f"\n{name}: {type(module).__name__}")
    #
    #     # 如果是ModuleDict或ModuleList，打印其内容
    #     if isinstance(module, (nn.ModuleDict, nn.ModuleList)):
    #         for sub_name, sub_module in module.named_children():
    #             print(f"  {sub_name}: {type(sub_module).__name__}")
    #             if hasattr(sub_module, 'parameters') and list(sub_module.parameters()):
    #                 print(f"    Parameters: {sum(p.numel() for p in sub_module.parameters())}")

    # # 3. 打印模型初始化参数量与保存的参数量  FIXME: 推理状态下 tts_model不是nn.Model类型这里会打不出来
    # print("=" * 60)
    # print(">>> 初始化后参数量")
    # dump_param_count(unitaf.tts_model, "tts_model")
    # dump_param_count(unitaf.a2f_model, "a2f_model")
    # dump_param_count(unitaf.audio_feature_projector, "audio_feature_projector")
    # print("-" * 60)
    # if unitaf.mode == "train":
    #     print(">>> 自定义 state_dict 过滤后")
    #     sd = unitaf.state_dict()
    #     dump_state_dict_count({k: v for k, v in sd.items() if k.startswith('tts_model.')}, "tts_model")
    #     dump_state_dict_count({k: v for k, v in sd.items() if k.startswith('a2f_model.')}, "a2f_model")
    #     dump_state_dict_count({k: v for k, v in sd.items() if k.startswith('audio_feature_projector.')},
    #                           "audio_feature_projector")
    #     print("=" * 60)

    # FIXME: 这里打印出来tts_model的为什么过滤后比初始化时多？
    '''
    >>> 初始化后参数量
    tts_model                  init params : 865,980,639
    a2f_model                  init params : 1,225,213
    audio_feature_projector    init params : 2,098,176
    ------------------------------------------------------------
    >>> 自定义 state_dict 过滤后
    tts_model                  state_dict  : 871,100,639
    a2f_model                  state_dict  : 1,225,213
    audio_feature_projector    state_dict  : 2,098,176
    '''

    # # 4. 打印模型参数详情
    # print("\n" + "=" * 80)
    # print("Detailed parameters:")
    # print("=" * 80)
    # for name, param in unitaf.named_parameters():
    #     print(f"{name}: shape={param.shape}, dtype={param.dtype}, requires_grad={param.requires_grad}")


    # 5. 尝试推理
    if unitaf.mode == "inference":
        text = "清晨的阳光透过窗帘洒在书桌上，新的一天开始了。窗外鸟儿欢快地歌唱，空气中弥漫着淡淡的花香。"
        unitaf.indextts2_unitalker_inference(
            spk_audio_prompt='examples/voice_zhongli.wav',
            text=text,
            tts_output_path="outputs/UniTAF_output.wav",
            a2f_output_path="outputs/UniTAF_output.npz",
            emo_alpha=0.6,
            use_emo_text=True,
            emo_text=text,  # 情感控制选择从传入的情感文本中推断，不传额外用于推断的情感文本时则直接从目标文本中推断。
            verbose=False,   # 音频生成过程是否打印
            render=True,  #是否渲染表情
        )










