'''
25-12-16
这里实现UniTAF中推理专用的IndexTTS2模型，主要继承自官方IndexTTS2
但是允许加载自己的gpt模型并在推理过程中拿到a2f所需的audio feature
'''
import os
from subprocess import CalledProcessError

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import json
import re
import time
import librosa
import torch
import torchaudio
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer

from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram

from transformers import AutoTokenizer
from modelscope import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import safetensors
from transformers import SeamlessM4TFeatureExtractor
import random
import torch.nn.functional as F

from indextts.infer_v2 import IndexTTS2, QwenEmotion

# 导入工具方法的包
import hashlib, itertools
from collections import OrderedDict


class UniTAFIndexTTS2(IndexTTS2):
    def __init__(
            self, cfg, gpt_ckpt=None, model_dir="checkpoints", use_fp16=False, device=None,
            use_cuda_kernel=None,use_deepspeed=False, use_accel=False, use_torch_compile=False
    ):
        '''
        该初始化与官方IndexTTS2初始化唯一不同在于cfg在外部加载好，而不是传入路径在初始化时加载。
        这样方便直接修改配置文件中的权重路径

        除此之外的其他初始化应当与原始IndexTTS2保持一致！
        '''
        # # 先初始化nn.Module
        # nn.Module.__init__(self)  # 这里给UniTAFIndexTTS2加上nn.Module仅仅为了方便打印模型权重
        # 设备自动检测和设置
        if device is not None:
            self.device = device
            self.use_fp16 = False if device == "cpu" else use_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = False
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.use_fp16 = False  # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.use_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        # 加载配置文件
        self.cfg = cfg
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.use_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token  # 停止生成梅尔谱图的特殊token
        self.use_accel = use_accel
        self.use_torch_compile = use_torch_compile

        # 初始化情感分析模型（Qwen）
        self.qwen_emo = QwenEmotion(os.path.join(self.model_dir, self.cfg.qwen_emo_path))

        # 初始化GPT文本到语义模型
        self.gpt = UnifiedVoice(**self.cfg.gpt, use_accel=self.use_accel)
        if gpt_ckpt is not None:  # 这里增加支持外部直接指定gpt权重
            # 由于load_checkpoint配置文件默认加载路径不同，因此这里直接自己实现而不是调用官方load_checkpoint()
            self.gpt_path = gpt_ckpt
            checkpoint = torch.load(self.gpt_path, map_location='cpu')
            checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
            self.gpt.load_state_dict(checkpoint, strict=True)
        else:
            self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
            load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.use_fp16:
            self.gpt.eval().half()  # 设置为评估模式并使用半精度
        else:
            self.gpt.eval()
        print("[UniTAFIndexTTS2] GPT weights restored from:", self.gpt_path)

        # DeepSpeed支持性检查
        if use_deepspeed:
            try:
                import deepspeed
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(f">> Failed to load DeepSpeed. Falling back to normal inference. Error: {e}")

        # GPT后初始化配置
        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_fp16)

        # CUDA内核预加载（用于BigVGAN）
        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.s2mel.modules.bigvgan.alias_free_activation.cuda import activation1d

                print(">> Preload custom CUDA kernel for BigVGAN", activation1d.anti_alias_activation_cuda)
            except Exception as e:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                print(f"{e!r}")
                self.use_cuda_kernel = False

        # 初始化特征提取器
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )

        # 构建语义模型
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat))
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)

        # 构建语义编解码器
        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        self.semantic_codec.eval()
        print('>> semantic_codec weights restored from: {}'.format(semantic_code_ckpt))

        # 初始化语义到梅尔谱图模型（s2mel）
        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device)
        # 设置缓存以提高推理效率
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # 如果请求，启用torch.compile优化 / Enable torch.compile optimization if requested
        if self.use_torch_compile:
            print(">> Enabling torch.compile optimization")
            self.s2mel.enable_torch_compile()
            print(">> torch.compile optimization enabled successfully")

        self.s2mel.eval()
        print(">> s2mel weights restored from:", s2mel_path)

        # 加载说话人识别模型（CAMPPlus） load campplus_model
        campplus_ckpt_path = hf_hub_download(
            "funasr/campplus", filename="campplus_cn_common.bin"
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(self.device)
        self.campplus_model.eval()
        print(">> campplus_model weights restored from:", campplus_ckpt_path)

        # 初始化声码器（BigVGAN）
        bigvgan_name = self.cfg.vocoder.name
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()  # 移除权重归一化以提高推理速度
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", bigvgan_name)

        # 文本处理相关初始化
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)

        # 加载情感和说话人矩阵
        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
        self.emo_matrix = emo_matrix.to(self.device)
        self.emo_num = list(self.cfg.emo_num)  # 每种情感的数量

        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
        self.spk_matrix = spk_matrix.to(self.device)

        # 按情感类型分割矩阵
        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        # 梅尔谱图计算函数配置
        mel_fn_args = {
            "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        # 缓存参考音频相关变量（用于语音克隆和风格转换）
        self.cache_spk_cond = None  # 缓存的说话人条件
        self.cache_s2mel_style = None  # 缓存的s2mel风格
        self.cache_s2mel_prompt = None  # 缓存的s2mel提示
        self.cache_spk_audio_prompt = None  # 缓存的说话人音频提示
        self.cache_emo_cond = None  # 缓存的情感条件
        self.cache_emo_audio_prompt = None  # 缓存的情感音频提示
        self.cache_mel = None  # 缓存的梅尔谱图

        # 进度引用显示（可选，用于GUI应用）
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

    def infer(self,
              spk_audio_prompt,
              text,
              emo_audio_prompt=None,
              emo_cond_emb=None,  # 如果直接传入从音频中提取的情感嵌入，则替代从emo_audio_prompt音频中重新推断的操作。
              emo_alpha=1.0,
              emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_segment=120, stream_return=False, more_segment_before=0,
              **generation_kwargs):
        """
        覆盖原始IndexTTS2.infer() 移除output_path，保存步骤将不再该内部进行
        同时传出需要为后续 口型生成（a2f） 和 表情生成（expression model）的audio feature特征与情感条件emovec

        主要推理接口参数:
            spk_audio_prompt: 说话人参考音频路径
            text: 要合成的文本
            emo_audio_prompt: 情感参考音频路径
            emo_cond_emb: 从情感参考音频中推断出的情感嵌入，效果与emo_audio_prompt一致，省去从音频进行推断出emo_cond_emb的步骤
            emo_alpha: 情感混合强度
            emo_vector: 手动指定的情感向量
            use_emo_text: 是否从文本中提取情感
            emo_text: 情感分析文本
            use_random: 是否随机选择情感
            interval_silence: 片段间静音时长(ms)
            verbose: 是否输出详细信息
            max_text_tokens_per_segment: 每段最大文本token数
            stream_return: 是否流式返回结果
        """
        if stream_return:
            # 流式推理模式
            return self.infer_generator(
                spk_audio_prompt, text,
                emo_audio_prompt, emo_cond_emb, emo_alpha,
                emo_vector,
                use_emo_text, emo_text, use_random, interval_silence,
                verbose, max_text_tokens_per_segment, stream_return, more_segment_before, **generation_kwargs
            )
        else:
            # 普通推理模式，返回最终结果
            try:
                return list(self.infer_generator(
                    spk_audio_prompt, text,
                    emo_audio_prompt, emo_cond_emb, emo_alpha,
                    emo_vector,
                    use_emo_text, emo_text, use_random, interval_silence,
                    verbose, max_text_tokens_per_segment, stream_return, more_segment_before, **generation_kwargs
                ))[0]
            except IndexError:
                return

    def infer_generator(self, spk_audio_prompt, text,
            emo_audio_prompt=None, emo_cond_emb=None, emo_alpha=1.0,
            emo_vector=None,
            use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
            verbose=False, max_text_tokens_per_segment=120, stream_return=False, quick_streaming_tokens=0,
            **generation_kwargs):
        """
        覆盖父类IndexTTS2的方法:
        1. 增加将中间audio feature返回出来以供后续a2f模型使用。
            我们选择将 self.s2mel.models['gpt_layer'](latent) 输出的特征作为可被后续a2f模型接收的audio_feature
        2. 移除output_path，音频的保存功能在外部实现，而不再infer_generator中实现了。

        3. 增加情感控制条件下：直接传入从emo_audio_prompt提取好的emo_cond_emb。
            当接收到emo_cond_emb时，执行与emo_audio_prompt相同的从音频推断情感，但是直接应用emo_cond_emb结果与spk_cond_emb合并。

        原方法：生成器形式的推理函数，支持流式输出
        """
        print(">> starting inference...")
        self._set_gr_progress(0, "starting inference...")
        if verbose:
            print(f"origin text:{text}, spk_audio_prompt:{spk_audio_prompt}, "
                  f"emo_audio_prompt:{emo_audio_prompt}, emo_alpha:{emo_alpha}, "
                  f"emo_vector:{emo_vector}, use_emo_text:{use_emo_text}, "
                  f"emo_text:{emo_text}")
        start_time = time.perf_counter()

        # 情感处理逻辑 ------------------------------------------------
        if use_emo_text or emo_vector is not None:
            # / we're using a text or emotion vector guidance; so we must remove
            # / "emotion reference voice", to ensure we use correct emotion mixing!
            # 使用文本或情感向量引导时，移除情感参考音频以确保正确的情感混合
            emo_audio_prompt = None
            emo_cond_emb = None  # 移除情感参考音频的条件嵌入

        if use_emo_text:
            # / automatically generate emotion vectors from text prompt
            # 从文本提示自动生成情感向量
            if emo_text is None:
                emo_text = text  # 使用主文本作为情感分析文本 / use main text prompt
            emo_dict = self.qwen_emo.inference(emo_text)  # 调用情感分析模型
            if verbose:
                print(f"从文本检测到的情感向量: {emo_dict}")
            # / convert ordered dict to list of vectors; the order is VERY important!
            # 将有序字典转换为向量列表（顺序非常重要！）
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            # / we have emotion vectors; they can't be blended via alpha mixing
            # / in the main inference process later, so we must pre-calculate
            # / their new strengths here based on the alpha instead!
            # 处理情感向量缩放
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))  # 限制在0-1范围内
            if emo_vector_scale != 1.0:
                # / scale each vector and truncate to 4 decimals (for nicer printing)
                # 缩放每个向量并截断到4位小数（为了更美观的打印）
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]
                if verbose:
                    print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")

        if emo_audio_prompt is None:
            # / we are not using any external "emotion reference voice"; use
            # / speaker's voice as the main emotion reference audio.
            # 没有外部情感参考音频时，使用说话人音频作为主要情感参考
            emo_audio_prompt = spk_audio_prompt
            # / must always use alpha=1.0 when we don't have an external reference voice
            # 没有外部参考音频时必须使用alpha=1.0
            emo_alpha = 1.0

        # 参考音频缓存 ------------------------------------------------
        # 如果参考音频改变了，才需要重新生成, 提升速度
        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
            if self.cache_spk_cond is not None:
                # 清空缓存
                self.cache_spk_cond = None
                self.cache_s2mel_style = None
                self.cache_s2mel_prompt = None
                self.cache_mel = None
                torch.cuda.empty_cache()  # 清空GPU缓存

            # 加载和处理说话人音频
            audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)  # 重采样到22.05kHz
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)  # 重采样到16kHz

            # 提取音频特征
            inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"]
            attention_mask = inputs["attention_mask"]
            input_features = input_features.to(self.device)
            attention_mask = attention_mask.to(self.device)
            spk_cond_emb = self.get_emb(input_features, attention_mask)  # 获取说话人嵌入

            # 语义编码量化
            _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
            # 计算梅尔谱图
            ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)

            # 提取说话人风格特征
            feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device),
                                                     num_mel_bins=80,
                                                     dither=0,
                                                     sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
            style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]

            # 生成提示条件
            prompt_condition = self.s2mel.models['length_regulator'](S_ref,
                                                                     ylens=ref_target_lengths,
                                                                     n_quantizers=3,
                                                                     f0=None)[0]

            # 更新缓存
            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
        else:
            # 使用缓存的条件
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel

        # 情感向量处理 ------------------------------------------------
        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)
            if use_random:
                # 随机选择情感索引
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                # 基于余弦相似度选择最相似的情感
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

            # 构建情感矩阵
            emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
            emo_matrix = torch.cat(emo_matrix, 0)  # torch.Size([8, 1280])
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix  # 加权情感矩阵 torch.Size([8, 1280])
            emovec_mat = torch.sum(emovec_mat, 0)  # 求和得到最终情感向量 torch.Size([1280])
            emovec_mat = emovec_mat.unsqueeze(0)  # 添加批次维度 torch.Size([1, 1280])

        # 情感条件缓存处理 ------------------------------------------------
        # 显式传入的 emo_cond_emb > cache_emo_cond > emo_audio_prompt 重新计算

        # 如果直接传入 emo_cond_emb，直接用（最高优先级）
        if emo_cond_emb is not None:
            emo_cond_emb = emo_cond_emb.to(self.device)  # 确保 device 一致
            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = None

        # 没有显式 emo_cond_emb，才走 cache / audio
        elif self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
            if self.cache_emo_cond is not None:
                self.cache_emo_cond = None
                torch.cuda.empty_cache()

            # 加载和处理情感音频
            emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
            # print(f"[DEBUG] emo_audio shape:{emo_audio.shape}")  # torch.Size([1, 46289])
            emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
            emo_input_features = emo_inputs["input_features"]
            emo_attention_mask = emo_inputs["attention_mask"]
            emo_input_features = emo_input_features.to(self.device)
            emo_attention_mask = emo_attention_mask.to(self.device)
            # print(f"[DEBUG] emo_attention_mask: {emo_attention_mask.shape}")  # torch.Size([1, 144])
            # print(f"[DEBUG] emo_input_features: {emo_input_features.shape}")  # torch.Size([1, 144, 160])
            emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)
            # print(F"[DEBUG] emo_cond_emb {emo_cond_emb.shape}")  # torch.Size([1, 144, 1024])

            # 更新情感条件缓存
            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = emo_audio_prompt

        # cache 命中
        else:
            emo_cond_emb = self.cache_emo_cond

        # 文本处理 ------------------------------------------------
        self._set_gr_progress(0.1, "text processing...")
        text_tokens_list = self.tokenizer.tokenize(text)  # 文本分词
        segments = self.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment,
                                                 quick_streaming_tokens=quick_streaming_tokens)
        segments_count = len(segments)

        # 检查未知token
        text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        if self.tokenizer.unk_token_id in text_token_ids:
            print(
                f"  >> Warning: input text contains {text_token_ids.count(self.tokenizer.unk_token_id)} unknown tokens (id={self.tokenizer.unk_token_id}):")
            print("     Tokens which can't be encoded: ",
                  [t for t, id in zip(text_tokens_list, text_token_ids) if id == self.tokenizer.unk_token_id])
            print(f"     Consider updating the BPE model or modifying the text to avoid unknown tokens.")

        if verbose:
            print("text_tokens_list:", text_tokens_list)
            print("segments count:", segments_count)
            print("max_text_tokens_per_segment:", max_text_tokens_per_segment)
            print(*segments, sep="\n")

        # 生成参数设置 ------------------------------------------------
        do_sample = generation_kwargs.pop("do_sample", True)                # True
        top_p = generation_kwargs.pop("top_p", 0.8)                         # 0.8
        top_k = generation_kwargs.pop("top_k", 30)                          # 30
        temperature = generation_kwargs.pop("temperature", 0.8)             # 0.8
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)                   # 3 贪心解码
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        sampling_rate = 22050

        # 初始化变量
        wavs = []
        audio_features = []  # a2f 口型生成的音频特征输入 与 表情补全Expression model的音频特征输入
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        silence = None  # 用于流式返回的静音段 / for stream_return

        # 分段合成 ------------------------------------------------
        for seg_idx, sent in enumerate(segments):
            self._set_gr_progress(0.2 + 0.7 * seg_idx / segments_count,
                                  f"speech synthesis {seg_idx + 1}/{segments_count}...")

            # 文本token转换
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                # 调试分词器 / debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                print("text_token_syms is same as segment tokens", text_token_syms == sent)

            # GPT推理阶段 ------------------------------------------------
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):

                    # 合并情感向量
                    emovec = self.gpt.merge_emovec(
                        spk_cond_emb,
                        emo_cond_emb,
                        torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        alpha=emo_alpha
                    )
                    # print(f"[DEBUG] emovec.shape: {emovec.shape}")  # torch.Size([1, 1280])

                    # 如果提供了情感向量，进行混合
                    if emo_vector is not None:
                        emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec  # torch.Size([1, 1280])
                        # emovec = emovec_mat
                        # print(f"[DEBUG] 混合后emovec: {emovec.shape}")  # torch.Size([1, 1280])

                    # 获取此处emovec作为expression model的显示情感控制输入
                    emovec_contrl = emovec  # torch.Size([1, 1280])

                    # 语音推理生成语义编码
                    # codes 为生成的目标音频的codes，但是要拿去s2mel前还得过一边gpt前向去得到latent；speech_conditioning_latent 为条件
                    codes, speech_conditioning_latent = self.gpt.inference_speech(
                        spk_cond_emb,
                        text_tokens,
                        emo_cond_emb,
                        cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=autoregressive_batch_size,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                        **generation_kwargs
                    )

                    # print(f"GPT生成codes, codes:{codes[:30]}, speech_conditioning_latent:{speech_conditioning_latent[:30]}")

                gpt_gen_time += time.perf_counter() - m_start_time

                # 检查是否因超过最大长度而停止生成
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True

                # 计算实际编码长度
                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                #                 if verbose:
                #                     print(codes, type(codes))
                #                     print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                #                     print(f"code len: {code_lens}")

                # 长度裁剪 ------------------------------------------------
                code_lens = []
                max_code_len = 0
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0]
                        code_len = len_[0].item() if len_.numel() > 0 else len(code)
                    code_lens.append(code_len)
                    max_code_len = max(max_code_len, code_len)
                codes = codes[:, :max_code_len]
                code_lens = torch.LongTensor(code_lens)
                code_lens = code_lens.to(self.device)
                if verbose:
                    print(codes, type(codes))
                    print(
                        f"修正后的编码形状: {codes.shape}, 编码类型: {codes.dtype}")  # torch.Size([1, 304]), torch.int64
                    print(f"编码长度: {code_lens}")  # tensor([304], device='cuda:0')

                # GPT前向传播 ------------------------------------------------
                m_start_time = time.perf_counter()
                use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = self.gpt(
                        speech_conditioning_latent,
                        text_tokens,
                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                        codes,
                        torch.tensor([codes.shape[-1]], device=text_tokens.device),
                        emo_cond_emb,
                        cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        use_speed=use_speed,
                    )
                    gpt_forward_time += time.perf_counter() - m_start_time
                    # print(f"gpt前向生成latent：{latent}")

                # s2mel推理阶段 ------------------------------------------------
                dtype = None
                with torch.amp.autocast(text_tokens.device.type, enabled=dtype is not None, dtype=dtype):
                    m_start_time = time.perf_counter()
                    diffusion_steps = 25
                    inference_cfg_rate = 0.7
                    # print(f"--UnifiedVoice.forward()输出： {latent.shape}")  # torch.Size([1, 304, 1280])
                    latent = self.s2mel.models['gpt_layer'](latent)  # GPT层处理
                    # print(f"--s2mel.models['gpt_layer']输出： {latent}")
                    audio_feature = latent
                    # 我们将这里的特征作为后续a2f的输入，todo：暂时也将该特征作为expression model输入
                    audio_features.append(audio_feature)

                    # print(f"--s2mel.models['gpt_layer']输出： {latent.shape}")  # torch.Size([1, 304, 1024])
                    S_infer = self.semantic_codec.quantizer.vq2emb(
                        codes.unsqueeze(1))  # 编码转嵌入 torch.Size([1, 1024, 304])
                    S_infer = S_infer.transpose(1, 2)  # torch.Size([1, 304, 1024])
                    S_infer = S_infer + latent  # 融合潜在表示 torch.Size([1, 304, 1024])
                    target_lengths = (code_lens * 1.72).long()  # 计算目标长度  304*1.72 = 522

                    # 长度调节
                    cond = self.s2mel.models['length_regulator'](S_infer,
                                                                 ylens=target_lengths,
                                                                 n_quantizers=3,
                                                                 f0=None)[0]
                    # print(f"cond: {cond.shape}")  # [1, 522, 512]
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)  # 拼接条件  # [1, 670, 512]
                    # 条件流匹配推理
                    vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                                   torch.LongTensor([cat_condition.size(1)]).to(
                                                                       cond.device),
                                                                   ref_mel, style, None, diffusion_steps,
                                                                   inference_cfg_rate=inference_cfg_rate)
                    # print(f"vc_target: {vc_target}")  # [1, 80, 670]
                    vc_target = vc_target[:, :, ref_mel.size(-1):]  # 裁剪目标梅尔谱图
                    # print(f"vc_target: {vc_target}")  # [1, 80, 522]

                    # vc_target = torch.load('temp/vc_target_cache.pt', map_location='cuda:0')  # vc_target
                    # print('【排查TTS生成噪声-Debug】vc_target 加载自 temp/vc_target_cache.pt')

                    s2mel_time += time.perf_counter() - m_start_time

                    # 声码器阶段
                    m_start_time = time.perf_counter()
                    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)  # 梅尔谱图转波形
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                # 后处理 ------------------------------------------------
                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)  # 限制幅度范围
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())

                # wavs.append(wav[:, :-512])
                wavs.append(wav.cpu())  # 转移到CPU并保存 / to cpu before saving
                if stream_return:
                    yield sampling_rate, wav.cpu(), audio_feature, emovec_contrl  # 流式返回当前段

                    if interval_silence > 0:  # 当静音段为0时，不应该返回静音段
                        if silence == None:
                            silence = self.interval_silence(wavs, sampling_rate=sampling_rate,
                                                            interval_silence=interval_silence)
                        yield sampling_rate, silence, None, emovec_contrl  # 返回静音段

        # 合成结束 ------------------------------------------------
        end_time = time.perf_counter()

        # 如果不是流式调用，则一次性返回全量，如果是流式调用，前面已经增量流式返回了，不在此处继续返回。
        if not stream_return:

            self._set_gr_progress(0.9, "saving audio...")
            # 插入静音并合并所有音频段
            wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
            # print("wavs", wavs)
            wav = torch.cat(wavs, dim=1)  # 沿时间维度拼接
            wav_length = wav.shape[-1] / sampling_rate  # 计算音频时长
            # 合并所有音频特征段
            audio_feature = torch.cat(audio_features, dim=1)


            # 打印性能统计
            if verbose:
                print(f"tts部分推理记录:")
                print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
                print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
                print(f">> s2mel_time: {s2mel_time:.2f} seconds")
                print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
                print(f">> Total inference time: {end_time - start_time:.2f} seconds")
                print(f">> Generated audio length: {wav_length:.2f} seconds")
                print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

            # 保存音频 / save audio
            wav = wav.cpu()  # 确保在CPU上 / to cpu
            # # 返回以符合Gradio的格式要求
            # wav_data = wav.type(torch.int16)
            # wav_data = wav_data.numpy().T  # 转置为(采样点数, 声道数)格式
            yield sampling_rate, wav, audio_feature, emovec_contrl

    def get_emo_cond_emb_from_audio(self, emo_audio_prompt, verbose=False):
        '''
        从情感音频 emo_audio_prompt 中获取情感条件嵌入 emo_cond_emb
        用于为数据集中音频标注情感控制特征
        '''
        # 加载和处理情感音频
        emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
        print(f"[DEBUG] emo_audio shape:{emo_audio.shape}")  # torch.Size([1, 46289])

        emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
        emo_input_features = emo_inputs["input_features"]
        emo_attention_mask = emo_inputs["attention_mask"]
        emo_input_features = emo_input_features.to(self.device)
        emo_attention_mask = emo_attention_mask.to(self.device)
        print(f"[DEBUG] emo_attention_mask: {emo_attention_mask.shape}")  # torch.Size([1, 144])
        print(f"[DEBUG] emo_input_features: {emo_input_features.shape}")  # torch.Size([1, 144, 160])

        emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)
        return emo_cond_emb




def find_most_similar_cosine(query_vector, matrix):
    """
    使用余弦相似度在矩阵中查找与查询向量最相似的向量
    """
    query_vector = query_vector.float()  # 确保为float类型
    matrix = matrix.float()  # 确保为float类型

    # 计算余弦相似度 [num_vectors]
    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    # 找到相似度最高的索引
    most_similar_index = torch.argmax(similarities)
    return most_similar_index


# 工具方法 ============================================================================

def print_gpt_weights(model, max_rows=20):
    """
    打印 IndexTTS2 中 GPT 部分的权重
    max_rows : 最多打印多少行，防止刷屏
    """
    gpt = model.gpt          # 这是 nn.Module
    count = 0
    for name, param in gpt.named_parameters():
        if not param.requires_grad:      # 冻结的也打，可自己过滤
            continue
        print(f"{name:<60}  {tuple(param.shape)}  "
              f"min={param.min().item():+.4f}  "
              f"max={param.max().item():+.4f}  "
              f"mean={param.mean().item():+.4f}  "
              f"std={param.std().item():+.4f}")
        count += 1
        if count >= max_rows:
            print("... 已截断，继续打印请调大 max_rows ...")
            break
    print(f"\nGPT 权重总计 {count} 个 tensor.")

@torch.no_grad()
def compare_modules(mod_a, mod_b, name_a='A', name_b='B', rtol=1e-5, atol=1e-8):
    """
    比较两个 nn.Module 的 state_dict
    """
    sd_a = mod_a.state_dict()
    sd_b = mod_b.state_dict()

    # 1. 统一 key 集合
    keys_a, keys_b = set(sd_a.keys()), set(sd_b.keys())
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    common = keys_a & keys_b

    diff = 0
    # 2. 先报缺失
    for k in only_a:
        print(f'[KEY MISSING] {k} 只在 {name_a} 出现')
        diff += 1
    for k in only_b:
        print(f'[KEY MISSING] {k} 只在 {name_b} 出现')
        diff += 1

    # 3. 再比对公共 key
    for k in sorted(common):
        va, vb = sd_a[k], sd_b[k]
        # 防御：确保都是 Tensor
        if not isinstance(va, torch.Tensor) or not isinstance(vb, torch.Tensor):
            print(f'[TYPE DIFF]   {k}  {name_a}:{type(va)} vs {name_b}:{type(vb)}')
            diff += 1
            continue

        if va.shape != vb.shape:
            print(f'[SHAPE DIFF]  {k}  {name_a}:{va.shape} vs {name_b}:{vb.shape}')
            diff += 1
            continue

        if not torch.allclose(va, vb, rtol=rtol, atol=atol):
            ha = hashlib.sha256(va.cpu().numpy().tobytes()).hexdigest()[:8]
            hb = hashlib.sha256(vb.cpu().numpy().tobytes()).hexdigest()[:8]
            print(f'[VALUE DIFF]  {k}  hash {name_a}:{ha} vs {name_b}:{hb}')
            diff += 1
    return diff

@torch.no_grad()
def compare_cfm_branch(model_a, model_b, name_a='UniTAF', name_b='IndexTTS2'):
    """
    仅比对 model.s2mel.models['cfm'] 分支
    """
    # 1. 收集所有 nn.Module
    mods_a = collect_nn_modules(model_a)
    mods_b = collect_nn_modules(model_b)

    # 2. 只保留路径中带 s2mel.models.cfm 的模块
    def filter_cfm(d):
        return {k: v for k, v in d.items()
                if 's2mel.models.cfm' in k}

    cfm_a = filter_cfm(mods_a)
    cfm_b = filter_cfm(mods_b)

    # 3. 统一 key（去掉前缀差异）
    def norm(k):
        k = re.sub(r'^(unitaf_)?(indextts2\.)?', '', k, flags=re.I)
        return k

    keys_a = {norm(k): k for k in cfm_a}
    keys_b = {norm(k): k for k in cfm_b}

    common = sorted(set(keys_a) & set(keys_b))
    only_a = set(keys_a) - set(keys_b)
    only_b = set(keys_b) - set(keys_a)

    if only_a or only_b:
        print('>>> 仅存在于 {} 的 cfm 模块：{}'.format(name_a, list(only_a)))
        print('>>> 仅存在于 {} 的 cfm 模块：{}'.format(name_b, list(only_b)))

    total_diff = 0
    for k in common:
        ka, kb = keys_a[k], keys_b[k]
        diff = compare_modules(cfm_a[ka], cfm_b[kb],
                               name_a=f'{name_a}.{k}',
                               name_b=f'{name_b}.{k}')
        total_diff += diff
        if diff > 0:
            print(f'----- cfm 子模块 {k} 存在 {diff} 处权重差异 -----')
        else:
            print(f'----- cfm 子模块 {k} 权重完全一致 -----')

    print(f'\n>>> cfm 分支总计发现 {total_diff} 处权重差异')
    return total_diff

def deep_compare(container_a, container_b, name_a='UniTAF', name_b='IndexTTS2', verbose_same=False):
    """
    对两个任意容器（非 nn.Module）做深度模块比对
    verbose_same=True  才打印完全一致的模块
    """
    mods_a = collect_nn_modules(container_a)
    mods_b = collect_nn_modules(container_b)

    # 统一 key：去掉前缀，只保留相对路径
    def norm(k):
        k = re.sub(r'^(unitaf_)?(indextts2\.)?', '', k, flags=re.I)
        return k

    keys_a = {norm(k): k for k in mods_a.keys()}
    keys_b = {norm(k): k for k in mods_b.keys()}

    common = sorted(set(keys_a) & set(keys_b))
    only_a = set(keys_a) - set(keys_b)
    only_b = set(keys_b) - set(keys_a)

    if only_a or only_b:
        print('>>> 仅存在于 {} 的模块：{}'.format(name_a, list(only_a)))
        print('>>> 仅存在于 {} 的模块：{}'.format(name_b, list(only_b)))

    total_diff = 0
    for k in common:
        ka, kb = keys_a[k], keys_b[k]
        diff = compare_modules(mods_a[ka], mods_b[kb],
                               name_a=f'{name_a}.{k}', name_b=f'{name_b}.{k}')
        total_diff += diff
        # 关键：只有差异或 verbose 才打印
        if diff > 0 or verbose_same:
            print(f'\n===== 比对模块 {k} =====')
            if diff == 0:
                print('  权重完全一致')

    print(f'\n>>> 总计发现 {total_diff} 处权重差异')

def collect_nn_modules(obj, prefix=''):
    """
    递归扫描任意 Python 对象，收集所有 nn.Module 类型的属性（含嵌套）
    返回 OrderedDict:  相对路径 -> Module
    """
    ans = OrderedDict()
    if isinstance(obj, torch.nn.Module):
        # 先把自己挂上去（如果愿意）
        if prefix:
            ans[prefix] = obj
        # 再扫子模块
        for name, module in obj.named_children():
            ans.update(collect_nn_modules(module, prefix+f'{name}.'))
    elif hasattr(obj, '__dict__'):      # 普通对象，扫成员变量
        for name, attr in obj.__dict__.items():
            ans.update(collect_nn_modules(attr, prefix+f'{name}.'))
    return ans

def skeleton(obj, name='model'):
    """
    只保留「含参数」的叶子 Module，返回 路径 -> (类名, [para_shape, ...])
    """
    sk = OrderedDict()
    for path, mod in collect_nn_modules(obj).items():
        # 只保留叶子层（不含子节点的 Module）
        if not list(mod.children()) and list(mod.parameters()):
            cls   = mod.__class__.__name__
            shapes = [tuple(p.shape) for p in mod.parameters()]
            sk[path] = (cls, shapes)
    return sk

def print_diff(a_name, a_sk, b_name, b_sk):
    keys_a, keys_b = set(a_sk), set(b_sk)
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    common = keys_a & keys_b

    print(f'\n========== 仅存在于 {a_name} 的层 ==========')
    for k in sorted(only_a):
        cls, sha = a_sk[k]
        print(f'{k:60s}  {cls:15s}  {sha}')

    print(f'\n========== 仅存在于 {b_name} 的层 ==========')
    for k in sorted(only_b):
        cls, sha = b_sk[k]
        print(f'{k:60s}  {cls:15s}  {sha}')

    print(f'\n========== 双方都有但形状不同的层 ==========')
    for k in sorted(common):
        cls_a, sha_a = a_sk[k]
        cls_b, sha_b = b_sk[k]
        if sha_a != sha_b:
            print(f'{k:60s}  {a_name}:{sha_a}')
            print(f'{"":60s}  {b_name}:{sha_b}')

    print(f'\n========== 双方都有且形状一致的层（共 {len(common)} 个） ==========')
    count = 0
    for k in sorted(common):
        cls_a, sha_a = a_sk[k]
        cls_b, sha_b = b_sk[k]
        if sha_a == sha_b and cls_a == cls_b:
            count += 1
            if count <= 50:  # 防刷屏，只打前 50 条
                print(f'{k:60s}  {cls_a:15s}  {sha_a}')
    if count > 50:
        print(f'  ... 还有 {count - 50} 个完全一致，已省略 ...')
    print(f'>>> 双方完全一致的层共计 {count} 个')

def compare_tts_pairs():
    '''
    加载UniTAF实现的UniTAFIndexTTS2和原始IndexTTS2对比模型权重和结构
    '''
    from omegaconf import OmegaConf
    indextts2_cfg = OmegaConf.load("checkpoints/config.yaml")  # 加载IndexTTS2配置

    unitaf_tts_model = UniTAFIndexTTS2(
        cfg=indextts2_cfg,
        # gpt_ckpt="./unitaf_ckpt/lora_tts_and_a2f_25-12-17/checkpoint-20000/tts_model.pt",
        gpt_ckpt=None,  # 如果需要单独指定gpt_ckpt路径
        model_dir="checkpoints",
        use_cuda_kernel=False,  # 禁用CUDA内核
        use_torch_compile=True  # 启用torch.compile优化
    )

    tts_model = IndexTTS2(
        cfg_path="checkpoints/config.yaml",  # 配置文件路径
        model_dir="checkpoints",
        use_cuda_kernel=False,  # 禁用CUDA内核
        use_torch_compile=True  # 启用torch.compile优化
    )

    # 深度比对
    deep_compare(unitaf_tts_model, tts_model,
                 name_a='UniTAFIndexTTS2', name_b='IndexTTS2')

    # 比对 模型中 cfm 分支
    compare_cfm_branch(unitaf_tts_model, tts_model,
                       name_a='UniTAFIndexTTS2', name_b='IndexTTS2')

    # print_gpt_weights(tts_model)

    unitaf_sk = skeleton(unitaf_tts_model, 'UniTAFIndexTTS2')
    tts_sk = skeleton(tts_model, 'IndexTTS2')

    print_diff('UniTAFIndexTTS2', unitaf_sk, 'IndexTTS2', tts_sk)

    '''
    结果表明模型权重和结构均相同
    '''


if __name__ == "__main__":
    '''
    单独测试UniTAFIndexTTS2的音频推理 python -m unitaf_train_component.indextts2_inference_component
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # # 比较我们实现的UniTAFIndexTTS2和IndexTTS2在权重和结构上是否有区别
    # compare_tts_pairs()

    from omegaconf import OmegaConf
    indextts2_cfg = OmegaConf.load("checkpoints/config.yaml")  # 加载IndexTTS2配置

    unitaf_tts_model = UniTAFIndexTTS2(
        cfg=indextts2_cfg,
        gpt_ckpt=None,  # 如果需要单独指定gpt_ckpt路径
        model_dir="checkpoints",
        use_cuda_kernel=False,  # 禁用CUDA内核
        use_torch_compile=True  # 启用torch.compile优化
    )

    # 实际调用
    text = "清晨的阳光透过窗帘洒在书桌上，新的一天开始了。窗外鸟儿欢快地歌唱，空气中弥漫着淡淡的花香。"
    sr, wav, audio_feature = unitaf_tts_model.infer(
        spk_audio_prompt='examples/voice_zhongli.wav',
        text=text,
        emo_alpha=0.6,
        use_emo_text=True,
        emo_text=text,  # 情感控制选择从传入的情感文本中推断，不传额外用于推断的情感文本时则直接从目标文本中推断。
        verbose=False
    )

    tts_output_path = "outputs/UniTAF_output.wav"

    # 直接保存音频到指定路径中
    if os.path.isfile(tts_output_path):
        os.remove(tts_output_path)
        print(">> remove old wav file:", tts_output_path)
    if os.path.dirname(tts_output_path) != "":
        os.makedirs(os.path.dirname(tts_output_path), exist_ok=True)

    torchaudio.save(tts_output_path, wav, sr)  # 保存为16位PCM
    print(">> wav file saved to:", tts_output_path)






