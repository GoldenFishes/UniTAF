import os
from subprocess import CalledProcessError

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import json
import re
import time
import librosa
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from omegaconf import OmegaConf

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

class IndexTTS2:
    def __init__(
            self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, device=None,
            use_cuda_kernel=None,use_deepspeed=False, use_accel=False, use_torch_compile=False
    ):
        """
        Args:
            cfg_path (str): 配置文件路径
            model_dir (str): 模型目录路径
            use_fp16 (bool): 是否使用fp16精度
            device (str): 使用的设备（如'cuda:0', 'cpu'）。如果为None，将根据CUDA或MPS可用性自动设置
            use_cuda_kernel (None | bool): 是否使用BigVGan自定义融合激活CUDA内核，仅适用于CUDA设备
            use_deepspeed (bool): 是否使用DeepSpeed
            use_accel (bool): 是否对GPT2使用加速引擎
            use_torch_compile (bool): 是否使用torch.compile进行优化
        """
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
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.use_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token  # 停止生成梅尔谱图的特殊token
        self.use_accel = use_accel
        self.use_torch_compile = use_torch_compile

        # 初始化情感分析模型（Qwen）
        self.qwen_emo = QwenEmotion(os.path.join(self.model_dir, self.cfg.qwen_emo_path))

        # 初始化GPT文本到语义模型
        self.gpt = UnifiedVoice(**self.cfg.gpt, use_accel=self.use_accel)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.use_fp16:
            self.gpt.eval().half()  # 设置为评估模式并使用半精度
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)

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
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

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
        self.cache_spk_cond = None          # 缓存的说话人条件
        self.cache_s2mel_style = None       # 缓存的s2mel风格
        self.cache_s2mel_prompt = None      # 缓存的s2mel提示
        self.cache_spk_audio_prompt = None  # 缓存的说话人音频提示
        self.cache_emo_cond = None          # 缓存的情感条件
        self.cache_emo_audio_prompt = None  # 缓存的情感音频提示
        self.cache_mel = None               # 缓存的梅尔谱图

        # 进度引用显示（可选，用于GUI应用）
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        """获取语义嵌入"""
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std  # 标准化
        return feat

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        移除长静音段
        codes: [B, T] 语义编码序列
        silent_token: 静音token ID
        max_consecutive: 最大连续静音帧数
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False

        # 对每个batch进行处理
        for i in range(0, codes.shape[0]):
            code = codes[i]
            # 计算有效序列长度（遇到停止token前的长度）
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            # 统计静音token数量
            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                # 需要修复长静音的情况
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[
                               k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:  # 最多保留10个连续静音
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # 不需要修复，直接截取有效长度 shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)

        # 重新构建tensor
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # 否则保持原样 unchanged
            pass

        # 根据最大长度裁剪codes
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        生成要插入生成片段之间的静音段
        """
        # 如果没有音频或静音间隔<=0，直接返回
        if not wavs or interval_silence <= 0:
            return wavs

        # 获取声道数 get channel_size
        channel_size = wavs[0].size(0)
        # 计算静音持续时间（采样点数） get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        return torch.zeros(channel_size, sil_dur)

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        在生成的音频段之间插入静音
        wavs: 音频张量列表 [List[torch.tensor]]
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # 获取声道数 get channel_size
        channel_size = wavs[0].size(0)
        # 计算静音持续时间 get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)  # 创建静音张量

        # 构建包含静音的音频列表
        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)  # 添加音频段
            if i < len(wavs) - 1:  # 如果不是最后一个音频段，添加静音
                wavs_list.append(sil_tensor)

        return wavs_list

    def _set_gr_progress(self, value, desc):
        """设置Gradio进度条（如果可用）"""
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    def _load_and_cut_audio(self,audio_path,max_audio_length_seconds,verbose=False,sr=None):
        """加载并裁剪音频文件"""
        if not sr:
            audio, sr = librosa.load(audio_path)  # 使用librosa默认采样率加载
        else:
            audio, _ = librosa.load(audio_path,sr=sr)  # 使用指定采样率加载

        audio = torch.tensor(audio).unsqueeze(0)  # 添加批次维度
        max_audio_samples = int(max_audio_length_seconds * sr)  # 计算最大采样点数

        # 如果音频过长，进行裁剪
        if audio.shape[1] > max_audio_samples:
            if verbose:
                print(f"音频过长 ({audio.shape[1]} 采样点)，截断为 {max_audio_samples} 采样点")
            audio = audio[:, :max_audio_samples]
        return audio, sr
    
    def normalize_emo_vec(self, emo_vector, apply_bias=True):
        """
        标准化情感向量
        apply_bias: 是否应用情感偏置，减少可能导致奇怪结果的情感权重
        """
        # / apply biased emotion factors for better user experience,
        # / by de-emphasizing emotions that can cause strange results
        # 应用情感偏置以获得更好的用户体验
        if apply_bias:
            # [高兴  , 生气  , 悲伤, 害怕  , 厌恶      , 忧郁        , 惊讶      , 平静] 的情感偏置
            # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
            emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]

        # / the total emotion sum must be 0.8 or less
        # 情感总和必须小于等于0.8
        emo_sum = sum(emo_vector)
        if emo_sum > 0.8:
            scale_factor = 0.8 / emo_sum  # 计算缩放因子
            emo_vector = [vec * scale_factor for vec in emo_vector]  # 缩放情感向量

        return emo_vector

    # 原始推理模式
    def infer(self, spk_audio_prompt, text, output_path,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_segment=120, stream_return=False, more_segment_before=0, **generation_kwargs):
        """
        主要推理接口
        参数:
            spk_audio_prompt: 说话人参考音频路径
            text: 要合成的文本
            output_path: 输出音频路径
            emo_audio_prompt: 情感参考音频路径
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
                spk_audio_prompt, text, output_path,
                emo_audio_prompt, emo_alpha,
                emo_vector,
                use_emo_text, emo_text, use_random, interval_silence,
                verbose, max_text_tokens_per_segment, stream_return, more_segment_before, **generation_kwargs
            )
        else:
            # 普通推理模式，返回最终结果
            try:
                return list(self.infer_generator(
                    spk_audio_prompt, text, output_path,
                    emo_audio_prompt, emo_alpha,
                    emo_vector,
                    use_emo_text, emo_text, use_random, interval_silence,
                    verbose, max_text_tokens_per_segment, stream_return, more_segment_before, **generation_kwargs
                ))[0]
            except IndexError:
                return None

    def infer_generator(self, spk_audio_prompt, text, output_path,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_segment=120, stream_return=False, quick_streaming_tokens=0, **generation_kwargs):
        """生成器形式的推理函数，支持流式输出"""
        print(">> starting inference...")
        self._set_gr_progress(0, "starting inference...")
        if verbose:
            print(f"origin text:{text}, spk_audio_prompt:{spk_audio_prompt}, "
                  f"emo_audio_prompt:{emo_audio_prompt}, emo_alpha:{emo_alpha}, "
                  f"emo_vector:{emo_vector}, use_emo_text:{use_emo_text}, "
                  f"emo_text:{emo_text}")
        start_time = time.perf_counter()

        # 情感处理逻辑
        if use_emo_text or emo_vector is not None:
            # / we're using a text or emotion vector guidance; so we must remove
            # / "emotion reference voice", to ensure we use correct emotion mixing!
            # 使用文本或情感向量引导时，移除情感参考音频以确保正确的情感混合
            emo_audio_prompt = None

        if use_emo_text:
            # / automatically generate emotion vectors from text prompt
            # 从文本提示自动生成情感向量
            if emo_text is None:
                emo_text = text  # 使用主文本作为情感分析文本 / use main text prompt
            emo_dict = self.qwen_emo.inference(emo_text)  # 调用情感分析模型
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
                print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")

        if emo_audio_prompt is None:
            # / we are not using any external "emotion reference voice"; use
            # / speaker's voice as the main emotion reference audio.
            # 没有外部情感参考音频时，使用说话人音频作为主要情感参考
            emo_audio_prompt = spk_audio_prompt
            # / must always use alpha=1.0 when we don't have an external reference voice
            # 没有外部参考音频时必须使用alpha=1.0
            emo_alpha = 1.0

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
            audio,sr = self._load_and_cut_audio(spk_audio_prompt,15,verbose)
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

        # 情感向量处理
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
            emo_matrix = torch.cat(emo_matrix, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix  # 加权情感矩阵
            emovec_mat = torch.sum(emovec_mat, 0)  # 求和得到最终情感向量
            emovec_mat = emovec_mat.unsqueeze(0)   # 添加批次维度

        # 情感条件缓存处理
        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
            if self.cache_emo_cond is not None:
                self.cache_emo_cond = None
                torch.cuda.empty_cache()

            # 加载和处理情感音频
            emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt,15,verbose,sr=16000)
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
        else:
            emo_cond_emb = self.cache_emo_cond

        # 文本处理
        self._set_gr_progress(0.1, "text processing...")
        text_tokens_list = self.tokenizer.tokenize(text)  # 文本分词
        segments = self.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment, quick_streaming_tokens = quick_streaming_tokens)
        segments_count = len(segments)

        # 检查未知token
        text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        if self.tokenizer.unk_token_id in text_token_ids:
            print(f"  >> Warning: input text contains {text_token_ids.count(self.tokenizer.unk_token_id)} unknown tokens (id={self.tokenizer.unk_token_id}):")
            print( "     Tokens which can't be encoded: ", [t for t, id in zip(text_tokens_list, text_token_ids) if id == self.tokenizer.unk_token_id])
            print(f"     Consider updating the BPE model or modifying the text to avoid unknown tokens.")
                  
        if verbose:
            print("text_tokens_list:", text_tokens_list)
            print("segments count:", segments_count)
            print("max_text_tokens_per_segment:", max_text_tokens_per_segment)
            print(*segments, sep="\n")

        # 生成参数设置
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        sampling_rate = 22050

        # 初始化变量
        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        silence = None # 用于流式返回的静音段 / for stream_return

        # 分段处理文本
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

            # GPT推理阶段
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
                        emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec
                        # emovec = emovec_mat
                        # print(f"[DEBUG] 混合后emovec: {emovec.shape}")  # torch.Size([1, 1280])

                    # 语音推理生成语义编码
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
                    print(f"修正后的编码形状: {codes.shape}, 编码类型: {codes.dtype}")
                    print(f"编码长度: {code_lens}")

                # GPT前向传播
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

                # s2mel推理阶段
                dtype = None
                with torch.amp.autocast(text_tokens.device.type, enabled=dtype is not None, dtype=dtype):
                    m_start_time = time.perf_counter()
                    diffusion_steps = 25
                    inference_cfg_rate = 0.7
                    latent = self.s2mel.models['gpt_layer'](latent)  # GPT层处理
                    S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))  # 编码转嵌入
                    S_infer = S_infer.transpose(1, 2)
                    S_infer = S_infer + latent  # 融合潜在表示
                    target_lengths = (code_lens * 1.72).long()  # 计算目标长度

                    # 长度调节
                    cond = self.s2mel.models['length_regulator'](S_infer,
                                                                 ylens=target_lengths,
                                                                 n_quantizers=3,
                                                                 f0=None)[0]
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)  # 拼接条件

                    # 条件流匹配推理
                    vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                                   torch.LongTensor([cat_condition.size(1)]).to(
                                                                       cond.device),
                                                                   ref_mel, style, None, diffusion_steps,
                                                                   inference_cfg_rate=inference_cfg_rate)
                    vc_target = vc_target[:, :, ref_mel.size(-1):]  # 裁剪目标梅尔谱图
                    s2mel_time += time.perf_counter() - m_start_time

                    # 声码器阶段
                    m_start_time = time.perf_counter()
                    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)  # 梅尔谱图转波形
                    print(wav.shape)
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                # 后处理
                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)  # 限制幅度范围
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())

                # wavs.append(wav[:, :-512])
                wavs.append(wav.cpu())  # 转移到CPU并保存 / to cpu before saving
                if stream_return:
                    yield wav.cpu()  # 流式返回当前段
                    if silence == None:
                        silence = self.interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
                    yield silence  # 返回静音段

        end_time = time.perf_counter()

        self._set_gr_progress(0.9, "saving audio...")
        # 插入静音并合并所有音频段
        wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
        wav = torch.cat(wavs, dim=1)  # 沿时间维度拼接
        wav_length = wav.shape[-1] / sampling_rate  # 计算音频时长

        # 打印性能统计
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> s2mel_time: {s2mel_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # 保存音频 / save audio
        wav = wav.cpu()  # 确保在CPU上 / to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)  # 保存为16位PCM
            print(">> wav file saved to:", output_path)
            if stream_return:
                return None
            yield output_path
        else:
            if stream_return:
                return None
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T  # 转置为(采样点数, 声道数)格式
            yield (sampling_rate, wav_data)


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

class QwenEmotion:
    """Qwen情感分析模型封装类"""
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype="float16",  # "auto"
            device_map="auto"  # 自动设备映射
        )
        self.prompt = "文本情感分类"  # 系统提示词
        # 中文情感关键词到英文的映射
        self.cn_key_to_en = {
            "高兴": "happy",
            "愤怒": "angry",
            "悲伤": "sad",
            "恐惧": "afraid",
            "反感": "disgusted",
            # TODO: the "低落" (melancholic) 情感总是会被QwenEmotion映射到
            #    # "悲伤" (sad)。即使用户输入确切的关键词，模型也无法区分这些情感。
            #    # 参见: `self.melancholic_words` 了解当前的解决方案
            "低落": "melancholic",
            "惊讶": "surprised",
            "自然": "calm",
        }
        # 期望的情感向量顺序
        self.desired_vector_order = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
        # 忧郁相关关键词集合（用于情感映射修正）
        self.melancholic_words = {
            # / emotion text phrases that will force QwenEmotion's "悲伤" (sad) detection
            # / to become "低落" (melancholic) instead, to fix limitations mentioned above.
            # 这些文本短语会强制将QwenEmotion检测到的"悲伤" (sad)
            # 转换为"低落" (melancholic)，以解决上述提到的限制
            "低落",
            "melancholy",
            "melancholic",
            "depression",
            "depressed",
            "gloomy",
        }
        # 情感分数范围限制
        self.max_score = 1.2
        self.min_score = 0.0

    def clamp_score(self, value):
        """将情感分数限制在允许的范围内"""
        return max(self.min_score, min(self.max_score, value))

    def convert(self, content):
        '''
        将原始情感分析结果转换为标准化的情感向量字典
        '''
        # / generate emotion vector dictionary:
        # / - insert values in desired order (Python 3.7+ `dict` remembers insertion order)
        # / - convert Chinese keys to English
        # / - clamp all values to the allowed min/max range
        # / - use 0.0 for any values that were missing in `content`
        # 生成情感向量字典:
        # - 按期望顺序插入值（Python 3.7+ 的`dict`会记住插入顺序）
        # - 将中文键转换为英文
        # - 将所有值限制在允许的最小/最大范围内
        # - 对于`content`中缺失的值使用0.0
        emotion_dict = {
            self.cn_key_to_en[cn_key]: self.clamp_score(content.get(cn_key, 0.0))
            for cn_key in self.desired_vector_order
        }

        # / default to a calm/neutral voice if all emotion vectors were empty
        # 如果所有情感向量都为空，默认使用平静/中性的声音
        if all(val <= 0.0 for val in emotion_dict.values()):
            print(">> 未检测到情感；使用默认的平静/中性声音")
            emotion_dict["calm"] = 1.0

        return emotion_dict

    def inference(self, text_input):
        start = time.time()
        # 构建对话消息
        messages = [
            {"role": "system", "content": f"{self.prompt}"},
            {"role": "user", "content": f"{text_input}"}
        ]
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,          # 不进行tokenize
            add_generation_prompt=True,  # 添加生成提示
            enable_thinking=False,    # 禁用思考模式
        )
        # 对输入进行tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 进行文本生成（情感分析） / conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,     # 最大新生成token数
            pad_token_id=self.tokenizer.eos_token_id  # 填充token ID
        )
        # 提取生成的输出（去除输入部分）
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # 解析思考内容（如果存在） / parsing thinking content
        try:
            # 反向查找151668 (</think>) token的位置 / rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0  # 如果找不到思考结束标记，从头开始

        # 解码输出内容
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)

        # 尝试将输出解析为JSON格式的情感检测字典 / decode the JSON emotion detections as a dictionary
        try:
            content = json.loads(content)
        except json.decoder.JSONDecodeError:
            # invalid JSON; fallback to manual string parsing
            # print(">> parsing QwenEmotion response", content)
            content = {
                m.group(1): float(m.group(2))  # 使用正则表达式提取键值对
                for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)
            }
            # print(">> dict result", content)

        # / workaround for QwenEmotion's inability to distinguish "悲伤" (sad) vs "低落" (melancholic).
        # / if we detect any of the IndexTTS "melancholic" words, we swap those vectors
        # / to encode the "sad" emotion as "melancholic" (instead of sadness).
        # 解决QwenEmotion无法区分"悲伤" (sad) 和 "低落" (melancholic) 的问题
        # 如果检测到任何IndexTTS的"melancholic"关键词，我们交换这些向量
        # 将"sad"情感编码为"melancholic"（而不是悲伤）
        text_input_lower = text_input.lower()
        if any(word in text_input_lower for word in self.melancholic_words):
            # print(">> before vec swap", content)
            content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get("悲伤", 0.0)
            # print(">>  after vec swap", content)

        return self.convert(content)  # 返回转换后的标准格式


if __name__ == "__main__":
    prompt_wav = "examples/voice_01.wav"
    text = '欢迎大家来体验indextts2，并给予我们意见与反馈，谢谢大家。'

    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",     # 配置文件路径
        model_dir="checkpoints", 
        use_cuda_kernel=False,                  # 禁用CUDA内核
        use_torch_compile=True                  # 启用torch.compile优化
    )

    # 进行语音合成推理
    tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True)

    # 性能测试：生成不同长度文本的语音并计时
    char_size = 5  # 测试文本字符数
    import string
    time_buckets = []  # 时间记录列表
    for i in range(10):  # 进行10次测试
        # 生成随机文本
        text = ''.join(random.choices(string.ascii_letters, k=char_size))
        start_time = time.time()  # 开始计时
        # 进行语音合成
        tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True)
        time_buckets.append(time.time() - start_time)  # 记录耗时

    print(time_buckets)
