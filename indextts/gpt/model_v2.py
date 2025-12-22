import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import GPT2Config, LogitsProcessorList
from indextts.gpt.transformers_gpt2 import GPT2PreTrainedModel, GPT2Model

# from transformers import GPT2Config, GPT2PreTrainedModel, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import (assert_device_map,
                                                     get_device_map)

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.utils.arch_util import AttentionBlock
from indextts.utils.typical_sampling import TypicalLogitsWarper


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


class ResBlock(nn.Module):
    """
    Basic residual convolutional block that uses GroupNorm.
    """

    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan),
            nn.ReLU(),
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan)
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class GPT2InferenceModel(GPT2PreTrainedModel):
    def __init__(self, config, gpt, text_pos_emb, embeddings, norm, linear, kv_cache=False):
        super().__init__(config)
        # Note: the argument named `text_pos_emb` here actually represents the mel position embedding
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.cached_mel_emb = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(max(1, torch.cuda.device_count())))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def store_mel_emb(self, mel_emb):
        self.cached_mel_emb = mel_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        assert self.cached_mel_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # Create embedding
        mel_len = self.cached_mel_emb.shape[1]
        if input_ids.shape[1] != 1:
            text_inputs = input_ids[:, mel_len:]
            text_emb = self.embeddings(text_inputs)
            text_emb = text_emb + self.text_pos_embedding(text_emb)
            if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                mel_emb = self.cached_mel_emb.repeat_interleave(
                    text_emb.shape[0] // self.cached_mel_emb.shape[0], 0
                )
            else:  # this outcome only occurs once per loop in most cases
                mel_emb = self.cached_mel_emb
            emb = torch.cat([mel_emb, text_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.text_pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - mel_len, attention_mask.device
            )
        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            if torch.backends.mps.is_available():
                self.to(self.transformer.first_device)
            else:
                torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=4,
                 do_checkpointing=False,
                 mean=False):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mean = mean

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        if self.mean:
            return h.mean(dim=2)
        else:
            return h
            # return h[:, :, 0]


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


def build_hf_gpt_transformer(layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing):
    """
    GPT-2 implemented by the HuggingFace library.
    """
    from transformers import GPT2Config, GPT2Model
    gpt_config = GPT2Config(vocab_size=256,  # Unused.
                            n_positions=max_mel_seq_len + max_text_seq_len,
                            n_ctx=max_mel_seq_len + max_text_seq_len,
                            n_embd=model_dim,
                            n_layer=layers,
                            n_head=heads,
                            gradient_checkpointing=checkpointing,
                            use_cache=not checkpointing)
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte
    return gpt, LearnedPositionEmbeddings(max_mel_seq_len, model_dim), LearnedPositionEmbeddings(max_text_seq_len, model_dim), \
        None, None


class MelEncoder(nn.Module):
    def __init__(self, channels, mel_channels=80, resblocks_per_reduction=2):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(nn.Conv1d(mel_channels, channels // 4, kernel_size=3, padding=1),
                                     nn.Sequential(*[ResBlock(channels // 4) for _ in range(resblocks_per_reduction)]),
                                     nn.Conv1d(channels // 4, channels // 2, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels // 16, channels // 2),
                                     nn.ReLU(),
                                     nn.Sequential(*[ResBlock(channels // 2) for _ in range(resblocks_per_reduction)]),
                                     nn.Conv1d(channels // 2, channels, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels // 8, channels),
                                     nn.ReLU(),
                                     nn.Sequential(*[ResBlock(channels) for _ in range(resblocks_per_reduction)]),
                                     )
        self.reduction = 4

    def forward(self, x):
        for e in self.encoder:
            x = e(x)
        return x.permute(0, 2, 1)


class UnifiedVoice(nn.Module):
    def __init__(self, layers=8, model_dim=512, heads=8, max_text_tokens=120, max_mel_tokens=250, max_conditioning_inputs=1,
                 mel_length_compression=1024, number_text_tokens=256,
                 start_text_token=0, stop_text_token=1, number_mel_codes=8194, start_mel_token=8192, stop_mel_token=8193,
                 train_solo_embeddings=False, use_mel_codes_as_input=True,
                 checkpointing=True, types=1,
                 condition_num_latent=32, condition_type="perceiver", condition_module=None, emo_condition_module=None, use_accel=False):
        """
        Args:
            layers: Number of layers in transformer stack.
            model_dim: Operating dimensions of the transformer
            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
            number_text_tokens:
            start_text_token:
            stop_text_token:
            number_mel_codes:
            start_mel_token:
            stop_mel_token:
            train_solo_embeddings:
            use_mel_codes_as_input:
            checkpointing:
            condition_type: perceiver, gst or default encoder
        """
        super().__init__()
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.condition_type = condition_type
        self.cond_num = condition_num_latent
        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)
        if condition_type == "perceiver":
            self.conditioning_encoder = ConditioningEncoder(1024, model_dim, num_attn_heads=heads)
            self.perceiver_encoder = PerceiverResampler(model_dim, dim_context=model_dim, num_latents=self.cond_num)
        elif condition_type == "conformer_perceiver" or condition_type == "conformer_encoder":
            self.conditioning_encoder = ConformerEncoder(input_size=1024,
                                                         output_size=condition_module['output_size'],
                                                         linear_units=condition_module['linear_units'],
                                                         attention_heads=condition_module['attention_heads'],
                                                         num_blocks=condition_module['num_blocks'],
                                                         input_layer=condition_module['input_layer'])
            if condition_type == "conformer_perceiver":
                self.perceiver_encoder = PerceiverResampler(model_dim, dim_context=condition_module['output_size'],
                                                            ff_mult=condition_module['perceiver_mult'],
                                                            heads=condition_module['attention_heads'],
                                                            num_latents=self.cond_num)
        else:
            self.conditioning_encoder = ConditioningEncoder(1024, model_dim, num_attn_heads=heads, mean=True)

        self.emo_conditioning_encoder = ConformerEncoder(input_size=1024,
                                                         output_size=emo_condition_module['output_size'],
                                                         linear_units=emo_condition_module['linear_units'],
                                                         attention_heads=emo_condition_module['attention_heads'],
                                                         num_blocks=emo_condition_module['num_blocks'],
                                                         input_layer=emo_condition_module['input_layer'])
        self.emo_perceiver_encoder = PerceiverResampler(1024, dim_context=emo_condition_module['output_size'],
                                                            ff_mult=emo_condition_module['perceiver_mult'],
                                                            heads=emo_condition_module['attention_heads'],
                                                            num_latents=1)



        self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim)
        self.emo_layer = nn.Linear(model_dim, model_dim)
        self.emovec_layer = nn.Linear(1024, model_dim)

        if use_mel_codes_as_input:
            self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        else:
            self.mel_embedding = MelEncoder(model_dim, resblocks_per_reduction=1)
        self.gpt, self.mel_pos_embedding, self.text_pos_embedding, self.mel_layer_pos_embedding, self.text_layer_pos_embedding = \
            build_hf_gpt_transformer(layers, model_dim, heads, self.max_mel_tokens + 2 + self.max_conditioning_inputs,
                                     self.max_text_tokens + 2, checkpointing)
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * .02, requires_grad=True)
            self.text_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * .02, requires_grad=True)
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        self.speed_emb = nn.Embedding(2, model_dim)
        self.speed_emb.weight.data.normal_(mean=0.0, std=0.0)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding]
        if use_mel_codes_as_input:
            embeddings.append(self.mel_embedding)
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=.02)

        self.use_accel = use_accel
        self.accel_engine = None  # Will be initialized in post_init_gpt2_config

    def post_init_gpt2_config(self, use_deepspeed=False, kv_cache=False, half=False):
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=self.number_mel_codes,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            gradient_checkpointing=False,
            use_cache=True,
        )

        if self.use_accel and torch.cuda.is_available():
            # Check if flash attention is available
            try:
                import flash_attn
            except ImportError:
                raise ImportError("flash_attn is required for acceleration but not installed. Please install from https://github.com/Dao-AILab/flash-attention/releases/")

            from indextts.accel import GPT2AccelModel, AccelInferenceEngine

            # Create accel model
            accel_gpt = GPT2AccelModel(gpt_config)
            accel_gpt.load_state_dict(self.gpt.state_dict(), strict=False)

            if half:
                accel_gpt = accel_gpt.half().cuda()
            else:
                accel_gpt = accel_gpt.cuda()
            accel_gpt.eval()

            lm_head_with_norm = nn.Sequential(self.final_norm, self.mel_head)
            self.accel_engine = AccelInferenceEngine(
                model=accel_gpt,
                lm_head=lm_head_with_norm,
                num_layers=self.layers,
                num_heads=self.heads,
                head_dim=self.model_dim // self.heads,
                block_size=256,
                num_blocks=16,  # Reduce to save memory (16*256 = 4096 tokens capacity)
                use_cuda_graph=True,
            )
            print("acceleration engine initialized")
        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        if use_deepspeed and half and torch.cuda.is_available():
            import deepspeed
            self.ds_engine = deepspeed.init_inference(model=self.inference_model,
                                                      mp_size=1,
                                                      replace_with_kernel_inject=True,
                                                      dtype=torch.float16)
            self.inference_model = self.ds_engine.module.eval()
        elif use_deepspeed and torch.cuda.is_available():
            import deepspeed
            self.ds_engine = deepspeed.init_inference(model=self.inference_model,
                                                      mp_size=1,
                                                      replace_with_kernel_inject=True,
                                                      dtype=torch.float32)
            self.inference_model = self.ds_engine.module.eval()
        else:
            self.inference_model = self.inference_model.eval()

        # self.inference_model = PrunedGPT2InferenceModel(gpt_config, self.gpt, self.mel_pos_embedding, self.mel_embedding, self.final_norm, self.mel_head)
        self.gpt.wte = self.mel_embedding


    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        """
        假设原始音频token序列为 [t1, t2, t3]

        输入序列 (inp): [start, t1, t2, t3]
        目标序列 (tar): [t1, t2, t3, stop]
        """

        inp = F.pad(input, (1, 0), value=start_token)  # 在开头添加start_token
        tar = F.pad(input, (0, 1), value=stop_token)   # 在末尾添加stop_token
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, mel_lengths):
        """
        给定从填充音频片段派生的mel tokens和实际每个batch元素的音频长度，
        用STOP_MEL_TOKEN替换零填充。这是创建可工作TTS模型所需的预格式化
        /
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(mel_lengths)):
            # 由于这些token生成的卷积性质， / Due to the convolutional nature of how these tokens are generated,
            # 最好让模型预测实际最后一个token之后的一个token。 / it would be best if the model predicts a token past the actual last token.
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def set_text_padding(self, text_input_tokens, text_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(text_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = text_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

    def get_logits(self, speech_conditioning_inputs, first_inputs, first_head, second_inputs=None, second_head=None, get_attns=False, return_latent=False, return_both=False):
        '''
        调用该方法的forward为训练过程，因为会传入second_inputs，second_inputs即为ground truth。返回的latent也是ground truth的latent。
        '''

        # 拼接所有输入：参考音频条件 + 文本输入 + 音频输入（如果存在）
        if second_inputs is not None:
            emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)
        else:
            emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)

        # 核心前向过程：将拼接后的embedding输入到GPT/Transformer中
        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)
        if get_attns:
            return gpt_out.attentions

        # 计算偏移量，跳过参考音频条件部分（只取文本和音频的输出）
        offset = speech_conditioning_inputs.shape[1]
        enc = gpt_out.last_hidden_state[:, offset:]  # 去掉条件部分，只保留文本和音频的隐藏状态
        enc = self.final_norm(enc)  # 应用层归一化

        # 只返回latent
        if return_latent:
            return enc[:, :first_inputs.shape[1]], enc[:, -second_inputs.shape[1]:]

        # [UniTextAudioFace训练新增逻辑] 同时返回latent（用于获取后续audio feature）和logits
        if return_both:
            # 分割输出：文本部分的logits
            first_logits = enc[:, :first_inputs.shape[1]]
            first_logits = first_head(first_logits)  # 文本head投影到文本词表
            first_logits = first_logits.permute(0, 2, 1)  # 调整维度用于cross_entropy
            if second_inputs is not None:
                # 分割输出：音频部分的logits
                second_logits = enc[:, -second_inputs.shape[1]:]
                second_logits = second_head(second_logits)  # 音频head投影到音频token词表
                second_logits = second_logits.permute(0, 2, 1)  # 调整维度用于cross_entropy  # (batch_size, sequence_length, audio_vocab_size)
                # 返回text_logits, mel_logits, text_latent, mel_latent
                return first_logits, second_logits, enc[:, :first_inputs.shape[1]], enc[:, -second_inputs.shape[1]:]
            else:
                return first_logits, enc[:, :first_inputs.shape[1]]

        # 原始逻辑：只返回logits
        # 分割输出：文本部分的logits
        first_logits = enc[:, :first_inputs.shape[1]]
        first_logits = first_head(first_logits)  # 文本head投影到文本词表
        first_logits = first_logits.permute(0, 2, 1)  # 调整维度用于cross_entropy
        if second_inputs is not None:
            # 分割输出：音频部分的logits
            second_logits = enc[:, -second_inputs.shape[1]:]
            second_logits = second_head(second_logits)  # 音频head投影到音频token词表
            second_logits = second_logits.permute(0, 2, 1)  # 调整维度用于cross_entropy  # (batch_size, sequence_length, audio_vocab_size)
            return first_logits, second_logits
        else:
            return first_logits

    def get_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        """
        获取语音条件特征
        Args:
            speech_conditioning_input: 语音条件输入，形状通常为 (B, D, T) 或 (B, 1, D, T)
            cond_mel_lengths: 条件mel长度，用于掩码处理
        Returns:
            conds: 条件特征，形状通常为 (B, 32, D) 或 (B, 1, D)
        """

        # 1. Perceiver编码器条件提取方式
        if self.condition_type == "perceiver":
            # 如果输入是4维，去除多余的维度
            if speech_conditioning_input.ndim == 4:
                speech_conditioning_input = speech_conditioning_input.squeeze(1)
            # 先通过条件编码器处理: (B, D, S) -> (B, D, S)
            speech_conditioning_input = self.conditioning_encoder(speech_conditioning_input)
            # 再通过Perceiver编码器: (B, S, D) -> (B, 32, D) 输出固定32个token
            conds = self.perceiver_encoder(speech_conditioning_input.transpose(1, 2))

        # 2. Conformer + Perceiver编码器条件提取方式
        elif self.condition_type == "conformer_perceiver":
            # 通过Conformer条件编码器处理，同时返回掩码: (B, S, D), (B, 1, S)
            speech_conditioning_input, mask = self.conditioning_encoder(
                speech_conditioning_input.transpose(1, 2), cond_mel_lengths
            )
            # 为条件特征生成掩码
            conds_mask = self.cond_mask_pad(mask.squeeze(1))
            # 使用带掩码的Perceiver编码器: (B, S, D) -> (B, 32, D)
            conds = self.perceiver_encoder(speech_conditioning_input, conds_mask)

        # 3. GST（全局风格token）条件提取方式
        elif self.condition_type == "gst":
            # 如果输入是4维，去除多余的维度
            if speech_conditioning_input.ndim == 4:
                speech_conditioning_input = speech_conditioning_input.squeeze(1)
            # 通过GST编码器提取全局风格特征: (B, S, D) -> (B, 1, D)
            conds = self.gst_encoder(speech_conditioning_input.transpose(1, 2))

        # 4. 默认条件提取方式（平均池化）
        else:
            # 确保输入是4维的: (B, 1, D, T)
            speech_conditioning_input = (
                speech_conditioning_input.unsqueeze(1)
                if len(speech_conditioning_input.shape) == 3
                else speech_conditioning_input
            )
            conds = []
            # 对每个通道分别提取条件特征
            for j in range(speech_conditioning_input.shape[1]):
                conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
            # 堆叠所有通道的条件特征
            conds = torch.stack(conds, dim=1)
            # 对所有通道求平均，得到全局条件特征
            conds = conds.mean(dim=1)
            # 添加序列维度: (B, D) -> (B, 1, D)
            conds = conds.unsqueeze(1)

        return conds


    def get_emo_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        speech_conditioning_input, mask = self.emo_conditioning_encoder(speech_conditioning_input.transpose(1, 2),
                                                                        cond_mel_lengths)  # (b, s, d), (b, 1, s)
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 1, d)
        return conds.squeeze(1)


    def forward(self, speech_conditioning_latent, text_inputs, text_lengths, mel_codes, mel_codes_lengths, emo_speech_conditioning_latent,
                cond_mel_lengths=None, emo_cond_mel_lengths=None, emo_vec=None, use_speed=None, do_spk_cond=False):
        """
        前向传播，同时使用文本和语音条件，可在文本条件模式或语音条件模式下工作

        speech_conditioning_input: MEL浮点张量, (batch,1024)
        text_inputs: 长整型张量, (batch, text长度)
        text_lengths: 长整型张量, (batch,)
        mel_inputs: 长整型张量, (batch, mel长度)
        wav_lengths: 长整型张量, (batch,)

        如果指定return_attentions，只返回logits。
        如果指定return_latent，不计算或返回loss和logits，只返回预测的潜在表示。
        """
        # 处理说话人条件特征
        if do_spk_cond:
            # 如果需要重新计算说话人条件，通过条件编码器获取
            speech_conditioning_latent = self.get_conditioning(speech_conditioning_latent.transpose(1,2), cond_mel_lengths)
        else:
            # 否则直接使用传入的说话人条件特征
            speech_conditioning_latent = speech_conditioning_latent

        # 处理情感向量
        if emo_vec is None:
            # 从情感语音条件计算情感向量
            emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1,2), emo_cond_mel_lengths)
            emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)  # 情感向量层处理
            emo_vec = self.emo_layer(emo_vec_syn)            # 情感层处理

        # 文本输入处理
        text_inputs = self.set_text_padding(text_inputs, text_lengths)  # 设置文本填充
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)  # 在末尾添加停止文本token

        # mel编码处理
        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)  # 设置mel填充
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)  # 在末尾添加停止mel token

        # 速度控制嵌入
        duration_emb = self.speed_emb(torch.zeros_like(use_speed))        # 正常速度嵌入
        duration_emb_half = self.speed_emb(torch.ones_like(use_speed))    # 半速嵌入

        # 合并所有条件特征：说话人条件 + 情感向量 + 速度控制
        conds = torch.cat((speech_conditioning_latent + emo_vec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)

        # 构建文本的输入和目标对齐
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        # 文本嵌入 = 词嵌入 + 位置嵌入
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        # 构建mel编码的输入和目标对齐
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token, self.stop_mel_token)

        # mel嵌入 = mel词嵌入 + mel位置嵌入
        mel_emb = self.mel_embedding(mel_codes)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

        # 获取文本和mel的logits（实际上是潜在表示）
        text_logits, mel_logits = self.get_logits(conds, text_emb, self.text_head, mel_emb, self.mel_head, get_attns=False, return_latent=True)
        # 返回mel的logits，去掉前向传播中添加的两个token
        # 这里推理代码中mel_logits返回的实际上是 经过GPT编码 + final_norm 后的潜在表示
        return mel_logits[:, :-2]  # 尽管名字叫logits，但这些不是真正的logits / Despite the name, these are not logits. Strip off the two tokens added by this forward pass.

    def prepare_gpt_inputs(
        self,
        conditional_latents: torch.Tensor,
        text_inputs: torch.Tensor,
    ):
        
        """
        为GPT2推理模型准备输入
        Args:
            conditional_latents: (b, 32, dim) 通过get_conditioning()得到的音频条件嵌入
            text_inputs: (b, L) 文本输入token
        Returns:
            input_ids: (b, s+1) 用于GPT2InferenceModel.generate()的输入ID
            inputs_embeds: (b, s+1, dim) 用于GPT2InferenceModel.forward()的输入嵌入
            attention_mask: (b, s+1) 用于GPT2InferenceModel.generate()的注意力掩码
        """
        # 获取批次大小和文本序列长度
        b, L = text_inputs.shape[:2]
        device = text_inputs.device

        # 检查是否为单一样本的条件（适用于所有文本使用相同语音条件的情况）
        single_cond = conditional_latents.ndim == 3 and conditional_latents.shape[0] == 1
        # 如果不是单一样本条件，确保条件与文本批次大小匹配
        if not single_cond:
            assert conditional_latents.shape[0] == b, f"batch size mismatch: {conditional_latents.shape[0]} vs {b}"

        # 初始化存储列表
        batched_mel_emb = []      # 存储每个样本的mel嵌入
        attention_masks = []      # 存储每个样本的注意力掩码

        # 计算目标长度：条件token数 + 文本token数 + 2（起始和结束文本token）
        target_len = conditional_latents.shape[1] + L + 2

        # 对批次中的每个样本分别处理
        for i in range(b):
            # 创建有效掩码，过滤掉起始和结束文本token
            valid_mask = (text_inputs[i] != self.stop_text_token) & (text_inputs[i] != self.start_text_token)

            # 应用掩码，获取有效的文本输入
            text_input = text_inputs[i][valid_mask]
            # 在文本开头添加起始文本token
            text_input = F.pad(text_input, (1, 0), value=self.start_text_token)
            # 在文本末尾添加结束文本token
            text_input = F.pad(text_input, (0, 1), value=self.stop_text_token)

            # 创建文本位置编码
            text_input_pos = torch.arange(0, text_input.size(-1), device=device)
            text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(text_input_pos)

            # 拼接条件潜在表示和文本嵌入 [conditional latents][text embeddings]
            conds_text_emb = [
                conditional_latents.squeeze(0) if single_cond else conditional_latents[i],
                text_emb,
            ]

            # 创建注意力掩码（目标长度+1，+1是为了起始mel token）
            attention_mask = torch.ones(target_len+1, dtype=torch.long, device=device)

            # 检查文本输入是否有填充
            padding: int = L + 2 - text_input.size(-1)

            # 如果存在填充，在条件嵌入前添加零填充 / pad left of [cond][text] -> [pad][cond][text]
            if padding > 0:
                pad = torch.zeros((padding, conditional_latents.size(-1)), dtype=text_emb.dtype, device=device) # [p, dim]
                conds_text_emb.insert(0, pad)
                attention_mask[:padding] = 0

            # 拼接所有嵌入：[pad(可选)][cond][text]
            mel_emb = torch.cat(conds_text_emb) #[s, dim]
            assert mel_emb.shape[0] == target_len, f"mel_emb.shape: {mel_emb.shape}, target_len: {target_len}"

            # 将处理好的嵌入和掩码添加到列表中
            batched_mel_emb.append(mel_emb)
            attention_masks.append(attention_mask)
        # 将列表中的嵌入堆叠成批次张量 [batch_size, sequence_length, embedding_dim]
        batched_mel_emb = torch.stack(batched_mel_emb, dim=0)
        # 将列表中的注意力掩码堆叠成批次张量 [batch_size, sequence_length+1]
        attention_mask = torch.stack(attention_masks, dim=0)

        # 创建伪输入ID（实际生成时不会使用，但HuggingFace API需要） [batch_size, sequence_length+1]
        fake_inputs = torch.ones(
            (
                batched_mel_emb.shape[0],
                batched_mel_emb.shape[1] + 1,  # +1 for the start_mel_token
            ),
            dtype=torch.long,
            device=device,
        )
        # 在序列末尾设置起始mel token
        fake_inputs[:, -1] = self.start_mel_token
        return fake_inputs, batched_mel_emb, attention_mask

    def inference_speech(self, speech_condition, text_inputs, emo_speech_condition=None, cond_lengths=None, emo_cond_lengths=None, emo_vec=None, use_speed=False, input_tokens=None, num_return_sequences=1,
                         max_generate_length=None, typical_sampling=False, typical_mass=.9, **hf_generate_kwargs):
        """
        推理时对模型输入的条件与文本的预处理

        Args:
            speech_condition: 语音条件特征 (batch, dim, frames) 或 (dim, frames)
            text_inputs: 文本输入 (batch, length)
            emo_speech_condition: 情感语音条件特征
            cond_lengths: 条件mel谱图的长度 (batch,) 或 (1,)
            emo_cond_lengths: 情感条件mel谱图的长度
            emo_vec: 情感向量
            use_speed: 是否使用速度控制
            input_tokens: 用于生成的额外token (batch, seq) 或 (seq,)
            num_return_sequences: 返回的序列数量
            max_generate_length: 生成token的最大数量限制
            typical_sampling: 是否使用典型采样
            typical_mass: 典型采样的质量参数
            hf_generate_kwargs: 传递给GPT2生成模型的参数
        """
        # # ===== 临时取消随机性 =====
        # hf_generate_kwargs['do_sample'] = False
        # hf_generate_kwargs['temperature'] = 1.0
        # hf_generate_kwargs['top_p'] = 1.0
        # hf_generate_kwargs['top_k'] = 0
        # hf_generate_kwargs['num_beams'] = 1
        # # ========================
        # # ===== 打印开始 =====
        # import inspect, pprint
        # frame = inspect.currentframe()
        # args, _, _, values = inspect.getargvalues(frame)
        #
        # print("【inference_speech】调用参数一览")
        # for arg in args:
        #     if arg == "self":
        #         continue
        #     print(f"  {arg:<25} = {values[arg]}")
        # # 单独把 **hf_generate_kwargs 打出来
        # print("  hf_generate_kwargs        =", end=" ")
        # pprint.pprint(hf_generate_kwargs, width=120)
        # # ===== 打印结束 =====

        if speech_condition.ndim == 2:  # 处理单样本情况，添加批次维度
            speech_condition = speech_condition.unsqueeze(0)
        if emo_speech_condition is None:  # 如果没有提供情感语音条件，使用普通语音条件
            emo_speech_condition = speech_condition
        if cond_lengths is None:  # 如果没有提供条件长度，使用语音条件的最后一维大小
            cond_lengths = torch.tensor([speech_condition.shape[-1]], device=speech_condition.device)
        if emo_cond_lengths is None:  # 如果没有提供情感条件长度，使用情感语音条件的最后一维大小
            emo_cond_lengths = torch.tensor([emo_speech_condition.shape[-1]], device=speech_condition.device)

        # 获取语音条件潜在表示
        speech_conditioning_latent = self.get_conditioning(speech_condition.transpose(1,2), cond_lengths)

        # 处理情感向量
        if emo_vec is None:
            print('compute emo vec')
            # 从情感语音条件计算情感向量
            emo_vec = self.get_emo_conditioning(emo_speech_condition.transpose(1,2), emo_cond_lengths)
            emo_vec = self.emovec_layer(emo_vec)  # 情感向量层处理
            emo_vec = self.emo_layer(emo_vec)     # 情感层处理
        else:
            print('Use the specified emotion vector')

        # 创建持续时间嵌入（速度控制相关）
        tmp = torch.zeros(text_inputs.size(0)).to(text_inputs.device)
        duration_emb =  self.speed_emb(torch.zeros_like(tmp).long())        # 正常速度嵌入
        duration_emb_half = self.speed_emb(torch.ones_like(tmp).long())     # 半速嵌入

        # 合并所有条件：语音条件 + 情感向量 + 速度嵌入
        conds_latent = torch.cat((speech_conditioning_latent + emo_vec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)
        # 准备GPT模型的输入
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(conds_latent, text_inputs)

        # # 如果想看具体数值，把下面几行放开即可（只打前 5 列，防止刷屏）
        # if input_ids.numel() > 0:
        #     print("input_ids[0, :5]      =", input_ids[0, :5].tolist())
        # if inputs_embeds.numel() > 0:
        #     print("inputs_embeds[0, :5, 0:3] (前 3 维)=", inputs_embeds[0, :5, 0:3].tolist())
        # if attention_mask.numel() > 0:
        #     print("attention_mask[0, :5] =", attention_mask[0, :5].tolist())

        # 存储mel嵌入到推理模型中
        self.inference_model.store_mel_emb(inputs_embeds)

        # 处理输入token
        if input_tokens is None:
            inputs = input_ids
        else:
            # 处理单样本情况
            if input_tokens.ndim == 1:
                input_tokens = input_tokens.unsqueeze(0)

            # 检查返回序列数量的可除性
            assert num_return_sequences % input_tokens.shape[0] == 0, \
                    "返回序列数量必须能被输入token的批次数量整除"
            assert num_return_sequences % text_inputs.shape[0] == 0, \
                    "返回序列数量必须能被文本输入的批次数量整除"

            # 扩展输入以匹配返回序列数量
            b = num_return_sequences // input_ids.shape[0]
            if b > 1:
                input_ids = input_ids.repeat(b, 1)
                attention_mask = attention_mask.repeat(b, 1)

            # 重复输入token并拼接
            input_tokens = input_tokens.repeat(num_return_sequences // input_tokens.shape[0], 1)
            inputs = torch.cat([input_ids, input_tokens], dim=1)
            attention_mask = F.pad(attention_mask, (0, input_tokens.shape[1]), value=1)

        # 记录截断索引（条件部分的结束位置）
        trunc_index = inputs.shape[1]

        # 创建logits处理器列表
        logits_processor = LogitsProcessorList()
        # 如果使用典型采样，添加对应的处理器
        if typical_sampling:
            # 使用自定义典型采样 / employ custom typical sampling
            if not (typical_mass > 0.0 and typical_mass < 1.0):
                raise ValueError(f"`typical_mass` has to be a float > 0 and < 1, but is {typical_mass}")
            min_tokens_to_keep = 2 if hf_generate_kwargs.get("num_beams", 1) > 1 else 1
            logits_processor.append(TypicalLogitsWarper(mass=typical_mass, min_tokens_to_keep=min_tokens_to_keep))
        # 计算最大生成长度
        max_length = (trunc_index + self.max_mel_tokens - 1) if max_generate_length is None else trunc_index + max_generate_length
        
        # 使用加速引擎（仅支持单序列） / Use accel engine if available (single sequence only)
        if self.accel_engine is not None and num_return_sequences == 1:
            output = self.accel_engine.generate(
                inputs,  # 伪输入ID（全是1 + 起始mel token） / fake input_ids (all 1s + start_mel_token)
                max_new_tokens=max_length - trunc_index,  # 最大新生成token数
                attention_mask=attention_mask,  # 注意力掩码
                temperature=hf_generate_kwargs.get('temperature', 1),  # 温度参数
                stop_tokens=[self.stop_mel_token],  # 停止token
                tts_embeddings=inputs_embeds,  # TTS嵌入 [pad][cond][text] 嵌入 / [pad][cond][text] embeddings (87 tokens, NO start_mel_token)
                tts_mel_embedding=self.inference_model.embeddings,  # mel_embedding layer
                tts_text_pos_embedding=self.inference_model.text_pos_embedding,  # text_pos_embedding layer
            )
        else:
            # 使用标准推理模型生成
            output = self.inference_model.generate(inputs,
                                                bos_token_id=self.start_mel_token, pad_token_id=self.stop_mel_token,
                                                eos_token_id=self.stop_mel_token, attention_mask=attention_mask,
                                                max_length=max_length, logits_processor=logits_processor,
                                                num_return_sequences=num_return_sequences,
                                                **hf_generate_kwargs)

            # # 打印输出
            # print("\n【generate 输出】")
            # print(f"output shape : {output.shape}")  # [B*num_return, seq]
            # print(f"output dtype : {output.dtype}")
            # print(f"output device: {output.device}")
            # if output.numel():
            #     print("output", output)


        # 处理输出结果，移除条件部分
        if isinstance(output, torch.Tensor):
            return output[:, trunc_index:], speech_conditioning_latent  # 返回生成的mel token和语音条件
        # 如果是GenerateOutput对象，同样处理序列
        output.sequences = output.sequences[:, trunc_index:]
        return output, speech_conditioning_latent

    def get_emovec(self, emo_speech_conditioning_latent, emo_cond_lengths):
        emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1,2), emo_cond_lengths)
        emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
        emo_vec = self.emo_layer(emo_vec_syn)
        return emo_vec

    def merge_emovec(self, speech_conditioning_latent, emo_speech_conditioning_latent, cond_lengths, emo_cond_lengths, alpha = 1.0):
        emo_vec = self.get_emovec(emo_speech_conditioning_latent, emo_cond_lengths)
        base_vec = self.get_emovec(speech_conditioning_latent, cond_lengths)

        out = base_vec + alpha * (emo_vec - base_vec)
        return out
