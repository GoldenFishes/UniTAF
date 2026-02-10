'''
25.12.2
这里实现UniTextAudioFace联合模型的训练脚本

UniTAFTrainer继承使用transformers Trainer类。
- 所有的训练配置都从UniTAFTrainer.train_config中获取。
- 数据集使用UniTAFDataset，UniTAFTrainer.setup_dataset() 已实现组装并创建合并的train/val dataset。
- UniTAFTrainer.collate_batch() 和 UniTAFTrainer.compute_losses() 已实现相应的功能。
- UniTAFTrainer.set_model_training_mode() 用于设置模型的训练模式

- 期望实现为模型中tts模块和a2f分别应用各自的loss进行更新。故而：
    尝试重写create_optimizer和create_scheduler为每个模块生成单独的优化器和调度器（创建配置需要根据UniTAFTrainer.train_config）

整个UniTAFTrainer的作用是训练 tts模块和 a2f(包含projector)模块。
我们为tts和a2f分别创建optimizor，并在training_step中手动分别更新两个部分。

'''
import sys
import os
import copy

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # unitaf_train的父目录
sys.path.insert(0, project_root)

# ----进行包修复----
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
# -----------------

from omegaconf import OmegaConf
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import ConcatDataset
from transformers import get_cosine_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence

from transformers import Trainer
from transformers import TrainingArguments
from transformers.trainer_pt_utils import get_parameter_names
import wandb

from peft import LoraConfig, get_peft_model, PeftModel

# UniTextAudioFace的dataset
from unitaf_dataset import UniTAFDataset
# 组装联合模型UniTextAudioFace的类
from UniTAF import UniTextAudioFaceModel

class AverageMeter:
    """最简单的均值累计器"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.
        self.sum = 0.
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.

class MultiOptimizer:
    """包装多个优化器，使其像单个优化器一样工作"""
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def step(self, closure=None):
        """执行每个optimizer的step"""
        for i, opt in enumerate(self.optimizers):
            opt.step(closure)  # 原来的逻辑

    def zero_grad(self, set_to_none=True):
        """清零所有优化器的梯度"""
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """获取所有优化器的状态"""
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        """加载所有优化器的状态"""
        for optimizer, state_dict in zip(self.optimizers, state_dicts):
            optimizer.load_state_dict(state_dict)

class MultiScheduler:
    """包装多个调度器"""
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def step(self):
        """执行所有调度器的step"""
        for scheduler in self.schedulers:
            scheduler.step()

    def get_last_lr(self):
        """获取所有调度器的学习率"""
        return [scheduler.get_last_lr() for scheduler in self.schedulers]

    def state_dict(self):
        """获取所有调度器的状态"""
        return [scheduler.state_dict() for scheduler in self.schedulers]

    def load_state_dict(self, state_dicts):
        """加载所有调度器的状态"""
        for scheduler, state_dict in zip(self.schedulers, state_dicts):
            scheduler.load_state_dict(state_dict)

class UniTAFTrainer(Trainer):
    '''
    UniTextAudioFace 训练器
    1. 为 TTS 与 A2F 分别建立优化器 / scheduler
    2. 重写 training_step，实现「不同模块应用各自loss 各自反向传播更新」
    3. 兼容 AMP、梯度累积、梯度裁剪、断点保存
    '''
    def __init__(self, train_config, model, device, train_dataset, val_dataset):
        self.train_config = train_config
        self.device = device

        # 是否为TTS模块添加LoRA
        if self.train_config["train_tts"] and self.train_config["train_tts_lora"]:
            model = self.apply_lora_to_tts(model)

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.collate_batch,  # 我们用自定义 collate_batch
            args=TrainingArguments(
                output_dir=train_config.get("output_dir", "./unitaf_ckpt"),
                num_train_epochs=train_config["epochs"],
                per_device_train_batch_size=train_config["batch_size"],
                per_device_eval_batch_size=train_config["batch_size"], # 验证batch size暂定与训练一致
                gradient_accumulation_steps=train_config["grad_accumulation"],
                warmup_steps=train_config.get("warmup_steps", 50),
                learning_rate=train_config["tts_train_cfg"]["lr"],  # 这里我们实际上后续会使用.create_optimizer()来重新构建优化器
                fp16=train_config.get("use_amp", False),
                logging_steps=train_config.get("log_interval", 50),
                save_steps=train_config.get("save_interval", 1000),
                eval_steps=train_config.get("val_interval", 500),
                eval_strategy="steps" if train_config.get("val_interval", 0) > 0 else "no",
                report_to=["wandb"] if train_config.get("use_wandb", True) else None,
                remove_unused_columns=False,  # 我们自己写 collate
                dataloader_drop_last=True,
                dataloader_pin_memory=False,  # 关闭。 pin_memory 只能对CPU Tensor操作，我们在dataset、collate等多个阶段均有将张量移到GPU中。
            )
        )

    def compute_losses(self, batch, eval_step=False):
        '''
        compute_losses会在training step和eval step中同时调用compute_losses，
        当为eval_step时，应当将audio feature与 out motion传递出来
        '''
        device = next(self.model.parameters()).device   # 真实设备 而非指定self.device, 因为实际运行时Trainer 会按 accelerate 策略重新分配设备
        loss = {}
        # 1. TTS部分计算出TTS核心输出
        if "IndexTTS2" in self.train_config["tts_model"]:
            """
            对于IndexTTS2部分,参考来源：https://github.com/JarodMica/index-tts/blob/training_v2/trainers/train_gpt_v2.py 中 compute_loss()
                1.从batch中获取原始的codes
                2.进行padding处理
                3.添加stop token
                4.通过 build_aligned_inputs_and_targets 构建最终的mel_targets
                5.计算loss时使用 mel_ce = F.cross_entropy(mel_logits, mel_targets, reduction="none")
            这里使用的self.model.tts_model等价于UnifiedVoice
            """
            use_duration_control = self.train_config["IndexTTS2"]["use_duration_control"]
            duration_dropout = self.train_config["IndexTTS2"]["duration_dropout"]

            tts_condition = batch["tts_condition"].to(device)
            text_ids = batch["text_ids"].to(device)
            audio_codes = batch["audio_codes"].to(device)
            # print("[compute loss] audio_codes", audio_codes.shape)  # torch.Size([16, 479])
            emo_vec = batch["emotion_vector"].to(device)
            text_ids_len = batch["text_ids_len"].to(device)
            audio_codes_len = batch["audio_codes_len"].to(device)

            batch_size = text_ids.size(0)
            use_speed = torch.zeros(batch_size, dtype=torch.long, device=device)

            # (1) 处理成模型输入

            text_inputs = self.model.tts_model.set_text_padding(text_ids.clone(), text_ids_len)
            text_inputs = F.pad(text_inputs, (0, 1), value=self.model.tts_model.stop_text_token)
            text_inputs, text_targets = self.model.tts_model.build_aligned_inputs_and_targets(
                text_inputs, self.model.tts_model.start_text_token, self.model.tts_model.stop_text_token
            )

            # 用stop_token替换零填充 [t1, t2, t3, stop, stop, stop]
            mel_inputs = self.model.tts_model.set_mel_padding(audio_codes.clone(), audio_codes_len)
            # 在末尾再添加一个stop_token [t1, t2, t3, stop, stop, stop, stop]
            mel_inputs = F.pad(mel_inputs, (0, 1), value=self.model.tts_model.stop_mel_token)
            # 构建对齐的输入和目标
            mel_inputs, mel_targets = self.model.tts_model.build_aligned_inputs_and_targets(
                mel_inputs, self.model.tts_model.start_mel_token, self.model.tts_model.stop_mel_token
            )
            # mel_inputs = [start, t1, t2, t3, stop, stop, stop, stop]  ← 模型输入
            # mel_targets = [t1, t2, t3, stop, stop, stop, stop, stop]  ← 真正的GT

            duration_free = self.model.tts_model.speed_emb(torch.zeros_like(use_speed))
            if use_duration_control:
                duration_ctrl = self.model.tts_model.get_duration_embeddings(audio_codes_len)
                if duration_dropout > 0.0:
                    drop_mask = torch.rand(audio_codes_len.size(0), device=device) < duration_dropout
                    if drop_mask.any():
                        duration_ctrl = torch.where(drop_mask.unsqueeze(1), duration_free, duration_ctrl)
            else:
                duration_ctrl = self.model.tts_model.speed_emb(torch.ones_like(use_speed))
            conds = torch.cat(
                (tts_condition + emo_vec.unsqueeze(1), duration_ctrl.unsqueeze(1), duration_free.unsqueeze(1)),
                dim=1,
            )  # tts_condition (B, 32, 1280) 与 emo_vec (B, 1, 1280)

            text_emb = self.model.tts_model.text_embedding(text_inputs) + self.model.tts_model.text_pos_embedding(text_inputs)
            mel_emb = self.model.tts_model.mel_embedding(mel_inputs) + self.model.tts_model.mel_pos_embedding(mel_inputs)

            if self.train_config["train_tts_lora"]:
                '''
                如果是LoRA微调的话，base_model均不会开梯度，只会给LoRA矩阵开梯度。
                但是为了保证正常训练，还需要额外给text_emb/mel_emb开梯度，这样才能够顺利反向传播
                '''
                text_emb = text_emb.requires_grad_(True)
                mel_emb = mel_emb.requires_grad_(True)
            # print('[DEBUG] text_emb.requires_grad:', text_emb.requires_grad,
            #       'mel_emb.requires_grad:', mel_emb.requires_grad)

            # (2) IndexTTS中gpt模型的前向过程
            text_logits, mel_logits, text_latent, mel_latent = self.model.tts_model.get_logits(
                conds, text_emb, self.model.tts_model.text_head, mel_emb, self.model.tts_model.mel_head,
                return_both=True  # 我们于indextts.gpt.model_v2.UnifiedVoice.get_logits()新增该return_both分支，
                # 用于同时返回tts训练用的logits和gt_audio的latent。logits用于计算tts loss，latent用于得到后续audio feature
            )  # 这里返回的mel_latent就是GT对应的latent
            # print('[DEBUG] mel_logits.requires_grad:', mel_logits.requires_grad,
            #       'mel_latent.requires_grad:', mel_latent.requires_grad)

            # (3) 根据gpt前向得到的logits计算loss
            text_mask = (
                    torch.arange(text_targets.size(1), device=device).unsqueeze(0)
                    < (text_ids_len + 1).unsqueeze(1)
            )
            mel_mask = (
                    torch.arange(mel_targets.size(1), device=device).unsqueeze(0)
                    < (audio_codes_len + 1).unsqueeze(1)
            )

            text_ce = F.cross_entropy(text_logits, text_targets, reduction="none")
            mel_ce = F.cross_entropy(mel_logits, mel_targets, reduction="none")
            # mel_targets (B,T), mel_logits (B,T,audio_vocab_size)

            text_loss = (text_ce * text_mask).sum() / text_mask.sum().clamp_min(1)
            mel_loss = (mel_ce * mel_mask).sum() / mel_mask.sum().clamp_min(1)

            metrics = {}
            mel_logits_flat = mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.size(1))
            mel_targets_flat = mel_targets.reshape(-1)
            mel_mask_flat = mel_mask.reshape(-1)
            if mel_mask_flat.any():
                valid_logits = mel_logits_flat[mel_mask_flat]
                valid_targets = mel_targets_flat[mel_mask_flat]
                top1 = (valid_logits.argmax(dim=-1) == valid_targets).float().mean().item()
            else:
                top1 = 0.0
            metrics["mel_top1"] = top1
            # ---

            loss["tts_text_loss"] = text_loss
            loss["tts_mel_loss"] = mel_loss
            loss["tts_metrics"] = metrics

            # (4) 获得GT latent继续IndexTTS生成音频的过程以获取audio_feature
            mel_latent = mel_latent[:, :-2]  # torch.Size([B, L, 1280]) 此时处理后的mel_latent与UnifiedVoice.forward()的返回一样了
            # print("[compute loss] mel_latent", mel_latent.shape)  # torch.Size([16, 479, 1280])

            # 这里 s2mel 不参与更新，在optimizer中排除，但这里不适用with torch.no_grad():以避免中断计算图
            mel_latent = self.model.s2mel.models["gpt_layer"](mel_latent)  # torch.Size([B, L, 1024])
            # print("[compute loss] mel_latent after gpt_layer", mel_latent.shape)  # torch.Size([16, 479, 1024])
            # print('[DEBUG] after s2mel mel_latent.requires_grad:', mel_latent.requires_grad)

        # 在验证步骤中，将后续生成音频波形的所需特征记录到loss字典中
        if eval_step:
            # TODO: 我们在验证中仅传出GT，暂不使用TTS预测的音频token来生成波形。如果要得到预测的波形则需要实例化额外的组件
            loss["gt_waveform"] = batch["gt_waveform"]  # List


            # 2. 获得TTS部分核心输出 处理audio feature
        if "IndexTTS2" in self.train_config["tts_model"] and "UniTalker" in self.train_config["a2f_model"]:
            '''
            1. 获取我们需要的audio_feature： torch.Size([B, L, 1024])
                一般情况下，在IndexTTS2 24000 (24K)音频采样率下, 实际产生的token数为 T(秒) * 50 -1, 可以几乎认为是50个token表示一秒.
                而UniTalker Decoder我们设置帧率为25fps.
            2. 我们需要确保audio_feature与2倍的face_data长度一致
            3. 因此我们在porjector层只需要将获取的音频长度padding一个token后, 降采样到1/2即可与GT Face Data的25fps 同步
            '''
            # (1) 暂定使用 经过s2mel.models["gpt_layer"]处理后的 mel_latent
            audio_feature = mel_latent

            # (2) 为audio_feature的长度 L 增加 n 与2倍的face_data长度一致. (B, L, 1024) -> (B, L+n, 1024), 以便正好被整每秒50个token数整除
            face_frames = batch["face_data"].size(1)
            need_tokens = face_frames * 2

            if audio_feature.size(1) < need_tokens:
                pad_len = need_tokens - audio_feature.size(1)
                audio_feature = F.pad(audio_feature, (0, 0, 0, pad_len), mode='constant', value=0)  # (B, L+n, 1024)
            elif audio_feature.size(1) > need_tokens:
                audio_feature = audio_feature[:, :need_tokens, :]  # 如果 token 比需求多，直接截断（也可插值，但截断最快）

            # (3) 使用额外的projector层，将长度降采样到1/2， (B, L, 1024) -> (B, L/2, 1024)
            audio_feature = self.model.audio_feature_projector(audio_feature)
            # print("[compute loss] audio_feature.shape", audio_feature.shape)  # torch.Size([16, 240, 1024])

            # print("[DEBUG] audio_feature.requires_grad:", audio_feature.requires_grad)  # True
            # print('audio_feature.grad_fn:', audio_feature.grad_fn)  # 且存在grad_fn

            # print('audio_feature  mean/absmean/max :',
            #       audio_feature.mean().item(),
            #       audio_feature.abs().mean().item(),
            #       audio_feature.max().item())  # 检查audio_feature是否纯在数值太小的量级问题


        # 3. A2F Decoder接收audio feature
        if "UniTalker" in self.train_config["a2f_model"]:
            """
            从batch中获得：
                speaker_idx 说话人idx
                face_data 表情数据
                face_template 表情数据格式的偏置
                face_type 表情数据的格式
                face_fps 表情帧率
            """
            face_data = batch["face_data"].to(device, non_blocking=True)
            face_type = batch["face_type"][0]
            identity = batch["speaker_idx"].to(device, non_blocking=True)
            face_template = batch["face_template"].to(device, non_blocking=True)

            if self.train_config["train_a2f_adapter_only"]:
                '''
                这里获取self.model.unitalker_encoder的特征与audio_feature计算loss
                '''
                frame_num = face_data.shape[1]  # 直接使用真实面部运动数据的帧数

                gt_waveform_list = batch["gt_waveform"]  # 获取GT波形的列表 这里列表中的每个音频都是24k hz采样的
                gt_audio = self._batch_waveform_to_tensor(gt_waveform_list)  # 获得16k采样下的音频tensor
                # 获得UniTalker Encoder输出的GT audio feature
                unitalker_encoder_output = self.model.unitalker_encoder(
                    gt_audio, frame_num=frame_num, interpolate_pos=train_config["UniTalker"]["interpolate_pos"])
                # 返回的是 Wav2Vec2BaseModelOutput
                gt_audio_feature = unitalker_encoder_output.last_hidden_state

                # print(f"[Debug] gt_audio_feature.shape = {gt_audio_feature.shape}")  # torch.Size([2, 200, 768]
                # print(f"[Debug] audio_feature.shape = {audio_feature.shape}")  # torch.Size([2, 200, 768]

                # 计算Adapter Feature 与 UniTalker Feature之间的L2 loss
                a2f_adapter_loss = F.mse_loss(audio_feature, gt_audio_feature, reduction='mean')

                loss["a2f_adapter_loss"] = a2f_adapter_loss

            out_motion, out_pca, gt_pca = self.model.a2f_model(
                audio_feature=audio_feature, template=face_template, face_motion=face_data,
                style_idx=identity ,annot_type=face_type,
            )

            # # ===== 关键检查点1：检查A2F输出的梯度状态 =====
            # print("\n[DEBUG] 检查A2F输出梯度状态:")  # 梯度状态正常
            # print(f"out_motion.requires_grad: {out_motion.requires_grad}")
            # print(f"out_motion.shape: {out_motion.shape}")
            # if out_motion.grad_fn is not None:
            #     print(f"out_motion.grad_fn: {out_motion.grad_fn}")
            #     # 追溯梯度链
            #     node = out_motion.grad_fn
            #     depth = 0
            #     while node is not None and depth < 3:
            #         print(f"  L{depth}: {str(node)[:80]}")
            #         node = node.next_functions[0][0] if node.next_functions else None
            #         depth += 1
            # else:
            #     print("❌ out_motion.grad_fn 是 None!")

            # TODO:计算loss的时候带上batch["face_data_len"]来生成有效部分的mask
            # 由我们单独实现的unitalker_decoder_component.UniTalkerDecoder.compute_loss()中计算loss
            rec_loss, pca_loss = self.model.a2f_model.compute_loss(out_motion, face_data, face_type, out_pca, gt_pca)

            loss["a2f_rec_loss"] = rec_loss
            loss["a2f_pca_loss"] = pca_loss

            # 在验证步骤中，将模型生成的out_motion和gt表情记录到loss字典里
            if eval_step:
                loss["out_motion"] = out_motion
                loss["gt_motion"] = face_data

        return loss

    def collate_batch(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        '''合并成tensor时需要padding为等长序列'''
        output = {}
        output["sample_id"] = [item["sample_id"] for item in batch]

        if "IndexTTS2" in self.train_config["tts_model"]:
            '''
            接收到batch
                条件音频token      tts_condition,tts_condition_len
                文本token         text_ids,text_ids_len
                GT音频token       audio_codes,audio_codes_len
                情感控制向量        emotion_vector
                
                GT音频波形         gt_waveform   仅供在验证步骤中使用
            '''
            text_tensors = [item["text_ids"] for item in batch]
            code_tensors = [item["audio_codes"] for item in batch]    # 这就是GT音频的离散编码
            condition_tensors = [item["tts_condition"] for item in batch]
            emo_tensors = [item["emotion_vector"] for item in batch]

            # 对变长序列进行填充（padding）
            text_padded = pad_sequence(text_tensors, batch_first=True, padding_value=0)
            code_padded = pad_sequence(code_tensors, batch_first=True, padding_value=0)  # 音频token序列padding：这些0在后续的set_mel_padding中会被替换为stop_token
            condition_stacked = torch.stack(condition_tensors, dim=0)  # 语音条件特征堆叠：形状从 [D] -> [B, D]
            emo_stacked = torch.stack(emo_tensors, dim=0)  # 情感向量堆叠：形状从 [E] -> [B, E]

            # 提取序列长度信息（用于masking和损失计算）
            text_lengths = torch.stack([item["text_ids_len"] for item in batch])
            code_lengths = torch.stack([item["audio_codes_len"] for item in batch])
            cond_lengths = torch.stack([item["tts_condition_len"] for item in batch])

            output["tts_condition"] = condition_stacked  # (B, 32, 1280)
            output["tts_condition_len"] = cond_lengths  # (B)
            output["text_ids"] = text_padded  # (B, L)
            output["text_ids_len"] = text_lengths  # (B)
            output["audio_codes"] = code_padded  # (B, L)
            output["audio_codes_len"] = code_lengths  # (B)
            output["emotion_vector"] = emo_stacked  # (B, 1280)

            output["gt_waveform"] = [item["gt_waveform"] for item in batch]  # 采样率均为24000

            # print(f"[collate_batch] tts_condition: {condition_stacked.shape} ")
            # print(f"[collate_batch] tts_condition_len: {cond_lengths.shape}")
            # print(f"[collate_batch] text_ids: {text_padded.shape}")
            # print(f"[collate_batch] text_ids_len: {text_lengths.shape}")
            # print(f"[collate_batch] audio_codes: {code_padded.shape}")
            # print(f"[collate_batch] audio_codes_len: {code_lengths.shape}")
            # print(f"[collate_batch] emotion_vector: {emo_stacked.shape}")

        if "UniTalker" in self.train_config["a2f_model"]:
            '''
            接收到batch
                说话人idx     speaker_idx
                表情数据      face_data
                表情数据的偏置  face_template
                表情标注格式   face_type
                表情帧率      face_fps
            '''
            # TODO：检查这里padding方式是否与UniTalker训练代码中一致
            face_tensors = [item["face_data"] for item in batch]  # List[Tensor[T, C]]
            tmpl_tensors = [item["face_template"] for item in batch]  # 等长，可 stack
            type_tensors = [item["face_type"] for item in batch]
            fps_tensors = [item["face_fps"] for item in batch]
            speaker_tensors = [item["speaker_idx"] for item in batch]

            # 变长维度 T 做 pad
            face_padded = pad_sequence(face_tensors, batch_first=True, padding_value=0.0)  # [B, T_max, C]
            # 同时返回长度
            face_lengths = torch.tensor([f.size(0) for f in face_tensors], dtype=torch.long)

            output["face_data"] = face_padded  # padding后的face data
            output["face_data_len"] = face_lengths  # face data 有效长度
            output["face_template"] = torch.stack(tmpl_tensors, dim=0)
            output["face_type"] = type_tensors  # 列表即可，后面用索引时再转 tensor
            output["face_fps"] = torch.stack(fps_tensors, dim=0)
            output["speaker_idx"] = torch.stack(speaker_tensors, dim=0)

        return output

    def apply_lora_to_tts(self, model):
        '''
        训练tts且使用LoRA训练时，返回peft model
        '''
        assert self.train_config["train_tts"] == True and self.train_config["train_tts_lora"] == True
        # 只给 TTS 模型加 LoRA
        lora_config = LoraConfig(
            r=self.train_config["tts_lora_cfg"].get("lora_rank", 16),
            lora_alpha=self.train_config["tts_lora_cfg"].get("lora_alpha", 32),
            target_modules=self.train_config["tts_lora_cfg"].get("lora_target_modules"),
            lora_dropout=self.train_config["tts_lora_cfg"].get("lora_dropout", 0.1),
            bias="none",
            task_type="FEATURE_EXTRACTION",  # 或无特定任务
        )

        model.tts_model = get_peft_model(model.tts_model, lora_config)
        print("[apply_lora_to_tts] TTS LoRA可训练参数量：")
        model.tts_model.print_trainable_parameters()  # 打印 LoRA 参数量
        return model

    def set_model_training_mode(self, model):
        """
        设置各部分模型的训练模式，
        固定包含tts_model，a2f_model 可能包含 audio_feature_projector 与 tts的后续波形生成模块

        设置模式直接 model.train() / model.eval() 即可，不必要再手动设置参数param.requires_grad = True / False
        """
        if self.train_config["train_tts"]:
            model.tts_model.train()
            if self.train_config["train_tts_lora"]:
                # TTS-LoRA部分需要训练
                for name, param in model.tts_model.named_parameters():
                    if "lora_A" in name or "lora_B" in name:
                        param.requires_grad = True
                # # 打印确认
                # print("[set_model_training_mode] LoRA 可训练参数:")
                # for name, p in model.tts_model.named_parameters():
                #     if p.requires_grad:
                #         print("  ", name)
            else:
                # TTS部分需要训练
                for param in model.tts_model.parameters():
                    param.requires_grad = True
        else:
            # TTS部分不需要训练，设置为eval模式并冻结
            model.tts_model.eval()
            for param in model.tts_model.parameters():
                param.requires_grad = False

        if self.train_config["train_a2f"]:
            # A2F部分需要训练
            model.a2f_model.train()
            for param in model.a2f_model.parameters():
                param.requires_grad = True
            # 若同时存在音频特征投影层，也同步打开
            if hasattr(self.model, 'audio_feature_projector'):
                self.model.audio_feature_projector.train()
                for param in self.model.audio_feature_projector.parameters():
                    param.requires_grad = True
        else:
            # A2F部分不需要训练
            model.a2f_model.eval()
            for param in model.a2f_model.parameters():
                param.requires_grad = False
            # 若同时存在音频特征投影层，也同步关闭
            if hasattr(self.model, 'audio_feature_projector'):
                self.model.audio_feature_projector.eval()
                for param in self.model.audio_feature_projector.parameters():
                    param.requires_grad = False

        if "IndexTTS2" in self.train_config["tts_model"]:
            # IndexTTS2的s2mel部分始终冻结，不参与训练
            model.s2mel.eval()  # 设置为eval模式
            for param in model.s2mel.parameters():
                param.requires_grad = False

    def create_optimizer(self):
        """
        重写optimizer创建逻辑，支持多个优化器。
        为联合模型中tts和a2f设置独立的优化器，会在训练前调用一次。
        """
        if hasattr(self, "optimizer") and self.optimizer is not None:
            return  # 已经创建过

        opt_list = []
        # ---- TTS ----
        if self.train_config["train_tts"]:
            cfg = self.train_config["tts_train_cfg"]
            tts_params = list(self.model.tts_model.parameters())
            opt_list.append(AdamW(tts_params,
                                  lr=cfg.get("lr", 2e-5),
                                  betas=cfg.get("betas", (0.9, 0.999)),
                                  weight_decay=cfg.get("weight_decay", 0.01),
                                  eps=cfg.get("eps", 1e-8)))
        # ---- A2F 与 投影层----
        if self.train_config["train_a2f"]:
            cfg = self.train_config["a2f_train_cfg"]
            # 如果只训练a2f adapter,则只包括audio_feature_projector
            if self.train_config["train_a2f_adapter_only"]:
                a2f_params = list(self.model.audio_feature_projector.parameters())
            # 如果不训练a2f adapter,则只包括a2f_model
            elif self.train_config["train_except_a2f_adapter"]:
                a2f_params = list(self.model.a2f_model.parameters())
            # 否则audio_feature_projector和a2f_model都参与训练
            else:
                a2f_params = list(self.model.a2f_model.parameters())
                a2f_params.extend(self.model.audio_feature_projector.parameters())

            opt_list.append(AdamW(a2f_params,
                                  lr=cfg.get("lr", 2e-5),
                                  betas=cfg.get("betas", (0.9, 0.999)),
                                  weight_decay=cfg.get("weight_decay", 0.01),
                                  eps=cfg.get("eps", 1e-8)))

        self.optimizer = MultiOptimizer(opt_list)  # 将多个优化器打包成一个

        # print(f"[DEBUG] - 传入optimizer的opt_list:{opt_list}")
        # for i, opt in enumerate(self.optimizer.optimizers):
        #     print(f"opt[{i}] 参数数:", sum(p.numel() for p in opt.param_groups[0]['params']))

        # ---------------- 调试打印 ----------------
        from itertools import chain
        for i, opt in enumerate(self.optimizer.optimizers):
            params = list(chain.from_iterable(g['params'] for g in opt.param_groups))
            # 根据你 append 的顺序，i==0 一定是 TTS，i==1 一定是 A2F
            name = 'TTS' if i == 0 and self.train_config.get("train_tts") else \
                'A2F' if self.train_config.get("train_a2f") else f'opt[{i}]'
            print(f"[Optimizer] {name} 可优化参数数量: {sum(p.numel() for p in params):,}")
            if self.train_config.get("debug_param_names", False):  # 想打名字就额外开开关
                for p in params:
                    print("   ", p.shape, p.name if hasattr(p, 'name') else "")

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        为每个优化器创建独立的调度器，会在训练前调用一次
        """
        if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
            return

        warmup_steps = self.train_config.get('warmup_steps', 50)
        schedulers = []
        for opt in self.optimizer.optimizers:
            # 统一用 cosine + warmup
            sch = get_cosine_schedule_with_warmup(
                opt,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps)
            schedulers.append(sch)
        self.lr_scheduler = MultiScheduler(schedulers)

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        """
        各自反传、各自 step，返回标量 loss 给 Trainer 做日志
        这里不会用到Trainer传入的num_items_in_batch。

        TTS模型 → (audio_feature) → A2F模型
          ↓                          ↓
        tts_loss                 a2f_loss

        a2f_loss同时依赖TTS和A2F，需要避免使用a2f_loss反传时，tts的梯度被清空
        """
        self.set_model_training_mode(model)  # 设置模型训练模式
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        # 1. 前向并获得loss
        with self.compute_loss_context_manager():
            loss_dict = self.compute_losses(inputs)

        # # 在反向传播前，检查计算图 ------------------------------------------
        # print("[training_step] 检查计算图连接")
        # if self.train_config["train_a2f"] and "tts_mel_loss" in loss_dict:  # 检查TTS输出是否连接到A2F输入
        #     tts_mel_loss = loss_dict["tts_mel_loss"]  # 假设这是TTS的输出
        #     print(f"TTS输出requires_grad: {tts_mel_loss.requires_grad}")
        #     print(f"TTS输出grad_fn: {tts_mel_loss.grad_fn}")
        # # ---------------------------------------------------------------

        # 2. 计算各自 loss
        tts_loss = torch.tensor(0., device=self.device)
        a2f_loss = torch.tensor(0., device=self.device)
        if self.train_config["train_tts"]:
            if "IndexTTS2" in self.train_config["tts_model"]:
                tts_loss = (self.train_config["IndexTTS2"]["text_loss_weight"] * loss_dict["tts_text_loss"] +
                            self.train_config["IndexTTS2"]["mel_loss_weight"] * loss_dict["tts_mel_loss"])
        if self.train_config["train_a2f"]:
            if "UniTalker" in self.train_config["a2f_model"]:
                if self.train_config["train_a2f_adapter_only"]:  # 只训练Adapter时，更新a2f部分用a2f_adapter_loss，此时的a2f部分优化器中也只有Adapter部分参数
                    a2f_loss = loss_dict["a2f_adapter_loss"]
                else:
                    a2f_loss = (loss_dict["a2f_rec_loss"] +
                                self.train_config["UniTalker"]["pca_weight"] * loss_dict["a2f_pca_loss"])
        total_loss = tts_loss + a2f_loss

        # 3. 各自反向
        use_amp = self.train_config["use_amp"] and torch.cuda.is_available()

        # 3-a TTS 反传
        if self.train_config["train_tts"] and tts_loss.item() != 0:

            # # 分离A2F相关参数，防止被TTS loss影响
            # if self.train_config["train_a2f"] and a2f_loss.item() != 0:
            #     with torch.no_grad():
            #         a2f_grad_states = {}  # 备份A2F参数梯度状态
            #         for name, param in self.model.a2f_model.named_parameters():
            #             if param.requires_grad:
            #                 a2f_grad_states[name] = param.grad

            # tts反向传播
            if use_amp:
                with torch.cuda.amp.autocast():
                    tts_loss.backward(retain_graph=True)  # 保留梯度给 A2F 用
            else:
                tts_loss.backward(retain_graph=True)  # 保留梯度给 A2F 用

            # # 恢复A2F梯度状态
            # if self.train_config["train_a2f"] and a2f_loss.item() != 0:
            #     with torch.no_grad():
            #         for name, param in self.model.a2f_model.named_parameters():
            #             if param.requires_grad and name in a2f_grad_states:
            #                 param.grad = a2f_grad_states[name]

            # # 打印 TTS 梯度 ---------------------------------------------------
            # print("[training_step] TTS grad sample:")
            # for name, p in model.tts_model.named_parameters():
            #     if p.grad is not None:
            #         print("✅", name[:60], p.grad.norm().item())
            # # ----------------------------------------------------------------

        # 3-b A2F 反传
        if self.train_config["train_a2f"] and a2f_loss.item() != 0:
            # a2f反向传播
            if use_amp:
                with torch.cuda.amp.autocast():
                    a2f_loss.backward()  # A2F 最后一次，不需要再 retain graph
            else:
                a2f_loss.backward()  # A2F 最后一次，不需要再 retain graph

            # # 打印 A2F 梯度 ---------------------------------------------------
            # print("[training_step] A2F grad sample:")
            # for name, p in model.a2f_model.named_parameters():
            #     if p.grad is not None:
            #         print("✅", name[:60], p.grad.norm().item())
            # # ----------------------------------------------------------------

            # ===== 立刻检查 A2F 梯度 =====
            # print("[training_step] A2F 梯度统计")
            # total, zero, tiny = 0, 0, 0
            # for name, p in self.model.a2f_model.named_parameters():
            #     if p.grad is None:
            #         continue
            #     total += 1
            #     g_norm = p.grad.norm().item()
            #     if g_norm == 0:
            #         zero += 1
            #     elif g_norm < 1e-6:
            #         tiny += 1
            # print(f"A2F 总参数 {total} | 零梯度 {zero} | 极小梯度 {tiny}")  # A2F 总参数 19 | 零梯度 0 | 极小梯度 0

        # 日志记录tts和a2f各模块的梯度
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            log_dict={}
            with torch.no_grad():
                '''
                用 unsqueeze(0) 把标量变成 1-D 张量，避免 stack 抱怨空列表；列表为空时返回 0.0 占位，日志里就能正常显示。
                (在只训练adapter时，不更新a2f参数，不存在a2f梯度但是日志会仍然尝试记录)
                '''
                if self.train_config["train_tts"]:
                    grads = [p.grad.norm().unsqueeze(0) for p in self.model.tts_model.parameters()
                             if p.grad is not None]
                    tts_norm = torch.norm(torch.cat(grads)) if grads else torch.tensor(0.0, device=self.args.device)
                    log_dict["grad_norm/tts"] = tts_norm.item()
                if self.train_config["train_a2f"]:
                    grads = [p.grad.norm().unsqueeze(0) for p in self.model.a2f_model.parameters()
                             if p.grad is not None]
                    a2f_norm = torch.norm(torch.cat(grads)) if grads else torch.tensor(0.0, device=self.args.device)
                    log_dict["grad_norm/a2f"] = a2f_norm.item()
            self.log(log_dict)


        # 4. 梯度裁剪
        grad_clip = self.train_config.get("grad_clip", 0)
        if grad_clip > 0:
            if self.train_config["train_tts"]:
                torch.nn.utils.clip_grad_norm_(
                    self.model.tts_model.parameters(),
                    grad_clip
                )
            if self.train_config["train_a2f"]:
                torch.nn.utils.clip_grad_norm_(
                    self.model.a2f_model.parameters(),
                    grad_clip
                )

        # # ===== 全量扫描 A2F 梯度 =====
        # print("[DEBUG] A2F 全梯度扫描（backward 后）")
        # zero_count = 0
        # for name, p in self.model.a2f_model.named_parameters():
        #     if p.grad is None:
        #         continue  # 跳过无图
        #     g_norm = p.grad.norm().item()
        #     if g_norm == 0.0:
        #         zero_count += 1
        #         print(f"❌ ZERO  {name[:70]}  norm={g_norm}")
        #     elif g_norm < 1e-7:
        #         print(f"⚠️ 极小 {name[:70]}  norm={g_norm}")
        #     else:
        #         # 只打印第一条非零梯度，作为「图连通」证据
        #         if zero_count == 0:
        #             print(f"✅ OK    {name[:70]}  norm={g_norm}")
        # print(f"[DEBUG] A2F 中 norm=0 的参数共 {zero_count} 个")

        # 5. 将各个模块的loss也记录到日志中
        if (self.state.global_step + 1 ) % self.args.logging_steps == 0:
            log_dict = {}  # 先收集 loss_dict 里所有叶子标量

            def _collect(key, val):
                if isinstance(val, dict):
                    for k, v in val.items():
                        _collect(f"{key}_{k}", v)
                elif isinstance(val, torch.Tensor) and val.numel() == 1:
                    log_dict[key] = val.detach().item()
                elif isinstance(val, (int, float)):
                    log_dict[key] = float(val)

            for k, v in loss_dict.items():
                _collect(k, v)
            # 再加加权 loss
            log_dict["tts_loss"] = tts_loss.detach().item()
            log_dict["a2f_loss"] = a2f_loss.detach().item()
            # 一次发给 Trainer（它会自动加前缀 train_ ）
            self.log(log_dict)

        # # ===== 训练步末尾：打印同一层权重，确保参数有更新 =====
        # if self.state.global_step % self.args.logging_steps == 0:
        #     # 选一条“代表层”即可
        #     p = self.model.a2f_model.decoder.tcn.first_net.conv_layers[0].conv.weight[0, :5]  # 只取前 5 维，防止刷屏
        #     print(f"[step={self.state.global_step}] "
        #           f"style_emb[0,:5] = {p.detach().cpu().tolist()}  "
        #           f"grad_norm = {p.grad.norm().item() if p.grad is not None else 'None'}")

        return total_loss.detach()      # 返回标量给 Trainer 做日志

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        覆盖Trainer.evaluate，遍历一遍eval_dataset并收集评估指标（主要是各个部分的loss）

        - 评估过程中对结果进行渲染可视化：
            对每次评估集遍历的首个Batch中的样本，我们会渲染GT音频下GT表情与预测表情的对比视频，并保存于outputs/evaluate中
            此时调用self.compute_losses时指定eval_step=True以保证GT音频与GT表情的顺利获取。

        """
        # 1. 准备模型和验证集
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("evaluate 时需要 eval_dataset")
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        self.model.eval()

        # meter_names = ["tts_loss", "a2f_loss", "total_loss"]  # 初始这三个指标，后根据返回的loss_dict扩容
        meters = defaultdict(AverageMeter)

        has_saved_render = False  # 用与标记是否已经可视化过一个batch的样本

        # 2. 遍历验证集
        for batch in tqdm(eval_dataloader, desc=f"Eval bsz={self.args.eval_batch_size}"):
            '''
            在eval_dataloader中获取到的batch是已经经过collate_batch的已经处理成模型输入的了
            '''
            # 2.1 前向并获得loss
            with self.compute_loss_context_manager():
                loss_dict = self.compute_losses(batch, eval_step=True)

            # 2.2 计算各自 loss
            tts_loss = torch.tensor(0., device=self.device)
            a2f_loss = torch.tensor(0., device=self.device)
            if self.train_config["train_tts"]:
                if "IndexTTS2" in self.train_config["tts_model"]:
                    tts_loss = (self.train_config["IndexTTS2"]["text_loss_weight"] * loss_dict["tts_text_loss"] +
                                self.train_config["IndexTTS2"]["mel_loss_weight"] * loss_dict["tts_mel_loss"])
            if self.train_config["train_a2f"]:
                if "UniTalker" in self.train_config["a2f_model"]:
                    a2f_loss = (loss_dict["a2f_rec_loss"] +
                                self.train_config["UniTalker"]["pca_weight"] * loss_dict["a2f_pca_loss"])
            total_loss = tts_loss + a2f_loss

            # 2.3 记录所有标量
            for k, v in loss_dict.items():  # 添加loss_dict中所有量
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    meters[k].update(v.item())
                elif isinstance(v, dict):  # 处理嵌套 dict，例如 tts_metrics
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float, torch.Tensor)):
                            meters[f"{k}_{sub_k}"].update(float(sub_v))
            # 添加加权 loss
            bsz = batch["face_data"].size(0)
            meters["tts_loss"].update(tts_loss.item(), bsz)
            meters["a2f_loss"].update(a2f_loss.item(), bsz)
            meters["total_loss"].update(total_loss.item(), bsz)

            # 2.4 可视化渲染
            if has_saved_render is not True:  # 用于控制在evaluate遍历中只渲染一个batch的样本
                '''
                我们已经在训练step中通过Loss字典传递出：
                loss_dict["gt_waveform"]  gt 音频波形  List
                loss_dict["gt_motion"]    gt 表情数据  tensor
                loss_dict["out_motion"]   模型预测表情数据  tensor 
                
                遍历loss_dict["gt_waveform"]列表，对batch下每一个样本单独获取其 gt音频/gt表情/预测表情 三元组
                '''
                gt_waveform_list = batch["gt_waveform"]  # GT 音频 从batch中获得
                batch_size = len(gt_waveform_list)

                # 遍历每个样本
                for i in range(batch_size):
                    gt_waveform = gt_waveform_list[i]  # 音频波形
                    gt_motion = loss_dict["gt_motion"][i]  # GT 表情  从loss_compute中获得
                    out_motion = loss_dict["out_motion"][i]  # 预测表情  从loss_compute中获得
                    sample_id = batch["sample_id"][i]  # 样本id   从batch中获得

                    self._render_and_save(
                        gt_waveform=gt_waveform,
                        gt_motion=gt_motion,
                        out_motion=out_motion,
                        annot_type="qxsk_inhouse_blendshape_weight",  # TODO:这里根据实际annot_type传递
                        # 保存于输出路径下 evaluate 文件夹中
                        output_video_path=Path(f"{self.train_config['output_dir']}/evaluate/step_{self.state.global_step}/{sample_id}_{i}.mp4")
                    )

                has_saved_render = True


        # 4. 加前缀 & 过滤
        metrics = {}
        for k, meter in meters.items():
            if ignore_keys and k in ignore_keys:
                continue
            metrics[f"{metric_key_prefix}_{k}"] = meter.avg

        self.log(metrics)
        return metrics

    def _render_and_save(self, gt_waveform, gt_motion, out_motion, annot_type, output_video_path):
        '''
        传入单条样本的GT音频路径，GT表情数据和预测表情数据，调用渲染方法进行渲染并保存
        '''
        from unitaf_train_component.render import render_video_for_evaluate
        # 获取表情的顶点数据
        out_motion_vertics = self.model.a2f_model.loss_module.get_vertices(out_motion.cuda(),
                                                            annot_type=annot_type)
        gt_motion_vectics = self.model.a2f_model.loss_module.get_vertices(gt_motion.cuda(),
                                                        annot_type=annot_type)

        # 确保输出路径存在
        output_video_path.parent.mkdir(parents=True, exist_ok=True)   # 递归建目录

        render_video_for_evaluate(
            gt_waveform=gt_waveform,
            gt_motion=gt_motion_vectics,
            out_motion=out_motion_vertics,
            annot_type=annot_type,
            output_video_path=output_video_path
        )

    def _save_checkpoint(self, model, trial, metrics=None):
        '''
        重写Trainer._save_checkpoint()保存逻辑
        只保存UniTAF.state_dict()实现的指定的部分模块权重
        # TODO:尚未验证加载保存的权重是否有误，训练中加载保存的权重/推理中加载保存的权重（推理脚本未实现）
        '''
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR  # "checkpoint"
        output_dir = os.path.join(self.args.output_dir,
                                  f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")  # 保证和官方权重保存文件夹一致
        os.makedirs(output_dir, exist_ok=True)

        # 1. TTS 权重
        if hasattr(model, 'tts_model') and any(p.requires_grad for p in model.tts_model.parameters()):
            print("[UniTAFTrainer._save_checkpoint] 准备保存tts")
            tts_path = os.path.join(output_dir, "tts_model.pt")
            # 如果是LoRA训练
            if hasattr(model.tts_model, 'merge_and_unload'):   # PeftModel 的标志
                # print("尝试保存合并后的LoRA tts")
                with torch.no_grad():  # 这里创建拷贝来合并模型，而非在原有模型上合并，以防止合并后优化器指向错误模型。
                    merge_copy = copy.deepcopy(model.tts_model)
                    merged_base = merge_copy.merge_and_unload()
                    torch.save(merged_base.state_dict(), tts_path)
                    del merge_copy, merged_base                       # 释放显存
            else:
                torch.save(model.tts_model.state_dict(), tts_path)

        # 2. A2F 权重（不含 loss_module）
        if hasattr(model, 'a2f_model') and any(p.requires_grad for p in model.a2f_model.parameters()):
            # print("[UniTAFTrainer._save_checkpoint] 准备保存a2f")
            a2f_sd = {k: v for k, v in model.a2f_model.state_dict().items() if 'loss_module' not in k}
            torch.save(a2f_sd, os.path.join(output_dir, "a2f_model.pt"))

        # 3. 投影层
        if hasattr(model, 'audio_feature_projector') and any(p.requires_grad for p in model.audio_feature_projector.parameters()):
            # print("[UniTAFTrainer._save_checkpoint] 准备保存projector")
            torch.save(model.audio_feature_projector.state_dict(),
                       os.path.join(output_dir, "audio_feature_projector.pt"))

        # 4. 让父类继续保存 optimizer / scheduler / args 等
        # 根据签名决定怎么调父类
        sig = inspect.signature(super()._save_checkpoint)
        if 'metrics' in sig.parameters:
            super()._save_checkpoint(model, trial, metrics=metrics)
        else:
            super()._save_checkpoint(model, trial)

    def _batch_waveform_to_tensor(self, wave_list, target_len=None):
        """
        专门为UniTalkerEncoder准备GT音频输入，
        以获取其在UniTalker空间下的输出Feature用于约束UniTAF Adapter产生的Feature

        wave_list: list[np.ndarray] 或 list[Tensor]，每个 shape (T,)
        return: Tensor[B, L]  16 kHz，float32，已归一化到 [-1,1]
        """
        resampled = []
        for w in wave_list:
            if isinstance(w, np.ndarray):
                w = torch.from_numpy(w).float()
            w = w.to(self.device)
            w = self.model.resample(w).squeeze(0)  # (T_16k,)
            # print("w shape",w.shape)
            resampled.append(w)

        # 2. 统一长度
        if target_len is None:  # 默认取最长
            target_len = max(w.shape[0] for w in resampled)

        padded = []
        for w in resampled:
            if w.shape[0] > target_len:
                w = w[:target_len]
            else:
                w = torch.nn.functional.pad(w, (0, target_len - w.shape[0]))
            padded.append(w)

        audio = torch.stack(padded, dim=0)  # [B, L]
        # print("audio shape", audio.shape)
        return audio

def setup_dataset(train_config):  # 支持多数据集训练，对应unitaf_dataset_support_config中具体数据集
    '''
    在这里为每个dataset 例如"D11","D12" ;以及为每个dataset_type 例如"train","val"，都创建数据集并合并
    最终返回总体合并后的train/val dataset
    '''
    # 1. 组装dataset config
    dataset_config = {}
    dataset_config["tts_model"] = train_config["tts_model"]
    dataset_config["a2f_model"] = train_config["a2f_model"]
    dataset_config["dataset_root_path"] = train_config["dataset_config"]["dataset_root_path"]
    dataset_config["device"] = train_config["device"]

    train_datasets = []
    val_datasets = []

    # 2. 遍历所有子数据集
    for subdataset in train_config["dataset_config"]["dataset_list"]:
        try:
            train_dataset = UniTAFDataset(dataset_config, dataset_name=subdataset, dataset_type="train")
            train_datasets.append(train_dataset)
        except Exception as e:
            print(f"[UniTAFTrainer][setup_dataset] 数据集{subdataset}没有train训练集，跳过。\n Error:{e}")
        try:
            val_dataset = UniTAFDataset(dataset_config, dataset_name=subdataset, dataset_type="val")
            val_datasets.append(val_dataset)
        except Exception as e:
            print(f"[UniTAFTrainer][setup_dataset] 数据集{subdataset}没有val验证集，跳过。\n Error:{e}")

    # 3. 合并dataset
    if len(train_datasets) > 1:
        combined_train_dataset = ConcatDataset(train_datasets)
    else:
        combined_train_dataset = train_datasets[0]
    if len(val_datasets) > 1:
        combined_val_dataset = ConcatDataset(val_datasets)
    else:
        combined_val_dataset = val_datasets[0]

    return combined_train_dataset, combined_val_dataset

def main(train_config):
    # 1. 初始化dataset
    train_dataset, val_dataset = setup_dataset(train_config)
    total_steps = len(train_dataset) * train_config["epochs"] // train_config["grad_accumulation"]

    # 2. 实例化基础联合模型（Trainer 需要）
    device = torch.device(train_config["device"] if torch.cuda.is_available() else "cpu")
    model = UniTextAudioFaceModel(cfg=train_config, device=device, mode="train")

    # 初始化Trainer
    trainer = UniTAFTrainer(
        train_config=train_config,
        model=model,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    # 设置UniTAF实现的优化器与调度器
    trainer.create_optimizer()
    trainer.create_scheduler(total_steps)

    # # DEBUG
    # print("Optimizer param groups:", len(trainer.optimizer.optimizers))  # 2
    # print("Total trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))  # 972972716

    # 开始训练
    print("[UniTAFTrainer] 开始训练...")
    trainer_stats = trainer.train()
    print("[UniTAFTrainer] 训练结束...")


if __name__ == '__main__':
    '''
    python unitaf_train/train_unitaf.py 
    '''
    # 设置使得Trainer限制在固定卡上
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只用第 0 号卡，此时会从可见卡开始编码cuda:0,cuda:1...

    train_config = {
        # 模型类型，这里用于指导训练器类训练哪些模型
        "tts_model": ["IndexTTS2"],
        "a2f_model": ["UniTalker"],

        # 为上面模型类型中包含的模型进行配置----------------------------------------------------------------------------------
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
            # 以下需要参数需要根据实际情况更新：
            "audio_encoder_feature_dim": 768,  # 原始UniTalker-L-D0-D7.pt接收特征维度是1024，UniTalker-B-D0-D7.pt接收特征维度768。我们需要经过projector使得音频特征输出相同维度
            "identity_num": 20,  # 暂定为20，实际在UniTalker Decoder权重加载时会从权重中得到这里的值并更新
        },
        "expression_model": {  # 提供从口型到完整面部的情感表情残差
            "mode": "cross_attn", # "film"情感通过FiLM控制, "cross_attn"情感通过CrossAttention控制
            # FiLM参数
            "hidden_dim": 256,
            "num_layers": 4,
            # Cross Attn参数
            "nhead": 4,
            "dropout": 0.1,
        },
        # 数据集类-------------------------------------------------------------------------------------------------------
        "dataset_config": {
            "dataset_root_path": "/home/zqg/project/data/UniTAF Dataset",  # 使用绝对路径
            # 支持多数据集训练，对应unitaf_dataset_support_config中具体数据集
            "dataset_list": ["D13"],  # "D12" , "D13"情感控制
            # unitaf_dataset_support_config是经过数据集格式转换UniTAFDataset能够支持的数据集
            # （UniTalker本身还有一个 a2f/dataset/dataset_config 记录UniTalker Decoder支持的数据集，
            # 但我们只会以 unitaf_dataset_support_config 为准）
        },
        # 设备
        "device": "cuda:0",  # 注: 需要与外部的CUDA_VISIBLE_DEVICES一致!但只有一个可见卡时，硬编码则应该是cuda：0
        # 训练设置-------------------------------------------------------------------------------------------------------
        "batch_size": 2,
        "epochs": 10,
        "grad_accumulation": 1,
        "grad_clip": 1.0,
        "use_amp": True,
        "warmup_steps": 50, # 所有调度器统一参数 warmup 为50
        "log_interval": 2,  # training_step 里打印 loss 的步长         # 20
        "val_interval": 2000,  # 每隔多少 step 做一次验证               # 2000
        "save_interval": 20000,  # 每隔多少 step 存一次 ckpt              # 20000
        "output_dir": "./unitaf_ckpt/UniTAF-A2F_Expression-加载A2d预训练权重(口型状态系数空间加权_260109)_260206",  # 断点 & 日志保存根目录
        "resume_path": None,  # 如需断点续训，填 ckpt 路径或 True
        # 分别训练tts和a2f的配置
        "train_tts": False,
        "train_tts_lora": True,  # 仅 train_tts 时有效
        "train_a2f": True,  # 只要训练a2f,则必须训练audio feature projector. 故不在额外增加投影层是否训练的参数判断了
        "train_a2f_adapter_only": False,  # 训练a2f时，是否只训练adapter部分（a2f.audio_feature_projector）
        "train_except_a2f_adapter": False,  # 训练a2f时，排除adapter部分（a2f.audio_feature_projector）
        "train_a2f_expression_model": True,  #  只训练a2f中表情残差模型，冻结口型模型  目前a2f生成口型，expression model生成表情，最终合并成完整面部
        # 优化器设置，为不同模块设置不同优化器
        "tts_train_cfg": {
            "lr": 5e-7,
            "betas": (0.9, 0.999),
            "weight_decay": 0.01,
            "eps": 1e-08,
        },
        "tts_lora_cfg": {
            "lora_target_modules": ["c_attn", "c_proj", "c_fc"],  # 只针对TTS中gpt,包括attn和mlp层
            "lora_rank": 128,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
        },
        "a2f_train_cfg": {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "weight_decay": 0.01,
            "eps": 1e-08,
        },
        # 日志配置：
        "report_to": "wandb",
        # 加载指定模块的自定义权重用于替代官方预训练权重：
        "finetune_checkpoint": {
            # "tts_model":
            #     "./unitaf_ckpt/UniTAF-A2F(lr_1e-4)- LoRA-TTS(lr_5e-7_rank_128)/checkpoint-20000/tts_model.pt",
            "audio_feature_projector":
                "./unitaf_ckpt/UniTAF-A2F(口型状态顶点空间加权loss_加权仅作用于嘴部顶点)-加载Adapter预训练权重(约束AudioFeature_step_74140)_260109/checkpoint-74140/audio_feature_projector.pt",
            "a2f_model":
                "./unitaf_ckpt/UniTAF-A2F(口型状态顶点空间加权loss_加权仅作用于嘴部顶点)-加载Adapter预训练权重(约束AudioFeature_step_74140)_260109/checkpoint-74140/a2f_model.pt",
        }
    }
    train_config = OmegaConf.create(train_config)

    # wandb.init(
    #     project="UniTAF",
    #     name=train_config.get("test UniTAF Train", None),  # 可再补一个 run_name 字段
    #     config=OmegaConf.to_container(train_config, resolve=True),
    #     resume="allow" if train_config.get("resume_path") else None
    # )

    main(train_config)

