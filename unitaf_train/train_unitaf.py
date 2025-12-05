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
    并在training_step()中手动完成各自反向，各自更新也由我们手动opt.step()。Trainer只负责调用 training_step，写日志，验证，保存，不会多管闲事。

    Optimizer 只能看到「参数 + 参数的 .grad 字段」，它永远把当前 .grad 当成“最终梯度”去做更新。
    如果我们想让 optimizer 自己决定用哪部分梯度”，就必须在反向之前把两份梯度拼到同一份 .grad 上。
    这等于又回到了“先各自反传、再各自 step”的复杂度，甚至更难维护：
        这需要分别保存tts_loss与a2f_loss到临时的buffer，把两份梯度手动写回对应参数的 .grad ,再让 optimizer 去做一次统一step；
        同时还要处理 retain_graph=True, AMP, 梯度累积, 裁剪, zero_grad的时机。

    因此在 training_step里手动各自 backward; 直接调用各自 optimizer.step();

'''
import sys
import os

from omegaconf import OmegaConf
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

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

# UniTextAudioFace的dataset
from unitaf_dataset import UniTAFDataset
# 组装联合模型UniTextAudioFace的类
from UniTAF import UniTextAudioFaceModel


class MultiOptimizer:
    """包装多个优化器，使其像单个优化器一样工作"""
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def step(self):
        """我们已经在外面 step 过了，这里什么都不做"""
        pass

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
        # for scheduler in self.schedulers:
        #     scheduler.step()
        # 我们已经在training_step手工操作过了
        pass

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

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.collate_batch,  # 我们用自定义 collate_batch
            args=TrainingArguments(
                output_dir=train_config.get("output_dir", "./unitaf_ckpt"),
                num_train_epochs=train_config["epochs"],
                per_device_train_batch_size=train_config["batch_size"],
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
            )
        )

    def compute_losses(self, batch):
        device = self.device
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
            emo_vec = batch["emo_vec"].to(device)
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
            )

            text_emb = self.model.tts_model.text_embedding(text_inputs) + self.model.tts_model.text_pos_embedding(text_inputs)
            mel_emb = self.model.tts_model.mel_embedding(mel_inputs) + self.model.tts_model.mel_pos_embedding(mel_inputs)

            # (2) IndexTTS中gpt模型的前向过程
            text_logits, mel_logits, text_latent, mel_latent = self.model.tts_model.get_logits(
                conds, text_emb, self.model.tts_model.text_head, mel_emb, self.model.tts_model.mel_head,
                return_both=True  # 我们于indextts.gpt.model_v2.UnifiedVoice.get_logits()新增该return_both分支，
                # 用于同时返回tts训练用的logits和gt_audio的latent。logits用于计算tts loss，latent用于得到后续audio feature
            )  # 这里返回的mel_latent就是GT对应的latent


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
            with torch.no_grad():
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

            loss["tts_text_loss"] = text_loss
            loss["tts_mel_loss"] = mel_loss
            loss["tts_metrics"] = metrics

            # (4) 获得GT latent继续IndexTTS生成音频的过程以获取audio_feature
            mel_latent = mel_latent[:, :-2]  # torch.Size([B, L, 1280]) 此时处理后的mel_latent与UnifiedVoice.forward()的返回一样了

            with torch.no_grad():
                mel_latent = self.model.s2mel.models["gpt_layer"](mel_latent)  # torch.Size([B, L, 1024])

                # # TODO: 这里会不会有没有剔除干净的只有训练时才会添加的pad残留？
                # # TODO: 这里是否应该传入audio_codes？
                # S_infer = self.semantic_codec.quantizer.vq2emb(audio_codes.unsqueeze(1))  # 编码转嵌入 torch.Size([B, 1024, L])
                # S_infer = S_infer.transpose(1, 2)  # torch.Size([1, L, 1024])
                # S_infer = S_infer + mel_latent  # 融合潜在表示 torch.Size([1, L, 1024])
                # target_lengths = (mel_latent.shape[1] * 1.72).long()  # 计算目标长度  L*1.72
                #
                # # 长度调节
                # cond = self.s2mel.models['length_regulator'](S_infer,
                #                                              ylens=target_lengths,
                #                                              n_quantizers=3,
                #                                              f0=None)[0]
                # cat_condition = torch.cat([prompt_condition, cond], dim=1)  # 拼接条件  # [1, 670, 512]
                # # 条件流匹配推理
                # vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                #                                                torch.LongTensor([cat_condition.size(1)]).to(
                #                                                    cond.device),
                #                                                ref_mel, style, None, diffusion_steps=25,
                #                                                inference_cfg_rate=0.7)
                # # print(f"vc_target: {vc_target.shape}")  # [1, 80, 670]
                # vc_target = vc_target[:, :, ref_mel.size(-1):]  # 裁剪目标梅尔谱图


        # 2. 获得TTS部分核心输出 处理成audio feature
        if "IndexTTS2" in self.train_config["tts_model"]:
            # 暂定使用 经过s2mel.models["gpt_layer"]处理后的 mel_latent
            audio_feature = mel_latent  # torch.Size([B, L, 1024])
            # 暂定不需要额外的projector层，直接训练UniTalkerDecoder中相应projector来适应即可。


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

            out_motion, out_pca, gt_pca = self.model.a2f_model(
                audio_feature=audio_feature, template=face_template, face_motion=face_data,
                style_idx=identity ,annot_type=face_type,
            )
            # 由我们单独实现的unitalker_decoder_component.UniTalkerDecoder.compute_loss()中计算loss
            rec_loss, pca_loss = self.model.a2f_model.compute_loss(out_motion, face_data, face_type, out_pca, gt_pca)

            loss["a2f_rec_loss"] = rec_loss
            loss["a2f_pca_loss"] = pca_loss

        return loss

    def collate_batch(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        '''合并成tensor时需要padding为等长序列'''
        output = {}
        if "IndexTTS2" in self.train_config["tts_model"]:
            '''
            接收到batch
                条件音频token      tts_condition,tts_condition_len
                文本token         text_ids,text_ids_len
                GT音频token       audio_codes,audio_codes_len
                情感控制向量        emotion_vector
            '''
            text_tensors = [item["text_ids"] for item in batch]
            code_tensors = [item["audio_codes"] for item in batch]    # 这就是GT音频的离散编码
            condition_tensors = [item["tts_condition"] for item in batch]
            emo_tensors = [  # 如果遇到item["emo_vec"]为None则替换为代表clam的向量
                item["emo_vec"] if (item.get("emo_vec") is not None)
                else torch.tensor([0., 0., 0., 0., 0., 0., 0., 1.], dtype=torch.float32)
                for item in batch
            ]

            # 对变长序列进行填充（padding）
            text_padded = pad_sequence(text_tensors, batch_first=True, padding_value=0)
            code_padded = pad_sequence(code_tensors, batch_first=True, padding_value=0)  # 音频token序列padding：这些0在后续的set_mel_padding中会被替换为stop_token
            condition_stacked = torch.stack(condition_tensors, dim=0)  # 语音条件特征堆叠：形状从 [D] -> [B, D]
            emo_stacked = torch.stack(emo_tensors, dim=0)  # 情感向量堆叠：形状从 [E] -> [B, E]

            # 提取序列长度信息（用于masking和损失计算）
            text_lengths = torch.stack([item["text_ids_len"] for item in batch])
            code_lengths = torch.stack([item["audio_codes_len"] for item in batch])
            cond_lengths = torch.stack([item["tts_condition_len"] for item in batch])

            output["tts_condition"] = condition_stacked
            output["tts_condition_len"] = cond_lengths
            output["text_ids"] = text_padded
            output["text_ids_len"] = text_lengths
            output["audio_codes"] = code_padded
            output["audio_codes_len"] = code_lengths
            output["emotion_vector"] = emo_stacked

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

            output["face_data"] = face_padded
            output["face_data_len"] = face_lengths
            output["face_template"] = torch.stack(tmpl_tensors, dim=0)
            output["face_type"] = type_tensors  # 列表即可，后面用索引时再转 tensor
            output["face_fps"] = torch.stack(fps_tensors, dim=0)
            output["speaker_idx"] = torch.stack(speaker_tensors, dim=0)

        return output

    def set_model_training_mode(self):
        """
        设置各部分模型的训练模式
        # TODO：部分冻结的情况
        """
        if self.train_config["train_tts"]:
            # TTS部分需要训练
            self.model.tts_model.train()
            for param in self.model.tts_model.parameters():
                param.requires_grad = True
        else:
            # TTS部分不需要训练，设置为eval模式并冻结
            self.model.tts_model.eval()
            for param in self.model.tts_model.parameters():
                param.requires_grad = False

        if self.train_config["train_a2f"]:
            # A2F部分需要训练
            self.model.a2f_model.train()
            for param in self.model.a2f_model.parameters():
                param.requires_grad = True
        else:
            # A2F部分不需要训练
            self.model.a2f_model.eval()
            for param in self.model.a2f_model.parameters():
                param.requires_grad = False

        if "IndexTTS2" in self.train_config["tts_model"]:
            # IndexTTS2的s2mel部分始终冻结，不参与训练
            self.model.s2mel.eval()  # 设置为eval模式
            for param in self.model.s2mel.parameters():
                param.requires_grad = False
            # 确保s2mel中的子模块也冻结
            for module in self.model.s2mel.modules():
                if hasattr(module, 'weight'):
                    module.weight.requires_grad = False
                if hasattr(module, 'bias'):
                    module.bias.requires_grad = False

    def create_optimizer(self):
        """
        重写optimizer创建逻辑，支持多个优化器。
        为联合模型中tts和a2f设置独立的优化器，会在训练前调用一次。
        """
        if hasattr(self, "optimizer") and self.optimizer is not None:
            return  # 已经创建过

        opt_list = []
        # ---- TTS ----
        if self.train_config["train_a2f"]:
            cfg = self.train_config["tts_train_cfg"]
            params = list(self.model.tts_model.parameters())
            opt_list.append(AdamW(params,
                                  lr=cfg.get("lr", 2e-5),
                                  betas=cfg.get("betas", (0.9, 0.999)),
                                  weight_decay=cfg.get("weight_decay", 0.01),
                                  eps=cfg.get("eps", 1e-8)))

        if self.train_config["train_a2f"]:
            cfg = self.train_config["a2f_train_cfg"]
            params = list(self.model.a2f_model.parameters())
            opt_list.append(AdamW(params,
                                  lr=cfg.get("lr", 2e-5),
                                  betas=cfg.get("betas", (0.9, 0.999)),
                                  weight_decay=cfg.get("weight_decay", 0.01),
                                  eps=cfg.get("eps", 1e-8)))
        self.optimizer = MultiOptimizer(opt_list)  # 将多个优化器打包成一个

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        为每个优化器创建独立的调度器，会在训练前调用一次
        """
        if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
            return

        schedulers = []
        for opt in self.optimizer.optimizers:
            # 统一用 cosine + warmup
            sch = get_cosine_schedule_with_warmup(
                opt,
                num_warmup_steps=self.train_config.get("warmup_steps", 50),
                num_training_steps=num_training_steps)
            schedulers.append(sch)
        self.lr_scheduler = MultiScheduler(schedulers)

    def training_step(self, model, inputs) -> torch.Tensor:
        """
        各自反传、各自 step，返回标量 loss 给 Trainer 做日志
        """
        self.set_model_training_mode()

        # 1. 前向
        with self.compute_loss_context_manager():
            loss_dict = self.compute_losses(inputs)

        # 2. 计算各子 loss
        tts_loss = torch.tensor(0., device=self.device)
        a2f_loss = torch.tensor(0., device=self.device)

        if self.train_config["train_tts"]:
            if "IndexTTS2" in self.train_config["tts_model"]:
                tts_loss = (
                        self.train_config["IndexTTS2"]["text_loss_weight"] * loss_dict["tts_text_loss"] +
                        self.train_config["IndexTTS2"]["mel_loss_weight"] * loss_dict["tts_mel_loss"]
                )
        if self.train_config["train_a2f"]:
            if "UniTalker" in self.train_config["a2f_model"]:
                a2f_loss = (
                        loss_dict["a2f_rec_loss"] +
                        self.train_config["UniTalker"]["pca_weight"] * loss_dict["a2f_pca_loss"]
                )
        total_loss = tts_loss + a2f_loss / self.train_config["grad_accumulation"]

        # 3. 各自反向
        # use_last_step = (self.state.global_step + 1) % self.train_config["grad_accumulation"] == 0  # Trainer 已维护
        # # TTS 反传
        # if self.train_config["train_tts"] and tts_loss.item() != 0:
        #     tts_loss.backward(retain_graph=not use_last_step)  # 只有非最后一步才保留图
        # # A2F 反传
        # if self.train_config["train_a2f"] and a2f_loss.item() != 0:
        #     a2f_loss.backward(retain_graph=False)  # A2F 总是最后一段，无需保留

        grad_clip = self.train_config["grad_clip"]
        use_amp = self.train_config["use_amp"] and torch.cuda.is_available()

        # 3-a TTS 反传
        if self.train_config["train_tts"] and tts_loss.item() != 0:
            if use_amp:
                with torch.cuda.amp.autocast():
                    tts_loss.backward(retain_graph=True)
            else:
                tts_loss.backward(retain_graph=True)

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.tts_model.parameters(), grad_clip)

        # 3-b A2F 反传
        if self.train_config["train_a2f"] and a2f_loss.item() != 0:
            if use_amp:
                with torch.cuda.amp.autocast():
                    a2f_loss.backward()
            else:
                a2f_loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.a2f_model.parameters(), grad_clip)

        # 4. 各自 step（Trainer 会在外面再调一次，我们把它覆盖掉）
        #    注意：Trainer 默认在 optimizer.step() 里会把所有优化器都 step，
        #    所以这里我们先 step 完，把梯度清掉，后面 Trainer 再调一次时梯度为 0 就不会重复更新。
        if self.train_config["train_tts"] and tts_loss.item() != 0:
            opt_tts = self.optimizer.optimizers[0]
            opt_tts.step()
            opt_tts.zero_grad(set_to_none=True)

        if self.train_config["train_a2f"] and a2f_loss.item() != 0:
            # 当 train_tts=False 时 a2f 优化器是第 0 个
            idx = 1 if self.train_config["train_tts"] else 0
            opt_a2f = self.optimizer.optimizers[idx]
            opt_a2f.step()
            opt_a2f.zero_grad(set_to_none=True)

        return total_loss.detach()      # 返回标量给 Trainer 做日志

    # TODO：梯度裁剪适合用哪个？
    def clip_gradients(self, parameters, max_norm, optimizer):
        """ Trainer 会在 optimizer.step 前自动调用 """
        # 这里直接调用官方实现即可
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    # def clip_gradients(self, optimizer, gradient_clip_val=None, gradient_clip_norm=None):
    #     """optimizer 是 MultiOptimizer，这里对每块单独裁"""
    #     clip_val = gradient_clip_val or self.train_config["grad_clip"]
    #     if clip_val <= 0:
    #         return
    #     # 顺序与 create_optimizer 保持一致
    #     if self.train_config["train_tts"]:
    #         torch.nn.utils.clip_grad_norm_(self.model.tts_model.parameters(), clip_val)
    #     if self.train_config["train_a2f"]:
    #         torch.nn.utils.clip_grad_norm_(self.model.a2f_model.parameters(), clip_val)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """占位：只返回空 dict，后续再填具体指标"""
        # TODO
        return {}

    def log_validation(self, metrics, step):
        """占位：把验证指标打印出来，后续可接 wandb"""
        # TODO
        print(f"[Validation @ step {step}] {metrics}")

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
            print(f"[UniTAFTrainer][setup_dataloader] 数据集{subdataset}没有train训练集，跳过。\n Error:{e}")
        try:
            val_dataset = UniTAFDataset(dataset_config, dataset_name=subdataset, dataset_type="val")
            val_datasets.append(val_dataset)
        except Exception as e:
            print(f"[UniTAFTrainer][setup_dataloader] 数据集{subdataset}没有val验证集，跳过。\n Error:{e}")

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

    # 2. 实例化模型（Trainer 需要）
    device = torch.device(train_config["device"] if torch.cuda.is_available() else "cpu")
    model = UniTextAudioFaceModel(cfg=train_config, device=device)

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

    # 开始训练
    print("[UniTAFTrainer] 开始训练...")
    trainer_stats = trainer.train()
    print("[UniTAFTrainer] 训练结束...")



if __name__ == '__main__':
    '''
    python unitaf_train/train_unitaf.py 
    '''
    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # unitaf_train的父目录
    sys.path.insert(0, project_root)

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
            "audio_encoder_feature_dim": 1024,  # 由我们选择获取的TTS中间结果Audio Feature的dim决定
            "identity_num": 20,  # 暂定为20，实际在UniTalker Decoder权重加载时会从权重中得到这里的值并更新
        },
        # 数据集类-------------------------------------------------------------------------------------------------------
        "dataset_config": {
            "dataset_root_path": "/home/zqg/project/data/UniTAF Dataset",  # 使用绝对路径
            "dataset_list": ["D12"],  # 支持多数据集训练，对应unitaf_dataset_support_config中具体数据集
            # unitaf_dataset_support_config是经过数据集格式转换UniTAFDataset能够支持的数据集
            # （UniTalker本身还有一个 a2f/dataset/dataset_config 记录UniTalker Decoder支持的数据集，
            # 但我们只会以 unitaf_dataset_support_config 为准）
        },
        # dataloader设置
        "num_workers": 2,
        # 设备
        "device": "cuda:0",
        # 训练设置-------------------------------------------------------------------------------------------------------
        "batch_size": 2,
        "epochs": 2,
        "grad_accumulation": 1,
        "grad_clip": 1.0,
        "use_amp": True,
        "warmup_steps": 50, # 所有调度器统一参数 warmup 为50
        "log_interval": 1,  # training_step 里打印 loss 的步长
        "val_interval": 500,  # 每隔多少 step 做一次验证
        "save_interval": 1000,  # 每隔多少 step 存一次 ckpt
        "output_dir": "./unitaf_ckpt",  # 断点 & 日志保存根目录
        "resume_path": None,  # 如需断点续训，填 ckpt 路径或 True
        # 分别训练tts和a2f的配置
        "train_tts": True,
        "train_a2f": True,
        # 优化器设置，为不同模块设置不同优化器
        "tts_train_cfg": {
            "lr": 2e-5,
            "betas": (0.9, 0.999),
            "weight_decay": 0.01,
            "eps": 1e-08,
        },
        "a2f_train_cfg": {
            "lr": 2e-5,
            "betas": (0.9, 0.999),
            "weight_decay": 0.01,
            "eps": 1e-08,
        },
        # 日志配置：
        "report_to": "wandb",
    }
    train_config = OmegaConf.create(train_config)

    # wandb.init(
    #     project="UniTAF",
    #     name=train_config.get("test UniTAF Train", None),  # 可再补一个 run_name 字段
    #     config=OmegaConf.to_container(train_config, resolve=True),
    #     resume="allow" if train_config.get("resume_path") else None
    # )

    main(train_config)

