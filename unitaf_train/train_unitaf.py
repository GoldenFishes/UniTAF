'''
25.12.2
这里实现UniTextAudioFace联合模型的训练脚本
'''
import sys
import os

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from transformers import get_cosine_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence

from FunASR.funasr.models.eres2net.eres2net import conv1x1
# UniTextAudioFace的dataset
from unitaf_dataset import UniTAFDataset
# 组装联合模型UniTextAudioFace的类
from UniTAF import UniTextAudioFaceModel


class UniTAFTrainer():
    '''
    UniTextAudioFace 训练器
    '''
    def __init__(
        self,
        train_config,
    ):
        self.train_config = train_config

        self.device = torch.device(self.train_config["device"] if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.model = UniTextAudioFaceModel(
            cfg=train_config,
            device=self.device,
        )


    def compute_losses(
        self,
        batch,
    ):
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

            # 这里是tts模型的前向过程
            text_logits, mel_logits = self.model.tts_model.get_logits(
                conds, text_emb, self.model.tts_model.text_head, mel_emb, self.model.tts_model.mel_head
            )

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

            # TODO: IndexTTS取哪里的feature？

        # 2. TODO：获得TTS部分核心输出 处理成audio feature

        audio_feature=

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
            emo_tensors = [item["emo_vec"] for item in batch]

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
            # 这里似乎不需要对UniTalker的内容做额外的计算
            pass


        return output


    def setup_dataloader(
        self,
        dataset_list: list,  # 支持多数据集训练，对应unitaf_dataset_support_config中具体数据集
    ):
        '''
        初始化一个dataloader
        在这里为每个dataset 例如"D11","D12" ;以及为每个dataset_type 例如"train","val"，都创建数据集并合并
        最终返回总体train/val dataloader
        '''
        # 1. 组装dataset config
        dataset_config = {}
        dataset_config["tts_model"] = self.train_config["tts_model"]
        dataset_config["a2f_model"] = self.train_config["a2f_model"]
        dataset_config["dataset_root_path"] = self.train_config["dataset_root_path"]
        dataset_config["device"] = self.train_config["device"]

        use_cuda = torch.cuda.is_available()

        train_datasets = []
        val_datasets = []

        # 2. 遍历所有子数据集
        for subdataset in dataset_list:

            try:
                train_dataset = UniTAFDataset(dataset_config, dataset_type="train", dataset_name=subdataset)
                train_datasets.append(train_dataset)
            except Exception as e:
                print(f"[UniTAFTrainer][setup_dataloader] 数据集{subdataset}没有train训练集，跳过。")

            try:
                val_dataset = UniTAFDataset(dataset_config, dataset_type="val", dataset_name=subdataset)
                val_datasets.append(val_dataset)
            except Exception as e:
                print(f"[UniTAFTrainer][setup_dataloader] 数据集{subdataset}没有val验证集，跳过。")

        # 3. 合并dataset
        if len(train_datasets) > 1:
            combined_train_dataset = ConcatDataset(train_datasets)
        else:
            combined_train_dataset = train_datasets[0]

        if len(val_datasets) > 1:
            combined_val_dataset = ConcatDataset(val_datasets)
        else:
            combined_val_dataset = val_datasets[0]

        # 4. 创建dataloader
        train_dataloader = DataLoader(
            dataset=combined_train_dataset,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
            num_workers=self.train_config["num_workers"],
            collate_fn=self.collate_batch,
            pin_memory=use_cuda,
        )
        val_dataloader = DataLoader(
            dataset=combined_val_dataset,
            batch_size=self.train_config["batch_size"],
            shuffle=False,
            num_workers=self.train_config["num_workers"],
            collate_fn=self.collate_batch,
            pin_memory=use_cuda,
        )

        return train_dataloader, val_dataloader


    def main(self):
        # 根据dataset_config创建训练和验证的dataloader
        dataset_list = self.train_config["dataset_config"]["dataset_list"]
        train_dataloader, val_dataloader = self.setup_dataloader(dataset_list)

        # 初始化优化器
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.train_config["learning_rate"],
            weight_decay=self.train_config["weight_decay"],
        )

        total_steps = self.train_config["epochs"] * max(1, len(train_dataloader)) // max(1, self.train_config["grad_accumulation"])
        total_steps = max(total_steps, 1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.train_config["warmup_steps"],
            num_training_steps=total_steps,
        )
        use_amp = self.train_config["use_amp"] and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        global_step = 0
        start_epoch = 0
        recent_checkpoints: List[str] = []
        last_saved_step: int | None = None

        # TODO: resume train恢复训练设置

        self.model.train()  # UniTAF联合模型
        optimizer.zero_grad(set_to_none=True)

        grad_accumulation = self.train_config["grad_accumulation"]
        grad_clip = self.train_config["grad_clip"]

        for epoch in range(start_epoch, self.train_config["epochs"]):
            for batch_idx, batch in enumerate(train_dataloader):
                with torch.cuda.amp.autocast(use_amp):
                    # 计算loss
                    loss = self.compute_losses(batch)  # 返回一个Dict


                    if "IndexTTS2" in self.train_config["tts_model"]:
                        tts_loss = self.train_config["IndexTTS2"]["text_loss_weight"] * loss["text_loss"] + self.train_config["IndexTTS2"]["mel_loss_weight"] * loss["mel_loss"]
                    if "UniTalker" in self.train_config["a2f_model"]:
                        a2f_loss = loss["a2f_rec_loss"] + self.train_config["UniTalker"]["pca_weight"] * loss["a2f_pca_loss"]


                    # TODO: a2f loss
                    if use_amp:
                        scaler.scale(tts_loss / grad_accumulation).backward()

                    else:
                        (tts_loss / grad_accumulation).backward()

                    # TODO:不同的loss作用于不同部分TTS/A2F
                    if (batch_idx + 1) % grad_accumulation == 0:
                        if grad_clip > 0:
                            if use_amp:
                                scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.tts_model.parameters(), grad_clip)
                        if use_amp:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                        global_step += 1

                        # 验证步骤
                        if self.train_config["val_interval"] > 0 and global_step > 0 and global_step % self.train_config["val_interval"] == 0:
                            # TODO
                            val_metrics = self.evaluate(
                                val_dataloader
                            )





if __name__ == '__main__':
    '''
    '''
    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # unitaf_train的父目录
    sys.path.insert(0, project_root)

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
        },

        # 数据集类
        "dataset_config": {
            "dataset_root_path": "/home/zqg/project/data/UniTAF Dataset",  # 使用绝对路径
            "dataset_list": ["D12"],  # 支持多数据集训练，对应unitaf_dataset_support_config中具体数据集
            # unitaf_dataset_support_config是经过数据集格式转换UniTAFDataset能够支持的数据集
            # UniTalker本身还有一个 a2f/dataset/dataset_config 记录UniTalker Decoder支持的数据集
        },
        # dataloader设置
        "num_workers": 2,
        # 设备
        "device": "cuda:0",
        # 训练设置：
        "batch_size": 2,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 50,
        "epochs": 10,
        "grad_accumulation": 1,
        "grad_clip": 1.0,
        "use_amp": True,



    }

