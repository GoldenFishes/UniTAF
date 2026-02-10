'''
25.12.31
使用系数空间loss+基于口型状态的顶点空间加权loss
Loss区别位于 a2f.loss.loss.BlendShapeLoss_61.forward()

在只训练adapter的权重上只训练a2f+adapter部分，eval_step过程中模型结果渲染可视化

只训练adapter时，我们使用audio_feature_projector产生特征 和 UniTalker Encoder产生特征的L2 loss来约束
Adapter： a2f.audio_feature_projector
'''

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
    # 数据集类-------------------------------------------------------------------------------------------------------
    "dataset_config": {
        "dataset_root_path": "/home/zqg/project/data/UniTAF Dataset",  # 使用绝对路径
        "dataset_list": ["D12"],  # 支持多数据集训练，对应unitaf_dataset_support_config中具体数据集
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
    "log_interval": 20,  # training_step 里打印 loss 的步长         # 20
    "val_interval": 2000,  # 每隔多少 step 做一次验证               # 2000
    "save_interval": 20000,  # 每隔多少 step 存一次 ckpt              # 20000
    "output_dir": "./unitaf_ckpt/UniTAF-A2F(系数空间loss+口型状态顶点空间加权loss)-加载Adapter预训练权重(约束AudioFeature_step_74140)_251231",  # 断点 & 日志保存根目录
    "resume_path": None,  # 如需断点续训，填 ckpt 路径或 True
    # 分别训练tts和a2f的配置
    "train_tts": False,
    "train_tts_lora": True,  # 仅 train_tts 时有效
    "train_a2f": True,  # 只要训练a2f,则必须训练audio feature projector. 故不在额外增加投影层是否训练的参数判断了
    "train_a2f_adapter_only": False,  # 训练a2f时，是否只训练adapter部分（a2f.audio_feature_projector）
    "train_except_a2f_adapter": False,  # 训练a2f时，排除adapter部分（a2f.audio_feature_projector）
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
        # "tts_model": "./unitaf_ckpt/UniTAF-A2F(lr_1e-4)- LoRA-TTS(lr_5e-7_rank_128)/checkpoint-20000/tts_model.pt",
        "audio_feature_projector": "./unitaf_ckpt/UniTAF-A2F-Adapter_Only(lr_1e-4)：计算audio feature的L2 loss/checkpoint-74140/audio_feature_projector.pt",
        # "a2f_model": "./unitaf_ckpt/UniTAF-A2F-Adapter(lr_1e-4)/checkpoint-7414/a2f_model.pt",
    }
}