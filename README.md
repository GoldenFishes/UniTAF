# UniTAF

.

### TODO:

- 实现情感控制的表情生成（训练expression_model）

.

## 1. Install

.

以下三种安装方式任选其一

### A. 使用UV安装

**1.克隆仓库**

```python
git clone https://github.com/GoldenFishes/UniTAF.git
cd UniTAF
# 下载大文件
git lfs pull
```

> ```
> # linux安装lfs
> apt update && apt install git-lfs -y
> # 验证
> git lfs install
> ```

.

**2.下载权重**

下载UniTalker权重

[UniTalker-B-[D0-D7\]](https://drive.google.com/file/d/1PmF8I6lyo0_64-NgeN5qIQAX6Bg0yw44/view?usp=sharing): The base model in UniTalker.
[UniTalker-L-[D0-D7\]](https://drive.google.com/file/d/1sH2T7KLFNjUnTM-V1eRMM1Tytxd2sYAp/view?usp=sharing): The default model in UniTalker.

```python
# Download it and place it in ./a2f/pretrained_models
├── a2f/pretrained_models
│   ├── UniTalker-B-D0-D7.pt
│   ├── UniTalker-L-D0-D7.pt
```

.

**3.确保安装包管理工具uv**

```
pip install -U uv
```

> *所有* *`uv`* *命令都会自动激活每个项目的**虚拟环境**。在运行* *`UV`* *命令之前 ，千万不要手动激活任何环境，因为这可能导致依赖冲突！*

.

**4.安装必须依赖**

```
# 会自动创建一个 .venv 项目目录，然后安装正确版本的 Python 及所有必要的依赖：
uv sync --all-extras

# (可选) 安装web ui
uv sync --extra webui
# (可选) 安装deepspeed加速推理
uv sync --extra deepspeed

# 如果下载缓慢可以尝试中国本地镜像
uv sync --all-extras --default-index "https://mirrors.aliyun.com/pypi/simple"
uv sync --all-extras --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
```

> 如果在Windows安装过程中 DeepSpeed 库报错，则可以通过移除 --all-extras来跳过它

> 如果安装过程中出现CUDA错误，请确保CUDA Toolkit版本为12.8或更高！

.

**5.通过uv工具下载模型**

```Shell
# 安装hf
uv tool install "huggingface-hub[cli,hf_xet]"
# 下载indextts2预训练模型
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
# 下载UniTAF训练的权重
hf download ATA-space/UniTAF --local-dir=unitaf_ckpt
```

> 如果上述命令不可用，请阅读uv tool输出。它会告诉你如何将这些工具添加到系统路径中。

> 如果网络环境对HuggingFace访问较慢，可以在运行代码前执行以下命令：
>
> ```Shell
> # Linux
> export HF_ENDPOINT="https://hf-mirror.com"
> # Windows
> $env:HF_ENDPOINT = "https://hf-mirror.com"
> ```

.

**6.诊断 PyTorch GPU 加速**

```Shell
# 运行官方提供的诊断工具
uv run tools/gpu_check.py
```

> windows如果运行推理示例时报错 _kaldifst 动态链接库初始化失败
>
> ```PowerShell
> >> bigvgan weights restored from: nvidia/bigvgan_v2_22khz_80band_256x
> Traceback (most recent call last):
>   File "D:\Project\index-tts\inference.py", line 9, in <module>
>     tts = IndexTTS2(
>   File "D:\Project\index-tts\indextts\infer_v2.py", line 186, in __init__
>     self.normalizer.load()
>   File "D:\Project\index-tts\indextts\utils\front.py", line 95, in load
>     from wetext import Normalizer
>   File "D:\Project\index-tts\.venv\lib\site-packages\wetext\__init__.py", line 15, in <module>
>     from wetext.utils import (
>   File "D:\Project\index-tts\.venv\lib\site-packages\wetext\utils.py", line 18, in <module>
>     from wetext.constants import FSTS
>   File "D:\Project\index-tts\.venv\lib\site-packages\wetext\constants.py", line 17, in <module>
>     from kaldifst import TextNormalizer as normalizer
>   File "D:\Project\index-tts\.venv\lib\site-packages\kaldifst\__init__.py", line 1, in <module>
>     from _kaldifst import (
> ImportError: DLL load failed while importing _kaldifst: 动态链接库(DLL)初始化例程失败。
> ```
>
> 则参考官方仓库下issue：https://github.com/index-tts/index-tts/issues/356 中方法。
>
> 下载并安装（Windows专用）： https://aka.ms/vs/17/release/vc_redist.x64.exe 。安装完成后重启终端或电脑即可解决。

.

**7.安装补充包**

**pytorch3d**，手动安装并编译：

在project同级目录下：

```Python
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
```

安装编译：

```Python
# 清空上次编译留下的临时文件
python setup.py clean
# 把 C/CUDA 扩展编译成当前目录下的共享库（*.so 或 *.pyd），但不复制到 site-packages；
# --inplace 让生成的 .so 直接躺在源码目录，方便调试/即时 import。
python setup.py build_ext --inplace
# 把编译好的包复制到当前环境的 site-packages，并注册入口脚本，使 import pytorch3d 全局可用。
python setup.py install
```

> 编译完验证：
>
> ```Python
> python -c "import pytorch3d; print(pytorch3d.__version__)"
> ```

.

**安装 Flash Attn 加速** https://github.com/Dao-AILab/flash-attention/releases

找到 **UV环境中：**对应cuda版本，torch版本，python版本的 whl 安装包。下载并移动到Linux服务器中，在路径下执行：

```Python
# 使用你正确的安装包的名字，我们这里下载的安装包是对应python-3.10，torch-2.8.0+cu128的
uv pip install ./flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
  --no-deps \
  --force-reinstall
# 验证是否成功安装
uv run python - << 'EOF'
import flash_attn
print("flash_attn import OK")
print(flash_attn.__version__)
EOF
```

.

**安装 ffmpeg**

```python
# Ubuntu / Debian
sudo apt update
sudo apt install -y ffmpeg
```

.

### B. 使用pip安装

示例环境为：python 3.11 ， torch 2.8.0+cu128

```Python
conda create -n unitaf python=3.11 -y
conda activate unitaf
```

.

**1.安装 pytorch**

只使用官方CUDA源

```Python
pip install torch==2.8.* torchaudio==2.8.* --index-url https://download.pytorch.org/whl/cu128 --no-deps
```

.

**2.安装其他依赖**

```Python
# 该requirements.txt从pyproject.toml中转换出来
pip install -r requirements.txt
```

> 这里如果有清华源，则会报错证书问题。请确保删除清华源，仅使用官方源。

.

**3.下载模型**

下载IndexTTS2模型

```Python
# 安装hf
pip install -U "huggingface-hub<1.0"
# 下载indextts2预训练模型
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
# 下载UniTAF训练的权重
hf download ATA-space/UniTAF --local-dir=unitaf_ckpt
```

> 服务器网络证书故障时直接本地下载模型然后上传
>
> 或运行仓库中的 hf_download.py脚本

下载UniTalker权重：

[UniTalker-B-[D0-D7\]](https://drive.google.com/file/d/1PmF8I6lyo0_64-NgeN5qIQAX6Bg0yw44/view?usp=sharing): The base model in UniTalker.
[UniTalker-L-[D0-D7\]](https://drive.google.com/file/d/1sH2T7KLFNjUnTM-V1eRMM1Tytxd2sYAp/view?usp=sharing): The default model in UniTalker.

```python
# Download it and place it in ./a2f/pretrained_models
├── a2f/pretrained_models
│   ├── UniTalker-B-D0-D7.pt
│   ├── UniTalker-L-D0-D7.pt
```

.

**4.安装补充包**

**pytorch3d**，手动安装并编译：

在project同级目录下：

```Python
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
```

安装编译：

```Python
# 清空上次编译留下的临时文件
python setup.py clean
# 把 C/CUDA 扩展编译成当前目录下的共享库（*.so 或 *.pyd），但不复制到 site-packages；
# --inplace 让生成的 .so 直接躺在源码目录，方便调试/即时 import。
python setup.py build_ext --inplace
# 把编译好的包复制到当前环境的 site-packages，并注册入口脚本，使 import pytorch3d 全局可用。
python setup.py install
```

> 编译完验证：
>
> ```Python
> python -c "import pytorch3d; print(pytorch3d.__version__)"
> ```

.

**安装 Flash Attn** 加速 https://github.com/Dao-AILab/flash-attention/releases

找到对应cuda版本，torch版本，python版本的 whl 安装包。下载并移动到Linux服务器中，在路径下执行：

```Python
# 使用你正确的安装包的名字，我们这里下载的安装包是对应python-3.11，torch-2.8.0+cu128的
pip install ./flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \
  --no-deps \
  --force-reinstall
# 验证是否成功安装
python - << 'EOF'
import flash_attn
print("flash_attn import OK")
print(flash_attn.__version__)
EOF
```

.

**安装 ffmpeg**

```python
# Ubuntu / Debian
sudo apt update
sudo apt install -y ffmpeg
```

.

### C. 使用AutoDL镜像（推荐）

前往AutoDL镜像界面，已上传公开镜像：

https://www.autodl.art/i/GoldenFishes/UniTAF/UniTAF

.

## 2. Inference

快速调用示例：

```python
# UV 环境
uv run python unitaf_train/UniTAF.py
# conda 环境
python unitaf_train/UniTAF.py
```

.

### 2.1 UniTAF流式调用

UniTAF模型的推理入口于 `unitaf_train/UniTAF.py` 中：

其中流式推理入口函数为：
`UniTextAudioFaceModel.indextts2_unitalker_stream_inference()`

流式推理返回：

```python
yield {
    "sr": sr,
    "wav": wav_chunk,
    "fps": 25,
    "motion": motion_vertices,  # (B, T_new, V, 3) or None
}
```

需要在外部接收并额外保存。

.

非流式推理入口函数为：
`UniTextAudioFaceModel.indextts2_unitalker_inference()`

非流式推理直接在函数内部保存音频，表情与渲染的适配

.

### 2.2 UniTAF情感控制

UniTAF沿用与IndexTTS2相同的情感调用方式

在以上推理入口 `indextts2_unitalker_stream_inference()` 与 `indextts2_unitalker_inference()` 传入以下参数即可进行可控情感生成。

.

| 参数               | 类型        | 说明                                                         | 使用场景                                                     |
| ------------------ | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `spk_audio_prompt` | str         | 说话人参考音频文件路径，用于声音克隆                         | 必填，通常是一个 WAV 文件，用来指定声音特征                  |
| `text`             | str         | 合成语音的文本内容                                           | 必填，可支持中文、英文或拼音标注                             |
| `emo_audio_prompt` | str         | 情感参考音频文件路径                                         | 可选，用来控制合成语音的情绪；配合 `emo_alpha` 调节强度      |
| `emo_alpha`        | float       | 情感参考音频对合成的影响程度，范围 0.0–1.0                   | 当 `emo_audio_prompt` 或 `use_emo_text` 被使用时，调节情感强度 |
| `emo_vector`       | List[float] | 8 维浮点向量指定情绪强度 `[开心、愤怒、悲伤、害怕、厌恶、忧郁、惊讶、平静]` | 可直接控制语音情绪，不依赖参考音频；范围 0.0–1.0；`use_random=False` 时可保证声音克隆精度 |
| `use_random`       | bool        | 推理时是否引入随机性                                         | 随机采样可增加自然感，但可能降低声音克隆精度                 |
| `use_emo_text`     | bool        | 是否根据文本脚本自动推断情感向量                             | True 时，文本内容会被模型转换为情绪向量                      |
| `emo_text`         | str         | 文本情感描述                                                 | 配合 `use_emo_text=True` 使用，可直接指定文本情绪，模型会自动转换为情感向量 |

**参考示例**

> 示例使用非流式生成，流式生成输入参数略有不同（流式生成不在函数内保存文件故而不需要传入输出路径）。

1. 使用一个参考文件合成新语音

```python
UniTextAudioFaceModel.indextts2_unitalker_inference(
	spk_audio_prompt='examples/voice_01.wav',
    text="Translate for me, what is a surprise!"
    tts_output_path="outputs/UniTAF_output.wav",
    a2f_output_path="outputs/UniTAF_output.npz",
)
```

2. 使用一个独立的情感参考音频文件来调节语音合成

```python
UniTextAudioFaceModel.indextts2_unitalker_inference(
	spk_audio_prompt='examples/voice_01.wav',
    text="Translate for me, what is a surprise!"
    tts_output_path="outputs/UniTAF_output.wav",
    a2f_output_path="outputs/UniTAF_output.npz",
    # 用于指定情感的参考音频
    emo_audio_prompt="examples/emo_sad.wav",
    # 当指定情感参考音频文件时，你可以选择设置 emo_alpha 来调整对输出的影响程度。
    emo_alpha=0.9,  # 有效范围为 0.0 - 1.0，默认值为 1.0
)
```

3. 也可以省略情感参考音频，改用一个8浮点表，指定每种情绪的强度，顺序如下：

```python
[开心、  愤怒、  悲伤、  害怕、    厌恶、       忧郁、       惊讶、     平静]
[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
```

```python
UniTextAudioFaceModel.indextts2_unitalker_inference(
	spk_audio_prompt='examples/voice_01.wav',
    text="Translate for me, what is a surprise!"
    tts_output_path="outputs/UniTAF_output.wav",
    a2f_output_path="outputs/UniTAF_output.npz",
    # 使用显示的情感向量来控制
    emo_vector=[0, 0.2, 0, 0.2, 0, 0, 0.45, 0],
    use_random=False, # 还可以使用 use_random 参数在推理过程中引入随机性; 随机采样会降低语音合成的声音克隆精度.
)
```

4. 或者，也可以启用 use_emo_text 根据提供的文字脚本引导情绪。

```python
UniTextAudioFaceModel.indextts2_unitalker_inference(
	spk_audio_prompt='examples/voice_01.wav',
    text="Translate for me, what is a surprise!"
    tts_output_path="outputs/UniTAF_output.wav",
    a2f_output_path="outputs/UniTAF_output.npz",
    # 根据提供的文字脚本引导情绪
    use_emo_text=True,
    emo_alpha=0.6, # 该文字脚本会自动转换为情感向量。建议在使用文本情感模式时使用大约 0.6（或更低）的 emo_alpha，这样语音听起来更自然。
)
```

5. 也可以通过 emo_text 参数直接提供特定的文本情感描述。

你的情感文本随后会自动转换为情感向量。这样你就能分别控制文本脚本和文本情感描述：

```python
UniTextAudioFaceModel.indextts2_unitalker_inference(
	spk_audio_prompt='examples/voice_01.wav',
    text="Translate for me, what is a surprise!"
    tts_output_path="outputs/UniTAF_output.wav",
    a2f_output_path="outputs/UniTAF_output.npz",
    use_emo_text=True,
    emo_text="你吓死我了！你是鬼吗？"
    emo_alpha=0.6,
)
```

6. 支持IndexTTS的拼音功能，支持输入的文本精确标注发音

```python
text = "之前你做DE5很好，所以这一次也DEI3做DE2很好才XING2，如果这次目标完成得不错的话，我们就直接打DI1去银行取钱。"
```

.

### 2.3 其他脚本

单独IndexTTS2批量实验脚本于 `./inference.py`

UniTAF批量实验推理脚本于 `./unitaf_inference.py`

调用并接收UniTAF流式生成结果的示例脚本于 `./unitaf_utils/test_stream_UniTAF.py`

用于实时查看表情表示ARKit渲染的WebUI脚本于 `./unitaf_train_component/render_webui.py`

.

## 3. Train

训练脚本入口位于 `./unitaf_train/train_unitaf.py` ，配置好该脚本的参数后直接运行该脚本即可开始训练

.

### 3.1 UniTAF Dataset

UniTAF模型训练使用特定格式的数据，数据集示例可以从[GoldenFishes/UniTAF-Dataset: UniTextAudioFace Dataset, a dataset for model combined TTS and A2F](https://github.com/GoldenFishes/UniTAF-Dataset)下载得到。

该数据集由UniTalker数据集基础上修改得到，其包含文本-音频-表情数据对。数据集格式和预处理方法见数据集仓库的自述文件。

.

### 3.2 training config

配置文件十分重要！我们仅使用该一个配置文件来控制所有的参数与变量，不仅训练器 `UniTAFTrainer` 需要接收 `train_config`，推理时 `UniTextAudioFaceModel` 也需要接收相同格式的 `cfg` 。

其中UniTAFTrainer接收的配置字典是最完整的配置字典，其余的单纯推理类 或 Dataset类接收的配置字典均是其与自身相关部分的子集。完整的配置字典示例如下（参考`unitaf_train/train_unitaf.py`中的配置字典）：

```python
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
        "dataset_root_path": "/autodl-tmp/UniTAF-Dataset",  # 使用绝对路径
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
```

.

其中支持分别加载TTS部分、Projector部分与A2F部分的自定义权重：

```python
train_config = {
    ...
    "finetune_checkpoint": {
        # "tts_model": "...",
        "audio_feature_projector": "...",
        "a2f_model": "...",
    }
}
```

如果某个模块不加载自定义权重而使用官方权重（若audio_feature_projector无自定义权重则会初始化）则注释`train_config["finetune_checkpoint"]`中该模块的key即可。

.

其中支持分别训练tts和a2f的配置（我们为tts和a2f两个模块分别使用两个不同的优化器）：

```python
train_config = {
    "train_tts": False,
    "train_tts_lora": True,  # 仅 train_tts 时有效
    # 只要训练a2f,则必须训练audio feature projector. 故不在额外增加投影层是否训练的参数判断了
    "train_a2f": True,
    # 训练a2f时，是否只训练adapter部分（a2f.audio_feature_projector）
    "train_a2f_adapter_only": False,
    # 训练a2f时，排除adapter部分（a2f.audio_feature_projector）
    "train_except_a2f_adapter": False,
    #  只训练a2f中表情残差模型，冻结口型模型。目前a2f生成口型，expression model生成表情，最终合并成完整面部
    "train_a2f_expression_model": True,
}
```

.

其中在配置文件中指定数据集：

```python
train_config = {
	"dataset_config": {
        "dataset_root_path": "/autodl-tmp/UniTAF-Dataset",  # 使用绝对路径
        # 支持多数据集训练，对应unitaf_dataset_support_config中具体数据集
        "dataset_list": ["D13"],  # "D12" , "D13"情感控制
        # unitaf_dataset_support_config是经过数据集格式转换UniTAFDataset能够支持的数据集
        # （UniTalker本身还有一个 a2f/dataset/dataset_config 记录UniTalker Decoder支持的数据集，
        # 但我们只会以 unitaf_dataset_support_config 为准）
    },
}
```

数据集子集的配置会根据 `unitaf_train/unitaf_dataset_support_config.py` 获取。 `unitaf_dataset_support_config` 是经过数据集格式转换UniTAFDataset能够支持的数据集

> UniTalker本身还有一个 `a2f/dataset/dataset_config` 记录UniTalker Decoder支持的数据集，但我们只会以 `unitaf_dataset_support_config` 为准。

如果要新增自己的数据集，则需要：

1）先参考UniTAF Dataset制作标准[GoldenFishes/UniTAF-Dataset](https://github.com/GoldenFishes/UniTAF-Dataset)；

2）在 `unitaf_train/unitaf_dataset_support_config.py` 中新增子集的配置；

3）在传入的训练配置`train_config`中增加子集的名称，`train_config["dataset_config"]["dataset_list"]`

.

### 3.3 训练组件

我们简要介绍UniTAF相关训练组件：

```python
UniTAF/
├── unitaf_train
│   ├── train_unitaf.py  	# 这里实现UniTAF Trainer与训练脚本入口
│   ├── UniTAF.py   		# 模型加载器，这里区分以训练模式和推理模式加载模型，并实现推理逻辑
│   ├── unitaf_dataset.py  	# UniTAF Dataset类，负责数据集读取和预处理
│   └── unitaf_dataset_support_config.py  # 数据集配置
└── unitaf_train_component
    ├── audio_feature_projector.py  	# 将TTS的音频特征投影到A2F空间
    ├── expression_model_component.py	# 用于学习情感表情残差的组件
    ├── indextts2_inference_component.py	# IndexTTS2适配UniTAF的推理组件
    ├── indextts2_train_component.py	# IndexTTS2适配UniTAF的训练组件
    └── unitalker_decoder_component.py	# UniTAF中使用的UniTAlker Decoder组件
...
```

.

## 4. License

本项目在[IndexTTS2](https://github.com/index-tts/index-tts)与[UniTalker](https://github.com/X-niper/UniTalker)基础上构建，本项目作学术用途，本项目开源协议将遵守且沿用IndexTTS2与UniTalker的开源协议。

