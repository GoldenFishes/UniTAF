# UniTAF

 

## 1. Install

 

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

 

**3.确保安装包管理工具uv**

```
pip install -U uv
```

> *所有* *`uv`* *命令都会自动激活每个项目的**虚拟环境**。在运行* *`UV`* *命令之前 ，千万不要手动激活任何环境，因为这可能导致依赖冲突！*

 

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

 

**安装 ffmpeg**

```python
# Ubuntu / Debian
sudo apt update
sudo apt install -y ffmpeg
```

 

### B. 使用pip安装

示例环境为：python 3.11 ， torch 2.8.0+cu128

```Python
conda create -n unitaf python=3.11 -y
conda activate unitaf
```

 

**1.安装 pytorch**

只使用官方CUDA源

```Python
pip install torch==2.8.* torchaudio==2.8.* --index-url https://download.pytorch.org/whl/cu128 --no-deps
```

 

**2.安装其他依赖**

```Python
# 该requirements.txt从pyproject.toml中转换出来
pip install -r requirements.txt
```

> 这里如果有清华源，则会报错证书问题。请确保删除清华源，仅使用官方源。

 

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

 

**安装 ffmpeg**

```python
# Ubuntu / Debian
sudo apt update
sudo apt install -y ffmpeg
```

 

### C. 使用AutoDL镜像（推荐）

前往AutoDL镜像界面，已上传公开镜像：

https://www.autodl.art/i/GoldenFishes/UniTAF/UniTAF

 

## 2. Inference

快速调用示例：

```python
# UV 环境
uv run python unitaf_train/UniTAF.py
# conda 环境
python unitaf_train/UniTAF.py
```

 

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

 

非流式推理入口函数为：
`UniTextAudioFaceModel.indextts2_unitalker_inference()`

非流式推理直接在函数内部保存音频，表情与渲染的适配

 

### 2.2 UniTAF情感控制

UniTAF沿用与IndexTTS2相同的情感调用方式

在以上推理入口 `indextts2_unitalker_stream_inference()` 与 `indextts2_unitalker_inference()` 传入以下参数即可进行可控情感生成。

 

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

 

### 2.3 其他推理脚本

单独IndexTTS2批量实验脚本于 `./inference.py`

UniTAF批量实验推理脚本于 `./unitaf_inference.py`





## 3. Train







## 4. License

本项目在[IndexTTS2](https://github.com/index-tts/index-tts)与[UniTalker](https://github.com/X-niper/UniTalker)基础上构建，本项目作学术用途，本项目开源协议将遵守且沿用IndexTTS2与UniTalker的开源协议。



