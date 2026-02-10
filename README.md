# UniTAF



## 1. Install



以下三种安装方式任选其一

### A. 使用UV安装



1.克隆仓库

```
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

2.确保安装包管理工具uv

```
pip install -U uv
```



> *所有* *`uv`* *命令都会自动激活每个项目的**虚拟环境**。在运行* *`UV`* *命令之前 ，千万不要手动激活任何环境，因为这可能导致依赖冲突！*

3.安装必须依赖

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

4.通过uv工具

