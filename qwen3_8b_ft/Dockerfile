# ==============================================
# 第一阶段：基础依赖安装（无 CUDA 编译，仅装系统/基础 Python 依赖）
# ==============================================
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel AS base

# 安装系统依赖（Git、编译工具等，与手动安装一致）
RUN apt-get update --fix-missing || true && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    make \
    libopenblas-dev \
    && apt-get clean || true \
    && rm -rf /var/cache/apt/archives/* || true \
    && rm -rf /var/lib/apt/lists/* || true

# 禁用 python 进度条、版本检查、缓存
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_PROGRESS_BAR=off

# 升级 pip 并安装基础 Python 依赖（避免后续安装冲突）
RUN pip install --upgrade pip \
    && pip install \
    numpy \
    setuptools \
    wheel \
    pyproject-hooks \
    huggingface-hub \
    datasets \
    peft \
    trl

# 修复 CUDA 软链接（提前配置，避免安装时路径识别错误）
RUN ln -s /usr/local/cuda-11.8 /usr/local/cuda

# 工作目录
WORKDIR /workspace

# ==============================================
# 第二阶段：在 GPU 运行时上下文安装 Axolotl+DeepSpeed（关键步骤）
# ==============================================
# 1. 先构建基础镜像（第一阶段）
# 2. 启动临时容器（带 GPU 上下文），执行安装命令
# 3. 将安装后的依赖目录复制到最终镜像

# 构建基础镜像（需先执行：docker build --target base -t axolotl-base .）
# 以下步骤通过脚本封装，Dockerfile 内通过多阶段复制实现

# 最终镜像（基于基础镜像，复制运行时安装的依赖）
FROM base AS final

# 从临时运行容器中复制安装好的 Axolotl+DeepSpeed 依赖（核心）
# 注：实际构建时，需先通过脚本启动临时容器执行安装，再复制依赖，此处通过 Docker 多阶段+本地挂载实现
# 简化方案：直接在最终镜像中通过“模拟运行时”命令安装（依赖 Docker 支持 GPU 构建，或宿主机已配置 nvidia-container-runtime）
# 安装 axolotl 和 deepspeed
ENV OMP_NUM_THREADS=1
RUN pip install --no-cache-dir packaging setuptools wheel ninja \
    && pip install --no-cache-dir axolotl[deepspeed]

# # 验证安装（可选，构建时可注释，避免构建时间过长）
RUN python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())" \
    && python -c "import deepspeed; print('DeepSpeed version:', deepspeed.__version__)" \
    && python -c "import axolotl; print('Axolotl version:', axolotl.__version__)"

WORKDIR /workspace

# 启动命令（默认进入 bash，可修改为直接启动训练脚本）
CMD ["/bin/bash"]
