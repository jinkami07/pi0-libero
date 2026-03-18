FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# ── system deps ──────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        tmux \
        htop \
        unzip \
        ffmpeg \
        libegl1 \
        libgles2 \
        libgl1 \
        libopengl0 \
        libglib2.0-0 \
        libglfw3 \
        libglew-dev \
        ca-certificates \
        gosu \
        supervisor \
        # VNC + desktop
        tigervnc-standalone-server \
        tigervnc-common \
        xfce4 \
        xfce4-terminal \
        dbus-x11 \
        x11-utils \
        # noVNC web client
        novnc \
        websockify \
    && rm -rf /var/lib/apt/lists/*

# ── VNC env vars ──────────────────────────────────────────────────────────────
# PASSWORD : VNC password (default: sandbox)
# RESOLUTION: desktop resolution (default: 1280x800)
ENV RESOLUTION=1280x800

# ── entrypoint / scripts ──────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
COPY physical-ai-sandbox/src/03_pi0_libero_finetune.py /src/03_pi0_libero_finetune.py
RUN chmod +x /entrypoint.sh

# ── Miniforge (conda-forge ベース、ToS 不要) ──────────────────────────────────
ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && ${CONDA_DIR}/bin/conda clean -afy
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# ── pi0 conda env (Python 3.11) ───────────────────────────────────────────────
RUN conda create -n pi0 python=3.11 -y

# すべての pip コマンドを pi0 env 内で実行
SHELL ["conda", "run", "-n", "pi0", "/bin/bash", "-c"]

# ── 1. numpy pin ──────────────────────────────────────────────────────────────
RUN pip install --upgrade "numpy<2.0.0,>=1.22.4"

# ── 2. PyTorch (CUDA 12.1) ───────────────────────────────────────────────────
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ── 3. uv インストール + openpi (uv sync + editable install) ─────────────────
RUN pip install uv

WORKDIR /opt
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
        https://github.com/Physical-Intelligence/openpi.git

WORKDIR /opt/openpi
RUN GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT=/opt/conda/envs/pi0 uv sync \
    && GIT_LFS_SKIP_SMUDGE=1 uv pip install --python /opt/conda/envs/pi0/bin/python -e .

# ── 4. base deps (uv sync の後に入れて上書き防止) ────────────────────────────
RUN pip install --upgrade "numpy<2.0.0,>=1.22.4"

RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install \
        mujoco \
        mujoco-python-viewer \
        gymnasium \
        gym-aloha \
        "imageio[ffmpeg]" \
        websockets \
        jupyterlab \
        pytest

RUN pip install -U \
        "gym==0.26.2" \
        gym-notices \
        "robosuite==1.4.0" \
        bddl \
        "easydict>=1.10" \
        termcolor \
        yacs \
        matplotlib \
        Pillow \
        opencv-python-headless \
        imageio

# ── 5. LIBERO (editable install) ─────────────────────────────────────────────
WORKDIR /opt
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

WORKDIR /opt/LIBERO
RUN pip install -e .

# ── env vars ──────────────────────────────────────────────────────────────────
ENV LIBERO_ROOT=/opt/LIBERO
ENV PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"
ENV MUJOCO_GL=egl
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
ENV MPLBACKEND=Agg
ENV CONDA_DEFAULT_ENV=pi0

# 5900: TigerVNC  |  6080: noVNC (web browser)
EXPOSE 5900 6080

WORKDIR /workspace/physical-ai-sandbox
ENTRYPOINT ["/entrypoint.sh"]
