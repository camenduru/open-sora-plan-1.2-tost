FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod && \
    pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 torchtext==0.16.0 torchdata==0.7.0 --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install https://download.pytorch.org/whl/cu121/xformers-0.0.22.post7-cp310-cp310-manylinux2014_x86_64.whl && \
    git clone -b v1.2.1 https://github.com/camenduru/Open-Sora-Plan /content/Open-Sora-Plan && cd /content/Open-Sora-Plan && pip install -e .

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/mt5-xxl/raw/main/config.json -d /content/mt5-xxl -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/mt5-xxl/raw/main/generation_config.json -d /content/mt5-xxl -o generation_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/mt5-xxl/raw/main/special_tokens_map.json -d /content/mt5-xxl -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/mt5-xxl/resolve/main/spiece.model -d /content/mt5-xxl -o spiece.model && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/mt5-xxl/resolve/main/pytorch_model.bin -d /content/mt5-xxl -o pytorch_model.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/mt5-xxl/raw/main/tokenizer_config.json -d /content/mt5-xxl -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/raw/main/93x720p/config.json -d /content/osp/video -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/resolve/main/93x720p/diffusion_pytorch_model.safetensors -d /content/osp/video -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/resolve/main/vae/checkpoint.ckpt -d /content/osp/vae -o checkpoint.ckpt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/raw/main/vae/config.json -d /content/osp/vae -o config.json

COPY ./worker_runpod.py /content/Open-Sora-Plan/worker_runpod.py
WORKDIR /content/Open-Sora-Plan
CMD python worker_runpod.py