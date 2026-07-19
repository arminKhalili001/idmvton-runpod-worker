# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS system
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface IDM_VTON_PATH=/opt/IDM-VTON \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git ca-certificates libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 build-essential ninja-build && \
    rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3.10 /usr/local/bin/python

FROM system AS official-source
ARG IDM_VTON_COMMIT=0d5f3ec2d737487a9bb24e4100936ad254780383
RUN git clone https://github.com/yisol/IDM-VTON.git /opt/IDM-VTON && \
    git -C /opt/IDM-VTON checkout --detach ${IDM_VTON_COMMIT} && \
    test "$(git -C /opt/IDM-VTON rev-parse HEAD)" = "${IDM_VTON_COMMIT}" && \
    rm -rf /opt/IDM-VTON/.git

FROM official-source AS python-dependencies
WORKDIR /worker
COPY requirements.txt .
RUN python -m pip install --upgrade pip==23.3.1 setuptools==69.0.3 wheel==0.42.0 && \
    python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
      torch==2.0.1+cu118 torchvision==0.15.2+cu118 && \
    python -m pip install -r requirements.txt && \
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
RUN python -c "import torch,torchvision,diffusers,transformers,runpod; print('torch',torch.__version__); print('torchvision',torchvision.__version__); print('diffusers',diffusers.__version__); print('transformers',transformers.__version__); print('runpod',runpod.__version__)"

FROM python-dependencies AS worker
WORKDIR /worker
COPY handler.py model_loader.py inference.py image_utils.py schemas.py ./
RUN mkdir -p /models/huggingface && chmod 777 /models/huggingface
CMD ["python", "-u", "handler.py"]
