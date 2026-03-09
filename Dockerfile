FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/models
ENV MODEL=Qwen/Qwen3.5-4B
ENV ENGINE=vllm
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTILIZATION=0.9
ENV PORT=8000
ENV EXTRA_ARGS=""

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git curl wget && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install vLLM (includes torch + CUDA)
RUN pip install vllm --no-cache-dir

# Install SGLang + FlashInfer
RUN pip install "sglang[all]" --no-cache-dir && \
    pip install flashinfer-python --no-cache-dir || true

# Install huggingface-cli for model downloads
RUN pip install huggingface_hub[cli] --no-cache-dir

# Copy scripts
COPY start.sh /app/start.sh
COPY models.yaml /app/models.yaml
RUN chmod +x /app/start.sh

WORKDIR /app
EXPOSE ${PORT}

ENTRYPOINT ["/app/start.sh"]
