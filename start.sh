#!/bin/bash
set -e

# Use persistent venv from /workspace
VENV="/workspace/.persist/venv"
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
else
    echo "ERROR: venv not found at $VENV"
    echo "Run: python -m venv $VENV && $VENV/bin/pip install vllm huggingface_hub[cli]"
    exit 1
fi

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
ENGINE="${ENGINE:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
PORT="${PORT:-8000}"
HF_HOME="${HF_HOME:-/workspace/models}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
QUANTIZATION="${QUANTIZATION:-}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

# Thinking: enabled by default for Qwen3.5-9B
# Set ENABLE_THINKING=false to disable <think> blocks (useful for 4B)
if [ "${ENABLE_THINKING}" = "false" ]; then
    THINKING_ARGS="--default-chat-template-kwargs {\"enable_thinking\":false}"
else
    THINKING_ARGS=""
fi

# Enforce eager: required for Qwen3.5-9B on RTX 4090 (CUDA graph capture bug)
# Set ENFORCE_EAGER=false for smaller models like 4B that work with CUDA graphs
if [ "${ENFORCE_EAGER}" = "true" ]; then
    EAGER_ARGS="--enforce-eager"
else
    EAGER_ARGS=""
fi

# Quantization: awq, gptq, fp8, etc. Leave empty for no quantization (bf16)
if [ -n "${QUANTIZATION}" ]; then
    QUANT_ARGS="--quantization ${QUANTIZATION}"
else
    QUANT_ARGS=""
fi

# Tensor parallelism: for multi-GPU setups
if [ "${TENSOR_PARALLEL}" -gt 1 ] 2>/dev/null; then
    TP_ARGS="--tensor-parallel-size ${TENSOR_PARALLEL}"
else
    TP_ARGS=""
fi

export HF_HOME

echo "==========================================="
echo " gpu-inference"
echo "==========================================="
echo " Engine:    ${ENGINE}"
echo " Model:     ${MODEL}"
echo " Max len:   ${MAX_MODEL_LEN}"
echo " GPU util:  ${GPU_MEMORY_UTILIZATION}"
echo " Port:      ${PORT}"
echo " Thinking:  ${ENABLE_THINKING}"
echo " Eager:     ${ENFORCE_EAGER}"
echo " Quant:     ${QUANTIZATION:-none}"
echo " TP size:   ${TENSOR_PARALLEL}"
echo " Cache dir: ${HF_HOME}"
echo "==========================================="

# Pre-download model if not cached
echo "[1/2] Checking model cache..."
if python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}')" 2>/dev/null; then
    echo "      Model ready."
else
    echo "      Downloading ${MODEL}..."
    huggingface-cli download "${MODEL}"
    echo "      Download complete."
fi

# Launch engine
echo "[2/2] Starting ${ENGINE} on port ${PORT}..."

if [ "${ENGINE}" = "vllm" ]; then
    exec python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --port "${PORT}" \
        --trust-remote-code \
        ${THINKING_ARGS} \
        ${EAGER_ARGS} \
        ${QUANT_ARGS} \
        ${TP_ARGS} \
        ${EXTRA_ARGS}

elif [ "${ENGINE}" = "sglang" ]; then
    exec python -m sglang.launch_server \
        --model-path "${MODEL}" \
        --context-length "${MAX_MODEL_LEN}" \
        --mem-fraction-static "${GPU_MEMORY_UTILIZATION}" \
        --port "${PORT}" \
        --trust-remote-code \
        ${EXTRA_ARGS}

else
    echo "ERROR: Unknown engine '${ENGINE}'. Use 'vllm' or 'sglang'."
    exit 1
fi
