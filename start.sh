#!/bin/bash
set -e

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
ENGINE="${ENGINE:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
SWAP_SPACE="${SWAP_SPACE:-2}"
PORT="${PORT:-8000}"
HF_HOME="${HF_HOME:-/workspace/models}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export HF_HOME

echo "==========================================="
echo " gpu-inference"
echo "==========================================="
echo " Engine:    ${ENGINE}"
echo " Model:     ${MODEL}"
echo " Max len:   ${MAX_MODEL_LEN}"
echo " GPU util:  ${GPU_MEMORY_UTILIZATION}"
echo " Port:      ${PORT}"
echo " Swap space: ${SWAP_SPACE}GB"
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
        --swap-space "${SWAP_SPACE}" \
        --port "${PORT}" \
        --trust-remote-code \
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
