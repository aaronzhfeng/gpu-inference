# gpu-inference

Minimal Docker setup for self-hosted LLM inference on RunPod (or any GPU cloud). Supports **vLLM** and **SGLang** as interchangeable backends, both exposing an OpenAI-compatible API.

## Quick Start

### Local / Any GPU Machine

```bash
docker build -t gpu-inference .
docker run --gpus all -p 8000:8000 \
  -v ./model-cache:/workspace/models \
  -e MODEL=Qwen/Qwen3.5-4B \
  -e ENGINE=vllm \
  gpu-inference
```

### RunPod

1. **Push image:**
   ```bash
   docker build -t youruser/gpu-inference .
   docker push youruser/gpu-inference
   ```

2. **Create RunPod template:**
   - Docker image: `youruser/gpu-inference`
   - Container disk: 20GB
   - Volume mount: `/workspace/models` (network volume for model cache)
   - Expose port: 8000

3. **Set env vars on the template:**
   ```
   ENGINE=vllm
   MODEL=Qwen/Qwen3.5-4B
   MAX_MODEL_LEN=4096
   GPU_MEMORY_UTILIZATION=0.9
   ```

4. **Start a pod** with the template. First boot downloads the model (~30s-2min depending on size). Subsequent boots use the cached model on the network volume (~30s to ready).

### Use It

The server exposes an OpenAI-compatible API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3.5-4B", "messages": [{"role": "user", "content": "Hello"}]}'
```

Point any OpenAI SDK client at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://<pod-ip>:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Configuration

All configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGINE` | `vllm` | `vllm` or `sglang` |
| `MODEL` | `Qwen/Qwen3.5-4B` | Any HuggingFace model repo |
| `MAX_MODEL_LEN` | `4096` | Max context length |
| `GPU_MEMORY_UTILIZATION` | `0.9` | Fraction of GPU memory to use |
| `PORT` | `8000` | API server port |
| `EXTRA_ARGS` | `` | Additional CLI args passed to engine |

See `models.yaml` for preset reference configs.

## Switching Engines

```bash
# vLLM
docker run --gpus all -e ENGINE=vllm -e MODEL=Qwen/Qwen3.5-4B ...

# SGLang
docker run --gpus all -e ENGINE=sglang -e MODEL=Qwen/Qwen3.5-4B ...
```

Both expose the same OpenAI-compatible `/v1/chat/completions` endpoint — no client changes needed.

## GPU Sizing

| Model | VRAM (BF16) | Recommended GPU |
|-------|-------------|-----------------|
| Qwen3.5-0.8B | ~1.6GB | Any |
| Qwen3.5-2B | ~4GB | RTX 3090 / T4 |
| Qwen3.5-4B | ~8GB | RTX 4090 / A10G |
| Qwen3.5-9B | ~18GB | RTX 4090 / A100 |
| Qwen2.5-7B | ~14GB | RTX 4090 / A100 |
