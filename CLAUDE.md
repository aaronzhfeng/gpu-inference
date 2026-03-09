# CLAUDE.md

## Project Overview

`gpu-inference` is a minimal Docker setup for self-hosted LLM inference on GPU clouds (RunPod, Vast.ai, or local). Supports **vLLM** and **SGLang** as interchangeable backends, both exposing an OpenAI-compatible `/v1/chat/completions` API.

Primary use case: serving Qwen3.5 small models (0.8B–9B) for high-concurrency batch experiments, replacing OpenRouter API calls.

## Structure

```
Dockerfile      # CUDA 12.4 + vLLM + SGLang, single image
start.sh        # Entrypoint: cache check → model download → launch engine
models.yaml     # Reference presets (model sizes, VRAM estimates)
.env.example    # Environment variable template
```

## How It Works

1. All config via env vars: `ENGINE`, `MODEL`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `PORT`, `EXTRA_ARGS`
2. `start.sh` checks if model weights are cached at `$HF_HOME` (`/workspace/models` by default)
3. Downloads from HuggingFace if not cached
4. Launches vLLM or SGLang based on `$ENGINE`
5. Both engines serve the same OpenAI-compatible API on `$PORT`

## RunPod Deployment

- Use a **network volume** mounted at `/workspace` so model weights persist across pod restarts
- Recommended GPU: **RTX 4090** ($0.44/hr) for Qwen3.5-4B/9B
- First boot: ~2min (model download). Subsequent boots: ~30s (cached)
- Build and push the Docker image, create a RunPod template pointing to it

## Integration with RRMC

This server is consumed by the [RRMC](https://github.com/aaronzhfeng/RRMC) project:
- RRMC's `configs/providers/selfhosted.yaml` points at this server's URL
- Use `--provider selfhosted --base_url http://<pod-ip>:8000/v1` when running RRMC experiments
- The OpenAI SDK client in RRMC's `LLMWrapper` works identically with vLLM/SGLang endpoints

## Development Notes

- No test suite. Validate by starting the container and hitting the `/v1/models` endpoint
- vLLM and SGLang are installed in the same image; if dependency conflicts arise, isolate with separate conda envs
- `EXTRA_ARGS` env var passes arbitrary CLI flags to the engine (e.g., `--quantization awq`)
