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

1. All config via env vars: `ENGINE`, `MODEL`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `PORT`, `ENABLE_THINKING`, `EXTRA_ARGS`
2. `setup-workspace.sh` bootstraps the full environment (system packages, Python venv, Claude Code)
3. `start.sh` checks if model weights are cached at `$HF_HOME` (`/workspace/models` by default)
4. Downloads from HuggingFace if not cached
5. Launches vLLM or SGLang based on `$ENGINE`
6. Both engines serve the same OpenAI-compatible API on `$PORT`
7. `ENABLE_THINKING` defaults to `false` — Qwen3.5 reasoning/`<think>` blocks are disabled. Set to `true` to enable.

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

## Quick Deploy: Qwen3.5-4B (No Reasoning)

### Prerequisites
- venv at `/workspace/.persist/venv` with `vllm` and `huggingface_hub[cli]` installed
- Model weights cached at `/workspace/models` (HF_HOME)
- GPU: RTX 4090 (24 GB) — model uses ~8.6 GB VRAM

### Launch Command

```bash
source /workspace/.persist/venv/bin/activate && \
export HF_HOME=/workspace/models && \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-4B \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --swap-space 2 \
    --port 8000 \
    --trust-remote-code \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

**Key flags:**
- `--default-chat-template-kwargs '{"enable_thinking": false}'` — disables Qwen3.5 reasoning/thinking mode so responses are direct (no `<think>` blocks)
- Do NOT use `--chat-template-kwargs` (invalid in vLLM 0.17.0; the correct flag is `--default-chat-template-kwargs`)

### Startup Time
- ~3 min on first launch (model load + torch.compile + CUDA graph capture)
- Subsequent launches faster if torch compile cache exists at `~/.cache/vllm/torch_compile_cache/`

### Verify It's Running

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
curl -s http://localhost:8000/health
```

### Connect to the Endpoint

| Field | Value |
|---|---|
| Base URL | `http://localhost:8000/v1` |
| Model name | `Qwen/Qwen3.5-4B` |
| Max context | 4,096 tokens |
| Max concurrency | ~61 parallel requests at full context |
| Auth | None (no API key required) |

**Chat completions:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-4B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

**Python (OpenAI SDK):**
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

### Using start.sh (Recommended)

`start.sh` now has native `ENABLE_THINKING` support (defaults to `false`):

```bash
# Default: no reasoning, Qwen3.5-4B, port 8000
bash /workspace/gpu-inference/start.sh

# With reasoning enabled
ENABLE_THINKING=true bash /workspace/gpu-inference/start.sh

# Different model
MODEL=Qwen/Qwen3.5-9B ENABLE_THINKING=true bash /workspace/gpu-inference/start.sh
```

## New Workspace Bootstrap (From Scratch)

### Quickest path (with Claude Code):

```bash
git clone <this-repo> /workspace/gpu-inference
cd /workspace/gpu-inference
npm install -g @anthropic-ai/claude-code
claude
# then type: /setup
```

The `/setup` slash command (`.claude/commands/setup.md`) handles everything:
1. Installs system packages (nvtop)
2. Creates Python venv with vllm at `/workspace/.persist/venv/`
3. Downloads Qwen3.5-4B model weights to `/workspace/models/`
4. Launches the inference server on port 8000 (thinking disabled)
5. Verifies health and reports connection info

### Manual path (without Claude Code):

```bash
bash /workspace/gpu-inference/setup-workspace.sh
```

Or step by step:
```bash
bash /workspace/gpu-inference/setup-workspace.sh --no-inference  # setup only
bash /workspace/gpu-inference/start.sh                           # launch server
```

### What persists across workspace resets (on /workspace disk):
- `/workspace/.persist/venv/` — Python venv with vllm
- `/workspace/.persist/claude-data/` — Claude Code binary
- `/workspace/.persist/claude-config/` — Claude credentials & settings
- `/workspace/models/` — HuggingFace model weights cache
- `/workspace/gpu-inference/` — this repo (scripts, config, docs)

### What does NOT persist (reinstalled by /setup or setup-workspace.sh):
- System packages (`nvtop`, etc.) — apt packages are ephemeral on pod restart
- PATH modifications in `.bashrc`

## Engine Choice: vLLM vs SGLang

**Use vLLM (default).** Benchmarked 2026-03-10 on RTX 4090 with Qwen3.5-4B, 128 max tokens:

| Concurrency | SGLang (tok/s) | vLLM (tok/s) | vLLM speedup |
|---|---|---|---|
| 1 | 90 | 95 | 1.1x |
| 8 | 530 | 615 | 1.2x |
| 16 | 974 | 1,157 | 1.2x |
| 32 | 970 | 1,969 | 2.0x |
| 64 | 1,147 | 2,951 | 2.6x |

vLLM is 1.2–2.6x faster, especially at high concurrency. Likely because Qwen3.5 is a **hybrid Mamba-attention architecture** (not pure transformer), and vLLM 0.17.0 has better torch.compile + CUDA graph support for it.

### SGLang caveats (if you still want to try it):
- SGLang and vLLM have **conflicting torch versions** (SGLang needs 2.9.1, vLLM needs 2.10.0). Installing both in one venv breaks one of them. Use separate venvs if needed.
- SGLang requires extra system deps: `libnuma1`, `nvidia-cudnn-cu12>=9.16`
- SGLang has no `--default-chat-template-kwargs` flag. To disable thinking, use `--chat-template` with the patched template at `chat_templates/qwen3.5-no-thinking.jinja`
- SGLang binds to `127.0.0.1` by default (not `0.0.0.0`); add `--host 0.0.0.0` for external access
- The 50 GB network volume quota is tight with both engines installed (~23 GB venv + 27 GB models)

## Development Notes

- No test suite. Validate by starting the container and hitting the `/v1/models` endpoint
- `EXTRA_ARGS` env var passes arbitrary CLI flags to the engine (e.g., `--quantization awq`)
- Benchmark script: `/workspace/benchmark.py` — async concurrency sweep against the running server
